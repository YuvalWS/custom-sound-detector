"""The running service: audio in, sound events to the clip store, detections to MQTT."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE, __version__
from .audio import LiveSource
from .classifier import load_targets, model_path
from .config import stream_url
from .events import Event, Segmenter
from .features import clip_features, event_windows, suggest_span, sustained_score
from .store import ClipStore
from .yamnet import WINDOW, YamNet

log = logging.getLogger(__name__)


class Ring:
    """The last `seconds` of audio, addressed by absolute sample index."""

    def __init__(self, seconds: float = 30):
        self.buf = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)
        self.pos = 0  # absolute index of the next sample

    def append(self, x: np.ndarray) -> None:
        n, size = len(x), len(self.buf)
        if n >= size:
            self.buf[:] = x[-size:]
        else:
            i = self.pos % size
            first = min(n, size - i)
            self.buf[i:i + first] = x[:first]
            self.buf[:n - first] = x[first:]
        self.pos += n

    def get(self, start: int, end: int) -> np.ndarray:
        size = len(self.buf)
        start = max(start, self.pos - size, 0)
        end = min(end, self.pos)
        idx = np.arange(start, end) % size
        return self.buf[idx]


class TargetState:
    def __init__(self):
        self.on = False
        self.hits: deque[float] = deque()
        self.last_hit = -1e9
        self.last_score = 0.0
        self.run_start: float | None = None  # when the scores went above the threshold


def model_outputs(models: dict, feats: dict, samples: np.ndarray) -> dict:
    """Each model's clip score, and where in the clip it hears its sound (a suggested mark)."""
    emb = feats["emb"].astype(np.float32)
    scores, spans = {}, {}
    for t, m in models.items():
        p = m.predict(emb)
        scores[t] = round(sustained_score(p, feats["starts"], m.sustain), 3)
        span = suggest_span(p, feats["starts"], samples)
        if span:
            spans[t] = span
    return {"scores": scores, "suggested_spans": spans}


def describe_clip(yamnet: YamNet, models: dict, feats: dict, samples: np.ndarray) -> dict:
    sel = event_windows(feats["levels"])
    return {"hints": yamnet.top_labels(feats["scores"][sel].astype(np.float32)),
            **model_outputs(models, feats, samples)}


class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.source_name = cfg["source"]["name"]
        coll = dict(cfg["collection"])
        # Detection windows that still cover the event end within ~1 s of it; finishing
        # the event later means every hit is attached to the clip that holds it.
        coll["post_roll"] = max(float(coll["post_roll"]), WINDOW / SAMPLE_RATE + 0.05)
        self.segmenter = Segmenter.from_config(coll)
        self.store = ClipStore(cfg["data_dir"], coll["max_clips_per_hour"], coll["max_disk_mb"],
                               cfg.get("categories"))
        self.yamnet = YamNet(cfg["yamnet_model"])
        self.ring = Ring(30)
        self.models: dict = {}
        self.states: dict[str, TargetState] = {}
        self._models_mtime = None
        self._hits: deque = deque(maxlen=200)  # (absolute pos, target, score)
        self._since_hop = 0
        self.connected = False
        self.started = time.time()
        self.events_seen = 0
        self.publisher = None
        if cfg["mqtt"]["enabled"]:
            from .mqtt import Publisher

            self.publisher = Publisher(cfg["mqtt"], self.source_name, __version__)
        self._source = None
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self.reload_models(force=True)

    # ---- models ------------------------------------------------------------------------
    def _models_signature(self):
        sig = []
        for target in self.cfg["detection"]["targets"]:
            p = model_path(self.cfg["data_dir"], target)
            sig.append(p.stat().st_mtime if p.exists() else None)
        return tuple(sig)

    def reload_models(self, force: bool = False) -> None:
        sig = self._models_signature()
        if not force and sig == self._models_mtime:
            return
        self._models_mtime = sig
        det = self.cfg["detection"]
        models = load_targets(self.cfg["data_dir"], det["targets"], self.yamnet.sha256, det.get("thresholds"))
        with self._lock:
            self.models = models if det["enabled"] else {}
            self.states = {t: self.states.get(t, TargetState()) for t in self.models}
        log.info("detection models: %s", {t: m.threshold for t, m in self.models.items()} or "none (collecting only)")
        if self.publisher:
            self.publisher.set_targets({t: self._attrs(t) for t in self.models})

    def _attrs(self, target: str, **extra) -> dict:
        m = self.models.get(target)
        st = self.states.get(target)
        return {"score": round(st.last_score, 3) if st else 0.0, "threshold": m.threshold if m else None,
                "maybe_threshold": m.maybe_threshold if m else None,
                "model_trained_at": m.trained_at if m else None, **extra}

    # ---- audio loop --------------------------------------------------------------------
    def run(self) -> None:
        if self.publisher:
            self.publisher.start()
        src = self.cfg["source"]
        url = stream_url(src)
        if not url:
            # Keep the web UI up, so the stream can be set in Settings (saving restarts us).
            log.error("source.url is not set: set it in the web UI (Settings) or the config file")
            self._idle.wait()
            return
        self._source = LiveSource(url, src["stall_timeout"], src["reconnect_delay"],
                                  realtime=src.get("realtime", False), loop=src.get("loop", True))
        last_reload = time.time()
        for chunk in self._source:
            if chunk is None:
                self._disconnected()
                continue
            self.connected = True
            self.process(chunk)
            if time.time() - last_reload > 10:
                self.reload_models()
                last_reload = time.time()
        self._disconnected()

    def stop(self) -> None:
        self._idle.set()
        if self._source:
            self._source.stop()
        if self.publisher:
            self.publisher.stop()

    def _disconnected(self) -> None:
        self.connected = False
        self.segmenter.reset()
        self.segmenter.pos = self.ring.pos
        for target, st in list(self.states.items()):
            st.run_start = None
            self._set_state(target, False)

    def process(self, chunk: np.ndarray) -> None:
        self.ring.append(chunk)
        for event in self.segmenter.feed(chunk):
            self._event_done(event)
        self._detect(len(chunk))
        now = self.ring.pos / SAMPLE_RATE
        off_delay = float(self.cfg["detection"]["off_delay"])
        for target, st in list(self.states.items()):
            if st.on and now - st.last_hit >= off_delay:
                self._set_state(target, False)

    def _detect(self, n_new: int) -> None:
        if not self.models:
            return
        hop = max(160, int(float(self.cfg["detection"]["hop"]) * SAMPLE_RATE))
        self._since_hop += n_new
        if self._since_hop < hop:
            return
        self._since_hop %= hop
        seg = self.segmenter
        if not (seg.active or self.ring.pos - seg.last_active < WINDOW) or self.ring.pos < WINDOW:
            for st in self.states.values():  # a run of scores never spans a pause in scoring
                st.run_start = None
            return
        window = self.ring.get(self.ring.pos - WINDOW, self.ring.pos)
        emb, _ = self.yamnet.window(window)
        now = self.ring.pos / SAMPLE_RATE
        det = self.cfg["detection"]
        for target, model in self.models.items():
            p = float(model.predict(emb)[0])
            st = self.states[target]
            if p < model.threshold:
                st.run_start = None
                continue
            if st.run_start is None:
                st.run_start = now
            # A hit once the score has stayed above the threshold for the model's sustain time.
            if now - st.run_start < model.sustain - 1e-6:
                continue
            st.last_score = p
            st.last_hit = now
            st.hits.append(now)
            while st.hits and now - st.hits[0] > float(det["off_delay"]):
                st.hits.popleft()
            self._hits.append((self.ring.pos, target, round(p, 3)))
            log.info("%s hit, score %.2f", target, p)
            if not st.on and len(st.hits) >= int(det["min_hits"]):
                self._set_state(target, True)

    def _set_state(self, target: str, on: bool, **extra) -> None:
        st = self.states.get(target)
        if st is None or (st.on == on and not extra):
            return
        st.on = on
        if not on:
            st.hits.clear()
        log.info("%s %s", target, "ON" if on else "OFF")
        if self.publisher:
            self.publisher.publish_state(target, on, self._attrs(target, **extra))

    def _event_done(self, event: Event) -> None:
        self.events_seen += 1
        try:
            self.store.count_event()
        except OSError:
            log.exception("could not count the event")
        # A hit belongs to this clip when its window overlaps the sound itself.
        hits = [h for h in self._hits if h[0] > event.onset and h[0] - WINDOW < event.sound_end]
        save = self.cfg["collection"]["enabled"] or (hits and self.cfg["detection"]["save_clips"])
        if not save:
            return
        samples = self.ring.get(event.start, event.end)
        recorded_at = time.time() - (self.ring.pos - event.start) / SAMPLE_RATE
        meta = {
            "source": self.source_name,
            "recorded_at": recorded_at,
            "onset": round((event.onset - event.start) / SAMPLE_RATE, 2),
            "peak_dbfs": event.peak_dbfs,
            "background_dbfs": event.background_dbfs,
            "detections": [{"target": t, "score": s, "offset": round((pos - event.start) / SAMPLE_RATE, 2)}
                           for pos, t, s in hits],
        }
        try:
            saved = self.store.save(samples, meta, enforce_rate=not hits)
            if saved is None:
                return
            feats = clip_features(self.store, self.yamnet, saved["id"])
            saved = self.store.update(saved["id"], **describe_clip(self.yamnet, self.models, feats, samples))
            log.info("clip %s (%.1fs, peak %.0f dBFS) %s", saved["id"], saved["duration"], event.peak_dbfs,
                     ", ".join(f"{n} {s}" for n, s in saved["hints"]))
            if self.publisher:
                for target in {t for _, t, _ in hits}:
                    if target in self.models:
                        self.publisher.publish_state(target, self.states[target].on,
                                                     self._attrs(target, clip_id=saved["id"]))
                self._publish_clip(saved, {t for _, t, _ in hits})
        except Exception:  # a failed save must never stop the audio loop
            log.exception("could not save clip")

    def _publish_clip(self, clip: dict, detected: set) -> None:
        """Every saved clip, with each model's score: Home Assistant picks its own thresholds."""
        for target, m in self.models.items():
            score = (clip.get("scores") or {}).get(target)
            if score is None:
                continue
            self.publisher.publish_clip(target, {
                "clip_id": clip["id"],
                "score": score,
                "threshold": m.threshold,
                "maybe_threshold": m.maybe_threshold,
                # Whether the live detector turned the sensor on for this clip.
                "detected": target in detected,
                "recorded_at": clip.get("recorded_at"),
                "duration": clip.get("duration"),
                "peak_dbfs": clip.get("peak_dbfs"),
                "span": (clip.get("suggested_spans") or {}).get(target),
                # Relative to the web UI's address; add ?format=ogg for a Telegram voice note.
                "audio": f"api/clips/{clip['id']}/audio",
            })

    def status(self) -> dict:
        seg = self.segmenter
        return {
            "version": __version__,
            "source": self.source_name,
            "connected": self.connected,
            "level_dbfs": round(seg.level, 1),
            "background_dbfs": round(seg.background, 1) if seg.background is not None else None,
            "collecting": bool(self.cfg["collection"]["enabled"]),
            "events_seen": self.events_seen,
            "uptime_s": int(time.time() - self.started),
            "models": {t: {"threshold": m.threshold, "maybe_threshold": m.maybe_threshold, "sustain": m.sustain,
                           "trained_at": m.trained_at, "on": self.states[t].on} for t, m in self.models.items()},
            "yamnet": Path(self.yamnet.path).name,
        }
