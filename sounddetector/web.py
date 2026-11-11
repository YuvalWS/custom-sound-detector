"""Labeling UI and its JSON API, on the standard library's threaded HTTP server."""

from __future__ import annotations

import base64
import hmac
import json
import logging
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import SAMPLE_RATE
from .audio import decode_file
from .config import KNOCK, NOT_KNOCK, UNSURE, valid_label
from .events import rms_dbfs, segment

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 200 * 1024 * 1024


class Jobs:
    """One background job at a time (training, rescoring), with a log the UI polls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = None
        self.log: list[str] = []
        self.error = None
        self.finished_at = None

    def say(self, msg: str) -> None:
        log.info(msg)
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        del self.log[:-300]

    def start(self, name: str, fn) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running, self.log, self.error = name, [], None

        def run():
            try:
                fn(self.say)
            except Exception as exc:  # reported to the UI
                self.error = str(exc)
                self.say(f"failed: {exc}")
                log.debug(traceback.format_exc())
            finally:
                self.running = None
                self.finished_at = time.time()

        threading.Thread(target=run, daemon=True).start()
        return True

    def state(self) -> dict:
        return {"running": self.running, "log": self.log[-80:], "error": self.error, "finished_at": self.finished_at}


class Server:
    def __init__(self, cfg: dict, store, app=None):
        self.cfg = cfg
        self.store = store
        self.app = app  # None when the UI runs on its own (`sound-detector ui`)
        self.jobs = Jobs()
        self._yamnet = None
        self._yamnet_lock = threading.Lock()

    # A second YAMNet instance for imports and training: interpreters are not thread-safe.
    def yamnet(self):
        if self._yamnet is None:
            from .yamnet import YamNet

            self._yamnet = YamNet(self.cfg["yamnet_model"])
        return self._yamnet

    def models(self) -> dict:
        if self.app:
            return dict(self.app.models)
        from .classifier import load_targets

        det = self.cfg["detection"]
        return load_targets(self.cfg["data_dir"], det["targets"], self.yamnet().sha256, det.get("thresholds"))

    # ---- API -----------------------------------------------------------------------------
    def status(self) -> dict:
        reports = {}
        for p in (Path(self.cfg["data_dir"]) / "reports").glob("*.json"):
            try:
                r = json.loads(p.read_text())
                reports[r["target"]] = {k: r[k] for k in ("trained_at", "threshold", "cv", "clips",
                                                          "positive_clips", "negative_clips")}
            except (OSError, ValueError, KeyError):
                pass
        return {
            "service": self.app.status() if self.app else {"source": self.cfg["source"]["name"], "connected": None},
            "stats": self.store.stats(),
            "disk_mb": round(self.store.disk_mb(), 1),
            "categories": self.store.categories(),
            "fixed_labels": [KNOCK, NOT_KNOCK, UNSURE],
            "targets": self.cfg["detection"]["targets"],
            "reports": reports,
            "job": self.jobs.state(),
        }

    def list_clips(self, q: dict) -> dict:
        label = q.get("label", "unlabeled")
        clips = self.store.all()
        if label == "unlabeled":
            clips = [c for c in clips if not c.get("label")]
        elif label == "labeled":
            clips = [c for c in clips if c.get("label")]
        elif label == "negative":
            clips = [c for c in clips if c.get("label") and c["label"] not in (KNOCK, UNSURE)]
        elif label != "all":
            clips = [c for c in clips if c.get("label") == label]
        if q.get("detected") == "1":
            clips = [c for c in clips if c.get("detections")]
        if q.get("source") == "import":
            clips = [c for c in clips if c.get("imported_from")]
        if q.get("day"):
            clips = [c for c in clips if c.get("day") == q["day"]]
        if q.get("ids"):
            wanted = set(q["ids"].split(","))
            clips = [c for c in clips if c["id"] in wanted]
        sort = q.get("sort", "newest")
        target = q.get("target", KNOCK)
        if sort == "oldest":
            clips.sort(key=lambda c: c["recorded_at"])
        elif sort == "score":
            clips.sort(key=lambda c: -(c.get("scores") or {}).get(target, -1))
        elif sort == "loudest":
            clips.sort(key=lambda c: -(c.get("peak_dbfs") or -120))
        else:
            clips.sort(key=lambda c: c["recorded_at"], reverse=True)
        total = len(clips)
        offset = int(q.get("offset", 0))
        limit = min(int(q.get("limit", 200)), 2000)
        return {"total": total, "clips": clips[offset:offset + limit]}

    def import_audio(self, body: bytes, name: str, label: str | None, split: bool) -> dict:
        if label and not valid_label(label):
            raise ValueError("invalid label")
        suffix = Path(name).suffix[:10] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(body)
            tmp.flush()
            samples = decode_file(tmp.name)
        if not len(samples):
            raise ValueError("no audio in the file")
        coll = self.cfg["collection"]
        if split:
            spans = [(e.start, e.end, e.onset, e.peak_dbfs, e.background_dbfs) for e in segment(samples, coll)]
        else:
            spans = []
        if not spans:
            end = min(len(samples), int(30 * SAMPLE_RATE))
            frames = samples[: end - end % 800].astype(np.float64).reshape(-1, 800)  # 50 ms
            peak = round(max(rms_dbfs(f) for f in frames), 1) if len(frames) else None
            spans = [(0, end, 0, peak, None)]
        saved = []
        models = self.models()
        with self._yamnet_lock:
            yam = self.yamnet()
            from .app import describe_clip
            from .features import clip_features

            for start, end, onset, peak, bg in spans:
                meta = {"source": "import", "imported_from": Path(name).name[:120],
                        "import_offset": round(start / SAMPLE_RATE, 2),
                        "onset": round((onset - start) / SAMPLE_RATE, 2), "peak_dbfs": peak,
                        "background_dbfs": bg, "detections": []}
                clip = self.store.save(samples[start:end], meta, enforce_rate=False)
                if label:
                    self.store.set_label(clip["id"], label)
                feats = clip_features(self.store, yam, clip["id"])
                saved.append(self.store.update(clip["id"], **describe_clip(yam, models, feats)))
        return {"saved": len(saved), "clips": saved}

    def start_training(self, target: str) -> bool:
        if not valid_label(target):
            raise ValueError("invalid target")

        def job(say):
            from .classifier import load_targets
            from .train import rescore, train

            t = self.cfg["training"]
            with self._yamnet_lock:
                yam = self.yamnet()
                train(self.store, yam, target, self.cfg["data_dir"], C=float(t["C"]),
                      min_pos=int(t["min_positive_clips"]), min_neg=int(t["min_negative_clips"]), progress=say)
                det = self.cfg["detection"]
                targets = sorted(set(det["targets"]) | {target})
                models = load_targets(self.cfg["data_dir"], targets, yam.sha256, det.get("thresholds"))
                say("rescoring all clips with the new model")
                rescore(self.store, yam, models, progress=say)
            if self.app:
                self.app.reload_models(force=True)
            if target not in self.cfg["detection"]["targets"]:
                say(f"note: '{target}' is not in detection.targets, so it is not published to MQTT")

        return self.jobs.start(f"train {target}", job)

    # ---- HTTP plumbing -------------------------------------------------------------------
    def serve(self) -> ThreadingHTTPServer:
        server = self
        web = self.cfg["web"]
        expected = None
        if web.get("password"):
            token = base64.b64encode(f"{web.get('username') or 'admin'}:{web['password']}".encode()).decode()
            expected = f"Basic {token}"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                log.debug("web: " + fmt, *args)

            def _authorized(self) -> bool:
                if expected is None:
                    return True
                got = self.headers.get("Authorization", "")
                if hmac.compare_digest(got.encode(), expected.encode()):
                    return True
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="sound-detector"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False

            def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _json(self, obj, code: int = 200):
                self._send(code, json.dumps(obj).encode(), "application/json")

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_UPLOAD:
                    raise ValueError("upload too large")
                return self.rfile.read(n) if n else b""

            def do_GET(self):
                if not self._authorized():
                    return
                url = urlparse(self.path)
                q = {k: v[-1] for k, v in parse_qs(url.query).items()}
                parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
                try:
                    if not parts or parts == ["index.html"]:
                        self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                    elif parts == ["api", "status"]:
                        self._json(server.status())
                    elif parts == ["api", "clips"]:
                        self._json(server.list_clips(q))
                    elif len(parts) == 3 and parts[:2] == ["api", "clips"]:
                        self._json(server.store.get(parts[2]))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "audio":
                        wav = server.store.wav_path(parts[2])
                        self._send(200, wav.read_bytes(), "audio/wav")
                    elif parts == ["api", "report"]:
                        target = q.get("target", KNOCK)
                        if not valid_label(target):
                            raise ValueError("invalid target")
                        p = Path(server.cfg["data_dir"]) / "reports" / f"{target}.json"
                        self._json(json.loads(p.read_text()) if p.exists() else None)
                    else:
                        self._json({"error": "not found"}, 404)
                except KeyError:
                    self._json({"error": "no such clip"}, 404)
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                except Exception as exc:
                    log.exception("GET %s", self.path)
                    self._json({"error": str(exc)}, 500)

            do_HEAD = do_GET

            def do_POST(self):
                if not self._authorized():
                    return
                url = urlparse(self.path)
                q = {k: v[-1] for k, v in parse_qs(url.query).items()}
                parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
                try:
                    if parts == ["api", "import"]:
                        body = self._body()
                        self._json(server.import_audio(body, q.get("name", "upload"), q.get("label") or None,
                                                       q.get("split", "1") == "1"))
                        return
                    data = json.loads(self._body() or b"{}")
                    if len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "label":
                        self._json(server.store.set_label(parts[2], data.get("label") or None, data.get("note")))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "delete":
                        server.store.delete(parts[2])
                        self._json({"ok": True})
                    elif parts == ["api", "bulk-label"]:
                        label = data.get("label") or None
                        done = [server.store.set_label(i, label)["id"] for i in data.get("ids", [])]
                        self._json({"updated": len(done)})
                    elif parts == ["api", "categories"]:
                        self._json({"categories": server.store.add_category(str(data.get("name", "")).strip())})
                    elif parts == ["api", "train"]:
                        started = server.start_training(data.get("target", KNOCK))
                        self._json({"started": started, "job": server.jobs.state()}, 200 if started else 409)
                    else:
                        self._json({"error": "not found"}, 404)
                except KeyError:
                    self._json({"error": "no such clip"}, 404)
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                except Exception as exc:
                    log.exception("POST %s", self.path)
                    self._json({"error": str(exc)}, 500)

        httpd = ThreadingHTTPServer((web["host"], int(web["port"])), Handler)
        httpd.daemon_threads = True
        log.info("labeling UI on http://%s:%s/", web["host"], web["port"])
        return httpd
