"""Clip store: one WAV plus one JSON sidecar per clip, under <data_dir>/clips/<date>/.

The JSON sidecar is the source of truth for the label. An in-memory index of all
sidecars is kept for the UI; it is rebuilt from disk at startup.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .audio import read_wav, write_wav
from .config import DEFAULT_CATEGORIES, KNOCK, NOT_KNOCK, UNSURE, valid_label

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


class ClipStore:
    def __init__(self, data_dir, max_clips_per_hour: int = 120, max_disk_mb: float = 2000,
                 default_categories: list[str] | None = None):
        self.root = Path(data_dir)
        self.clips_dir = self.root / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.max_clips_per_hour = max_clips_per_hour
        self.max_disk_mb = max_disk_mb
        self.default_categories = list(default_categories or DEFAULT_CATEGORIES)
        self._lock = threading.RLock()
        self._index: dict[str, dict] = {}
        self._saved_times: list[float] = []
        self.dropped = 0  # clips not saved because of the hourly cap
        self._load_index()

    # ---- index -------------------------------------------------------------------------
    def _load_index(self) -> None:
        for meta_path in self.clips_dir.glob("*/*.json"):
            try:
                meta = json.loads(meta_path.read_text())
                self._index[meta["id"]] = meta
            except (OSError, ValueError, KeyError):
                log.warning("skipping unreadable %s", meta_path)
        log.info("clip store: %d clips", len(self._index))

    def _paths(self, clip_id: str) -> tuple[Path, Path]:
        if not ID_RE.match(clip_id) or clip_id not in self._index:
            raise KeyError(clip_id)
        day = self._index[clip_id]["day"]
        base = self.clips_dir / day / clip_id
        return base.with_suffix(".wav"), base.with_suffix(".json")

    def wav_path(self, clip_id: str) -> Path:
        with self._lock:
            return self._paths(clip_id)[0]

    def get(self, clip_id: str) -> dict:
        with self._lock:
            if clip_id not in self._index:
                raise KeyError(clip_id)
            return dict(self._index[clip_id])

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(m) for m in self._index.values()]

    def samples(self, clip_id: str) -> np.ndarray:
        return read_wav(self.wav_path(clip_id))

    def _write_meta(self, meta: dict) -> None:
        _, meta_path = self._paths(meta["id"])
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(meta_path)

    # ---- writing -----------------------------------------------------------------------
    def allow_new_clip(self) -> bool:
        now = time.time()
        self._saved_times = [t for t in self._saved_times if now - t < 3600]
        return len(self._saved_times) < self.max_clips_per_hour

    def save(self, samples: np.ndarray, meta: dict, enforce_rate: bool = True) -> dict | None:
        """Save a clip. meta must hold `source`; `recorded_at` (epoch s) defaults to now."""
        with self._lock:
            if enforce_rate and not self.allow_new_clip():
                self.dropped += 1
                return None
            recorded = float(meta.get("recorded_at") or time.time())
            dt = datetime.fromtimestamp(recorded).astimezone()
            source = re.sub(r"[^A-Za-z0-9_-]", "_", meta.get("source", "clip"))[:40]
            clip_id = f"{source}-{dt:%Y%m%d-%H%M%S}-{int(recorded * 1000) % 1000:03d}"
            n = 1
            while clip_id in self._index:
                clip_id = f"{clip_id.rsplit('~', 1)[0]}~{n}"
                n += 1
            day = f"{dt:%Y-%m-%d}"
            full = {
                "id": clip_id,
                "day": day,
                "recorded_at": dt.isoformat(timespec="milliseconds"),
                "duration": round(len(samples) / SAMPLE_RATE, 2),
                "label": None,
                "labeled_at": None,
                "note": "",
                **{k: v for k, v in meta.items() if k not in ("id", "day", "recorded_at")},
            }
            if full["label"] is not None and not valid_label(full["label"]):
                full["label"] = None
            (self.clips_dir / day).mkdir(parents=True, exist_ok=True)
            self._index[clip_id] = full
            wav, _ = self._paths(clip_id)
            write_wav(wav, samples)
            self._write_meta(full)
            if enforce_rate:
                self._saved_times.append(time.time())
            self.prune()
            return dict(full)

    def update(self, clip_id: str, **fields) -> dict:
        with self._lock:
            meta = self._index[clip_id]
            meta.update(fields)
            self._write_meta(meta)
            return dict(meta)

    def set_label(self, clip_id: str, label: str | None, note: str | None = None) -> dict:
        if label is not None and not valid_label(label):
            raise ValueError(f"invalid label {label!r}")
        fields = {"label": label, "labeled_at": _now_iso() if label else None}
        if note is not None:
            fields["note"] = str(note)[:500]
        if label and label not in (KNOCK, NOT_KNOCK, UNSURE):
            self.add_category(label)
        return self.update(clip_id, **fields)

    def delete(self, clip_id: str) -> None:
        with self._lock:
            wav, meta = self._paths(clip_id)
            for p in [wav, meta, *wav.parent.glob(f"{clip_id}.emb*.npz")]:
                p.unlink(missing_ok=True)
            del self._index[clip_id]

    # ---- categories --------------------------------------------------------------------
    @property
    def _categories_file(self) -> Path:
        return self.root / "categories.json"

    def categories(self) -> list[str]:
        cats = list(self.default_categories)
        if self._categories_file.exists():
            try:
                cats = json.loads(self._categories_file.read_text())
            except ValueError:
                pass
        with self._lock:
            used = {m["label"] for m in self._index.values() if m.get("label")}
        for label in sorted(used - set(cats) - {KNOCK, NOT_KNOCK, UNSURE}):
            cats.append(label)
        return cats

    def add_category(self, name: str) -> list[str]:
        if not valid_label(name) or name in (KNOCK, NOT_KNOCK, UNSURE):
            raise ValueError(f"invalid category {name!r}")
        cats = self.categories()
        if name not in cats:
            cats.append(name)
            self._categories_file.write_text(json.dumps(cats))
        return cats

    # ---- housekeeping ------------------------------------------------------------------
    def disk_mb(self) -> float:
        return sum(p.stat().st_size for p in self.clips_dir.glob("*/*") if p.is_file()) / 1e6

    def prune(self) -> int:
        """Delete the oldest unlabeled clips until under max_disk_mb. Labeled clips are kept."""
        if not self.max_disk_mb:
            return 0
        # Approximate: WAV size dominates; checked against the real total every call.
        size = self.disk_mb()
        if size <= self.max_disk_mb:
            return 0
        with self._lock:
            victims = sorted((m for m in self._index.values() if not m.get("label")),
                             key=lambda m: (bool(m.get("detections")), m["recorded_at"]))
            removed = 0
            for meta in victims:
                if size <= self.max_disk_mb * 0.95:
                    break
                wav, _ = self._paths(meta["id"])
                size -= (wav.stat().st_size if wav.exists() else 0) / 1e6
                self.delete(meta["id"])
                removed += 1
            if removed:
                log.info("pruned %d unlabeled clips to stay under %s MB", removed, self.max_disk_mb)
            for day in self.clips_dir.iterdir():
                if day.is_dir() and not any(day.iterdir()):
                    shutil.rmtree(day, ignore_errors=True)
            return removed

    def stats(self) -> dict:
        with self._lock:
            labels: dict[str, int] = {}
            unlabeled = detected = 0
            for m in self._index.values():
                if m.get("label"):
                    labels[m["label"]] = labels.get(m["label"], 0) + 1
                else:
                    unlabeled += 1
                if m.get("detections"):
                    detected += 1
            return {"total": len(self._index), "unlabeled": unlabeled, "detected": detected,
                    "labels": labels, "dropped_this_run": self.dropped}
