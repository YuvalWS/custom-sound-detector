"""A trained one-vs-rest classifier for one label: logistic regression on YAMNet embeddings.

Stored as plain JSON (a few kB), so the detector needs only numpy to run it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class TargetModel:
    def __init__(self, data: dict):
        self.data = data
        self.target: str = data["target"]
        self.mean = np.asarray(data["mean"], dtype=np.float32)
        self.scale = np.asarray(data["scale"], dtype=np.float32)
        self.coef = np.asarray(data["coef"], dtype=np.float32)
        self.intercept = float(data["intercept"])
        self.threshold = float(data["threshold"])
        # The looser "maybe" threshold (the threshold itself for models trained before it).
        self.maybe_threshold = min(float(data.get("maybe_threshold", self.threshold)), self.threshold)
        # Seconds the score must stay above the threshold (0 for models trained before it).
        self.sustain = float(data.get("sustain", 0.0))
        self.yamnet_sha256: str = data.get("yamnet_sha256", "")
        self.trained_at: str = data.get("trained_at", "")

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(embeddings) - self.mean) / self.scale
        z = x @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    @classmethod
    def load(cls, path) -> TargetModel:
        return cls(json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(path)


def model_path(data_dir, target: str) -> Path:
    return Path(data_dir) / "models" / f"{target}.json"


def report_path(data_dir, target: str) -> Path:
    return Path(data_dir) / "reports" / f"{target}.json"


def history_dir(data_dir) -> Path:
    return Path(data_dir) / "models" / "history"


def _check_id(model_id: str) -> None:
    if not model_id or not all(c.isalnum() or c in "-_" for c in model_id):
        raise ValueError("invalid model id")


def archive_current(data_dir, target: str) -> str | None:
    """Move the active model (and its report) to the history; returns its id there."""
    path = model_path(data_dir, target)
    if not path.exists():
        return None
    history = history_dir(data_dir)
    history.mkdir(parents=True, exist_ok=True)
    model_id = f"{target}-{datetime.fromtimestamp(path.stat().st_mtime):%Y%m%d-%H%M%S}"
    base, n = model_id, 1
    while (history / f"{model_id}.json").exists():
        model_id, n = f"{base}-{n}", n + 1
    path.replace(history / f"{model_id}.json")
    rp = report_path(data_dir, target)
    if rp.exists():
        rp.replace(history / f"{model_id}.report.json")
    return model_id


def _summary(data: dict, report: dict | None) -> dict:
    s = data.get("summary") or {}
    out = {"trained_at": data.get("trained_at"), "threshold": data.get("threshold"),
           "maybe_threshold": data.get("maybe_threshold"), "sustain": data.get("sustain", 0.0),
           "clips": s.get("clips"), "positive_clips": s.get("positive_clips"), "cv": s.get("cv"),
           "has_report": report is not None, "yamnet_sha256": data.get("yamnet_sha256", "")}
    rule = (report or {}).get("threshold_rule") or s.get("threshold_rule") or {}
    out["false_alarms_per_week"] = rule.get("estimated_per_week")
    return out


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def list_models(data_dir, target: str) -> list[dict]:
    """The active model, then the previous ones, newest first."""
    out = []
    active = _read(model_path(data_dir, target))
    if active:
        out.append({"id": "active", "active": True, **_summary(active, _read(report_path(data_dir, target)))})
    history = history_dir(data_dir)
    rows = []
    for p in history.glob(f"{target}-*.json") if history.exists() else []:
        if p.name.endswith(".report.json"):
            continue
        data = _read(p)
        if data and data.get("target") == target:
            rows.append({"id": p.stem, "active": False,
                         **_summary(data, _read(p.with_name(f"{p.stem}.report.json")))})
    rows.sort(key=lambda r: r.get("trained_at") or "", reverse=True)
    return out + rows


def model_report(data_dir, target: str, model_id: str = "active") -> dict | None:
    if model_id == "active":
        return _read(report_path(data_dir, target))
    _check_id(model_id)
    return _read(history_dir(data_dir) / f"{model_id}.report.json")


def activate(data_dir, target: str, model_id: str, yamnet_sha256: str) -> str | None:
    """Make a previous model the active one; the active one goes to the history (its id is returned)."""
    _check_id(model_id)
    src = history_dir(data_dir) / f"{model_id}.json"
    data = _read(src)
    if not data or data.get("target") != target:
        raise KeyError(model_id)
    if data.get("yamnet_sha256") and data["yamnet_sha256"] != yamnet_sha256:
        raise ValueError("that model was trained on a different YAMNet file")
    archived = archive_current(data_dir, target)
    src.replace(model_path(data_dir, target))
    rep = src.with_name(f"{model_id}.report.json")
    if rep.exists():
        report_path(data_dir, target).parent.mkdir(parents=True, exist_ok=True)
        rep.replace(report_path(data_dir, target))
    # A new modification time, so a running detector reloads it.
    model_path(data_dir, target).touch()
    return archived


def load_targets(data_dir, targets, yamnet_sha256: str, thresholds: dict | None = None) -> dict:
    """Load the models of the configured targets that exist and match the YAMNet file."""
    models = {}
    for target in targets:
        path = model_path(data_dir, target)
        if not path.exists():
            continue
        model = TargetModel.load(path)
        if model.yamnet_sha256 and model.yamnet_sha256 != yamnet_sha256:
            log.error("%s: trained on a different YAMNet file; retrain it", path)
            continue
        if thresholds and target in thresholds:
            model.threshold = float(thresholds[target])
            model.maybe_threshold = min(model.maybe_threshold, model.threshold)
        models[target] = model
    return models
