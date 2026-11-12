"""A trained one-vs-rest classifier for one label: logistic regression on YAMNet embeddings.

Stored as plain JSON (a few kB), so the detector needs only numpy to run it.
"""

from __future__ import annotations

import json
import logging
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
