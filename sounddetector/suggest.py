"""Label suggestions: one model over every label, to speed up labeling.

A multinomial logistic regression on the same YAMNet embeddings the knock model uses,
trained on all labeled clips (`unsure` left out), one class per label with at least
MIN_CLIPS clips. A clip's suggestion is the average of its loud windows' probabilities,
the labels ranked. It only helps labeling: detection never uses it.

Training runs with every knock training (seconds), and is cross-validated by clip, so the
report says how often the top suggestion, or one of the top three, is right, per label.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import UNSURE
from .features import clip_features, event_windows, training_windows
from .store import ClipStore

log = logging.getLogger(__name__)

MIN_CLIPS = 3
NAME = "label_suggester"


def model_file(data_dir) -> Path:
    return Path(data_dir) / "models" / f"{NAME}.json"


def report_file(data_dir) -> Path:
    return Path(data_dir) / "reports" / f"{NAME}.json"


class Suggester:
    def __init__(self, data: dict):
        self.data = data
        self.classes: list[str] = data["classes"]
        self.mean = np.asarray(data["mean"], dtype=np.float32)
        self.scale = np.asarray(data["scale"], dtype=np.float32)
        self.coef = np.asarray(data["coef"], dtype=np.float32)
        self.intercept = np.asarray(data["intercept"], dtype=np.float32)
        self.yamnet_sha256: str = data.get("yamnet_sha256", "")

    def probs(self, emb: np.ndarray) -> np.ndarray:
        z = ((np.atleast_2d(emb).astype(np.float32) - self.mean) / self.scale) @ self.coef.T + self.intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def suggest(self, feats: dict, top: int = 3) -> list:
        """[[label, probability], ...], best first, for a clip's loud windows."""
        sel = event_windows(feats["levels"])
        p = self.probs(feats["emb"][sel]).mean(axis=0)
        order = np.argsort(p)[::-1][:top]
        return [[self.classes[i], round(float(p[i]), 2)] for i in order]

    @classmethod
    def from_fit(cls, classes: list[str], params) -> Suggester:
        mean, scale, coef, b = params
        return cls({"classes": classes, "mean": mean, "scale": scale, "coef": coef, "intercept": b})

    @classmethod
    def load(cls, data_dir, yamnet_sha256: str) -> Suggester | None:
        p = model_file(data_dir)
        if not p.exists():
            return None
        try:
            s = cls(json.loads(p.read_text()))
        except (OSError, ValueError, KeyError):
            log.exception("could not load %s", p)
            return None
        if s.yamnet_sha256 and s.yamnet_sha256 != yamnet_sha256:
            log.error("%s: trained on a different YAMNet file; retrain", p)
            return None
        return s


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression

    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit((X - mean) / scale, y)
    coef, b = clf.coef_, clf.intercept_
    if len(clf.classes_) == 2:  # one row for two classes: as a softmax, the same probabilities
        coef, b = np.vstack([-coef / 2, coef / 2]), np.concatenate([-b / 2, b / 2])
    return mean, scale, coef, b


def train_suggester(store: ClipStore, yamnet, data_dir, C: float = 1.0, progress=None) -> dict | None:
    """Train and save the suggester; returns its report, or None with fewer than two labels."""
    from sklearn.model_selection import StratifiedGroupKFold

    say = progress or (lambda msg: log.info(msg))
    clips = [m for m in store.all() if m.get("label") and m["label"] != UNSURE]
    counts: dict[str, int] = {}
    for m in clips:
        counts[m["label"]] = counts.get(m["label"], 0) + 1
    classes = sorted(lab for lab, n in counts.items() if n >= MIN_CLIPS)
    if len(classes) < 2:
        say("label suggestions: need two labels with at least 3 clips each")
        return None
    index = {c: i for i, c in enumerate(classes)}
    clips = [m for m in clips if m["label"] in index]
    X, y, groups, used, loud = [], [], [], [], []
    for meta in clips:
        try:
            feats = clip_features(store, yamnet, meta["id"])
        except (OSError, ValueError, KeyError):
            continue
        sel = training_windows(feats, meta.get("span"))
        X.append(feats["emb"][sel].astype(np.float32))
        y += [index[meta["label"]]] * len(sel)
        groups += [len(used)] * len(sel)
        used.append(meta)
        loud.append(feats["emb"][event_windows(feats["levels"])].astype(np.float32))
    X, y, groups = np.concatenate(X), np.array(y), np.array(groups)
    truth = np.array([index[m["label"]] for m in used])

    # Cross-validated: how often the suggestion would have been right on clips it never saw.
    n_splits = int(min(5, np.bincount(truth).min()))
    ranks = np.full(len(used), -1)
    if n_splits >= 2:
        for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y, groups):
            if len(np.unique(y[tr])) < len(classes):
                continue
            s = Suggester.from_fit(classes, _fit(X[tr], y[tr], C))
            for g in np.unique(groups[te]):
                p = s.probs(loud[g]).mean(axis=0)
                ranks[g] = int(np.flatnonzero(np.argsort(p)[::-1] == truth[g])[0])
    scored = ranks >= 0
    per_label = []
    for c in classes:
        m = scored & (truth == index[c])
        per_label.append({"label": c, "clips": counts[c],
                          "top1": round(float(np.mean(ranks[m] == 0)), 3) if m.any() else None,
                          "top3": round(float(np.mean(ranks[m] < 3)), 3) if m.any() else None})
    top1 = round(float(np.mean(ranks[scored] == 0)), 3) if scored.any() else None
    top3 = round(float(np.mean(ranks[scored] < 3)), 3) if scored.any() else None

    mean, scale, coef, b = _fit(X, y, C)
    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    data = {"classes": classes, "trained_at": trained_at, "yamnet_sha256": yamnet.sha256, "C": C,
            "mean": mean.round(6).tolist(), "scale": scale.round(6).tolist(),
            "coef": np.asarray(coef).round(5).tolist(), "intercept": np.asarray(b).round(6).tolist()}
    report = {"kind": NAME, "trained_at": trained_at, "clips": len(used), "classes": classes,
              "folds": n_splits, "top1": top1, "top3": top3, "labels": per_label,
              "left_out": sorted(lab for lab, n in counts.items() if n < MIN_CLIPS)}
    path = model_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    report_file(data_dir).parent.mkdir(parents=True, exist_ok=True)
    report_file(data_dir).write_text(json.dumps(report, indent=1))
    say(f"label suggestions: {len(classes)} labels from {len(used)} clips; top suggestion right "
        f"{round(100 * (top1 or 0))}% of the time, in the top 3 {round(100 * (top3 or 0))}%")
    return report
