"""Label suggestions, to speed up labeling. Detection never uses them.

Two multinomial logistic regressions on the same YAMNet embeddings the knock model uses:

- **labels**: every label with at least MIN_CLIPS clips, `not_knock` included. Says whether
  a sound is a knock, a known kind of sound, or "not a knock".
- **kinds**: only the categories (no `knock`, no `not_knock`). `not_knock` is a catch-all:
  quiet noise, sounds nobody named, and named sounds labeled in bulk. So the labels model
  often prefers it even when the sound is plainly a door. The kinds model always answers
  "if it is a named sound, which one", and its answer is shown beside "not a knock".

A clip's suggestion averages its loud windows' probabilities. Training runs with every knock
training (seconds) and is cross-validated by clip, so the report says how often the first
suggestion, or one of the top three, is right, per label.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import KNOCK, NOT_KNOCK, UNSURE
from .features import clip_features, event_windows, training_windows
from .store import ClipStore

log = logging.getLogger(__name__)

MIN_CLIPS = 3
NAME = "label_suggester"
KIND_FIRST = 0.5  # a kind this sure goes before "not a knock"


def model_file(data_dir) -> Path:
    return Path(data_dir) / "models" / f"{NAME}.json"


def report_file(data_dir) -> Path:
    return Path(data_dir) / "reports" / f"{NAME}.json"


class Model:
    """One multinomial logistic regression: classes ranked by probability."""

    def __init__(self, data: dict):
        self.classes: list[str] = data["classes"]
        self.mean = np.asarray(data["mean"], dtype=np.float32)
        self.scale = np.asarray(data["scale"], dtype=np.float32)
        self.coef = np.asarray(data["coef"], dtype=np.float32)
        self.intercept = np.asarray(data["intercept"], dtype=np.float32)

    def probs(self, emb: np.ndarray) -> np.ndarray:
        z = ((np.atleast_2d(emb).astype(np.float32) - self.mean) / self.scale) @ self.coef.T + self.intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def clip_probs(self, loud_emb: np.ndarray) -> dict[str, float]:
        p = self.probs(loud_emb).mean(axis=0)
        return {c: float(v) for c, v in zip(self.classes, p, strict=True)}

    @classmethod
    def from_fit(cls, classes: list[str], params) -> Model:
        mean, scale, coef, b = params
        return cls({"classes": classes, "mean": mean, "scale": scale, "coef": coef, "intercept": b})

    def to_json(self) -> dict:
        return {"classes": self.classes, "mean": self.mean.round(6).tolist(), "scale": self.scale.round(6).tolist(),
                "coef": self.coef.round(5).tolist(), "intercept": self.intercept.round(6).tolist()}


class Suggester:
    def __init__(self, data: dict):
        self.data = data
        self.labels = Model(data)
        self.kinds = Model(data["kinds"]) if data.get("kinds") else None
        self.yamnet_sha256: str = data.get("yamnet_sha256", "")

    def suggest(self, feats: dict, top: int = 3) -> list:
        """[[label, probability], ...], best first, for a clip's loud windows.

        Without a sure knock, a sure kind of sound (door 80%) comes before "not a knock",
        even when the labels model prefers "not a knock": that one is a catch-all. A kind's
        probability is among the kinds only ("if it is a named sound, it is a door").
        """
        loud = feats["emb"][event_windows(feats["levels"])]
        pl = self.labels.clip_probs(loud)
        ranked = sorted(pl.items(), key=lambda kv: -kv[1])
        if not self.kinds:
            return [[c, round(p, 2)] for c, p in ranked[:top]]
        kinds = sorted(self.kinds.clip_probs(loud).items(), key=lambda kv: -kv[1])
        out: list = []
        if ranked[0][0] == KNOCK:
            out.append((KNOCK, pl[KNOCK]))
        if kinds[0][1] >= KIND_FIRST:
            out.append(kinds[0])
        for c, p in ranked:  # then the labels model's ranking, with each kind at its kinds probability
            out.append((c, dict(kinds).get(c, p)) if c not in (KNOCK, NOT_KNOCK) else (c, p))
        seen, result = set(), []
        for c, p in out:
            if c not in seen:
                seen.add(c)
                result.append([c, round(float(p), 2)])
        return result[:top]

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


def _train(data: list[tuple], counts: dict, C: float) -> tuple[Model, dict] | None:
    """Fit one model on (meta, windows, loud windows) items and cross-validate it by clip."""
    from sklearn.model_selection import StratifiedGroupKFold

    classes = sorted({m["label"] for m, _, _ in data if counts[m["label"]] >= MIN_CLIPS})
    if len(classes) < 2:
        return None
    index = {c: i for i, c in enumerate(classes)}
    data = [d for d in data if d[0]["label"] in index]
    X = np.concatenate([w for _, w, _ in data])
    y = np.concatenate([np.full(len(w), index[m["label"]]) for m, w, _ in data])
    groups = np.concatenate([np.full(len(w), i) for i, (_, w, _) in enumerate(data)])
    truth = np.array([index[m["label"]] for m, _, _ in data])
    n_splits = int(min(5, np.bincount(truth).min()))
    ranks = np.full(len(data), -1)
    if n_splits >= 2:
        for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y, groups):
            if len(np.unique(y[tr])) < len(classes):
                continue
            model = Model.from_fit(classes, _fit(X[tr], y[tr], C))
            for g in np.unique(groups[te]):
                p = model.probs(data[g][2]).mean(axis=0)
                ranks[g] = int(np.flatnonzero(np.argsort(p)[::-1] == truth[g])[0])
    scored = ranks >= 0

    def rate(mask, n):
        return round(float(np.mean(ranks[mask] < n)), 3) if mask.any() else None

    report = {"clips": len(data), "classes": classes, "folds": n_splits,
              "top1": rate(scored, 1), "top3": rate(scored, 3),
              "labels": [{"label": c, "clips": counts[c], "top1": rate(scored & (truth == index[c]), 1),
                          "top3": rate(scored & (truth == index[c]), 3)} for c in classes]}
    return Model.from_fit(classes, _fit(X, y, C)), report


def train_suggester(store: ClipStore, yamnet, data_dir, C: float = 1.0, progress=None) -> dict | None:
    """Train and save both models; returns the report, or None with fewer than two labels."""
    say = progress or (lambda msg: log.info(msg))
    counts: dict[str, int] = {}
    data = []
    for meta in store.all():
        if not meta.get("label") or meta["label"] == UNSURE:
            continue
        try:
            feats = clip_features(store, yamnet, meta["id"])
        except (OSError, ValueError, KeyError):
            continue
        counts[meta["label"]] = counts.get(meta["label"], 0) + 1
        data.append((meta, feats["emb"][training_windows(feats, meta.get("span"))].astype(np.float32),
                     feats["emb"][event_windows(feats["levels"])].astype(np.float32)))
    labels = _train(data, counts, C)
    if labels is None:
        say("label suggestions: need two labels with at least 3 clips each")
        return None
    kinds = _train([d for d in data if d[0]["label"] not in (KNOCK, NOT_KNOCK)], counts, C)
    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    model = {**labels[0].to_json(), "trained_at": trained_at, "yamnet_sha256": yamnet.sha256, "C": C,
             "kinds": kinds[0].to_json() if kinds else None}
    report = {"kind": NAME, "trained_at": trained_at, **labels[1], "kinds": kinds[1] if kinds else None,
              "left_out": sorted(lab for lab, n in counts.items() if n < MIN_CLIPS)}
    path = model_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model))
    tmp.replace(path)
    report_file(data_dir).parent.mkdir(parents=True, exist_ok=True)
    report_file(data_dir).write_text(json.dumps(report, indent=1))
    msg = (f"label suggestions: {len(labels[1]['classes'])} labels from {labels[1]['clips']} clips; first "
           f"suggestion right {round(100 * (labels[1]['top1'] or 0))}%, in the top 3 "
           f"{round(100 * (labels[1]['top3'] or 0))}%")
    if kinds:
        msg += (f"; kind of sound ({len(kinds[1]['classes'])} kinds) right "
                f"{round(100 * (kinds[1]['top1'] or 0))}%")
    say(msg)
    return report
