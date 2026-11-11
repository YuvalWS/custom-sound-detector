"""Train a one-vs-rest classifier for one label, and report where it gets confused.

Positives are clips labeled with the target. Negatives are all other labeled clips
(`not_knock` and every category). `unsure` and unlabeled clips are left out. A clip with a
marked span (the part the label is about) contributes only the windows inside it.

Evaluation is cross-validated by clip, so windows of one clip never sit on both sides,
and a clip's score is the max over its windows, the same as the live detector sees it.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .classifier import TargetModel, model_path
from .config import NOT_KNOCK, UNSURE
from .features import clip_features, training_windows
from .store import ClipStore

log = logging.getLogger(__name__)


class NotEnoughData(Exception):
    pass


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression

    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit((X - mean) / scale, y)
    return mean, scale, clf.coef_[0], float(clf.intercept_[0])


def _predict(params, X):
    mean, scale, coef, b = params
    z = ((X - mean) / scale) @ coef + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _events_per_day(store: ClipStore) -> float | None:
    """Sounds heard per day: counted by the service, else estimated from the saved live clips."""
    counted = store.events_per_day()
    if counted:
        return counted
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    per_day: dict[str, int] = {}
    for m in store.all():
        if not m.get("imported_from"):
            per_day[m["day"]] = per_day.get(m["day"], 0) + 1
    full = [n for d, n in per_day.items() if d != today]
    return float(np.median(full)) if full else None


FLOOR = 0.5  # where the model itself starts to say "yes"; the budget can only raise it


def budget_threshold(neg_scores: np.ndarray, events_per_day: float | None, budget_per_week: float) -> dict:
    """The lowest threshold (so the most knocks found) whose false alarms fit the weekly budget.

    Never below FLOOR: the labeled sounds are not every sound there is, and a low threshold
    would let kinds of sound nobody labeled alarm.

    The labeled everyday sounds are a sample of what the detector hears: if a share p of them
    would alarm, that is p * events_per_day * 7 false alarms a week. With few labeled sounds
    p is coarse, so the result says how many would be needed to check the budget at all.
    """
    n = len(neg_scores)
    allowed = int(np.floor(budget_per_week / 7.0 / events_per_day * n)) if events_per_day and n else 0
    ranked = np.sort(neg_scores)[::-1]
    t = FLOOR if allowed >= n else float(np.ceil(ranked[allowed] * 100 + 1e-6) / 100)  # just above it
    t = round(min(max(t, FLOOR), 0.99), 2)
    above = int(np.sum(neg_scores >= t))
    rule = {"budget_per_week": budget_per_week, "events_per_day": events_per_day,
            "everyday_sounds": n, "allowed_among_them": allowed, "above_threshold": above,
            "met": above <= allowed}
    if events_per_day and n:
        week = events_per_day * 7
        rule["estimated_per_week"] = round(above / n * week, 2)
        # One false alarm among the labeled sounds already means this many a week.
        rule["resolution_per_week"] = round(week / n, 2)
        rule["sounds_needed"] = int(np.ceil(week / budget_per_week))
    return {"threshold": t, "rule": rule}


def train(store: ClipStore, yamnet, target: str, data_dir, C: float = 1.0, min_pos: int = 5,
          min_neg: int = 10, false_alarms_per_week: float = 1.0, progress=None) -> dict:
    """Train, save <data_dir>/models/<target>.json and return the report (also saved)."""
    from sklearn.model_selection import StratifiedGroupKFold

    say = progress or (lambda msg: log.info(msg))
    clips = [m for m in store.all() if m.get("label") and m["label"] != UNSURE]
    pos_clips = [m for m in clips if m["label"] == target]
    neg_clips = [m for m in clips if m["label"] != target]
    if len(pos_clips) < min_pos or len(neg_clips) < min_neg:
        raise NotEnoughData(f"need at least {min_pos} '{target}' clips and {min_neg} other labeled clips; "
                            f"have {len(pos_clips)} and {len(neg_clips)}")

    say(f"computing features for {len(clips)} labeled clips")
    X, y, groups = [], [], []
    used = []
    t0 = time.time()
    for i, meta in enumerate(clips):
        try:
            feats = clip_features(store, yamnet, meta["id"])
        except (OSError, ValueError, KeyError) as exc:
            say(f"skipping {meta['id']}: {exc}")
            continue
        sel = training_windows(feats, meta.get("span"))
        X.append(feats["emb"][sel].astype(np.float32))
        y += [meta["label"] == target] * len(sel)
        groups += [len(used)] * len(sel)
        used.append(meta)
        if (i + 1) % 50 == 0:
            say(f"  {i + 1}/{len(clips)} clips")
    X = np.concatenate(X)
    y = np.array(y, dtype=bool)
    groups = np.array(groups)
    say(f"features ready in {time.time() - t0:.0f}s: {len(X)} windows from {len(used)} clips")

    truth = np.array([m["label"] == target for m in used])
    n_splits = int(min(5, truth.sum(), (~truth).sum()))
    oof = np.zeros(len(used))
    if n_splits >= 2:
        say(f"cross-validating ({n_splits} folds)")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for tr, te in cv.split(X, y, groups):
            params = _fit(X[tr], y[tr], C)
            p = _predict(params, X[te])
            for g in np.unique(groups[te]):
                oof[g] = p[groups[te] == g].max()
    # Every labeled non-target clip counts as an everyday sound, imported ones too: an imported
    # recording that scores high is a false alarm waiting to happen (and errs on the safe side).
    epd = _events_per_day(store)
    if n_splits >= 2:
        choice = budget_threshold(oof[~truth], epd, float(false_alarms_per_week))
    else:
        choice = {"threshold": 0.5, "rule": {"note": "too few clips to cross-validate"}}
    threshold = choice["threshold"]

    say("training the final model on all clips")
    mean, scale, coef, b = _fit(X, y, C)

    pred = oof >= threshold
    tp, fp, fn = int(np.sum(pred & truth)), int(np.sum(pred & ~truth)), int(np.sum(~pred & truth))
    per_label: dict[str, dict] = {}
    for meta, score, hit in zip(used, oof, pred, strict=True):
        row = per_label.setdefault(meta["label"], {"clips": 0, "scores": [], "above": 0})
        row["clips"] += 1
        row["scores"].append(float(score))
        row["above"] += int(hit)
    categories = []
    for label, row in sorted(per_label.items(), key=lambda kv: (kv[0] != target, -np.mean(kv[1]["scores"]))):
        s = np.array(row["scores"])
        categories.append({
            "label": label,
            "clips": row["clips"],
            "mean_score": round(float(s.mean()), 3),
            "max_score": round(float(s.max()), 3),
            # For the target: clips recognised. For anything else: clips that would be false alarms.
            "above_threshold": row["above"],
        })
    clip_scores = sorted(({"id": m["id"], "label": m["label"], "score": round(float(s), 3)}
                          for m, s in zip(used, oof, strict=True)),
                         key=lambda r: -r["score"])
    confusing = [r for r in clip_scores if (r["label"] == target) != (r["score"] >= threshold)]

    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "target": target,
        "trained_at": trained_at,
        "clips": len(used),
        "positive_clips": int(truth.sum()),
        "negative_clips": int((~truth).sum()),
        "windows": int(len(X)),
        "folds": n_splits,
        "threshold": threshold,
        "threshold_rule": choice["rule"],
        "cv": {"true_positives": tp, "false_positives": fp, "missed": fn,
               "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3)},
        "categories": categories,
        "confusing": confusing[:100],
        "clip_scores": clip_scores,
        "unspecified_negatives": per_label.get(NOT_KNOCK, {}).get("clips", 0) if target != NOT_KNOCK else 0,
    }
    data = {
        "target": target,
        "trained_at": trained_at,
        "threshold": threshold,
        "mean": mean.round(6).tolist(),
        "scale": scale.round(6).tolist(),
        "coef": np.asarray(coef).round(6).tolist(),
        "intercept": b,
        "yamnet_sha256": yamnet.sha256,
        "C": C,
        "summary": {k: report[k] for k in ("clips", "positive_clips", "negative_clips", "cv", "threshold_rule")},
        # Which recordings, with the label each had then: a model stays traceable after a
        # label changes or a clip is deleted.
        "training_set": [[m["id"], m["label"], *([m["span"]] if m.get("span") else [])] for m in used],
    }
    model = TargetModel(data)
    path = model_path(data_dir, target)
    report_path = Path(data_dir) / "reports" / f"{target}.json"
    if path.exists():  # keep the previous model, and the report that came with it
        history = path.parent / "history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
        path.replace(history / f"{target}-{stamp}.json")
        if report_path.exists():
            report_path.replace(history / f"{target}-{stamp}.report.json")
    model.save(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    rule = choice["rule"]
    fa = f"about {rule['estimated_per_week']} false alarms a week" if "estimated_per_week" in rule else \
        "false alarm rate unknown until a full day of listening is counted"
    say(f"saved {path}: threshold {threshold}, finds {round(100 * report['cv']['recall'])}% of "
        f"{target} clips, {fa}")
    return report


def rescore(store: ClipStore, yamnet, models: dict, progress=None) -> int:
    """Recompute every clip's model scores (after training)."""
    say = progress or (lambda msg: log.info(msg))
    from .app import model_outputs

    n = 0
    for meta in store.all():
        try:
            feats = clip_features(store, yamnet, meta["id"])
            store.update(meta["id"], **model_outputs(models, feats, store.samples(meta["id"])))
            n += 1
        except (OSError, ValueError, KeyError):
            continue
        if n % 200 == 0:
            say(f"  rescored {n} clips")
    say(f"rescored {n} clips")
    return n
