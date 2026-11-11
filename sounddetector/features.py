"""Per-clip YAMNet features, cached next to the clip so retraining takes seconds."""

from __future__ import annotations

import numpy as np

from .store import ClipStore
from .yamnet import YamNet

HOP = 0.1  # seconds between training/scoring windows
WINDOW_S = 0.96  # length of one YAMNet window


def clip_features(store: ClipStore, yamnet: YamNet, clip_id: str) -> dict:
    """{'emb', 'scores', 'levels', 'starts'} for every window of a clip (cached)."""
    wav = store.wav_path(clip_id)
    cache = wav.with_name(f"{clip_id}.emb{yamnet.sha256[:8]}.npz")
    if cache.exists():
        try:
            with np.load(cache) as data:
                return {k: data[k] for k in data.files}
        except (OSError, ValueError):
            cache.unlink(missing_ok=True)
    emb, scores, levels, starts = yamnet.windows(store.samples(clip_id), hop=HOP)
    feats = {"emb": emb.astype(np.float16), "scores": scores.astype(np.float16),
             "levels": levels, "starts": starts}
    tmp = cache.with_name(cache.name + ".tmp.npz")
    np.savez(tmp, **feats)
    tmp.replace(cache)
    return feats


def event_windows(levels: np.ndarray, within_db: float = 10.0) -> np.ndarray:
    """Indices of the windows that hold the sound itself: within `within_db` of the loudest."""
    return np.flatnonzero(levels >= levels.max() - within_db)


def span_windows(starts: np.ndarray, span) -> np.ndarray:
    """Indices of the windows that mostly cover the marked span [a, b] (seconds).

    A window counts when at least half of it, or half of the span when the span is
    shorter than a window, lies inside the span. Never empty: the best-overlapping
    window is kept if nothing passes.
    """
    a, b = float(span[0]), float(span[1])
    overlap = np.clip(np.minimum(starts + WINDOW_S, b) - np.maximum(starts, a), 0, None)
    need = 0.5 * min(WINDOW_S, b - a)
    sel = np.flatnonzero(overlap >= need - 1e-6)
    return sel if len(sel) else np.array([int(np.argmax(overlap))])


def training_windows(feats: dict, span=None) -> np.ndarray:
    """The windows a clip contributes to training: the loud ones, inside the marked span if any."""
    if not span:
        return event_windows(feats["levels"])
    inside = span_windows(np.asarray(feats["starts"], dtype=np.float64), span)
    return inside[event_windows(feats["levels"][inside])]
