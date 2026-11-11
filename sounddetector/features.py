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


def suggest_span(probs: np.ndarray, starts: np.ndarray, samples: np.ndarray, floor: float = 0.5):
    """Where in the clip a model hears its sound, as [start, end] seconds, or None.

    Takes the run of windows around the best one that score at least `floor`, then
    tightens it to the loud part inside them (windows are ~1 s and overlap, so the run
    alone is about twice as wide as the sound).
    """
    from . import SAMPLE_RATE
    from .events import FRAME, dbfs

    probs = np.asarray(probs, dtype=np.float64)
    if not len(probs) or probs.max() < floor:
        return None
    best = int(np.argmax(probs))
    i = j = best
    while i > 0 and probs[i - 1] >= floor:
        i -= 1
    while j < len(probs) - 1 and probs[j + 1] >= floor:
        j += 1
    duration = len(samples) / SAMPLE_RATE
    a, b = float(starts[i]), min(float(starts[j]) + WINDOW_S, duration)
    x = np.asarray(samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)], dtype=np.float64)
    n = len(x) // FRAME
    if n >= 5:
        energy = np.mean(x[: n * FRAME].reshape(n, FRAME) ** 2, axis=1)
        level = np.array([dbfs(e) for e in np.convolve(energy, np.ones(5) / 5, mode="same")])
        loud = np.flatnonzero(level >= level.max() - 12.0)
        a, b = a + max(0, loud[0] - 5) * FRAME / SAMPLE_RATE, min(b, a + (loud[-1] + 10) * FRAME / SAMPLE_RATE)
    if b - a < 0.3:  # never narrower than a single knock
        mid = (a + b) / 2
        a, b = max(0.0, mid - 0.15), min(duration, mid + 0.15)
    return [round(a, 2), round(b, 2)]


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
