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


def peak_dbfs(samples: np.ndarray) -> float:
    """The loudest 50 ms level, as the segmenter measures it."""
    from .events import FRAME, SHORT, dbfs

    x = np.asarray(samples, dtype=np.float64)
    n = len(x) // FRAME
    if n == 0:
        return -120.0
    energy = np.mean(x[: n * FRAME].reshape(n, FRAME) ** 2, axis=1)
    if n >= SHORT:
        energy = np.convolve(energy, np.ones(SHORT) / SHORT, mode="valid")
    return dbfs(float(energy.max()))


def quieter_copies(store: ClipStore, yamnet: YamNet, clip_id: str, gains, noise: list[np.ndarray]) -> list[dict]:
    """Features of the clip played `gains` dB quieter, over real background noise (cached).

    A soft knock is a loud knock heard from further away: quieter, with the same background.
    Each copy keeps its peak level, so the ones the detector could not hear can be dropped.
    """
    import zlib

    gains = [float(g) for g in gains]
    wav = store.wav_path(clip_id)
    cache = wav.with_name(f"{clip_id}.aug{yamnet.sha256[:8]}.npz")
    if cache.exists():
        try:
            with np.load(cache) as data:
                if list(data["asked"]) == gains:
                    return [{"emb": data[f"emb{k}"], "levels": data[f"levels{k}"], "starts": data[f"starts{k}"],
                             "peak": float(data[f"peak{k}"])} for k in range(int(data["n"]))]
        except (OSError, ValueError, KeyError):
            pass
        cache.unlink(missing_ok=True)
    x = store.samples(clip_id).astype(np.float64)
    seed = zlib.crc32(clip_id.encode())
    out = []
    for k, gain in enumerate(gains):
        y = x * 10 ** (gain / 20)
        if noise:
            n = noise[(seed + k) % len(noise)]
            y = y + np.resize(n, len(y))
        y = np.clip(y, -32768, 32767).astype(np.int16)
        emb, _, levels, starts = yamnet.windows(y, hop=HOP)
        out.append({"emb": emb.astype(np.float16), "levels": levels, "starts": starts, "peak": peak_dbfs(y)})
    arrays = {"asked": np.array(gains), "n": np.array(len(out))}
    for k, f in enumerate(out):
        arrays.update({f"emb{k}": f["emb"], f"levels{k}": f["levels"], f"starts{k}": f["starts"],
                       f"peak{k}": np.array(f["peak"])})
    tmp = cache.with_name(cache.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(cache)
    return out


def audible(copies: list[dict], min_dbfs: float | None) -> list[dict]:
    """The copies loud enough to start a clip (peak at or above the collection gate)."""
    return [c for c in copies if min_dbfs is None or c["peak"] >= min_dbfs]


def event_windows(levels: np.ndarray, within_db: float = 10.0) -> np.ndarray:
    """Indices of the windows that hold the sound itself: within `within_db` of the loudest."""
    return np.flatnonzero(levels >= levels.max() - within_db)


SUSTAIN_OPTIONS = (0.0, 0.2, 0.4, 0.6, 0.8)  # seconds; training picks the one that does best


def sustained_score(probs: np.ndarray, starts: np.ndarray, sustain: float = 0.0) -> float:
    """A clip's score: the highest level the window scores stay at for `sustain` seconds.

    With sustain 0 it is the best single window. A knock is a series of hits, so its score
    stays high across many overlapping windows; one click scores high in fewer. A clip
    passes a threshold when windows whose starts span `sustain` seconds all pass it, the
    same rule the live detector uses. A clip too short to hold that run is scored as a whole.
    """
    p = np.asarray(probs, dtype=np.float64)
    if not len(p):
        return 0.0
    if sustain <= 0:
        return float(p.max())
    t = np.asarray(starts, dtype=np.float64)
    ends = np.searchsorted(t, t + sustain - 1e-6)  # first window at least `sustain` after each
    best = None
    for i, k in enumerate(ends):
        if k >= len(p):
            break
        level = float(p[i:k + 1].min())
        best = level if best is None else max(best, level)
    return float(p.min()) if best is None else best


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
