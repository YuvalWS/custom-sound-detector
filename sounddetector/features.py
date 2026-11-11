"""Per-clip YAMNet features, cached next to the clip so retraining takes seconds."""

from __future__ import annotations

import numpy as np

from .store import ClipStore
from .yamnet import YamNet

HOP = 0.1  # seconds between training/scoring windows


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
