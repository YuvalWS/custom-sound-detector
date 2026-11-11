"""YAMNet feature extractor: per-window embeddings (1024) and stock class scores (521).

The trained classifiers sit on top of the embeddings, so a classifier is only valid with
the exact model file it was trained on; its SHA-256 is stored with every classifier.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
import zipfile
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE

log = logging.getLogger(__name__)

WINDOW = 15600  # 0.975 s, the input length YAMNet's TFLite export expects
EMBEDDING_SIZE = 1024

# The default model: Google's YAMNet as published for MediaPipe (521 AudioSet classes,
# labels embedded in the file). Pinned so a classifier always matches its features.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
MODEL_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(dest, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> None:
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:  # noqa: S310 (fixed URL)
        fh.write(resp.read())
    got = sha256_file(tmp)
    if sha256 and got != sha256:
        tmp.unlink()
        raise RuntimeError(f"YAMNet download checksum mismatch: {got}")
    tmp.replace(dest)


def _load_labels(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".txt"):
                    return zf.read(name).decode().splitlines()
    except zipfile.BadZipFile:
        pass
    side = path.with_name("yamnet_label_list.txt")
    if side.exists():
        return side.read_text().splitlines()
    return [f"class_{i}" for i in range(521)]


class YamNet:
    def __init__(self, model_path, num_threads: int = 1):
        from ai_edge_litert.interpreter import Interpreter

        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"YAMNet model not found at {self.path} (run `sound-detector download-model`)")
        self.sha256 = sha256_file(self.path)
        # The embedding is an intermediate tensor, so intermediates must be preserved
        # (LiteRT warns that this is meant for debugging; the model is small enough).
        warnings.filterwarnings("ignore", message=".*experimental_preserve_all_tensors.*")
        self._it = Interpreter(model_path=str(self.path), num_threads=num_threads,
                               experimental_preserve_all_tensors=True)
        self._it.allocate_tensors()
        inp = self._it.get_input_details()[0]
        if list(inp["shape"]) != [WINDOW]:
            self._it.resize_tensor_input(inp["index"], [WINDOW])
            self._it.allocate_tensors()
        self._in = inp["index"]
        self._scores = self._it.get_output_details()[0]["index"]
        self._emb = None
        for d in self._it.get_tensor_details():
            name = d["name"].lower()
            if int(np.prod(d["shape"])) == EMBEDDING_SIZE and ("reduce_mean" in name or "embedding" in name):
                self._emb = d
        if self._emb is None:
            raise RuntimeError("no 1024-wide embedding tensor found in this YAMNet model")
        self.labels = _load_labels(self.path)

    def window(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(embedding[1024], scores[521]) of one 0.975 s window of int16 samples."""
        x = np.asarray(samples, dtype=np.float32) / 32768.0
        if len(x) < WINDOW:
            x = np.pad(x, (0, WINDOW - len(x)))
        self._it.set_tensor(self._in, x[:WINDOW])
        self._it.invoke()
        scores = self._it.get_tensor(self._scores).reshape(-1).astype(np.float32)
        raw = self._it.get_tensor(self._emb["index"]).reshape(-1)
        scale, zero = self._emb.get("quantization", (0.0, 0))
        emb = (raw.astype(np.float32) - zero) * scale if scale else raw.astype(np.float32)
        return emb, scores

    def windows(self, samples: np.ndarray, hop: float = 0.1):
        """Slide over a clip. Returns (embeddings[n,1024], scores[n,521], rms_dbfs[n], starts[n] in s)."""
        from .events import rms_dbfs

        step = max(1, int(hop * SAMPLE_RATE))
        n = len(samples)
        starts = list(range(0, max(1, n - WINDOW + 1), step))
        if n > WINDOW and starts[-1] != n - WINDOW:
            starts.append(n - WINDOW)
        embs, scores, levels = [], [], []
        for s in starts:
            chunk = samples[s:s + WINDOW]
            e, sc = self.window(chunk)
            embs.append(e)
            scores.append(sc)
            levels.append(rms_dbfs(chunk))
        return (np.stack(embs), np.stack(scores), np.array(levels, dtype=np.float32),
                np.array(starts, dtype=np.float32) / SAMPLE_RATE)

    def top_labels(self, scores: np.ndarray, k: int = 3) -> list[list]:
        """Top-k stock YAMNet labels of a clip (max over its windows): hints for labeling."""
        best = scores.max(axis=0) if scores.ndim == 2 else scores
        idx = np.argsort(-best)[:k]
        return [[self.labels[i] if i < len(self.labels) else str(i), round(float(best[i]), 2)] for i in idx]
