"""Sound-event segmentation: find where something louder than the background happens.

Works on 10 ms frames. The background is the median frame level of the last few seconds,
frozen while an event is active, so a long sound doesn't raise its own threshold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE

FRAME = SAMPLE_RATE // 100  # 10 ms
SHORT = 5  # frames in the 50 ms short-term level
FULL_SCALE = 32768.0


def dbfs(energy: float) -> float:
    """dBFS of a mean-square energy of int16 samples."""
    return 10.0 * np.log10(max(energy, 1e-3) / FULL_SCALE**2)


def rms_dbfs(samples: np.ndarray) -> float:
    x = np.asarray(samples, dtype=np.float64)
    return dbfs(float(np.mean(x * x))) if len(x) else -120.0


@dataclass
class Event:
    start: int  # absolute sample index of the clip start (includes pre-roll)
    end: int  # absolute sample index of the clip end (exclusive)
    onset: int  # absolute sample index where the sound started
    sound_end: int  # absolute sample index of the last loud frame
    peak_dbfs: float  # loudest 50 ms level
    background_dbfs: float


class Segmenter:
    def __init__(self, trigger_db=15.0, min_dbfs=-55.0, release_db=6.0, background_window=10.0,
                 pre_roll=0.75, post_roll=1.0, min_clip=2.0, max_clip=6.0, **_ignored):
        self.trigger_db = trigger_db
        self.min_dbfs = min_dbfs
        self.release_db = release_db
        self.pre_roll = int(pre_roll * SAMPLE_RATE)
        self.post_roll = int(post_roll * SAMPLE_RATE)
        self.min_clip = int(min_clip * SAMPLE_RATE)
        self.max_clip = int(max_clip * SAMPLE_RATE)
        self._bg_frames: deque[float] = deque(maxlen=max(10, int(background_window * 100)))
        self._short: deque[float] = deque(maxlen=SHORT)
        self._pending = np.zeros(0, dtype=np.int16)
        self.pos = 0  # absolute index of the next sample to arrive
        self.background = None  # dBFS
        self._frames_since_bg = 0
        self.active = False
        self.level = -120.0  # current 50 ms level, dBFS
        self.last_active = -10**12  # absolute sample index of the last loud frame
        self._ev_onset = 0
        self._ev_peak = -120.0
        self._ev_bg = -120.0

    @classmethod
    def from_config(cls, collection_cfg: dict) -> Segmenter:
        return cls(**collection_cfg)

    def reset(self) -> None:
        """Forget the current event and background (after a disconnect)."""
        self._bg_frames.clear()
        self._short.clear()
        self._pending = np.zeros(0, dtype=np.int16)
        self.background = None
        self.active = False

    def _update_background(self) -> None:
        if self._bg_frames:
            self.background = float(np.median(np.fromiter(self._bg_frames, dtype=np.float64)))

    def feed(self, samples: np.ndarray) -> list[Event]:
        """Consume samples; return the events that finished."""
        data = np.concatenate([self._pending, np.asarray(samples, dtype=np.int16)])
        n_frames = len(data) // FRAME
        self._pending = data[n_frames * FRAME:]
        frames = data[: n_frames * FRAME].astype(np.float64).reshape(n_frames, FRAME)
        energies = np.mean(frames * frames, axis=1) if n_frames else []
        done: list[Event] = []
        for energy in energies:
            frame_end = self.pos + FRAME
            self._short.append(energy)
            level = dbfs(float(np.mean(self._short)))
            self.level = level
            if self.background is None:
                self.background = dbfs(energy)
            if not self.active:
                self._bg_frames.append(dbfs(energy))
                self._frames_since_bg += 1
                if self._frames_since_bg >= 50 or len(self._bg_frames) < 50:
                    self._update_background()
                    self._frames_since_bg = 0
                if level >= max(self.background + self.trigger_db, self.min_dbfs):
                    self.active = True
                    self._ev_onset = frame_end - SHORT * FRAME
                    self._ev_peak = level
                    self._ev_bg = self.background
                    self.last_active = frame_end
            else:
                if level >= self._ev_bg + self.release_db:
                    self.last_active = frame_end
                    self._ev_peak = max(self._ev_peak, level)
                too_long = frame_end - (self._ev_onset - self.pre_roll) >= self.max_clip
                quiet = frame_end - self.last_active >= self.post_roll
                if quiet or too_long:
                    done.append(self._finish(frame_end))
            self.pos = frame_end  # a partial frame is counted once it completes
        return done

    def _finish(self, now: int) -> Event:
        self.active = False
        start = max(0, self._ev_onset - self.pre_roll)
        end = min(now, start + self.max_clip)
        if end - start < self.min_clip:
            start = max(0, end - self.min_clip)
        return Event(start=start, end=end, onset=self._ev_onset, sound_end=min(self.last_active, end),
                     peak_dbfs=round(self._ev_peak, 1),
                     background_dbfs=round(self._ev_bg, 1))


def segment(samples: np.ndarray, collection_cfg: dict) -> list[Event]:
    """Offline segmentation of a whole recording (used for imports)."""
    seg = Segmenter.from_config(collection_cfg)
    events = []
    for i in range(0, len(samples), SAMPLE_RATE // 10):
        events += seg.feed(samples[i:i + SAMPLE_RATE // 10])
    if seg.active:
        events.append(seg._finish(seg.pos))
    return events
