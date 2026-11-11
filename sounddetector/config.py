"""Configuration: a YAML file deep-merged over the defaults below."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

# Labels with a fixed meaning. Everything else is a user category, and every category
# except the target counts as a negative when a model for that target is trained.
KNOCK = "knock"
NOT_KNOCK = "not_knock"
UNSURE = "unsure"  # kept, but never used for training

DEFAULT_CATEGORIES = [
    "footsteps",
    "voice",
    "keys",
    "lock_motor",
    "door",
    "doorbell_chime",
    "elevator",
    "vehicle",
    "animal",
    "rain_wind",
    "music",
    "other",
]

LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")

DEFAULTS: dict = {
    "data_dir": "/var/lib/sound-detector",
    # YAMNet .tflite used to compute features. Default: <data_dir>/yamnet.tflite
    "yamnet_model": None,
    "source": {
        # A name for this microphone, used in clip ids and MQTT topics.
        "name": "front_door",
        # Anything ffmpeg can read: rtsp://, http(s)://, a file path.
        "url": None,
        # Seconds without audio before ffmpeg is restarted.
        "stall_timeout": 15,
        "reconnect_delay": 5,
    },
    "collection": {
        # Save a clip of every sound event. Turn off once the model is good enough;
        # clips that trigger a detection are saved regardless (see detection.save_clips).
        "enabled": True,
        # A sound event starts when the 50 ms level rises this many dB above the
        # background (the median level of the last background_window seconds)...
        "trigger_db": 15.0,
        # ...and is at least this loud (dBFS). Use `sound-detector check` to see your levels.
        "min_dbfs": -55.0,
        # It lasts while the level stays release_db above the background.
        "release_db": 6.0,
        "background_window": 10.0,
        "pre_roll": 0.75,
        "post_roll": 1.0,
        "min_clip": 2.0,
        "max_clip": 6.0,
        "max_clips_per_hour": 120,
        # Oldest unlabeled clips are deleted first; labeled clips are never deleted.
        "max_disk_mb": 2000,
    },
    "detection": {
        "enabled": True,
        # Seconds between classifier runs while a sound event is active.
        "hop": 0.25,
        # Seconds without a hit before the MQTT state returns to OFF.
        "off_delay": 10.0,
        # Hits needed (within off_delay) before the state turns ON.
        "min_hits": 1,
        # Labels with a trained model to load and publish.
        "targets": [KNOCK],
        # Per-target threshold overrides, e.g. {knock: 0.7}. Default: chosen by training.
        "thresholds": {},
        "save_clips": True,
    },
    "mqtt": {
        "enabled": True,
        "host": "localhost",
        "port": 1883,
        "username": None,
        "password": None,
        "topic_prefix": "sound_detector",
        "discovery": True,
        "discovery_prefix": "homeassistant",
    },
    "web": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        # Basic auth for the labeling UI. The clips are recordings of your doorstep:
        # leave a password set unless the UI is unreachable from anything you don't trust.
        "username": "admin",
        "password": None,
    },
    "training": {
        # Inverse regularisation strength of the logistic regression.
        "C": 1.0,
        "min_positive_clips": 5,
        "min_negative_clips": 10,
    },
    "categories": DEFAULT_CATEGORIES,
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: str | os.PathLike | None) -> dict:
    data = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, data)
    cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser())
    if not cfg["yamnet_model"]:
        cfg["yamnet_model"] = str(Path(cfg["data_dir"]) / "yamnet.tflite")
    for target in cfg["detection"]["targets"]:
        if not LABEL_RE.match(target):
            raise ValueError(f"invalid target label: {target!r}")
    return cfg


def valid_label(label: str) -> bool:
    return bool(LABEL_RE.match(label or ""))
