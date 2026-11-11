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
        # Anything ffmpeg can read: rtsp://, http(s)://, a file path. Without a login in it:
        # put the login in username/password below, which are percent-encoded for you.
        "url": None,
        "username": None,
        "password": None,
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
        # The threshold is set so that the model would have raised at most this many false
        # alarms per week on the sounds you labeled, scaled to how many sounds a day it hears.
        "false_alarms_per_week": 1.0,
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


# Settings changed in the web UI. They live in the data directory (the service can't write
# /etc) and override the config file, key by key.
SETTINGS_FILE = "settings.yaml"

# What the web UI may change: section -> keys. Secrets are write-only there.
EDITABLE = {
    "source": ["name", "url", "username", "password"],
    "mqtt": ["enabled", "host", "port", "username", "password"],
    "collection": ["enabled", "min_dbfs", "trigger_db", "max_clips_per_hour", "max_disk_mb"],
    "detection": ["targets", "off_delay", "min_hits", "thresholds"],
    "training": ["false_alarms_per_week"],
    "web": ["password"],
}
SECRETS = {("source", "password"), ("mqtt", "password"), ("web", "password")}


def _read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def settings_path(cfg: dict) -> Path:
    return Path(cfg["data_dir"]) / SETTINGS_FILE


def load(path: str | os.PathLike | None, with_settings: bool = True) -> dict:
    data = _read_yaml(path) if path else {}
    cfg = _merge(DEFAULTS, data)
    cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser())
    overrides = settings_path(cfg)
    if with_settings and overrides.exists():
        ui = _read_yaml(overrides)
        cfg = _merge(cfg, {s: {k: v for k, v in (ui.get(s) or {}).items() if k in keys}
                           for s, keys in EDITABLE.items()})
    if not cfg["yamnet_model"]:
        cfg["yamnet_model"] = str(Path(cfg["data_dir"]) / "yamnet.tflite")
    for target in cfg["detection"]["targets"]:
        if not LABEL_RE.match(target):
            raise ValueError(f"invalid target label: {target!r}")
    return cfg


def valid_label(label: str) -> bool:
    return bool(LABEL_RE.match(label or ""))


def stream_url(source: dict) -> str | None:
    """The source URL with username/password inserted, percent-encoded."""
    from urllib.parse import quote, urlsplit, urlunsplit

    url = source.get("url")
    if not url or not source.get("username"):
        return url
    parts = urlsplit(url)
    if not parts.scheme or "@" in parts.netloc:
        return url  # a file path, or a login already in the URL
    login = quote(str(source["username"]), safe="")
    if source.get("password"):
        login += ":" + quote(str(source["password"]), safe="")
    return urlunsplit(parts._replace(netloc=f"{login}@{parts.netloc}"))


def _coerce(default, value):
    if value is None or value == "":
        return None if default is None or isinstance(default, str) else default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        items = value if isinstance(value, list) else str(value).replace(" ", "").split(",")
        items = [i for i in items if i]
        for item in items:
            if not LABEL_RE.match(item):
                raise ValueError(f"invalid label {item!r}")
        return items
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError("expected a mapping")
        out = {}
        for k, v in value.items():
            if not LABEL_RE.match(k):
                raise ValueError(f"invalid label {k!r}")
            if v not in (None, ""):
                out[k] = float(v)
        return out
    return str(value)


def editable_view(cfg: dict) -> dict:
    """The UI-editable settings, with secrets replaced by whether they are set."""
    view = {}
    for section, keys in EDITABLE.items():
        view[section] = {}
        for key in keys:
            value = cfg[section].get(key)
            view[section][key] = {"set": bool(value)} if (section, key) in SECRETS else value
    return view


def save_settings(cfg: dict, changes: dict) -> dict:
    """Validate UI changes and merge them into the settings file. Returns the new file content.

    A secret left empty keeps its current value; send {"clear": true} to remove it.
    """
    path = settings_path(cfg)
    current = _read_yaml(path) if path.exists() else {}
    for section, values in (changes or {}).items():
        if section not in EDITABLE or not isinstance(values, dict):
            raise ValueError(f"not editable: {section}")
        for key, value in values.items():
            if key not in EDITABLE[section]:
                raise ValueError(f"not editable: {section}.{key}")
            if (section, key) in SECRETS:
                if isinstance(value, dict) and value.get("clear"):
                    current.setdefault(section, {})[key] = None
                elif value:
                    current.setdefault(section, {})[key] = str(value)
                continue
            current.setdefault(section, {})[key] = _coerce(DEFAULTS[section][key], value)
    if current.get("source", {}).get("name") is not None and not LABEL_RE.match(current["source"]["name"]):
        raise ValueError("source.name: lowercase letters, digits and _ only")
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("# Written by the web UI (Settings). Overrides the config file key by key.\n")
        yaml.safe_dump(current, fh, sort_keys=True)
    tmp.replace(path)
    return current
