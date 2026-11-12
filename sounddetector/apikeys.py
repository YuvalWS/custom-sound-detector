"""API keys for the web API, for programs (Home Assistant, scripts) instead of the UI password.

A key is sent as `Authorization: Bearer <key>` (or `X-API-Key: <key>`). Only its SHA-256 is
stored, in <data_dir>/api_keys.json (readable by the service alone), so a key is shown once,
when it is created. Each key has a scope:

- `labeling`: read clips, their audio and the reports, and set labels and marks. Enough for
  a Home Assistant alert that sends the recording and labels it from a button.
- `full`: everything the UI can do, except managing keys: that needs the UI password, so a
  leaked key can't mint more keys or hide itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

SCOPES = {
    "labeling": "read clips, audio and reports; set labels and marks",
    "full": "everything except managing API keys",
}
PREFIX = "sdk_"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeys:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "api_keys.json"
        self._lock = threading.Lock()
        self._keys: list[dict] = []
        try:
            self._keys = json.loads(self.path.read_text())["keys"]
        except (OSError, ValueError, KeyError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({"keys": [{k: v for k, v in r.items() if not k.startswith("_")} for r in self._keys]},
                      fh, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _public(rec: dict) -> dict:
        return {k: v for k, v in rec.items() if k != "hash" and not k.startswith("_")}

    def list(self) -> list[dict]:
        with self._lock:
            return [self._public(r) for r in self._keys]

    def create(self, name: str, scope: str) -> tuple[dict, str]:
        name = (name or "").strip()[:60]
        if not name:
            raise ValueError("give the key a name, e.g. Home Assistant")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
        key = PREFIX + secrets.token_urlsafe(32)
        rec = {"id": secrets.token_hex(4), "name": name, "scope": scope, "prefix": key[:10],
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "last_used": None, "hash": _hash(key)}
        with self._lock:
            self._keys.append(rec)
            self._save()
        return self._public(rec), key

    def revoke(self, key_id: str) -> None:
        with self._lock:
            keep = [r for r in self._keys if r["id"] != key_id]
            if len(keep) == len(self._keys):
                raise ValueError("no such key")
            self._keys = keep
            self._save()

    def check(self, key: str) -> dict | None:
        """The key's record if it is valid (and note when it was used), else None."""
        if not key.startswith(PREFIX):
            return None
        h = _hash(key)
        with self._lock:
            for rec in self._keys:
                if hmac.compare_digest(rec["hash"], h):
                    now = time.time()
                    if now - rec.get("_saved_use", 0) > 300:  # don't rewrite the file on every request
                        rec["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                        self._save()
                        rec["_saved_use"] = now
                    return self._public(rec)
        return None


def allowed(scope: str, method: str, parts: list[str]) -> bool:
    """Whether a key with this scope may make this request (parts: the URL path, split)."""
    if parts[:2] == ["api", "keys"]:
        return False  # the UI password only
    if scope == "full":
        return True
    if method in ("GET", "HEAD"):
        return (parts in (["api", "status"], ["api", "clips"], ["api", "report"])
                or (len(parts) == 3 and parts[:2] == ["api", "clips"])
                or (len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "audio"))
    return len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] in ("label", "span")
