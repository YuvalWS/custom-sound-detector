"""The last log lines in memory, for the web UI's Logs tab (the service can't read the journal)."""

from __future__ import annotations

import logging
import threading
from collections import deque

from .audio import redact

MAX_LINES = 2000


class BufferHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: deque[dict] = deque(maxlen=MAX_LINES)
        self.seq = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = redact(record.getMessage())
            if record.exc_info:
                msg += "\n" + redact(logging.Formatter().formatException(record.exc_info))
            with self._lock:
                self.seq += 1
                self.lines.append({"seq": self.seq, "time": record.created, "level": record.levelname,
                                   "name": record.name.removeprefix("sounddetector."), "msg": msg})
        except Exception:  # logging must never raise
            self.handleError(record)

    def since(self, after: int = 0, limit: int = 500) -> list[dict]:
        with self._lock:
            return [line for line in self.lines if line["seq"] > after][-limit:]


HANDLER = BufferHandler()


def install(level: int = logging.INFO) -> None:
    HANDLER.setLevel(level)
    logging.getLogger().addHandler(HANDLER)
