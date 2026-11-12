"""Audio input through ffmpeg: a reconnecting live source, file decoding and WAV I/O."""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import wave
from collections.abc import Iterator

import numpy as np

from . import SAMPLE_RATE

log = logging.getLogger(__name__)

CHUNK = SAMPLE_RATE // 10  # 100 ms of samples per read

_CREDENTIALS = re.compile(r"(://)[^/@\s]+@")


def redact(text: str) -> str:
    """Hide user:password in URLs, so a stream login never reaches the log."""
    return _CREDENTIALS.sub(r"\1***@", text)


def _ffmpeg_cmd(url: str, realtime: bool = False) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if url.startswith("rtsp://") or url.startswith("rtsps://"):
        cmd += ["-rtsp_transport", "tcp"]
    if realtime:
        cmd += ["-re"]
    cmd += ["-i", url, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    return cmd


def decode_file(path: str) -> np.ndarray:
    """Decode any audio/video file ffmpeg understands to 16 kHz mono int16."""
    proc = subprocess.run(_ffmpeg_cmd(path), capture_output=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg could not decode the file: {redact(proc.stderr.decode(errors='replace')[-300:])}")
    return np.frombuffer(proc.stdout, dtype=np.int16).copy()


def to_ogg(wav_path) -> bytes:
    """A WAV file as Ogg/Opus, the format Telegram plays as a voice note."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(wav_path),
                           "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "-"], capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"ffmpeg could not encode Ogg: {proc.stderr.decode(errors='replace')[-300:]}")
    return proc.stdout


def write_wav(path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).copy()


class LiveSource:
    """Yields 100 ms int16 chunks from ffmpeg, restarting it when it exits or stalls.

    Yields None once after every disconnect, so the caller can reset state.
    """

    def __init__(self, url: str, stall_timeout: float = 15, reconnect_delay: float = 5,
                 realtime: bool = False, loop: bool = True):
        self.url = url
        self.stall_timeout = stall_timeout
        self.reconnect_delay = reconnect_delay
        self.realtime = realtime  # pace a file at real speed (testing)
        self.loop = loop  # reconnect forever; False stops at the end of the input
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _reader(self, proc: subprocess.Popen, out: queue.Queue) -> None:
        nbytes = CHUNK * 2
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if not buf:
                    break
                out.put(buf)
        finally:
            out.put(b"")

    def __iter__(self) -> Iterator[np.ndarray | None]:
        while not self._stop.is_set():
            log.info("connecting to audio source")
            self._proc = subprocess.Popen(_ffmpeg_cmd(self.url, self.realtime), stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            chunks: queue.Queue = queue.Queue(maxsize=600)
            threading.Thread(target=self._reader, args=(self._proc, chunks), daemon=True).start()
            pending = b""
            got_audio = False
            while not self._stop.is_set():
                try:
                    buf = chunks.get(timeout=self.stall_timeout)
                except queue.Empty:
                    log.warning("no audio for %ss, restarting ffmpeg", self.stall_timeout)
                    break
                if not buf:
                    break
                if not got_audio:
                    log.info("receiving audio")
                    got_audio = True
                pending += buf
                usable = len(pending) - len(pending) % 2
                if usable:
                    yield np.frombuffer(pending[:usable], dtype=np.int16).copy()
                    pending = pending[usable:]
            if self._proc.poll() is None:
                self._proc.kill()
            err = self._proc.stderr.read().decode(errors="replace").strip() if self._proc.stderr else ""
            self._proc.wait()
            if err:
                log.warning("ffmpeg: %s", redact(err[-500:]))
            yield None
            if not self.loop:
                return
            self._stop.wait(self.reconnect_delay)
