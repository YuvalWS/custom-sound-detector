"""Connection tests for the Settings tab: can we read the stream, can we log in to MQTT."""

from __future__ import annotations

import json
import subprocess
import threading

import numpy as np

from . import SAMPLE_RATE
from .audio import redact
from .events import rms_dbfs


def _rtsp_args(url: str) -> list[str]:
    return ["-rtsp_transport", "tcp"] if url.startswith(("rtsp://", "rtsps://")) else []


def test_stream(url: str, seconds: float = 3.0, timeout: float = 20.0) -> dict:
    """Open the stream, list its audio tracks and measure a few seconds of audio."""
    if not url:
        return {"ok": False, "error": "no stream URL"}
    probe = ["ffprobe", "-v", "error", *_rtsp_args(url), "-show_entries",
             "stream=index,codec_type,codec_name,sample_rate,channels", "-of", "json", url]
    try:
        out = subprocess.run(probe, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"no answer within {timeout:.0f}s (host, port or firewall?)"}
    if out.returncode != 0:
        return {"ok": False, "error": redact(out.stderr.decode(errors="replace").strip()[-400:]) or "ffprobe failed"}
    streams = json.loads(out.stdout or b"{}").get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        return {"ok": False, "error": "the stream has no audio track", "streams": streams}
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *_rtsp_args(url), "-i", url,
           "-t", str(seconds), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    try:
        rec = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "connected, but no audio arrived", "audio_tracks": audio}
    samples = np.frombuffer(rec.stdout[: len(rec.stdout) // 2 * 2], dtype=np.int16)
    if not len(samples):
        return {"ok": False, "error": redact(rec.stderr.decode(errors="replace").strip()[-400:]) or "no audio",
                "audio_tracks": audio}
    frames = samples[: len(samples) // 800 * 800].astype(np.float64).reshape(-1, 800)  # 50 ms
    result = {
        "ok": True,
        "audio_tracks": [{k: a.get(k) for k in ("codec_name", "sample_rate", "channels")} for a in audio],
        "seconds": round(len(samples) / SAMPLE_RATE, 1),
        "level_dbfs": round(rms_dbfs(samples), 1),
        "loudest_dbfs": round(max(rms_dbfs(f) for f in frames), 1) if len(frames) else None,
    }
    if len(audio) > 1:
        result["warning"] = (f"{len(audio)} audio tracks: ffmpeg reads the first. A stream with a single "
                             "track is safer (see the README, 'Getting a stream').")
    return result


def test_mqtt(host: str, port: int, username: str | None, password: str | None, timeout: float = 8.0) -> dict:
    """Log in to the broker and disconnect again; publishes nothing."""
    import paho.mqtt.client as mqtt

    if not host:
        return {"ok": False, "error": "no broker host"}
    done = threading.Event()
    result: dict = {}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        result["ok"] = not reason_code.is_failure
        if reason_code.is_failure:
            result["error"] = f"the broker refused the login: {reason_code}"
        done.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sound-detector-test")
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    try:
        client.connect(host, int(port), keepalive=10)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"cannot reach {host}:{port}: {exc}"}
    client.loop_start()
    try:
        if not done.wait(timeout):
            return {"ok": False, "error": f"no answer from {host}:{port} within {timeout:.0f}s"}
        return result
    finally:
        client.loop_stop()
        client.disconnect()
