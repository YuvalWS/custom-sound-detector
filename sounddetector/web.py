"""Labeling UI and its JSON API, on the standard library's threaded HTTP server."""

from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import tarfile
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import SAMPLE_RATE, __version__
from .apikeys import SCOPES, ApiKeys, allowed
from .audio import decode_file
from .config import KNOCK, NOT_KNOCK, UNSURE, valid_label
from .events import rms_dbfs, segment

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 200 * 1024 * 1024
# What a backup holds, under the data directory. settings.yaml is left out: it holds passwords.
# Installed by proxmox/install.sh: runs `update` as root when the UI writes the request file.
UPDATE_UNIT = Path("/etc/systemd/system/sound-detector-update.path")
BACKUP_PARTS = ("clips", "models", "reports", "categories.json", "events_per_day.json", "yamnet.tflite")


class Jobs:
    """One background job at a time (training, rescoring), with a log the UI polls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = None
        self.log: list[str] = []
        self.error = None
        self.finished_at = None

    def say(self, msg: str) -> None:
        log.info(msg)
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        del self.log[:-300]

    def start(self, name: str, fn) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running, self.log, self.error = name, [], None

        def run():
            try:
                fn(self.say)
            except Exception as exc:  # reported to the UI
                self.error = str(exc)
                self.say(f"failed: {exc}")
                log.debug(traceback.format_exc())
            finally:
                self.running = None
                self.finished_at = time.time()

        threading.Thread(target=run, daemon=True).start()
        return True

    def state(self) -> dict:
        return {"running": self.running, "log": self.log[-80:], "error": self.error, "finished_at": self.finished_at}


class Server:
    def __init__(self, cfg: dict, store, app=None):
        self.cfg = cfg
        self.store = store
        self.app = app  # None when the UI runs on its own (`sound-detector ui`)
        self.jobs = Jobs()
        self.keys = ApiKeys(cfg["data_dir"])
        self._yamnet = None
        self._yamnet_lock = threading.Lock()

    # A second YAMNet instance for imports and training: interpreters are not thread-safe.
    def yamnet(self):
        if self._yamnet is None:
            from .yamnet import YamNet

            self._yamnet = YamNet(self.cfg["yamnet_model"])
        return self._yamnet

    def models(self) -> dict:
        if self.app:
            return dict(self.app.models)
        from .classifier import load_targets

        det = self.cfg["detection"]
        return load_targets(self.cfg["data_dir"], det["targets"], self.yamnet().sha256, det.get("thresholds"))

    def suggester(self):
        if self.app:
            return self.app.suggester
        from .suggest import Suggester

        return Suggester.load(self.cfg["data_dir"], self.yamnet().sha256)

    # ---- API -----------------------------------------------------------------------------
    def status(self) -> dict:
        reports = {}
        for p in (Path(self.cfg["data_dir"]) / "reports").glob("*.json"):
            try:
                r = json.loads(p.read_text())
                if "target" not in r:  # the label suggestions' report
                    continue
                reports[r["target"]] = {k: r[k] for k in ("trained_at", "threshold", "cv", "clips",
                                                          "positive_clips", "negative_clips")}
            except (OSError, ValueError, KeyError):
                pass
        return {
            "service": self.app.status() if self.app else {"source": self.cfg["source"]["name"], "connected": None},
            "stats": self.store.stats(),
            "disk_mb": round(self.store.disk_mb(), 1),
            "categories": self.store.categories(),
            "fixed_labels": [KNOCK, NOT_KNOCK, UNSURE],
            "targets": self.cfg["detection"]["targets"],
            "reports": reports,
            "job": self.jobs.state(),
        }

    # ---- software update -----------------------------------------------------------------
    def _installed(self) -> dict | None:
        conf = self.cfg.get("_config_path")
        if not conf:
            return None
        try:
            return json.loads((Path(conf).parent / "installed.json").read_text())
        except (OSError, ValueError):
            return None

    def _latest(self, installed: dict) -> dict | None:
        """The newest commit on the installed branch, and what changed since (cached 5 minutes)."""
        import re
        import subprocess
        import urllib.request

        key = (installed.get("repo"), installed.get("branch"), installed.get("commit"))
        cached = getattr(self, "_latest_cache", None)
        if cached and cached[0] == key and time.time() - cached[1] < 300:
            return cached[2]
        out = None
        try:
            proc = subprocess.run(["git", "ls-remote", key[0], f"refs/heads/{key[1]}"], capture_output=True,
                                  text=True, timeout=20, check=False)
            if proc.returncode == 0 and proc.stdout.split():
                out = {"commit": proc.stdout.split()[0], "changes": None}
        except (OSError, subprocess.SubprocessError):
            pass
        m = re.match(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", key[0] or "")
        if out and m and out["commit"] != key[2]:
            try:  # what changed, from GitHub (best effort)
                url = f"https://api.github.com/repos/{m[1]}/{m[2]}/compare/{key[2]}...{out['commit']}"
                req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                           "User-Agent": "custom-sound-detector"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                out["changes"] = [c["commit"]["message"].splitlines()[0] for c in data.get("commits", [])][-30:]
            except (OSError, ValueError, KeyError):
                pass
        self._latest_cache = (key, time.time(), out)
        return out

    def software(self) -> dict:
        root = Path(self.cfg["data_dir"])
        installed = self._installed()
        log_path = root / "update.log"
        try:
            tail = log_path.read_text(errors="replace").splitlines()[-40:]
            log_time = log_path.stat().st_mtime
        except OSError:
            tail, log_time = [], None
        return {
            "version": __version__,
            "can_update": UPDATE_UNIT.exists() and installed is not None,
            "installed": installed,
            "latest": self._latest(installed) if installed and installed.get("repo") else None,
            "requested": (root / "update-request").exists(),
            "log": tail,
            "log_time": log_time,
        }

    def request_update(self) -> dict:
        """Ask the root update unit to update and restart the service."""
        if not (UPDATE_UNIT.exists() and self._installed()):
            raise ValueError("updating from the web UI needs an install made by install.sh (run `update` once)")
        if self.jobs.running:
            raise ValueError(f"wait for the running job ({self.jobs.running}) to finish")
        (Path(self.cfg["data_dir"]) / "update-request").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        log.info("update requested from the web UI")
        self._latest_cache = None
        return {"requested": True}

    def export_backup(self, out) -> dict:
        """Write a .tar.gz of the dataset (clips and labels), models with their history, and reports."""
        root = Path(self.cfg["data_dir"])
        files = []
        for part in BACKUP_PARTS:
            p = root / part
            files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p] if p.is_file() else []
        # Feature caches (.npz) are rebuilt from the WAVs on the next training: leave them out.
        files = [f for f in files if f.suffix != ".npz" and ".tmp" not in f.name]
        manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "version": __version__,
                    "stats": self.store.stats(), "files": 0,
                    "restore": "stop the service, extract into the data directory, start it again"}
        with tarfile.open(fileobj=out, mode="w:gz", compresslevel=6) as tar:
            for f in files:
                try:  # a clip can be dropped while we read: skip it
                    tar.add(f, arcname=f.relative_to(root).as_posix())
                    manifest["files"] += 1
                except FileNotFoundError:
                    pass
            data = json.dumps(manifest, indent=1).encode()
            info = tarfile.TarInfo("backup.json")
            info.size, info.mtime = len(data), int(time.time())
            tar.addfile(info, io.BytesIO(data))
        return manifest

    def list_clips(self, q: dict) -> dict:
        label = q.get("label", "unlabeled")
        clips = self.store.all()
        if label == "unlabeled":
            clips = [c for c in clips if not c.get("label")]
        elif label == "labeled":
            clips = [c for c in clips if c.get("label")]
        elif label == "negative":
            clips = [c for c in clips if c.get("label") and c["label"] not in (KNOCK, UNSURE)]
        elif label != "all":
            clips = [c for c in clips if c.get("label") == label]
        if q.get("detected") == "1":
            clips = [c for c in clips if c.get("detections")]
        if q.get("source") == "import":
            clips = [c for c in clips if c.get("imported_from")]
        if q.get("find"):
            needle = q["find"].lower()
            clips = [c for c in clips if needle in c["id"].lower() or needle in (c.get("imported_from") or "").lower()]
        if q.get("day"):
            clips = [c for c in clips if c.get("day") == q["day"]]
        if q.get("suggested"):  # the label suggestions' first choice
            clips = [c for c in clips if (c.get("label_suggestions") or [[None]])[0][0] == q["suggested"]]
        if q.get("ids"):
            wanted = set(q["ids"].split(","))
            clips = [c for c in clips if c["id"] in wanted]
        sort = q.get("sort", "newest")
        target = q.get("target", KNOCK)
        if sort == "oldest":
            clips.sort(key=lambda c: c["recorded_at"])
        elif sort == "score":
            clips.sort(key=lambda c: -(c.get("scores") or {}).get(target, -1))
        elif sort == "suggestion":  # most confident suggestion first
            clips.sort(key=lambda c: -((c.get("label_suggestions") or [[None, -1]])[0][1]))
        elif sort == "loudest":
            clips.sort(key=lambda c: -(c.get("peak_dbfs") or -120))
        else:
            clips.sort(key=lambda c: c["recorded_at"], reverse=True)
        total = len(clips)
        offset = int(q.get("offset", 0))
        limit = min(int(q.get("limit", 200)), 2000)
        return {"total": total, "clips": clips[offset:offset + limit]}

    def import_audio(self, body: bytes, name: str, label: str | None, split: bool) -> dict:
        if label and not valid_label(label):
            raise ValueError("invalid label")
        suffix = Path(name).suffix[:10] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(body)
            tmp.flush()
            samples = decode_file(tmp.name)
        if not len(samples):
            raise ValueError("no audio in the file")
        coll = self.cfg["collection"]
        if split:
            spans = [(e.start, e.end, e.onset, e.peak_dbfs, e.background_dbfs) for e in segment(samples, coll)]
        else:
            spans = []
        if not spans:
            end = min(len(samples), int(30 * SAMPLE_RATE))
            frames = samples[: end - end % 800].astype(np.float64).reshape(-1, 800)  # 50 ms
            peak = round(max(rms_dbfs(f) for f in frames), 1) if len(frames) else None
            spans = [(0, end, 0, peak, None)]
        saved = []
        models, suggester = self.models(), self.suggester()
        with self._yamnet_lock:
            yam = self.yamnet()
            from .app import describe_clip
            from .features import clip_features

            for start, end, onset, peak, bg in spans:
                meta = {"source": "import", "imported_from": Path(name).name[:120],
                        "import_offset": round(start / SAMPLE_RATE, 2),
                        "onset": round((onset - start) / SAMPLE_RATE, 2), "peak_dbfs": peak,
                        "background_dbfs": bg, "detections": []}
                clip = self.store.save(samples[start:end], meta, enforce_rate=False)
                if label:
                    self.store.set_label(clip["id"], label)
                feats = clip_features(self.store, yam, clip["id"])
                saved.append(self.store.update(clip["id"], **describe_clip(yam, models, feats, samples[start:end],
                                                                               suggester)))
        return {"saved": len(saved), "clips": saved}

    def split_clip(self, clip_id: str, at) -> dict:
        """Split a clip at `at` seconds; each part gets its hints, scores and suggested mark."""
        try:
            at = float(at)
        except (TypeError, ValueError):
            raise ValueError("at must be seconds") from None
        parts = self.store.split(clip_id, at)
        models, suggester = self.models(), self.suggester()
        with self._yamnet_lock:
            yam = self.yamnet()
            from .app import describe_clip
            from .features import clip_features

            out = []
            for part in parts:
                feats = clip_features(self.store, yam, part["id"])
                out.append(self.store.update(part["id"], **describe_clip(yam, models, feats,
                                                                         self.store.samples(part["id"]), suggester)))
        return {"parts": out}

    def settings(self) -> dict:
        from .config import editable_view, load

        # Re-read what is on disk: it may hold saved changes that the next restart will apply.
        on_disk = load(self.cfg.get("_config_path"))
        running = {k: v for k, v in self.cfg.items() if k != "_config_path"}
        return {"settings": editable_view(on_disk), "restart_pending": on_disk != running,
                "config_file": self.cfg.get("_config_path")}

    def save_settings(self, changes: dict) -> dict:
        from .config import save_settings

        save_settings(self.cfg, changes)
        self.request_restart()
        return {"saved": True, "restarting": True}

    def run_test(self, kind: str, form: dict) -> dict:
        """Test the stream or MQTT with the values in the form; empty secrets mean the saved ones."""
        from . import checks
        from .config import load, stream_url

        saved = load(self.cfg.get("_config_path"))
        if kind == "stream":
            src = {**saved["source"], **{k: v for k, v in form.items() if k in ("url", "username") and v is not None}}
            if form.get("password"):
                src["password"] = form["password"]
            return checks.test_stream(stream_url(src))
        if kind == "mqtt":
            given = {k: v for k, v in form.items() if k in ("host", "port", "username") and v is not None}
            m = {**saved["mqtt"], **given}
            return checks.test_mqtt(m.get("host"), int(m.get("port") or 1883), m.get("username"),
                                    form.get("password") or m.get("password"))
        raise ValueError("unknown test")

    def request_restart(self) -> None:
        """Exit cleanly once the response is out; systemd (Restart=always) starts us again."""
        import os
        import signal

        log.info("settings changed, restarting")
        threading.Timer(0.7, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    def start_training(self, target: str) -> bool:
        if not valid_label(target):
            raise ValueError("invalid target")

        def job(say):
            from .classifier import load_targets
            from .suggest import Suggester, train_suggester
            from .train import options, rescore, train

            with self._yamnet_lock:
                yam = self.yamnet()
                train(self.store, yam, target, self.cfg["data_dir"], progress=say, **options(self.cfg))
                say("training the label suggestions")
                train_suggester(self.store, yam, self.cfg["data_dir"], C=float(self.cfg["training"]["C"]),
                                progress=say)
                det = self.cfg["detection"]
                targets = sorted(set(det["targets"]) | {target})
                models = load_targets(self.cfg["data_dir"], targets, yam.sha256, det.get("thresholds"))
                say("rescoring all clips with the new models")
                rescore(self.store, yam, models, progress=say,
                        suggester=Suggester.load(self.cfg["data_dir"], yam.sha256))
            if self.app:
                self.app.reload_models(force=True)
            if target not in self.cfg["detection"]["targets"]:
                say(f"note: '{target}' is not in detection.targets, so it is not published to MQTT")

        return self.jobs.start(f"train {target}", job)

    # ---- HTTP plumbing -------------------------------------------------------------------
    def serve(self) -> ThreadingHTTPServer:
        server = self
        web = self.cfg["web"]
        expected = None
        if web.get("password"):
            token = base64.b64encode(f"{web.get('username') or 'admin'}:{web['password']}".encode()).decode()
            expected = f"Basic {token}"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                log.debug("web: " + fmt, *args)

            def _authorized(self) -> bool:
                """The UI password (basic auth), or an API key allowed to make this request."""
                got = self.headers.get("Authorization", "")
                key = got[7:].strip() if got[:7].lower() == "bearer " else self.headers.get("X-API-Key", "").strip()
                if key:
                    rec = server.keys.check(key)
                    if rec is not None:
                        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
                        if allowed(rec["scope"], self.command, parts):
                            return True
                        self.close_connection = True  # the request body is left unread
                        self._json({"error": f"this API key ({rec['scope']}) may not do that"}, 403)
                        return False
                elif expected is None or hmac.compare_digest(got.encode(), expected.encode()):
                    return True
                self.close_connection = True
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="sound-detector"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False

            def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _send_backup(self):
                name = time.strftime("sound-detector-backup-%Y%m%d-%H%M.tar.gz")
                with tempfile.TemporaryFile() as tmp:
                    server.export_backup(tmp)
                    size = tmp.tell()
                    tmp.seek(0)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/gzip")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        while chunk := tmp.read(1 << 20):
                            self.wfile.write(chunk)

            def _json(self, obj, code: int = 200):
                self._send(code, json.dumps(obj).encode(), "application/json")

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_UPLOAD:
                    raise ValueError("upload too large")
                return self.rfile.read(n) if n else b""

            def do_GET(self):
                if not self._authorized():
                    return
                url = urlparse(self.path)
                q = {k: v[-1] for k, v in parse_qs(url.query).items()}
                parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
                try:
                    if not parts or parts == ["index.html"]:
                        self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                    elif parts == ["api", "status"]:
                        self._json(server.status())
                    elif parts == ["api", "clips"]:
                        self._json(server.list_clips(q))
                    elif len(parts) == 3 and parts[:2] == ["api", "clips"]:
                        self._json(server.store.get(parts[2]))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "audio":
                        wav = server.store.wav_path(parts[2])
                        if q.get("format") == "ogg":
                            from .audio import to_ogg

                            self._send(200, to_ogg(wav), "audio/ogg",
                                       {"Content-Disposition": f'inline; filename="{parts[2]}.ogg"'})
                        else:
                            self._send(200, wav.read_bytes(), "audio/wav")
                    elif parts == ["api", "settings"]:
                        self._json(server.settings())
                    elif parts == ["api", "logs"]:
                        from .logbuffer import HANDLER

                        self._json({"lines": HANDLER.since(int(q.get("after", 0)), int(q.get("limit", 500)))})
                    elif parts == ["api", "update"]:
                        self._json(server.software())
                    elif parts == ["api", "keys"]:
                        self._json({"keys": server.keys.list(), "scopes": SCOPES})
                    elif parts == ["api", "export"]:
                        self._send_backup()
                    elif parts == ["api", "report"]:
                        target = q.get("target", KNOCK)
                        if not valid_label(target):
                            raise ValueError("invalid target")
                        p = Path(server.cfg["data_dir"]) / "reports" / f"{target}.json"
                        self._json(json.loads(p.read_text()) if p.exists() else None)
                    else:
                        self._json({"error": "not found"}, 404)
                except KeyError:
                    self._json({"error": "no such clip"}, 404)
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                except Exception as exc:
                    log.exception("GET %s", self.path)
                    self._json({"error": str(exc)}, 500)

            do_HEAD = do_GET

            def do_POST(self):
                if not self._authorized():
                    return
                url = urlparse(self.path)
                q = {k: v[-1] for k, v in parse_qs(url.query).items()}
                parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
                try:
                    if parts == ["api", "import"]:
                        body = self._body()
                        self._json(server.import_audio(body, q.get("name", "upload"), q.get("label") or None,
                                                       q.get("split", "1") == "1"))
                        return
                    data = json.loads(self._body() or b"{}")
                    if len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "label":
                        self._json(server.store.set_label(parts[2], data.get("label") or None, data.get("note")))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "span":
                        self._json(server.store.set_span(parts[2], data.get("span")))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "split":
                        self._json(server.split_clip(parts[2], data.get("at")))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "unsplit":
                        self._json(server.store.unsplit(parts[2]))
                    elif len(parts) == 4 and parts[:2] == ["api", "clips"] and parts[3] == "delete":
                        server.store.delete(parts[2])
                        self._json({"ok": True})
                    elif parts == ["api", "bulk-label"]:
                        label = data.get("label") or None
                        done = [server.store.set_label(i, label)["id"] for i in data.get("ids", [])]
                        self._json({"updated": len(done)})
                    elif parts == ["api", "categories"]:
                        self._json({"categories": server.store.add_category(str(data.get("name", "")).strip())})
                    elif parts == ["api", "settings"]:
                        self._json(server.save_settings(data))
                    elif len(parts) == 3 and parts[:2] == ["api", "test"]:
                        self._json(server.run_test(parts[2], data))
                    elif parts == ["api", "update"]:
                        self._json(server.request_update())
                    elif parts == ["api", "keys"]:
                        rec, key = server.keys.create(str(data.get("name", "")), str(data.get("scope", "")))
                        log.info("API key %s (%s) created", rec["name"], rec["scope"])
                        self._json({"key": key, **rec})
                    elif len(parts) == 4 and parts[:2] == ["api", "keys"] and parts[3] == "revoke":
                        server.keys.revoke(parts[2])
                        log.info("API key %s revoked", parts[2])
                        self._json({"ok": True})
                    elif parts == ["api", "train"]:
                        started = server.start_training(data.get("target", KNOCK))
                        self._json({"started": started, "job": server.jobs.state()}, 200 if started else 409)
                    else:
                        self._json({"error": "not found"}, 404)
                except KeyError:
                    self._json({"error": "no such clip"}, 404)
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                except Exception as exc:
                    log.exception("POST %s", self.path)
                    self._json({"error": str(exc)}, 500)

        httpd = ThreadingHTTPServer((web["host"], int(web["port"])), Handler)
        httpd.daemon_threads = True
        log.info("labeling UI on http://%s:%s/", web["host"], web["port"])
        return httpd
