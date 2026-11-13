"""Tests that need neither the YAMNet model nor a network: synthetic audio and a fake embedder."""

import base64
import json
import threading
import urllib.request

import numpy as np
import pytest

from sounddetector import SAMPLE_RATE, config
from sounddetector.app import Ring
from sounddetector.classifier import TargetModel, load_targets
from sounddetector.events import Segmenter, segment
from sounddetector.store import ClipStore

RNG = np.random.default_rng(0)


def noise(seconds, level=30):
    return (RNG.normal(0, level, int(seconds * SAMPLE_RATE))).astype(np.int16)


def knock(audio, at, amp=4000, freq=600):
    """Add a decaying low thud at `at` seconds."""
    n = int(0.08 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    thud = amp * np.exp(-t * 60) * np.sin(2 * np.pi * freq * t)
    i = int(at * SAMPLE_RATE)
    audio[i:i + n] = np.clip(audio[i:i + n] + thud, -32768, 32767).astype(np.int16)
    return audio


def chime(audio, at, seconds=0.6, amp=6000, freq=3500):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    i = int(at * SAMPLE_RATE)
    audio[i:i + n] = np.clip(audio[i:i + n] + amp * np.sin(2 * np.pi * freq * t), -32768, 32767).astype(np.int16)
    return audio


class FakeYamNet:
    """Stands in for YAMNet: the 'embedding' is the share of energy in a few bands, plus noise."""

    sha256 = "f" * 64
    labels = [f"class_{i}" for i in range(521)]

    def windows(self, samples, hop=0.1):
        from sounddetector.events import rms_dbfs
        from sounddetector.yamnet import WINDOW

        step = int(hop * SAMPLE_RATE)
        starts = list(range(0, max(1, len(samples) - WINDOW + 1), step))
        embs, levels = [], []
        for s in starts:
            x = samples[s:s + WINDOW].astype(np.float64)
            spec = np.abs(np.fft.rfft(x, 2048)) ** 2
            bands = np.array([spec[a:b].sum() for a, b in [(0, 128), (128, 256), (256, 512), (512, 1025)]])
            emb = np.zeros(1024, dtype=np.float32)
            emb[:4] = bands / (bands.sum() + 1e-9)
            emb[4:] = RNG.normal(0, 0.01, 1020)
            embs.append(emb)
            levels.append(rms_dbfs(x))
        n = len(starts)
        return (np.stack(embs), np.zeros((n, 521), np.float32), np.array(levels, np.float32),
                np.array(starts, np.float32) / SAMPLE_RATE)

    def top_labels(self, scores, k=3):
        return []


def test_segmenter_finds_separate_events():
    audio = noise(12)
    knock(audio, 3.0)
    knock(audio, 3.3)
    chime(audio, 8.0)
    events = segment(audio, config.DEFAULTS["collection"])
    assert len(events) == 2
    first, second = events
    assert abs(first.onset / SAMPLE_RATE - 3.0) < 0.1
    assert abs(second.onset / SAMPLE_RATE - 8.0) < 0.1
    assert all((e.end - e.start) / SAMPLE_RATE >= 2.0 for e in events)  # min_clip
    assert first.peak_dbfs > first.background_dbfs + 15


def test_segmenter_ignores_quiet_sounds():
    audio = noise(6)
    knock(audio, 3.0, amp=40)
    assert segment(audio, config.DEFAULTS["collection"]) == []


def test_segmenter_caps_long_events():
    audio = noise(14)
    chime(audio, 2.0, seconds=10)
    events = segment(audio, config.DEFAULTS["collection"])
    assert events and max(e.end - e.start for e in events) <= 6 * SAMPLE_RATE


def test_streaming_matches_offline():
    audio = noise(8)
    knock(audio, 4.0)
    seg = Segmenter.from_config(config.DEFAULTS["collection"])
    events = []
    for i in range(0, len(audio), 1234):  # odd chunk sizes
        events += seg.feed(audio[i:i + 1234])
    assert [e.onset for e in events] == [e.onset for e in segment(audio, config.DEFAULTS["collection"])]


def test_ring_wraps():
    ring = Ring(1)
    data = np.arange(40000, dtype=np.int16)
    for i in range(0, len(data), 3000):
        ring.append(data[i:i + 3000])
    assert np.array_equal(ring.get(ring.pos - 100, ring.pos), data[-100:])


def test_store_labels_and_categories(tmp_path):
    store = ClipStore(tmp_path)
    clip = store.save(noise(2), {"source": "door"})
    assert store.get(clip["id"])["label"] is None
    store.set_label(clip["id"], "stroller")
    assert "stroller" in store.categories()
    with pytest.raises(ValueError):
        store.set_label(clip["id"], "../evil")
    reopened = ClipStore(tmp_path)
    assert reopened.get(clip["id"])["label"] == "stroller"
    assert reopened.stats()["labels"] == {"stroller": 1}


def test_store_rate_limit_and_prune_keep_labels(tmp_path):
    store = ClipStore(tmp_path, max_clips_per_hour=3, max_disk_mb=0)
    saved = [store.save(noise(2), {"source": "door"}) for _ in range(5)]
    assert sum(s is not None for s in saved) == 3 and store.dropped == 2
    labeled = saved[0]["id"]
    store.set_label(labeled, "knock")
    store.max_disk_mb = 0.07  # about one 2 s clip
    store.prune()
    assert labeled in {c["id"] for c in store.all()}
    assert len(store.all()) < 3


def _labeled_dataset(store):
    for i in range(8):
        a = knock(knock(noise(3), 1.0), 1.3 + 0.05 * i)
        store.set_label(store.save(a, {"source": "door"}, enforce_rate=False)["id"], "knock")
    for i in range(6):
        a = chime(noise(3), 1.0, freq=3000 + 100 * i)
        store.set_label(store.save(a, {"source": "door"}, enforce_rate=False)["id"], "doorbell_chime")
    for i in range(6):
        a = chime(noise(3), 1.0, freq=1800 + 50 * i, amp=3000)
        store.set_label(store.save(a, {"source": "door"}, enforce_rate=False)["id"], "not_knock")
    store.set_label(store.save(noise(3), {"source": "door"}, enforce_rate=False)["id"], "unsure")


def test_training_report_and_model(tmp_path):
    from sounddetector.train import NotEnoughData, train

    store = ClipStore(tmp_path)
    with pytest.raises(NotEnoughData):
        train(store, FakeYamNet(), "knock", tmp_path)
    _labeled_dataset(store)
    report = train(store, FakeYamNet(), "knock", tmp_path)
    assert report["clips"] == 20  # unsure left out
    # No full day of listening counted yet: no labeled everyday sound may alarm.
    rule = report["threshold_rule"]
    assert rule["events_per_day"] is None and rule["allowed_among_them"] == 0
    top_negative = max(c["score"] for c in report["clip_scores"] if c["label"] != "knock")
    assert report["threshold"] > top_negative or report["threshold"] == 0.99
    labels = {row["label"] for row in report["categories"]}
    assert labels == {"knock", "doorbell_chime", "not_knock"}
    # Training tried each sustain length and kept one; the model carries it to the detector.
    from sounddetector.features import SUSTAIN_OPTIONS

    assert [o["sustain"] for o in report["sustain_options"]] == list(SUSTAIN_OPTIONS)
    assert report["sustain"] in SUSTAIN_OPTIONS
    models = load_targets(tmp_path, ["knock"], FakeYamNet.sha256)
    assert set(models) == {"knock"}
    # A model trained on other features is refused.
    assert load_targets(tmp_path, ["knock"], "0" * 64) == {}
    # Each model lists its training set, with the labels the clips had then.
    data = json.loads((tmp_path / "models" / "knock.json").read_text())
    assert data["sustain"] == report["sustain"]
    assert len(data["training_set"]) == 20 and {lab for _, lab in data["training_set"]} == {
        "knock", "doorbell_chime", "not_knock"}
    # Retraining keeps the previous model and its report.
    train(store, FakeYamNet(), "knock", tmp_path)
    history = tmp_path / "models" / "history"
    assert len(list(history.glob("knock-*.report.json"))) == 1
    assert len([p for p in history.glob("knock-*.json") if not p.name.endswith(".report.json")]) == 1
    assert (tmp_path / "reports" / "knock.json").exists()


def test_training_on_quieter_copies(tmp_path):
    from sounddetector.train import train

    store = ClipStore(tmp_path)
    _labeled_dataset(store)
    report = train(store, FakeYamNet(), "knock", tmp_path, quieter_db=[-8, -16], min_dbfs=-120)
    assert report["quieter_copies_db"] == [-8, -16]
    # The copies are cached next to each clip, so the next training is quick.
    assert len(list(tmp_path.rglob("*.aug*.npz"))) == 20
    # The "maybe" threshold is never above the alarm threshold, and the model carries both.
    maybe = report["maybe"]
    assert maybe["threshold"] <= report["threshold"] and maybe["found"] >= report["cv"]["true_positives"]
    model = load_targets(tmp_path, ["knock"], FakeYamNet.sha256)["knock"]
    assert model.maybe_threshold == maybe["threshold"]
    # Copies too quiet for the collection gate are not learned from.
    from sounddetector.features import audible, quieter_copies

    clip = store.all()[0]["id"]
    copies = quieter_copies(store, FakeYamNet(), clip, [-8, -16], [])
    assert len(copies) == 2 and copies[0]["peak"] - copies[1]["peak"] == pytest.approx(8, abs=0.5)
    assert audible(copies, copies[0]["peak"] - 1) == copies[:1]


def test_each_saved_clip_is_published_with_its_score():
    from sounddetector.app import App

    class Pub:
        def __init__(self):
            self.sent = []

        def publish_clip(self, target, info):
            self.sent.append((target, info))

    class Model:
        threshold, maybe_threshold = 0.9, 0.4

    app = object.__new__(App)
    app.models, app.publisher = {"knock": Model()}, Pub()
    clip = {"id": "door-20261112-120000-000", "scores": {"knock": 0.55}, "recorded_at": "2026-11-12T12:00:00+02:00",
            "duration": 3.0, "peak_dbfs": -35.0, "suggested_spans": {"knock": [0.7, 1.5]}}
    app._publish_clip(clip, set())
    (target, info), = app.publisher.sent
    assert target == "knock" and info["score"] == 0.55 and not info["detected"]
    assert (info["threshold"], info["maybe_threshold"], info["span"]) == (0.9, 0.4, [0.7, 1.5])
    assert info["audio"] == "api/clips/door-20261112-120000-000/audio"


def test_score_sensor_is_an_event_log_not_a_statistic():
    from sounddetector.mqtt import Publisher

    class Client:
        def __init__(self):
            self.sent = {}

        def publish(self, topic, payload, retain=False):
            self.sent[topic] = payload

    pub = object.__new__(Publisher)
    pub.cfg = {"discovery": True, "discovery_prefix": "homeassistant", "topic_prefix": "sound_detector"}
    pub.source, pub.version, pub.client = "door", "0", Client()
    pub.base, pub.availability = "sound_detector/door", "sound_detector/door/availability"
    pub._discovery("knock")
    score = json.loads(pub.client.sent["homeassistant/sensor/sound_detector_door_knock_score/config"])
    # A state class makes Home Assistant compile 5-minute statistics for the sensor and hide it from the logbook.
    assert "state_class" not in score and "unit_of_measurement" not in score
    assert score["force_update"] is True


def test_binary_state_is_retained_so_home_assistant_has_one_after_its_restart():
    from sounddetector.mqtt import Publisher

    class Client:
        def __init__(self):
            self.sent = []

        def publish(self, topic, payload, retain=False):
            self.sent.append((topic, payload, retain))

    pub = object.__new__(Publisher)
    pub.base, pub.client = "sound_detector/door", Client()
    pub.publish_state("knock", True, {"score": 1.0})
    pub.publish_state("knock", False)
    states = [(p, r) for t, p, r in pub.client.sent if t == "sound_detector/door/knock"]
    assert states == [("ON", True), ("OFF", True)]


def test_label_suggestions(tmp_path):
    from sounddetector.app import model_outputs
    from sounddetector.features import clip_features
    from sounddetector.suggest import Suggester, report_file, train_suggester

    store = ClipStore(tmp_path)
    assert train_suggester(store, FakeYamNet(), tmp_path) is None  # nothing labeled yet
    _labeled_dataset(store)
    report = train_suggester(store, FakeYamNet(), tmp_path)
    assert set(report["classes"]) == {"knock", "doorbell_chime", "not_knock"}
    assert report["top1"] is not None and 0 <= report["top1"] <= report["top3"] <= 1
    assert report_file(tmp_path).exists()
    sug = Suggester.load(tmp_path, FakeYamNet.sha256)
    assert Suggester.load(tmp_path, "0" * 64) is None  # other YAMNet file
    knock_clip = next(m for m in store.all() if m["label"] == "knock")
    feats = clip_features(store, FakeYamNet(), knock_clip["id"])
    out = model_outputs({}, feats, store.samples(knock_clip["id"]), sug)
    (best, p), *rest = out["label_suggestions"]
    assert best == "knock" and 0 < p <= 1 and len(rest) == 2
    # Two labels: still a valid ranking (a binary fit, turned into two rows).
    s2 = ClipStore(tmp_path / "two")
    for _ in range(4):
        s2.set_label(s2.save(knock(noise(3), 1.0), {"source": "d"}, enforce_rate=False)["id"], "knock")
        s2.set_label(s2.save(chime(noise(3), 1.0), {"source": "d"}, enforce_rate=False)["id"], "door")
    assert train_suggester(s2, FakeYamNet(), tmp_path / "two")["classes"] == ["door", "knock"]
    s = Suggester.load(tmp_path / "two", FakeYamNet.sha256)
    assert s.probs(np.zeros((1, 1024))).sum() == pytest.approx(1)


def test_event_counts_give_events_per_day(tmp_path):
    import time as _t

    store = ClipStore(tmp_path)
    assert store.events_per_day() is None
    for days_ago, n in ((3, 100), (2, 120), (1, 80)):
        for _ in range(n):
            store.count_event(_t.time() - days_ago * 86400)
    store.count_event()  # today does not count until it is over
    assert store.events_per_day() == 100


def test_model_roundtrip(tmp_path):
    data = {"target": "knock", "mean": [0.0] * 3, "scale": [1.0] * 3, "coef": [1.0, 0.0, 0.0],
            "intercept": 0.0, "threshold": 0.5}
    TargetModel(data).save(tmp_path / "m.json")
    m = TargetModel.load(tmp_path / "m.json")
    p = m.predict(np.array([[10.0, 0, 0], [-10.0, 0, 0]]))
    assert p[0] > 0.99 and p[1] < 0.01


def test_web_api(tmp_path):
    from sounddetector.web import Server

    cfg = config.load(None)
    cfg["data_dir"] = str(tmp_path)
    cfg["web"].update({"host": "127.0.0.1", "port": 0, "password": "pw"})
    store = ClipStore(tmp_path)
    clip = store.save(noise(2), {"source": "door"})
    httpd = Server(cfg, store).serve()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    auth = {"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()}

    def call(path, data=None, headers=auth):
        req = urllib.request.Request(base + path, headers={**headers, "Content-Type": "application/json"},
                                     data=None if data is None else json.dumps(data).encode())
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    with pytest.raises(urllib.error.HTTPError) as err:
        call("/api/clips", headers={})
    assert err.value.code == 401
    listed = json.loads(call("/api/clips?label=unlabeled"))
    assert listed["total"] == 1
    call(f"/api/clips/{clip['id']}/label", {"label": "knock"})
    assert json.loads(call("/api/clips?label=knock"))["total"] == 1
    assert json.loads(call(f"/api/clips?label=all&find={clip['id'][-8:].upper()}"))["total"] == 1
    assert json.loads(call("/api/clips?label=all&find=no-such-file.mp4"))["total"] == 0
    assert call(f"/api/clips/{clip['id']}/audio")[:4] == b"RIFF"
    import shutil

    if shutil.which("ffmpeg"):  # Ogg/Opus, for a Telegram voice note
        assert call(f"/api/clips/{clip['id']}/audio?format=ogg")[:4] == b"OggS"
    assert b"Sound detector" in call("/")
    # API keys: made with the password, shown once, stored hashed.
    made = json.loads(call("/api/keys", {"name": "Home Assistant", "scope": "labeling"}))
    key, full = made["key"], json.loads(call("/api/keys", {"name": "Claude", "scope": "full"}))["key"]
    assert key not in (tmp_path / "api_keys.json").read_text()
    assert [k["name"] for k in json.loads(call("/api/keys"))["keys"]] == ["Home Assistant", "Claude"]
    bearer = {"Authorization": f"Bearer {key}"}
    assert call(f"/api/clips/{clip['id']}/audio", headers=bearer)[:4] == b"RIFF"
    labeled = call(f"/api/clips/{clip['id']}/label", {"label": "knock"}, headers={"X-API-Key": key})
    assert json.loads(labeled)["label"] == "knock"
    for path, data in (("/api/settings", None), ("/api/update", {}), ("/api/keys", None)):
        with pytest.raises(urllib.error.HTTPError) as err:
            call(path, data, headers=bearer)
        assert err.value.code == 403
    assert json.loads(call("/api/settings", headers={"Authorization": f"Bearer {full}"}))
    with pytest.raises(urllib.error.HTTPError) as err:  # not even a full key manages keys
        call("/api/keys", {"name": "x", "scope": "full"}, headers={"Authorization": f"Bearer {full}"})
    assert err.value.code == 403
    call(f"/api/keys/{made['id']}/revoke", {})
    with pytest.raises(urllib.error.HTTPError) as err:
        call("/api/clips", headers=bearer)
    assert err.value.code == 401
    # Not installed by install.sh: the UI can show the version, but not update.
    sw = json.loads(call("/api/update"))
    assert sw["can_update"] is False and sw["installed"] is None
    with pytest.raises(urllib.error.HTTPError) as err:
        call("/api/update", {})
    assert err.value.code == 400 and not (tmp_path / "update-request").exists()
    store.update(clip["id"], label_suggestions=[["door", 0.8], ["voice", 0.1]])
    assert json.loads(call("/api/clips?label=all&suggested=door"))["total"] == 1
    assert json.loads(call("/api/clips?label=all&suggested=voice"))["total"] == 0
    (tmp_path / "settings.yaml").write_text("mqtt: {password: secret}")
    (tmp_path / "models" / "history").mkdir(parents=True)
    (tmp_path / "models" / "history" / "knock-old.json").write_text("{}")
    store.wav_path(clip["id"]).with_name(f"{clip['id']}.embabc.npz").write_bytes(b"x")
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(call("/api/export")), mode="r:gz") as tar:
        names = tar.getnames()
        assert json.load(tar.extractfile("backup.json"))["stats"]["labels"] == {"knock": 1}
    assert {f"clips/{store.wav_path(clip['id']).parent.name}/{clip['id']}.{x}" for x in ("wav", "json")} <= set(names)
    assert "models/history/knock-old.json" in names
    assert not any(n.endswith(".npz") or "settings" in n for n in names)
    with pytest.raises(urllib.error.HTTPError) as err:
        call("/api/clips/..%2F..%2Fetc%2Fpasswd/audio")
    assert err.value.code == 404
    httpd.shutdown()


def test_redact_hides_stream_login():
    from sounddetector.audio import redact

    msg = "rtsp://restream:s3cr%40t@10.1.2.3:8554/door: Connection refused"
    assert "s3cr" not in redact(msg) and "rtsp://***@10.1.2.3:8554/door" in redact(msg)


def test_budget_threshold():
    from sounddetector.train import budget_threshold

    neg = np.array([0.9, 0.6, 0.3] + [0.05] * 697)  # 700 labeled everyday sounds
    # 100 sounds a day = 700 a week, so 1 false alarm a week allows 1 of the 700 above.
    r = budget_threshold(neg, 100, 1.0)
    assert r["threshold"] == 0.61 and r["rule"]["above_threshold"] == 1 and r["rule"]["met"]
    assert r["rule"]["estimated_per_week"] == 1.0
    # Ten times the sounds per day: none may alarm, so just above the highest one.
    assert budget_threshold(neg, 1000, 1.0)["threshold"] == 0.91
    # Too few labeled sounds to verify: say how many are needed.
    small = budget_threshold(np.array([0.2] * 20), 100, 1.0)
    assert small["rule"]["resolution_per_week"] == 35.0 and small["rule"]["sounds_needed"] == 700
    assert small["threshold"] == 0.5  # never below the model's own decision point
    # Unknown rate: nothing labeled may alarm.
    assert budget_threshold(neg, None, 1.0)["threshold"] == 0.91


def test_stream_url_encodes_login():
    from sounddetector.config import stream_url

    src = {"url": "rtsp://host:8554/door", "username": "restream", "password": "p@ss/w:rd#1"}
    assert stream_url(src) == "rtsp://restream:p%40ss%2Fw%3Ard%231@host:8554/door"
    assert stream_url({"url": "rtsp://u:p@host/x", "username": "a"}) == "rtsp://u:p@host/x"
    assert stream_url({"url": "/tmp/file.mp4", "username": "a"}) == "/tmp/file.mp4"
    assert stream_url({"url": "rtsp://host/x"}) == "rtsp://host/x"


def test_settings_override_and_secrets(tmp_path):
    import yaml

    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump({"data_dir": str(tmp_path), "mqtt": {"host": "a", "password": "old"}}))
    cfg = config.load(conf)
    config.save_settings(cfg, {"mqtt": {"host": "b", "password": "", "port": "1884"},
                               "detection": {"targets": "knock, footsteps", "thresholds": {"knock": "0.8"}}})
    cfg2 = config.load(conf)
    assert cfg2["mqtt"]["host"] == "b" and cfg2["mqtt"]["port"] == 1884
    assert cfg2["mqtt"]["password"] == "old"  # empty secret keeps the current one
    assert cfg2["detection"]["targets"] == ["knock", "footsteps"]
    assert cfg2["detection"]["thresholds"] == {"knock": 0.8}
    view = config.editable_view(cfg2)
    assert view["mqtt"]["password"] == {"set": True}  # never the value
    config.save_settings(cfg2, {"mqtt": {"password": {"clear": True}}})
    assert config.load(conf)["mqtt"]["password"] is None
    assert (tmp_path / "settings.yaml").stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError):
        config.save_settings(cfg2, {"data_dir": {"x": 1}})
    with pytest.raises(ValueError):
        config.save_settings(cfg2, {"detection": {"targets": "../x"}})


def test_clips_saved_in_the_same_millisecond_get_usable_ids(tmp_path):
    store = ClipStore(tmp_path)
    ids = [store.save(noise(1), {"source": "door", "recorded_at": 1700000000.0}, enforce_rate=False)["id"]
           for _ in range(3)]
    assert len(set(ids)) == 3
    for clip_id in ids:
        assert store.wav_path(clip_id).exists()
        store.set_label(clip_id, "knock")


def test_stream_check_measures_a_file(tmp_path):
    from sounddetector.audio import write_wav
    from sounddetector.checks import test_stream

    audio = knock(noise(4), 2.0)
    write_wav(tmp_path / "a.wav", audio)
    r = test_stream(str(tmp_path / "a.wav"))
    assert r["ok"] and r["audio_tracks"][0]["sample_rate"] == "16000"
    assert r["loudest_dbfs"] > r["level_dbfs"]
    bad = test_stream(str(tmp_path / "missing.wav"))
    assert not bad["ok"] and bad["error"]


def test_mqtt_check_reports_an_unreachable_broker():
    from sounddetector.checks import test_mqtt

    r = test_mqtt("127.0.0.1", 1, None, None, timeout=2)
    assert not r["ok"] and "cannot reach" in r["error"]


def test_log_buffer_masks_stream_logins():
    import logging

    from sounddetector.logbuffer import BufferHandler

    h = BufferHandler()
    log = logging.getLogger("sounddetector.test")
    log.addHandler(h)
    log.warning("ffmpeg: rtsp://restream:hunter2@10.0.0.5:8554/door: 401 Unauthorized")
    line = h.since(0)[-1]
    assert "hunter2" not in line["msg"] and line["name"] == "test" and line["level"] == "WARNING"
    assert h.since(line["seq"]) == []


def test_marked_span_limits_training_windows(tmp_path):
    from sounddetector.features import span_windows, training_windows

    starts = np.arange(0, 4.4, 0.1)  # a 5.36 s clip
    sel = span_windows(starts, [3.4, 4.2])
    # Only windows that sit mostly inside the knock, none from the talking at the start.
    assert starts[sel].min() >= 2.85 and starts[sel].max() <= 3.85
    assert len(span_windows(starts, [5.3, 5.36])) == 1  # never empty
    # A louder sound outside the span no longer decides which windows are used.
    levels = np.full(len(starts), -60.0, np.float32)
    levels[:10] = -30.0
    picked = training_windows({"levels": levels, "starts": starts}, [3.4, 4.2])
    assert len(picked) and set(picked) <= set(sel)

    store = ClipStore(tmp_path)
    clip = store.save(noise(5), {"source": "door"})
    assert store.set_span(clip["id"], [4.2, 3.4])["span"] == [3.4, 4.2]
    assert store.set_span(clip["id"], [-1, 99])["span"] == [0.0, 5.0]
    with pytest.raises(ValueError):
        store.set_span(clip["id"], [1, 1.05])
    with pytest.raises(ValueError):
        store.set_span(clip["id"], "all")
    assert store.set_span(clip["id"], None)["span"] is None


def test_model_suggests_where_the_knock_is():
    from sounddetector.features import suggest_span

    audio = knock(noise(5, level=20), 3.6, amp=8000)
    starts = np.arange(0, 4.05, 0.1)
    probs = np.where((starts > 2.7) & (starts < 3.6), 0.9, 0.05)  # windows holding the knock
    span = suggest_span(probs, starts, audio)
    assert span[0] <= 3.6 <= span[1] and span[1] - span[0] < 1.0
    assert suggest_span(np.full(len(starts), 0.2), starts, audio) is None


def test_sustained_score_needs_a_run_of_windows():
    from sounddetector.features import sustained_score

    starts = np.arange(0, 3.0, 0.1)
    click = np.full(len(starts), 0.01)
    click[10:13] = 0.999  # one short sound: high in a few windows
    series = np.full(len(starts), 0.01)
    series[10:20] = [0.99, 0.97, 0.995, 0.98, 0.99, 0.999, 0.96, 0.99, 0.98, 0.97]  # a knock series
    assert sustained_score(click, starts, 0.0) == pytest.approx(0.999)
    assert sustained_score(click, starts, 0.4) == pytest.approx(0.01)
    assert sustained_score(series, starts, 0.4) == pytest.approx(0.97)
    assert sustained_score(series, starts, 0.8) == pytest.approx(0.96)
    # A clip too short for the run is scored as a whole.
    assert sustained_score(np.array([0.9, 0.95]), np.array([0.0, 0.1]), 0.4) == pytest.approx(0.9)


def test_live_detector_waits_for_the_sustain_time():
    from sounddetector.app import App, TargetState

    class Seg:
        active, last_active = True, 0

    class Yam:
        def __init__(self):
            self.p = 0.0

        def window(self, samples):
            return np.array([self.p], dtype=np.float32), None

    class Model:
        threshold, sustain = 0.9, 0.4

        def predict(self, emb):
            return np.asarray(emb, dtype=np.float64)

    app = object.__new__(App)
    app.cfg = {"detection": {"hop": 0.1, "off_delay": 10.0, "min_hits": 1}}
    app.models, app.states = {"knock": Model()}, {"knock": TargetState()}
    app.ring, app.segmenter, app.yamnet = Ring(30), Seg(), Yam()
    app._since_hop, app._hits, app.publisher = 0, __import__("collections").deque(maxlen=200), None
    step = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.int16)

    def feed(p, n):
        app.yamnet.p = p
        for _ in range(n):
            app.ring.append(step)
            app._detect(len(step))

    feed(0.0, 12)
    feed(0.99, 3)  # a click: 0.2 s above the threshold
    feed(0.0, 2)
    assert not app.states["knock"].on and not app._hits
    feed(0.99, 5)  # a knock series: 0.4 s above it
    assert app.states["knock"].on and len(app._hits) == 1


def test_split_a_clip_and_undo_it(tmp_path):
    store = ClipStore(tmp_path)
    audio = knock(chime(noise(4), 0.5), 3.0)
    clip = store.save(audio, {"source": "import", "imported_from": "door.mp4", "import_offset": 10.0,
                              "detections": [{"target": "knock", "score": 0.99, "offset": 3.1}]})
    store.set_label(clip["id"], "doorbell_chime")
    a, b = store.split(clip["id"], 2.0)
    assert [p["duration"] for p in (a, b)] == [2.0, 2.0]
    assert a["label"] is None and b["label"] is None and a["split_from"] == clip["id"]
    assert (a["import_offset"], b["import_offset"]) == (10.0, 12.0)
    assert a["detections"] == [] and b["detections"][0]["offset"] == 1.1
    assert np.array_equal(np.concatenate([store.samples(a["id"]), store.samples(b["id"])]), audio)
    assert {m["id"] for m in store.all()} == {a["id"], b["id"]}
    with pytest.raises(ValueError):
        store.split(a["id"], 0.1)  # parts must be at least MIN_PART long
    # A part split again must be undone first.
    store.split(b["id"], 1.0)
    with pytest.raises(ValueError):
        store.unsplit(clip["id"])
    store.unsplit(b["id"])
    restored = store.unsplit(clip["id"])
    assert restored["label"] == "doorbell_chime"
    assert [m["id"] for m in store.all()] == [clip["id"]]
    assert np.array_equal(store.samples(clip["id"]), audio)
    # It survives a restart.
    assert [m["id"] for m in ClipStore(tmp_path).all()] == [clip["id"]]
