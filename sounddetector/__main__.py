"""Command line: `sound-detector <command> [-c config.yaml]`."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

from . import SAMPLE_RATE, __version__
from . import config as config_mod

DEFAULT_CONFIG = "/etc/sound-detector/config.yaml"


def _store(cfg):
    from .store import ClipStore

    c = cfg["collection"]
    return ClipStore(cfg["data_dir"], c["max_clips_per_hour"], c["max_disk_mb"], cfg.get("categories"))


def cmd_run(cfg, args):
    from .app import App
    from .web import Server

    app = App(cfg)
    if cfg["web"]["enabled"]:
        httpd = Server(cfg, app.store, app).serve()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def bye(*_):
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    app.run()
    app.stop()


def cmd_ui(cfg, args):
    """The labeling UI alone, without listening to the stream."""
    from .web import Server

    Server(cfg, _store(cfg)).serve().serve_forever()


def cmd_train(cfg, args):
    from .classifier import load_targets
    from .train import rescore, train
    from .yamnet import YamNet

    yam = YamNet(cfg["yamnet_model"])
    store = _store(cfg)
    t = cfg["training"]
    report = train(store, yam, args.target, cfg["data_dir"], C=float(t["C"]),
                   min_pos=int(t["min_positive_clips"]), min_neg=int(t["min_negative_clips"]),
                   false_alarms_per_week=float(t["false_alarms_per_week"]))
    print(f"\n{'label':<16}{'clips':>6}{'mean':>7}{'max':>7}{'above':>7}")
    for row in report["categories"]:
        print(f"{row['label']:<16}{row['clips']:>6}{row['mean_score']:>7.2f}{row['max_score']:>7.2f}"
              f"{row['above_threshold']:>7}")
    rule = report["threshold_rule"]
    print(f"\nthreshold {report['threshold']}: " + ", ".join(f"{k} {v}" for k, v in rule.items()))
    targets = sorted(set(cfg["detection"]["targets"]) | {args.target})
    rescore(store, yam, load_targets(cfg["data_dir"], targets, yam.sha256))
    print("\nA running service picks the new model up within 10 seconds.")


def cmd_import(cfg, args):
    from .web import Server

    server = Server(cfg, _store(cfg))
    for path in args.files:
        with open(path, "rb") as fh:
            res = server.import_audio(fh.read(), path, args.label, not args.whole)
        print(f"{path}: {res['saved']} clip(s)")


def cmd_check(cfg, args):
    """Print the stream's level once a second, to choose collection.min_dbfs."""
    import numpy as np

    from .audio import LiveSource
    from .events import Segmenter, rms_dbfs

    seg = Segmenter.from_config(cfg["collection"])
    src = LiveSource(config_mod.stream_url(cfg["source"]), cfg["source"]["stall_timeout"], 2, loop=False)
    end = time.time() + args.seconds
    buf = []
    for chunk in src:
        if chunk is None:
            print("stream ended or failed (see the log above)")
            break
        for ev in seg.feed(chunk):
            print(f"  event: peak {ev.peak_dbfs} dBFS over a {ev.background_dbfs} dBFS background, "
                  f"{(ev.end - ev.start) / SAMPLE_RATE:.1f}s")
        buf.append(chunk)
        if sum(map(len, buf)) >= SAMPLE_RATE:
            x = np.concatenate(buf)
            buf = []
            bg = f"{seg.background:6.1f}" if seg.background is not None else "   n/a"
            peak = 20 * np.log10(max(1, int(np.abs(x.astype(np.int32)).max())) / 32768)
            print(f"level {rms_dbfs(x):6.1f} dBFS   background {bg} dBFS   peak {peak:6.1f} dBFS")
        if time.time() > end:
            src.stop()
            break


def cmd_download_model(cfg, args):
    from .yamnet import download_model

    download_model(cfg["yamnet_model"])
    print(f"saved {cfg['yamnet_model']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sound-detector", description=__doc__)
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="listen, collect clips, detect, publish; serves the UI")
    sub.add_parser("ui", help="serve the labeling UI only")
    t = sub.add_parser("train", help="train a model for one label")
    t.add_argument("--target", default=config_mod.KNOCK)
    i = sub.add_parser("import", help="import audio/video files as clips")
    i.add_argument("files", nargs="+")
    i.add_argument("--label", default=None)
    i.add_argument("--whole", action="store_true", help="keep each file as one clip (max 30 s)")
    c = sub.add_parser("check", help="print the stream's levels, to tune collection.min_dbfs")
    c.add_argument("--seconds", type=int, default=60)
    sub.add_parser("download-model", help="download the pinned YAMNet model")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = args.config if os.path.exists(args.config) or args.config != DEFAULT_CONFIG else None
    cfg = config_mod.load(path)
    cfg["_config_path"] = path
    {"run": cmd_run, "ui": cmd_ui, "train": cmd_train, "import": cmd_import, "check": cmd_check,
     "download-model": cmd_download_model}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
