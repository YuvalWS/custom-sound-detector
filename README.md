# custom-sound-detector

Detect one specific sound, like a knock on your front door, from a camera's audio stream, with
a model trained on your own recordings. It collects every distinct sound it hears, gives you a
web UI to label them, trains a small classifier from those labels and publishes detections to
Home Assistant over MQTT.

## Why

Generic sound classifiers are trained on YouTube audio, not on your door through your camera's
microphone. Frigate's audio detection, for example, runs Google's stock
[YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) on fixed,
back-to-back ~1 second windows and needs a score of at least 0.5. A short knock gets a score that
depends heavily on where it falls inside the window, and a knock that YAMNet doesn't recognise
can't be fixed by any setting.

This project keeps YAMNet as a **feature extractor** (the 1,024 numbers it computes for each
window before picking a label) and trains a tiny logistic-regression classifier on top of
them, from your own clips. That needs dozens of examples, not thousands, trains in seconds on
a CPU, and gets better every time you label a missed sound or a false alarm.

## How it works

1. **Listen.** ffmpeg reads the audio stream (RTSP, HTTP or anything else ffmpeg reads) as
   16 kHz mono.
2. **Collect.** A sound event starts when the level rises well above the background, and is
   saved as a short WAV clip with half a second of context. Each clip gets YAMNet's own top
   guesses as hints ("Knock 0.59, Speech 0.41").
3. **Label.** In the web UI: knock, not a knock, or a category (footsteps, voice, keys, door,
   ...). Categories are optional; they show which sounds the model confuses, and any category
   can become a model of its own later.
4. **Train.** One click. The report shows cross-validated precision and recall, the score
   every category gets, and the clips the model got wrong (usually a wrong label).
5. **Detect.** While a sound is active, a 0.975 s window is classified every 0.25 s, so a
   short sound is always seen whole. A hit turns an MQTT binary sensor on; it turns off
   after a quiet period. The clip that triggered it is saved, so false alarms can be labeled.

Classification runs only while something is louder than the background. On a small x86 box
this is a fraction of one core.

## Install

### Proxmox VE (new LXC container)

On the Proxmox host shell, as root:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/YuvalWS/custom-sound-detector/main/proxmox/sound-detector.sh)"
```

It asks for the container settings (ID, storage, network) and the detector settings (stream
URL, MQTT broker), creates an unprivileged Debian container, installs everything, and prints
the address and password of the labeling UI.

To update later:

```bash
pct exec <ctid> -- update
```

### Any Debian or Ubuntu machine

```bash
curl -fsSL https://raw.githubusercontent.com/YuvalWS/custom-sound-detector/main/proxmox/install.sh | sudo bash
```

Then set `source.url`, the MQTT settings and `web.password` in
`/etc/sound-detector/config.yaml`, and run `sudo systemctl restart sound-detector`. Update
with `sudo update`.

### What the installer sets up

| Path | What |
|---|---|
| `/opt/sound-detector` | the code (a git checkout) and its Python virtualenv |
| `/etc/sound-detector/config.yaml` | the configuration (readable only by the service) |
| `/var/lib/sound-detector` | clips, labels, models, reports and the YAMNet model file |
| `sound-detector.service` | the systemd service, running as the `sound-detector` user |
| `/usr/local/bin/sound-detector` | the command line tool |
| `/usr/bin/update` | updates to the latest version and restarts |

## Getting a stream

Any URL ffmpeg can open works. With [Frigate](https://frigate.video), use its go2rtc restream
(`rtsp://<frigate-host>:8554/<stream>`). If that stream offers more than one audio track
(for example AAC plus an Opus copy for browsers), give the detector a stream with a single
track, the same way Frigate's own audio role needs one:

```yaml
# Frigate config.yml, go2rtc section
go2rtc:
  streams:
    front_door_audio:
      - ffmpeg:front_door#audio=copy
```

If the restream needs a login (go2rtc's `rtsp: username/password`), put it in the URL:
`rtsp://user:password@<frigate-host>:8554/<stream>`, with characters like `@ : / # ? %` in the
password percent-encoded. The detector never writes the login to its log.

Then check the levels, to see that audio arrives and to choose `collection.min_dbfs`:

```bash
sound-detector check --seconds 30
```

## Labeling

Open `http://<container-ip>:8080`. The **Label** tab plays one clip at a time:

| Key | Action |
|---|---|
| `K` | knock |
| `N` | not a knock |
| `1`-`9`, `0` | one of the first ten categories, instead of "not a knock" |
| `U` | unsure: kept, never used for training |
| `Space` | play / pause (click the spectrogram to play from a point) |
| `←` `→` | previous / next clip |
| `Z` | undo |
| `Del` | delete the clip |

On a phone the same buttons are big enough to tap. **Browse** lists clips with filters and
bulk labeling ("label these 200 clips as not a knock"). **Import** takes recordings you
already have, such as a Frigate clip of someone knocking, and splits them into sounds.

A good first dataset:

- **Knocks:** 20-30 knocking sessions: different people, soft and hard, knuckles and palm,
  day and night. Knock on purpose a few times a day for a week, then label those clips.
- **Everything else:** the clips collected during the same week. Mark them "not a knock" in
  bulk, and give a category to the ones you have time for.

## Training

**Train** tab, or `sound-detector train --target knock`. The rules:

- Positives are clips labeled with the target. Negatives are all other labeled clips,
  `not_knock` and every category. `unsure` and unlabeled clips are left out.
- Evaluation is 5-fold cross-validation by clip: the numbers come from clips the model did
  not see. A clip's score is its best window, the same as the live detector sees it.
- The threshold is the one with the best F1 on those scores; override it per target with
  `detection.thresholds`.
- At least 5 target clips and 10 others are needed.

The running service picks up a new model within seconds. The previous one is kept in
`models/history/`. A model is tied to the YAMNet file it was trained with, so a different
YAMNet file means retraining.

## Home Assistant

With MQTT discovery on (the default), every trained target appears as a binary sensor:
`binary_sensor.sound_detector_<source>_<target>`, with device class `sound`. The state is
`ON` from the first hit until `detection.off_delay` seconds without one. Attributes: `score`,
`threshold`, `model_trained_at`, and `clip_id` of the clip that triggered it.

```yaml
automation:
  - alias: Knock at the door
    triggers:
      - trigger: state
        entity_id: binary_sensor.sound_detector_front_door_knock
        to: "on"
    actions:
      - action: notify.notify
        data:
          message: Someone is knocking at the door.
```

Raw topics, under `topic_prefix` (default `sound_detector`):

| Topic | Payload |
|---|---|
| `sound_detector/<source>/<target>` | `ON` / `OFF` |
| `sound_detector/<source>/<target>/attributes` | JSON, retained |
| `sound_detector/<source>/availability` | `online` / `offline`, retained |

## Command line

```text
sound-detector run              listen, collect, detect, publish, serve the UI (the service)
sound-detector ui               serve the labeling UI only
sound-detector train --target knock
sound-detector import FILE...   [--label knock] [--whole]
sound-detector check            print the stream's levels
sound-detector download-model   fetch the pinned YAMNet model
```

The installed wrapper reads `/etc/sound-detector/config.yaml`; pass `-c` for another file.

## Configuration

All options, with defaults and explanations, are in
[`sounddetector/config.py`](sounddetector/config.py). The common ones are in
[`config.example.yaml`](config.example.yaml).

## Privacy and backups

Clips are recordings of whoever is near your door, voices included. They stay on the machine
running the detector: nothing is uploaded anywhere. Keep `web.password` set, don't expose the
UI to the internet, and never commit the data directory to git.

Your labels are the valuable part: back up `/var/lib/sound-detector` (with the Proxmox
installer, a backup of the container covers it). Labeled clips are never deleted by the disk
cap; the oldest unlabeled ones go first.

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
```

The tests need neither the YAMNet model nor a network.

## Credits and license

MIT, see [LICENSE](LICENSE). YAMNet is by Google, released under the Apache 2.0 license; the
installer downloads the model file from Google at install time, it is not part of this repository.
