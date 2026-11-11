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

### What we measured

On a 19-second recording from a doorbell camera, with four knocking spells, run through the same
pipeline as Frigate (0.975 s windows, the same YAMNet model, its 0.5 floor):

| Knocking spell | Best "knock" score, at any window position | Detected by Frigate? |
|---|---|---|
| 1st | 0.67 (0.11 where Frigate's windows actually fell) | only if a window happens to start right, about 1 time in 10 |
| 2nd | 0.02 | never |
| 3rd | 0.15 | never |
| 4th | 0.03 | never |

YAMNet heard the later spells as speech, animals or silence. So the window position is a real
problem, but the bigger one is that the stock model doesn't recognise these knocks at all.
A classifier trained on three of the spells, on YAMNet's features, recognised the fourth
(median scores 0.55 to 0.99 for each held-out spell), while the doorbell chime in the same
recording scored at most 0.11. One recording proves little; that is what the labeling is for.

### The approach

This project keeps YAMNet as a **feature extractor** (the 1,024 numbers it computes for each
window before picking a label) and trains a tiny logistic-regression classifier on top of
them, from your own clips. That needs dozens of examples, not thousands, trains in seconds on
a CPU, and gets better every time you label a missed sound or a false alarm.

## How it works

1. **Listen.** ffmpeg reads the audio stream (RTSP, HTTP or anything else ffmpeg reads) as
   16 kHz mono.
2. **Collect.** A sound event starts when the level rises 15 dB above the background and is
   at least `collection.min_dbfs` loud (-45 dBFS by default), and is saved as a short WAV clip
   with half a second of context. Nothing else is recorded: there is no continuous recording. Each clip gets YAMNet's own top
   guesses as hints ("Knock 0.59, Speech 0.41").
3. **Label.** In the web UI: knock, not a knock, or a category (footsteps, voice, keys, door,
   ...). Categories are optional; they show which sounds the model confuses, and any category
   can become a model of its own later.
4. **Train.** One click. The report shows how many knocks the model finds, how many false
   alarms a week to expect, the score every category gets, and the clips the model got wrong
   (usually a wrong label).
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

It works like the community Proxmox VE helper scripts: a text UI with **Default settings**,
**Advanced settings** (container ID, hostname, CPU, memory, disk, storage, bridge, IP) and,
from the second run on, **My saved defaults**. Advanced settings can be saved as your defaults
(`/usr/local/community-scripts/defaults/sound-detector.vars`, never the passwords). Then it asks
for the stream (URL, username, password) and MQTT (host, port, username, password), creates an
unprivileged Debian container, installs everything, and prints the address and the generated
password of the web UI. Every setting can be changed later in the web UI.

Without a terminal, or with `NONINTERACTIVE=1`, it takes the settings from environment
variables (listed at the top of the script).

To update later:

```bash
pct exec <ctid> -- update
```

An update replaces only the code in `/opt/sound-detector` and its Python packages. It never
touches the config file, the data directory (clips, labels, models, reports, `settings.yaml`)
or the YAMNet model file, and it restarts the service.

### Any Debian or Ubuntu machine

```bash
curl -fsSL https://raw.githubusercontent.com/YuvalWS/custom-sound-detector/main/proxmox/install.sh | sudo bash
```

It prints the web UI's generated password; set the stream and MQTT in its **Settings** tab.
Update with `sudo update`.

### What the installer sets up

| Path | What |
|---|---|
| `/opt/sound-detector` | the code (a git checkout) and its Python virtualenv |
| `/etc/sound-detector/config.yaml` | the configuration written at install time (readable only by root and the service) |
| `/var/lib/sound-detector` | clips, labels, models, reports, the YAMNet model file, and `settings.yaml` (changes made in the web UI) |
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

If the restream needs a login (go2rtc's `rtsp: username/password`), enter it as the stream
username and password, not in the URL: it is percent-encoded for you, so any characters work.

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
- **The threshold is set by false alarms, not by precision.** Precision on your labels depends
  on how many knocks you recorded on purpose, so it says little about real life. Instead,
  the share of labeled everyday sounds that would alarm, times how many sounds a day the
  detector hears (it counts them), gives false alarms per week. The threshold is the lowest
  one (so the most knocks found) that stays within `training.false_alarms_per_week`, 1 by
  default. With few labeled everyday sounds the estimate is coarse; the report says how many
  are needed to check the budget at all. Until a full day has been counted, no labeled
  everyday sound may alarm. The threshold never goes below 0.5, where the model itself starts
  saying "yes": the budget can only raise it, so kinds of sound nobody labeled don't slip
  through a low threshold. Override it per target with `detection.thresholds`.
- At least 5 target clips and 10 others are needed.

The running service picks up a new model within seconds. The previous one is kept in
`models/history/`, with the report it came with. Every model file (about 45 kB, plus about
50 bytes per training clip) lists its training set: each clip's id and the label it had at
training time, so any past model stays traceable after labels change or clips are deleted. A model is tied to the YAMNet file it was trained with, so a different
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

The **Settings** tab of the web UI changes the common settings: the stream and its login,
MQTT, collection, detection, the false-alarm budget and the UI's own password. "Test
connection" checks the stream (its audio tracks and a few seconds of level) and the MQTT
login with the values in the form, before you save. Saving restarts the service. The **Logs**
tab shows the service's recent log. The changes go to `settings.yaml` in the data directory and override
the config file key by key (the service can't write to `/etc`).

All options, with defaults and explanations, are in
[`sounddetector/config.py`](sounddetector/config.py); the common ones are in
[`config.example.yaml`](config.example.yaml).

### Passwords

The stream, MQTT and web UI passwords are stored in plain text in the config file and in
`settings.yaml`, readable only by root and the service user, the same way Frigate keeps its
camera passwords in its `.env`. They are never sent to the browser (the Settings tab only
shows whether one is set), never written to the log (a login in an ffmpeg error is masked),
and never part of the repository. The Proxmox helper's saved defaults leave them out.

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
