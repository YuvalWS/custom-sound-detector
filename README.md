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
   guesses as hints ("Knock 0.59, Speech 0.41"). Settings > Collection chooses what is saved:
   **all** sounds (to collect training data), **debug** (only what the knock model detects or
   scores at least its "maybe" threshold: the alerts and near-misses, to check and fine-tune
   the model), or **off** (nothing; Home Assistant then gets no recordings either).
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

Or press **Update now** in Settings → Software update, which shows what changed since the
installed version. The service runs unprivileged, so the button only writes a request file
in the data directory; a systemd path unit (`sound-detector-update.path`, set up by the
installer) sees it and runs `update` as root. It works on any install made by `install.sh`,
Proxmox or not. Installs from before the button need one `update` from the shell to set it up.

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
  not see. Every window of a held-out clip is scored, as the live detector hears it.
- **A knock must last.** A clip's score is the level its window scores stay at for a
  *sustain* time (0 to 0.8 s): a knock is a series of hits that keeps the score high, while
  one click or clack scores high only briefly. Training tries each sustain time and keeps
  the one that finds the most knocks within the false-alarm budget; the live detector then
  needs the score above the threshold for that long before it reports a knock.
- **Soft knocks.** Training also picks a lower `maybe_threshold`
  (`training.maybe_per_week`, 7 a week), for a quiet heads-up about knocks it is less sure
  of. Optionally, `training.quieter_copies_db: [-8, -16]` also learns from copies of every
  clip that much quieter, mixed with real background from your quietest `not_knock` clips:
  a soft knock is a loud one heard from further away. Copies too quiet to pass
  `collection.min_dbfs` are left out, every class gets copies (with knocks alone the model
  learns that quiet means knock), and evaluation always scores the clips as recorded. It is
  off by default: on 41 knocks and 577 other sounds it raised soft knocks and footsteps
  alike, and found 31 knocks instead of 32. The first training with it takes minutes.
- A clip with two sounds (a beep, then a knock) can be split in the Label tab: click
  between them and press S, then label each part. Undo puts the clip back.
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

**Label suggestions.** Every training also trains one more model, over all your labels at
once (a multinomial logistic regression on the same features, labels with at least 3 clips).
It suggests a label for every clip: in the Label tab press A to accept the first suggestion
or click another; in Browse, filter by suggested label (then select all and label them in
bulk) or sort by how sure it is. The Train tab shows how often it was right on clips it had
not seen. Detection never uses it. "Not a knock" is a catch-all (quiet noise, sounds nobody named,
and named sounds labeled in bulk), so a second model is trained on the named kinds of sound
alone; when it is sure (50% or more), its kind is suggested before "Not a knock". To find
bulk-labeled clips worth naming, filter Browse to "Not a knock" and "Suggested: Door".
On the first real dataset (614 clips, 9 labels) its first
suggestion was right 81% of the time, and the right label was in its top three 97%.

After training, the clips are rescored with the new models (their knock scores and suggested
labels). By default only the labeled clips and the last 24 hours of clips are; the choice under
the Train button can rescore every clip, or none.

The running service picks up a new model within seconds. The previous one is kept in
`models/history/`, with the report it came with. The **Models** list in the Train tab shows every
model trained, with what it found and its thresholds: "results" opens a model's full report, and
"Make active" switches the detector back to it (the current one stays in the list). Every model file (about 45 kB, plus about
50 bytes per training clip) lists its training set: each clip's id and the label it had at
training time, so any past model stays traceable after labels change or clips are deleted. A model is tied to the YAMNet file it was trained with, so a different
YAMNet file means retraining.

## Home Assistant

With MQTT discovery on (the default), every trained target appears as two entities:

- `binary_sensor.sound_detector_<source>_<target>` (device class `sound`): `ON` from the
  first live hit until `detection.off_delay` seconds without one. Attributes: `score`,
  `threshold`, `maybe_threshold`, `model_trained_at`, and `clip_id` of the clip that
  triggered it. This is the fastest signal, under a second after the knock.
- `sensor.sound_detector_<source>_<target>_score`: the model's score for every clip the
  detector saves, a second or two after the sound ends. Its attributes are the clip message
  below, so an automation can alert at two levels and fetch the recording.

Each saved clip is published on `sound_detector/<source>/<target>/clip`:

```json
{"clip_id": "front_door_audio-20261112-145338-277", "score": 0.981,
 "threshold": 0.95, "maybe_threshold": 0.42, "detected": true,
 "recorded_at": "2026-11-12T14:53:38.277+02:00", "duration": 2.43, "peak_dbfs": -30.1,
 "span": [0.22, 1.14], "audio": "api/clips/front_door_audio-20261112-145338-277/audio"}
```

- `threshold` is the alarm threshold training chose, within `training.false_alarms_per_week`.
- `maybe_threshold` is a lower one, within `training.maybe_per_week` (7 by default): soft
  or unusual knocks often land between the two. Use it for a quiet heads-up, not an alarm.
  Both come with every message, so a retrain moves them without touching Home Assistant.
- `audio` is relative to the web UI (`http://<host>:8080/`) and needs its login. Add
  `?format=ogg` for Ogg/Opus, which Telegram plays as a voice note.

The audio is fetched over HTTP rather than sent over MQTT: Home Assistant cannot attach an
MQTT payload to a notification, but its Telegram and notify actions can fetch a URL with a
login.

A knock alert on Telegram with the recording and two buttons that label the clip, so the
model keeps learning once it is live. As a Home Assistant package
(`packages/sound_detector.yaml`). Create a **labeling** API key in Settings → API keys,
put it in `secrets.yaml` as `sound_detector_api_key`, and replace `sound-detector.lan` with
the detector's address:

```yaml
rest_command:
  sound_detector_label:
    url: "http://sound-detector.lan:8080/api/clips/{{ clip_id }}/label"
    method: POST
    headers:
      X-API-Key: !secret sound_detector_api_key
    content_type: application/json
    payload: '{"label": "{{ label }}"}'

automation:
  - alias: "Sound detector: knock with the recording"
    mode: queued
    triggers:
      - trigger: mqtt
        topic: sound_detector/front_door_audio/knock/clip
    conditions:
      - condition: template
        value_template: "{{ trigger.payload_json.score >= trigger.payload_json.maybe_threshold }}"
    actions:
      - variables:
          clip: "{{ trigger.payload_json }}"
          sure: "{{ clip.score >= clip.threshold }}"
      - action: telegram_bot.send_voice
        data:
          url: "http://sound-detector.lan:8080/{{ clip.audio }}?format=ogg"
          authentication: bearer_token
          password: !secret sound_detector_api_key
          caption: "{{ 'Knock at the door' if sure else 'Maybe a knock' }} ({{ clip.score }})"
          disable_notification: "{{ not sure }}"
          inline_keyboard:
            - "Knock:/clip {{ clip.clip_id }} knock, Not a knock:/clip {{ clip.clip_id }} not_knock"

  - alias: "Sound detector: label from Telegram"
    triggers:
      - trigger: event
        event_type: telegram_callback
        event_data:
          command: /clip
    actions:
      - action: rest_command.sound_detector_label
        data:
          clip_id: "{{ trigger.event.data.args[0] }}"
          label: "{{ trigger.event.data.args[1] }}"
      - action: telegram_bot.answer_callback_query
        data:
          callback_query_id: "{{ trigger.event.data.id }}"
          message: "Labeled {{ trigger.event.data.args[1] }}"
```

Telegram limits a button's data to 64 bytes, which fits a clip id and a label. Labels set
from Telegram are used at the next training, like labels set in the web UI. For the strict
alert alone, compare the score with `threshold` instead of `maybe_threshold`.

Raw topics, under `topic_prefix` (default `sound_detector`):

| Topic | Payload |
|---|---|
| `sound_detector/<source>/<target>` | `ON` / `OFF`, retained |
| `sound_detector/<source>/<target>/attributes` | JSON, retained |
| `sound_detector/<source>/<target>/clip` | JSON, one message per saved clip |
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

### Signing in

The web UI asks for its password (user `admin`) once per browser: it then stays signed in for
90 days, renewed on each visit (`web.session_days`). The sign-in is an HttpOnly, SameSite=Strict
cookie signed with a random key in `session_secret` in the data directory. Changing the
password signs every browser out; so does deleting that file.

### API keys

For programs, create a key in **Settings → API keys** instead of handing out the UI
password. Send it as `Authorization: Bearer <key>` or `X-API-Key: <key>`. Two scopes:

- **labeling**: read clips, their audio and the reports, and set labels and marks. What a
  Home Assistant alert needs.
- **full**: everything the UI can do, except managing keys.

Creating, listing and revoking keys needs the UI password, so a leaked key can't make more
keys or hide itself. A key is shown once; only its SHA-256 is kept, in `api_keys.json` in the
data directory (readable only by the service user, and left out of backups). The list shows
when each key was last used. Revoking a key takes effect at once.

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
