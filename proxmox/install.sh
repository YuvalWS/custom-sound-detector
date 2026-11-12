#!/usr/bin/env bash
# Install or update custom-sound-detector on Debian (12 or newer) or Ubuntu, for example in
# the LXC that proxmox/sound-detector.sh creates. Safe to run again: it never overwrites
# the config, the YAMNet model or the clips.
#
#   install:  curl -fsSL https://raw.githubusercontent.com/YuvalWS/custom-sound-detector/main/proxmox/install.sh | bash
#   update:   update            (installed to /usr/bin/update by this script)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/YuvalWS/custom-sound-detector.git}"
BRANCH="${BRANCH:-main}"
APP_DIR=/opt/sound-detector
DATA_DIR=/var/lib/sound-detector
CONF_DIR=/etc/sound-detector
CONF="$CONF_DIR/config.yaml"
SVC_USER=sound-detector

msg() { printf '\e[1;34m==>\e[0m %s\n' "$*"; }
die() { printf '\e[1;31merror:\e[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root"
command -v apt-get >/dev/null || die "this installer needs Debian or Ubuntu (apt-get)"

msg "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3 python3-venv ffmpeg git ca-certificates curl >/dev/null

if ! id "$SVC_USER" >/dev/null 2>&1; then
  msg "Creating the $SVC_USER system user"
  useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SVC_USER"
fi

# Run from the checkout itself (as `update` does), the code is already current: never
# rewrite the file bash is executing.
if [[ "$(realpath "${BASH_SOURCE[0]:-/dev/null}" 2>/dev/null)" == "$APP_DIR/"* ]]; then
  msg "Installing from $APP_DIR ($(git -C "$APP_DIR" log -1 --format='%h %cs'))"
elif [[ -d "$APP_DIR/.git" ]]; then
  msg "Updating $APP_DIR"
  git -C "$APP_DIR" fetch --quiet origin
  git -C "$APP_DIR" reset --quiet --hard "origin/$(git -C "$APP_DIR" rev-parse --abbrev-ref HEAD)"
else
  msg "Downloading $REPO_URL ($BRANCH)"
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

msg "Installing Python packages (this takes a few minutes the first time)"
[[ -x "$APP_DIR/venv/bin/python" ]] || python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/venv/bin/pip" install --quiet --upgrade "$APP_DIR"

install -d -o "$SVC_USER" -g "$SVC_USER" -m 0750 "$DATA_DIR"
install -d -o root -g "$SVC_USER" -m 0750 "$CONF_DIR"
if [[ ! -f "$CONF" ]]; then
  web_password=$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-16)
  (umask 077; sed "s/^\(  password: \)WEB_PASSWORD$/\1$web_password/" "$APP_DIR/config.example.yaml" >"$CONF")
  msg "Created $CONF. Web UI: user admin, password $web_password (set the stream and MQTT in its Settings tab)"
fi
# Holds passwords: readable by the service, by nobody else.
chown root:"$SVC_USER" "$CONF"
chmod 0640 "$CONF"

# The command line tool runs as the service user, so files it writes stay readable to the service.
cat > /usr/local/bin/sound-detector <<EOF
#!/usr/bin/env bash
if [[ \$EUID -eq 0 ]]; then
  exec runuser -u $SVC_USER -- $APP_DIR/venv/bin/sound-detector -c $CONF "\$@"
fi
exec $APP_DIR/venv/bin/sound-detector -c $CONF "\$@"
EOF
chmod 0755 /usr/local/bin/sound-detector

if [[ ! -f "$DATA_DIR/yamnet.tflite" ]]; then
  msg "Downloading the YAMNet model"
  /usr/local/bin/sound-detector download-model
fi

cat > /usr/bin/update <<'EOF'
#!/usr/bin/env bash
# Update custom-sound-detector to the latest version and restart it.
set -euo pipefail
APP=/opt/sound-detector
git -C "$APP" fetch --quiet origin
git -C "$APP" reset --quiet --hard "origin/$(git -C "$APP" rev-parse --abbrev-ref HEAD)"
exec bash "$APP/proxmox/install.sh"
EOF
chmod 0755 /usr/bin/update

cat > /etc/systemd/system/sound-detector.service <<EOF
[Unit]
Description=custom-sound-detector
After=network-online.target
Wants=network-online.target

[Service]
User=$SVC_USER
Group=$SVC_USER
ExecStart=$APP_DIR/venv/bin/sound-detector -c $CONF run
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=$DATA_DIR

[Install]
WantedBy=multi-user.target
EOF
# The web UI's "Update" button. The service runs unprivileged, so it only writes a request
# file; this path unit sees it and runs `update` as root. The request is removed first, so
# the update runs once; its output goes to update.log, which the UI shows.
cat > /etc/systemd/system/sound-detector-update.service <<EOF
[Unit]
Description=Update custom-sound-detector (requested from its web UI)

[Service]
Type=oneshot
ExecStartPre=/bin/rm -f $DATA_DIR/update-request
ExecStart=/bin/bash -c '/usr/bin/update >$DATA_DIR/update.log 2>&1'
EOF
cat > /etc/systemd/system/sound-detector-update.path <<EOF
[Unit]
Description=Watch for an update request from the custom-sound-detector web UI

[Path]
PathExists=$DATA_DIR/update-request

[Install]
WantedBy=multi-user.target
EOF

# Which commit is installed, for the web UI (it can't run git in a checkout it doesn't own).
printf '{"commit": "%s", "date": "%s", "repo": "%s", "branch": "%s"}\n' \
  "$(git -C "$APP_DIR" rev-parse HEAD)" "$(git -C "$APP_DIR" log -1 --format=%cI)" \
  "$(git -C "$APP_DIR" remote get-url origin)" "$(git -C "$APP_DIR" rev-parse --abbrev-ref HEAD)" \
  > "$CONF_DIR/installed.json"
chown root:"$SVC_USER" "$CONF_DIR/installed.json"
chmod 0640 "$CONF_DIR/installed.json"

systemctl daemon-reload
systemctl enable --quiet sound-detector
systemctl enable --quiet --now sound-detector-update.path
systemctl restart sound-detector
msg "Running: web UI on port 8080. Logs: journalctl -u sound-detector -f"
msg "Done: version $("$APP_DIR/venv/bin/python" -c 'import sounddetector; print(sounddetector.__version__)')"
