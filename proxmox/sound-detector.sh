#!/usr/bin/env bash
# Create a Debian LXC on a Proxmox VE host and install custom-sound-detector in it.
#
# Run on the Proxmox host shell, as root:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/YuvalWS/custom-sound-detector/main/proxmox/sound-detector.sh)"
#
# It asks for each setting, with a default in [brackets]. Every setting can also be given
# as an environment variable (the names in capitals below); with NONINTERACTIVE=1 the
# defaults are used for anything not given.
set -euo pipefail

REPO="${REPO:-YuvalWS/custom-sound-detector}"
BRANCH="${BRANCH:-main}"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH"

msg() { printf '\e[1;34m==>\e[0m %s\n' "$*"; }
die() { printf '\e[1;31merror:\e[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root on the Proxmox host"
if ! command -v pct >/dev/null || ! command -v pveam >/dev/null; then
  die "pct/pveam not found: run this on a Proxmox VE host"
fi

INTERACTIVE=1
[[ -t 0 && "${NONINTERACTIVE:-0}" != 1 ]] || INTERACTIVE=0

# ask VAR "question" default [secret]
ask() {
  local var=$1 question=$2 default=$3 secret=${4:-} answer
  if [[ -n "${!var:-}" ]]; then return; fi
  if [[ $INTERACTIVE -eq 1 ]]; then
    if [[ -n $secret ]]; then
      read -rsp "$question: " answer; echo
    else
      read -rp "$question [$default]: " answer
    fi
    printf -v "$var" '%s' "${answer:-$default}"
  else
    printf -v "$var" '%s' "$default"
  fi
}

first_storage() { pvesm status -content "$1" 2>/dev/null | awk 'NR>1 && $3=="active" {print $1; exit}'; }

# YAML single-quoted string
yq() { local s=${1//\'/\'\'}; printf "'%s'" "$s"; }

echo
echo "custom-sound-detector: new LXC container"
echo
ask CTID "Container ID" "$(pvesh get /cluster/nextid)"
ask CT_HOSTNAME "Hostname" "sound-detector"
ask CT_STORAGE "Storage for the container disk" "$(first_storage rootdir)"
ask TEMPLATE_STORAGE "Storage for the Debian template" "$(first_storage vztmpl)"
ask DISK_GB "Disk size in GB (clips are capped at 2 GB by default)" "8"
ask MEMORY_MB "Memory in MB" "1024"
ask CORES "CPU cores" "2"
ask BRIDGE "Network bridge" "vmbr0"
ask IP_CONFIG "IPv4: dhcp, or address/prefix like 192.168.1.50/24" "dhcp"
GATEWAY_OPT=""
if [[ $IP_CONFIG != dhcp ]]; then
  ask GATEWAY "Gateway" ""
  [[ -n $GATEWAY ]] && GATEWAY_OPT=",gw=$GATEWAY"
fi
echo
echo "Detector settings (they go to /etc/sound-detector/config.yaml; change them there later)"
ask SOURCE_NAME "Name of this microphone (used in MQTT topics)" "front_door"
ask AUDIO_URL "Audio stream URL, e.g. rtsp://<frigate-host>:8554/<stream>" ""
ask MQTT_HOST "MQTT broker host (empty: no MQTT)" ""
if [[ -n $MQTT_HOST ]]; then
  ask MQTT_PORT "MQTT port" "1883"
  ask MQTT_USER "MQTT username" "sound-detector"
  ask MQTT_PASSWORD "MQTT password" "" secret
fi
UI_PASSWORD="${UI_PASSWORD:-$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 16)}"

[[ -n $CT_STORAGE && -n $TEMPLATE_STORAGE ]] || die "no storage found for containers or templates"
pct status "$CTID" >/dev/null 2>&1 && die "container $CTID already exists"

msg "Finding the Debian template"
pveam update >/dev/null || true
TEMPLATE=$(pveam available --section system | awk '{print $2}' | grep -E '^debian-1[3-9]-standard' | sort -V | tail -1)
[[ -n $TEMPLATE ]] || TEMPLATE=$(pveam available --section system | awk '{print $2}' | grep -E '^debian-12-standard' | sort -V | tail -1)
[[ -n $TEMPLATE ]] || die "no Debian template available from pveam"
if ! pveam list "$TEMPLATE_STORAGE" | grep -q "$TEMPLATE"; then
  msg "Downloading $TEMPLATE"
  pveam download "$TEMPLATE_STORAGE" "$TEMPLATE" >/dev/null
fi

msg "Creating container $CTID ($CT_HOSTNAME)"
pct create "$CTID" "$TEMPLATE_STORAGE:vztmpl/$TEMPLATE" \
  --hostname "$CT_HOSTNAME" \
  --cores "$CORES" --memory "$MEMORY_MB" --swap 512 \
  --rootfs "$CT_STORAGE:$DISK_GB" \
  --net0 "name=eth0,bridge=$BRIDGE,ip=$IP_CONFIG$GATEWAY_OPT" \
  --unprivileged 1 --features nesting=1 \
  --onboot 1 --timezone host --tags sound-detector \
  --description "custom-sound-detector: https://github.com/$REPO" >/dev/null
pct start "$CTID"

msg "Waiting for the network"
for _ in $(seq 1 30); do
  pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null 2>&1 && break
  sleep 2
done
pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null 2>&1 || die "container $CTID has no network/DNS"

msg "Writing the configuration"
CONF_TMP=$(mktemp)
chmod 600 "$CONF_TMP"
trap 'rm -f "$CONF_TMP"' EXIT
{
  echo "# Written by proxmox/sound-detector.sh. All options: config.example.yaml in the repository."
  echo "source:"
  echo "  name: $(yq "$SOURCE_NAME")"
  echo "  url: $(yq "${AUDIO_URL:-rtsp://frigate.local:8554/front_door}")"
  echo "mqtt:"
  if [[ -n $MQTT_HOST ]]; then
    echo "  host: $(yq "$MQTT_HOST")"
    echo "  port: ${MQTT_PORT:-1883}"
    echo "  username: $(yq "${MQTT_USER:-}")"
    echo "  password: $(yq "${MQTT_PASSWORD:-}")"
  else
    echo "  enabled: false"
  fi
  echo "web:"
  echo "  port: 8080"
  echo "  username: admin"
  echo "  password: $(yq "$UI_PASSWORD")"
} >"$CONF_TMP"
pct exec "$CTID" -- mkdir -p /etc/sound-detector
pct push "$CTID" "$CONF_TMP" /etc/sound-detector/config.yaml --perms 0600

msg "Installing custom-sound-detector in the container"
pct exec "$CTID" -- bash -c "apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null"
pct exec "$CTID" -- env REPO_URL="https://github.com/$REPO.git" BRANCH="$BRANCH" \
  bash -c "curl -fsSL '$RAW/proxmox/install.sh' | bash"

IP=$(pct exec "$CTID" -- hostname -I | awk '{print $1}')
echo
msg "Done."
echo "  Labeling UI:  http://$IP:8080   (user admin, password $UI_PASSWORD)"
echo "  Config:       pct enter $CTID, then edit /etc/sound-detector/config.yaml"
echo "  Logs:         pct exec $CTID -- journalctl -u sound-detector -f"
echo "  Update:       pct exec $CTID -- update"
[[ -n $AUDIO_URL ]] || echo "  No audio URL given: set source.url in the config, then: pct exec $CTID -- systemctl restart sound-detector"
