#!/usr/bin/env bash
# Create a Debian LXC on a Proxmox VE host and install custom-sound-detector in it.
# Laid out like the community Proxmox VE helper scripts: Default / Advanced / saved settings
# in a text UI, the choices saved as defaults for next time (never the passwords).
#
# On the Proxmox host shell, as root:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/YuvalWS/custom-sound-detector/main/proxmox/sound-detector.sh)"
#
# Inside an existing sound-detector container the same command updates it.
#
# Without a terminal (or with NONINTERACTIVE=1) every setting comes from the environment or
# the defaults: CTID, CT_HOSTNAME, CT_STORAGE, TEMPLATE_STORAGE, DISK_GB, MEMORY_MB, CORES,
# BRIDGE, IP_CONFIG, GATEWAY, SOURCE_NAME, STREAM_URL, STREAM_USER, STREAM_PASSWORD,
# MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASSWORD.
set -Eeuo pipefail

APP="Sound Detector"
REPO="${REPO:-YuvalWS/custom-sound-detector}"
BRANCH="${BRANCH:-main}"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH"
DEFAULTS_FILE="/usr/local/community-scripts/defaults/sound-detector.vars"
BACKTITLE="custom-sound-detector  ·  github.com/$REPO"

YW=$'\e[33m' GN=$'\e[1;92m' RD=$'\e[01;31m' BL=$'\e[36m' CL=$'\e[m'
msg_info() { printf ' %s… %s' "${YW}" "$1${CL}"; }
msg_ok() { printf '\r\e[K %s✔%s %s\n' "${GN}" "${CL}" "$1"; }
msg_error() { printf '\r\e[K %s✖%s %s\n' "${RD}" "${CL}" "$1" >&2; }

CREATED_CTID=""
ct_note() {
  if [[ -n $CREATED_CTID ]]; then
    echo "  Container $CREATED_CTID was created and is left as it is. To remove it: pct stop $CREATED_CTID; pct destroy $CREATED_CTID"
  fi
}
on_error() {
  local code=$? line=$1 cmd=$2
  msg_error "failed (exit $code) at line $line: $cmd"
  ct_note
  exit "$code"
}
trap 'on_error $LINENO "$BASH_COMMAND"' ERR
cancelled() { echo; msg_error "cancelled, nothing was changed"; exit 0; }

header() {
  clear 2>/dev/null || true
  cat <<'EOF'
   _____                        __   ____       __            __
  / ___/____  __  ______  ____/ /  / __ \___  / /____  _____/ /_____  _____
  \__ \/ __ \/ / / / __ \/ __  /  / / / / _ \/ __/ _ \/ ___/ __/ __ \/ ___/
 ___/ / /_/ / /_/ / / / / /_/ /  / /_/ /  __/ /_/  __/ /__/ /_/ /_/ / /
/____/\____/\__,_/_/ /_/\__,_/  /_____/\___/\__/\___/\___/\__/\____/_/

EOF
}

# ---- inside an existing container: update -----------------------------------------------
if ! command -v pct >/dev/null 2>&1; then
  if [[ -x /usr/bin/update && -d /opt/sound-detector ]]; then
    exec /usr/bin/update
  fi
  echo "Run this on a Proxmox VE host (or inside a sound-detector container to update it)." >&2
  exit 1
fi
[[ $EUID -eq 0 ]] || { echo "Run as root." >&2; exit 1; }

INTERACTIVE=1
if [[ ! -t 0 || "${NONINTERACTIVE:-0}" == 1 ]] || ! command -v whiptail >/dev/null; then
  INTERACTIVE=0
fi

# ---- saved defaults: KEY=VALUE lines, parsed (never sourced) ------------------------------
declare -A SAVED=()
if [[ -f $DEFAULTS_FILE ]]; then
  while IFS='=' read -r key value; do
    [[ $key =~ ^[a-z_]+$ ]] || continue
    value=${value%\"}; value=${value#\"}
    SAVED[$key]=$value
  done <"$DEFAULTS_FILE"
fi
saved() { local v=${SAVED[$1]:-}; printf '%s' "${v:-$2}"; }

first_storage() { pvesm status -content "$1" 2>/dev/null | awk 'NR>1 && $3=="active" {print $1; exit}'; }

# Built-in defaults, community-scripts style.
var_cpu=2 var_ram=1024 var_disk=8 var_brg=vmbr0 var_net=dhcp var_gateway=""
var_hostname=sound-detector
var_container_storage=$(first_storage rootdir)
var_template_storage=$(first_storage vztmpl)
sd_source_name=front_door sd_stream_url="" sd_stream_user="" sd_mqtt_host="" sd_mqtt_port=1883 sd_mqtt_user=""

load_saved() {
  for k in var_cpu var_ram var_disk var_brg var_net var_gateway var_hostname var_container_storage \
           var_template_storage sd_source_name sd_stream_url sd_stream_user sd_mqtt_host sd_mqtt_port sd_mqtt_user; do
    printf -v "$k" '%s' "$(saved "$k" "${!k}")"
  done
}

# ---- whiptail helpers (each returns the answer on stdout; Cancel ends the script) --------
input() { # title text default
  whiptail --backtitle "$BACKTITLE" --title "$1" --inputbox "$2" 10 72 "$3" 3>&1 1>&2 2>&3 || cancelled
}
secret() { # title text
  whiptail --backtitle "$BACKTITLE" --title "$1" --passwordbox "$2" 10 72 3>&1 1>&2 2>&3 || cancelled
}
storage_menu() { # content-type current
  local items=() name type avail
  while read -r name type _ _ _ avail _; do
    items+=("$name" "$type, $((avail / 1024 / 1024)) GB free")
  done < <(pvesm status -content "$1" 2>/dev/null | awk 'NR>1 && $3=="active"')
  if [[ ${#items[@]} -le 2 ]]; then printf '%s' "${items[0]:-$2}"; return; fi
  whiptail --backtitle "$BACKTITLE" --title "Storage" --default-item "$2" \
    --menu "Storage for the $3:" 16 72 6 "${items[@]}" 3>&1 1>&2 2>&3 || cancelled
}
bridge_menu() {
  local items=() b
  for b in $(ip -o link show type bridge 2>/dev/null | awk -F': ' '{print $2}' | cut -d@ -f1); do
    items+=("$b" "")
  done
  if [[ ${#items[@]} -le 2 ]]; then printf '%s' "${items[0]:-$1}"; return; fi
  whiptail --backtitle "$BACKTITLE" --title "Network" --default-item "$1" \
    --menu "Bridge:" 14 50 5 "${items[@]}" 3>&1 1>&2 2>&3 || cancelled
}

container_settings() {
  CTID=$(input "Container ID" "Container ID:" "$CTID")
  var_hostname=$(input "Hostname" "Hostname:" "$var_hostname")
  var_cpu=$(input "CPU" "CPU cores (training uses them; detection needs a fraction of one):" "$var_cpu")
  var_ram=$(input "Memory" "Memory in MB:" "$var_ram")
  var_disk=$(input "Disk" "Disk size in GB (clips are capped at 2 GB by default):" "$var_disk")
  var_container_storage=$(storage_menu rootdir "$var_container_storage" "container disk")
  var_template_storage=$(storage_menu vztmpl "$var_template_storage" "Debian template")
  var_brg=$(bridge_menu "$var_brg")
  var_net=$(input "IPv4" "dhcp, or a static address with prefix (192.168.1.50/24):" "$var_net")
  if [[ $var_net != dhcp ]]; then
    var_gateway=$(input "Gateway" "Gateway address:" "$var_gateway")
  fi
}

detector_settings() { # ask_names: also ask the non-secret detector settings
  if [[ $1 == yes ]]; then
    sd_stream_url=$(input "Audio stream" "Stream URL, WITHOUT the login (it comes next).
For Frigate: rtsp://<frigate-host>:8554/<stream>" "$sd_stream_url")
    sd_stream_user=$(input "Audio stream" "Stream username (empty if the stream needs no login):" "$sd_stream_user")
  fi
  STREAM_PASSWORD=""
  if [[ -n $sd_stream_user ]]; then
    STREAM_PASSWORD=$(secret "Audio stream" "Password for '$sd_stream_user' (any characters; it is encoded for you):")
  fi
  if [[ $1 == yes ]]; then
    sd_source_name=$(input "Microphone name" "A name for this microphone, used in MQTT topics and the Home Assistant entity id (a-z, 0-9, _):" "$sd_source_name")
    sd_mqtt_host=$(input "MQTT" "MQTT broker host (empty: don't publish to MQTT):" "$sd_mqtt_host")
    if [[ -n $sd_mqtt_host ]]; then
      sd_mqtt_port=$(input "MQTT" "MQTT port:" "$sd_mqtt_port")
      sd_mqtt_user=$(input "MQTT" "MQTT username:" "$sd_mqtt_user")
    fi
  fi
  MQTT_PASSWORD=""
  if [[ -n $sd_mqtt_host && -n $sd_mqtt_user ]]; then
    MQTT_PASSWORD=$(secret "MQTT" "Password for MQTT user '$sd_mqtt_user':")
  fi
}

save_defaults() {
  mkdir -p "$(dirname "$DEFAULTS_FILE")"
  {
    echo "# custom-sound-detector helper defaults (no passwords). Written $(date -Iseconds)."
    for k in var_cpu var_ram var_disk var_brg var_net var_gateway var_hostname var_container_storage \
             var_template_storage sd_source_name sd_stream_url sd_stream_user sd_mqtt_host sd_mqtt_port sd_mqtt_user; do
      printf '%s="%s"\n' "$k" "${!k}"
    done
  } >"$DEFAULTS_FILE"
  chmod 600 "$DEFAULTS_FILE"
}

# ---- gather the settings ------------------------------------------------------------------
header
CTID=${CTID:-$(pvesh get /cluster/nextid)}
if [[ $INTERACTIVE -eq 1 ]]; then
  whiptail --backtitle "$BACKTITLE" --title "$APP LXC" --yesno \
    "This creates a new $APP container: it listens to an audio stream, collects sounds for labeling and detects the ones you train it on.\n\nProceed?" 12 72 || cancelled
  menu=("1" "Default settings" "2" "Advanced settings")
  [[ -f $DEFAULTS_FILE ]] && menu+=("3" "My saved defaults")
  choice=$(whiptail --backtitle "$BACKTITLE" --title "Settings" --menu "Choose:" 13 60 3 "${menu[@]}" 3>&1 1>&2 2>&3) || cancelled
  case $choice in
    1) detector_settings yes ;;
    2) load_saved; container_settings; detector_settings yes ;;
    3) load_saved; detector_settings no ;;
  esac
  stream_login="none"
  [[ -n $sd_stream_user ]] && stream_login="$sd_stream_user, password $([[ -n $STREAM_PASSWORD ]] && echo set || echo empty)"
  summary="Container $CTID ($var_hostname): $var_cpu cores, $var_ram MB, $var_disk GB on $var_container_storage
Network: $var_brg, ${var_net}${var_gateway:+, gateway $var_gateway}

Stream: ${sd_stream_url:-(not set: set it later in the web UI)}
Stream login: $stream_login
Microphone name: $sd_source_name
MQTT: ${sd_mqtt_host:-off}${sd_mqtt_host:+:$sd_mqtt_port, user ${sd_mqtt_user:-none}}"
  whiptail --backtitle "$BACKTITLE" --title "Create it?" --yesno "$summary" 16 76 || cancelled
  if [[ $choice == 2 ]] && whiptail --backtitle "$BACKTITLE" --title "Save as defaults?" --yesno \
      "Save these settings as your defaults for next time?\n\nPasswords are never saved.\n($DEFAULTS_FILE)" 12 72; then
    save_defaults
  fi
else
  load_saved
  CT_ENV=(CT_HOSTNAME:var_hostname CORES:var_cpu MEMORY_MB:var_ram DISK_GB:var_disk CT_STORAGE:var_container_storage
          TEMPLATE_STORAGE:var_template_storage BRIDGE:var_brg IP_CONFIG:var_net GATEWAY:var_gateway
          SOURCE_NAME:sd_source_name STREAM_URL:sd_stream_url STREAM_USER:sd_stream_user MQTT_HOST:sd_mqtt_host
          MQTT_PORT:sd_mqtt_port MQTT_USER:sd_mqtt_user)
  for pair in "${CT_ENV[@]}"; do
    env_name=${pair%%:*} var=${pair#*:}
    if [[ -n ${!env_name:-} ]]; then printf -v "$var" '%s' "${!env_name}"; fi
  done
  STREAM_PASSWORD=${STREAM_PASSWORD:-} MQTT_PASSWORD=${MQTT_PASSWORD:-}
fi

[[ -n $var_container_storage && -n $var_template_storage ]] || { msg_error "no storage found for containers or templates"; exit 1; }
if pct status "$CTID" >/dev/null 2>&1; then msg_error "container $CTID already exists"; exit 1; fi
[[ $sd_source_name =~ ^[a-z0-9][a-z0-9_]{0,31}$ ]] || { msg_error "microphone name: a-z, 0-9 and _ only"; exit 1; }
# A finite source, so no pipe can be cut short (that killed the first version of this script).
UI_PASSWORD=$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-16)

# ---- create the container -------------------------------------------------------------
msg_info "Finding the Debian template"
pveam update >/dev/null 2>&1 || true
TEMPLATE=$(pveam available --section system | awk '{print $2}' | grep -E '^debian-1[3-9]-standard' | sort -V | tail -1 || true)
[[ -n $TEMPLATE ]] || TEMPLATE=$(pveam available --section system | awk '{print $2}' | grep -E '^debian-12-standard' | sort -V | tail -1 || true)
[[ -n $TEMPLATE ]] || { msg_error "no Debian template available from pveam"; exit 1; }
if ! pveam list "$var_template_storage" | grep -q "$TEMPLATE"; then
  pveam download "$var_template_storage" "$TEMPLATE" >/dev/null
fi
msg_ok "Template $TEMPLATE"

msg_info "Creating container $CTID"
GW_OPT=""
[[ $var_net != dhcp && -n $var_gateway ]] && GW_OPT=",gw=$var_gateway"
pct create "$CTID" "$var_template_storage:vztmpl/$TEMPLATE" \
  --hostname "$var_hostname" \
  --cores "$var_cpu" --memory "$var_ram" --swap 512 \
  --rootfs "$var_container_storage:$var_disk" \
  --net0 "name=eth0,bridge=$var_brg,ip=$var_net$GW_OPT" \
  --unprivileged 1 --features nesting=1 \
  --onboot 1 --timezone host --tags sound-detector \
  --description "custom-sound-detector: https://github.com/$REPO" >/dev/null
CREATED_CTID=$CTID
pct start "$CTID"
msg_ok "Created container $CTID"

msg_info "Waiting for the network"
for _ in $(seq 1 30); do
  pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null 2>&1 && break
  sleep 2
done
pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null 2>&1 || { msg_error "container $CTID has no network/DNS"; ct_note; exit 1; }
msg_ok "Network is up"

msg_info "Writing the configuration"
yq() { local s=${1//\'/\'\'}; printf "'%s'" "$s"; } # YAML single-quoted string
CONF_TMP=$(mktemp)
chmod 600 "$CONF_TMP"
trap 'rm -f "$CONF_TMP"' EXIT
{
  echo "# Written by proxmox/sound-detector.sh. Change settings in the web UI (Settings);"
  echo "# all options are in config.example.yaml in the repository."
  echo "source:"
  echo "  name: $(yq "$sd_source_name")"
  [[ -n $sd_stream_url ]] && echo "  url: $(yq "$sd_stream_url")"
  [[ -n $sd_stream_user ]] && echo "  username: $(yq "$sd_stream_user")"
  [[ -n $STREAM_PASSWORD ]] && echo "  password: $(yq "$STREAM_PASSWORD")"
  echo "mqtt:"
  if [[ -n $sd_mqtt_host ]]; then
    echo "  host: $(yq "$sd_mqtt_host")"
    echo "  port: ${sd_mqtt_port:-1883}"
    [[ -n $sd_mqtt_user ]] && echo "  username: $(yq "$sd_mqtt_user")"
    [[ -n $MQTT_PASSWORD ]] && echo "  password: $(yq "$MQTT_PASSWORD")"
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
msg_ok "Configuration written"

msg_info "Installing (a few minutes: Python packages and the YAMNet model)"
LOG=/tmp/sound-detector-install-$CTID.log
if ! { pct exec "$CTID" -- bash -c "apt-get update -qq && apt-get install -y -qq curl ca-certificates" &&
       pct exec "$CTID" -- env REPO_URL="https://github.com/$REPO.git" BRANCH="$BRANCH" \
         bash -c "curl -fsSL '$RAW/proxmox/install.sh' | bash"; } >"$LOG" 2>&1; then
  msg_error "installation failed; the last lines of $LOG:"
  tail -20 "$LOG" >&2
  ct_note
  exit 1
fi
msg_ok "Installed"

IP=$(pct exec "$CTID" -- hostname -I | awk '{print $1}')
echo
msg_ok "$APP is running in container $CTID"
echo "   ${BL}Web UI:${CL}   http://$IP:8080   user admin, password ${GN}$UI_PASSWORD${CL}"
echo "            (save it in your password manager; change it in Settings)"
echo "   ${BL}Settings:${CL} in the web UI, Settings tab"
echo "   ${BL}Logs:${CL}     pct exec $CTID -- journalctl -u sound-detector -f"
echo "   ${BL}Update:${CL}   pct exec $CTID -- update"
[[ -n $sd_stream_url ]] || echo "   No stream URL yet: set it in Settings, and the service starts listening."
