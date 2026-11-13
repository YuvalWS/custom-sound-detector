"""MQTT output, with Home Assistant discovery.

Per target: `<prefix>/<source>/<target>` is ON/OFF (a binary sensor), with attributes on
`.../attributes`; `.../clip` gets one JSON message per saved clip with the model's score for
it (a "score" sensor), so Home Assistant can alert at either threshold and fetch the audio.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


class Publisher:
    def __init__(self, cfg: dict, source: str, version: str):
        import paho.mqtt.client as mqtt

        self.cfg = cfg
        self.source = source
        self.version = version
        self.base = f"{cfg['topic_prefix']}/{source}"
        self.availability = f"{self.base}/availability"
        self.targets: dict[str, dict] = {}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sound-detector-{source}")
        if cfg.get("username"):
            self.client.username_pw_set(cfg["username"], cfg.get("password"))
        self.client.will_set(self.availability, "offline", retain=True)
        self.client.on_connect = self._on_connect
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)

    def start(self) -> None:
        self.client.connect_async(self.cfg["host"], int(self.cfg["port"]), keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.publish(self.availability, "offline", retain=True)
        self.client.loop_stop()
        self.client.disconnect()

    def set_targets(self, targets: dict[str, dict]) -> None:
        """targets: {label: attributes}. Publishes discovery for each, removes dropped ones."""
        old = set(self.targets) - set(targets)
        self.targets = dict(targets)
        if self.client.is_connected():
            for target in old:
                self._discovery(target, remove=True)
            self._announce()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            log.error("MQTT connect failed: %s", reason_code)
            return
        log.info("MQTT connected")
        self._announce()

    def _announce(self) -> None:
        self.client.publish(self.availability, "online", retain=True)
        for target, attrs in self.targets.items():
            self._discovery(target)
            self.publish_state(target, False, attrs)

    def _discovery(self, target: str, remove: bool = False) -> None:
        if not self.cfg.get("discovery"):
            return
        uid = f"sound_detector_{self.source}_{target}"
        name = target.replace("_", " ").capitalize()
        device = {
            "identifiers": [f"sound_detector_{self.source}"],
            "name": f"Sound detector {self.source.replace('_', ' ')}",
            "manufacturer": "custom-sound-detector",
            "sw_version": self.version,
        }
        entities = {
            f"binary_sensor/{uid}": {
                "name": name,
                "unique_id": uid,
                "object_id": uid,
                "device_class": "sound",
                "state_topic": f"{self.base}/{target}",
                "json_attributes_topic": f"{self.base}/{target}/attributes",
            },
            f"sensor/{uid}_score": {
                "name": f"{name} score",
                "unique_id": f"{uid}_score",
                "object_id": f"{uid}_score",
                "icon": "mdi:waveform",
                "state_topic": f"{self.base}/{target}/clip",
                "value_template": "{{ value_json.score }}",
                "json_attributes_topic": f"{self.base}/{target}/clip",
                "force_update": True,
            },
        }
        for path, payload in entities.items():
            topic = f"{self.cfg['discovery_prefix']}/{path}/config"
            if remove:
                self.client.publish(topic, "", retain=True)
            else:
                payload.update({"availability_topic": self.availability, "device": device})
                self.client.publish(topic, json.dumps(payload), retain=True)

    def publish_clip(self, target: str, info: dict) -> None:
        """One saved clip and the target model's score for it (not retained: an event)."""
        self.client.publish(f"{self.base}/{target}/clip", json.dumps(info))

    def publish_state(self, target: str, on: bool, attributes: dict | None = None) -> None:
        # Retained, so Home Assistant has a state after its own restart. A stale ON can't outlive the
        # detector: the availability last-will turns the entity unavailable, and _announce resets it to OFF.
        self.client.publish(f"{self.base}/{target}", "ON" if on else "OFF", retain=True)
        if attributes is not None:
            self.client.publish(f"{self.base}/{target}/attributes", json.dumps(attributes), retain=True)
