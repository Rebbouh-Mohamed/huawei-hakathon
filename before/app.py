"""
app.py — FireWatch AI

Flow: MQTT → predictor → Supabase POST

Nothing else. No WebSocket, no dashboard, no Flask templates.
Flask only exists for the /health endpoint so you can check if it's running.
"""

import json, logging, os, re, threading
from datetime import datetime

import requests
import paho.mqtt.client as mqtt
from flask import Flask, jsonify
from dotenv import load_dotenv

from config    import CONFIG
from predictor import FirePredictor

import numpy as np 
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
KEY_MAP = {
    "temperature":    "temp",
    "humidity":       "RH",
    "gas_resistance": "FFMC",
}

# ── Supabase ──────────────────────────────────────────────────────────────────
# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_KEY      = os.getenv("SUPABASE_KEY")

# ✅ Point to your Edge Function instead of the REST table
SUPABASE_ENDPOINT = f"{SUPABASE_URL}/functions/v1/ai-fire-insert"
SUPABASE_HEADERS  = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}

# ── Model ─────────────────────────────────────────────────────────────────────
log.info("Loading models…")
predictor = FirePredictor(CONFIG["model_dir"])
log.info("Models ready ✅")

# ── MAC validator ─────────────────────────────────────────────────────────────
MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


# ── Status label based on is_fire + confidence ────────────────────────────────
def get_status(is_fire: bool, confidence: str) -> str:
    if not is_fire:
        if confidence == "HIGH":   return "SAFE"
        if confidence == "MEDIUM": return "MONITOR"
        return                            "UNCERTAIN"
    else:
        if confidence == "HIGH":   return "FIRE_CRITICAL"
        if confidence == "MEDIUM": return "FIRE_WARNING"
        return                            "FIRE_LOW CONFIDENCE"


# ── MQTT callbacks ────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("fire/sensors/#", qos=1)
        log.info("MQTT connected ✅  listening on fire/sensors/#")
    else:
        log.error(f"MQTT connect failed rc={rc}")


def on_disconnect(client, userdata, rc):
    log.warning(f"MQTT disconnected rc={rc} — will reconnect…")

def to_python(v):
    if isinstance(v, (np.bool_,)):        return bool(v)
    if isinstance(v, (np.integer,)):      return int(v)
    if isinstance(v, (np.floating,)):     return float(v)
    return v

def on_message(client, userdata, msg):
    try:
        parts = msg.topic.split("/")
        if len(parts) < 3:
            return

        raw = json.loads(msg.payload.decode("utf-8"))
        mac = raw.get("mac_address", parts[2]).upper()

        if not MAC_RE.match(mac):
            log.warning(f"Invalid MAC: {mac}")
            return

        # fill missing fields with defaults
        sensor = {k: raw.get(k, v) for k, v in CONFIG["sensor_defaults"].items()}
        result = predictor.predict(sensor)

        status = get_status(result["is_fire"], result["confidence"])

        row = {
            **sensor,
            "mac_address": mac,
            "is_fire":     result["is_fire"],
            "confidence":  result["confidence"],
            "status":      status,
            "timestamp":   datetime.utcnow().isoformat(),
        }
        row = {k: to_python(v) for k, v in row.items()}
        _RENAME = {"temp": "temperature", "RH": "humidity", "FFMC": "gas_resistance"}
        row = {_RENAME.get(k, k): v for k, v in row.items()}
        print ("Row:",row)
        try:
            resp = requests.post(
                SUPABASE_ENDPOINT,
                headers=SUPABASE_HEADERS,
                json=row,
                timeout=5,
            )
            if resp.status_code in (200, 201):
                log.info(f"[{mac}]  {status}  → Supabase ✅")
            else:
                log.error(f"[{mac}] Supabase {resp.status_code}: {resp.text[:120]}")
        except requests.exceptions.RequestException as e:
            log.error(f"[{mac}] Supabase failed: {e}")

    except json.JSONDecodeError:
        log.error(f"Bad JSON on {msg.topic}")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")


# ── Flask — health check only ─────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": predictor.ready})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mqtt_client = mqtt.Client(client_id="firewatch_server", clean_session=True)
    mqtt_client.on_connect    = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message    = on_message

    if CONFIG["mqtt_username"]:
        mqtt_client.username_pw_set(CONFIG["mqtt_username"], CONFIG["mqtt_password"])

    mqtt_client.connect(CONFIG["mqtt_broker"], CONFIG["mqtt_port"], keepalive=60)
    threading.Thread(target=mqtt_client.loop_forever, daemon=True).start()

    log.info(f"Starting on http://0.0.0.0:{CONFIG['flask_port']}")
    app.run(host="0.0.0.0", port=CONFIG["flask_port"])