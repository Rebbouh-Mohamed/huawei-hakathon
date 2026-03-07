"""
simulator.py — Simulates multiple ESP32 sensors, each with a unique MAC address.

Each ESP32 publishes to: fire/sensors/{MAC_ADDRESS}
Payload includes mac_address + all sensor readings.

Usage:
  python simulator.py                           # 5 ESP32s, random readings
  python simulator.py --fire A4:CF:12:03:33:CC  # force fire on specific MAC
  python simulator.py --interval 2              # reading every 2s per device
"""

import json, time, random, argparse, threading
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT   = 1883

# ── Your ESP32 devices — add/remove as needed ─────────────────────────────────
DEVICES = {
    "A4:CF:12:98:AB:BB": "Zone C",
    "A4:CF:12:98:AB:AA": "Zone D",
    "A4:CF:12:98:AB:99": "Zone E",
    "A4:CF:12:98:AB:88": "Zone F",
    "A4:CF:12:98:AB:77": "Zone G",
    "A4:CF:12:98:AB:66": "Zone H",
    "A4:CF:12:98:AB:55": "Zone I",
    "A4:CF:12:98:AB:44": "Zone J",
}

MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
DAYS   = ["mon","tue","wed","thu","fri","sat","sun"]

NORMAL    = dict(FFMC=85.0, DMC=60.0,  DC=400.0, ISI=6.0,  temp=16.0, RH=55.0, wind=3.5, rain=0.0)
HIGH_RISK = dict(FFMC=95.0, DMC=200.0, DC=700.0, ISI=16.0, temp=32.0, RH=15.0, wind=8.0, rain=0.0)


def make_reading(mac: str, force_fire: bool = False) -> dict:
    # ~15% chance of random fire spike on any device
    fire_mode = force_fire or (random.random() < 0.15)
    base  = HIGH_RISK if fire_mode else NORMAL
    noise = 0.12
    return {
        # ── MAC is always included in the payload ──
        "mac_address": mac,
        "X":     random.randint(1, 9),
        "Y":     random.randint(2, 9),
        "month": random.choice(["jul","aug","sep"]) if fire_mode else random.choice(MONTHS),
        "day":   random.choice(DAYS),
        "FFMC":  round(max(18.7, base["FFMC"] + random.gauss(0, base["FFMC"] * noise)), 1),
        "DMC":   round(max(1.1,  base["DMC"]  + random.gauss(0, base["DMC"]  * noise)), 1),
        "DC":    round(max(7.9,  base["DC"]   + random.gauss(0, base["DC"]   * noise)), 1),
        "ISI":   round(max(0.0,  base["ISI"]  + random.gauss(0, base["ISI"]  * noise)), 1),
        "temp":  round(base["temp"] + random.gauss(0, 2.5), 1),
        "RH":    round(max(5, min(100, base["RH"] + random.gauss(0, 8))), 0),
        "wind":  round(max(0.4, base["wind"] + random.gauss(0, 1.0)), 1),
        "rain":  round(max(0.0, random.gauss(0, 0.05)) if not fire_mode else 0.0, 1),
    }


def device_loop(client, mac, label, interval, force_fire_mac):
    """Each ESP32 gets its own thread — publishes to fire/sensors/{MAC}"""
    topic = f"fire/sensors/{mac}"
    n = 0
    while True:
        n += 1
        payload = make_reading(mac, force_fire=(mac == force_fire_mac))
        client.publish(topic, json.dumps(payload), qos=1)
        print(f"📡 [{mac}] {label} | #{n:03d} "
              f"T={payload['temp']}°C  RH={payload['RH']}%  FFMC={payload['FFMC']}")
        # Stagger slightly so devices don't all publish at the exact same time
        time.sleep(interval + random.uniform(-0.3, 0.3))


def main():
    parser = argparse.ArgumentParser(description="Multi-ESP32 Forest Fire Simulator")
    parser.add_argument("--fire",     type=str,   default=None,
                        help="MAC address to force fire conditions on (e.g. A4:CF:12:03:33:CC)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between readings per device (default: 3)")
    parser.add_argument("--broker",   type=str,   default=BROKER)
    args = parser.parse_args()

    client = mqtt.Client(client_id="multi_esp32_sim")

    def on_connect(c, u, f, rc):
        if rc == 0:
            print(f"\n✅ Connected to MQTT broker at {args.broker}:{PORT}")
            print(f"📡 Simulating {len(DEVICES)} ESP32 devices\n")
            for mac, label in DEVICES.items():
                print(f"   {mac}  →  {label}")
            print()
        else:
            print(f"❌ Connection failed: rc={rc}")

    client.on_connect = on_connect
    client.connect(args.broker, PORT, keepalive=60)
    client.loop_start()
    time.sleep(0.8)

    # Spawn one thread per ESP32
    for mac, label in DEVICES.items():
        t = threading.Thread(
            target=device_loop,
            args=(client, mac, label, args.interval, args.fire),
            daemon=True
        )
        t.start()
        time.sleep(0.2)   # stagger thread starts

    print("Press Ctrl+C to stop\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Simulator stopped.")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()