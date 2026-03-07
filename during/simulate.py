"""
simulate.py
──────────────────────────────────────────────────────────────────────────────
Simulates multiple drones by looping through all images in ./test_images/
and posting each one to the Flask /detect endpoint.

Each image is assigned a random MAC address from the DRONE_MACS list,
mimicking different drones in the fleet.

Usage:
    python simulate.py                  # uses default API url
    python simulate.py --url http://... # custom url
    python simulate.py --delay 2        # seconds between requests
"""

import argparse
import random
import time
from pathlib import Path

import requests

# ── Fake drone MAC addresses ──────────────────────────────────────────────────
DRONE_MACS = [
    "A4:CF:12:98:AB:BB",
    "A4:CF:12:98:AB:AA",
    "A4:CF:12:98:AB:99",
    "A4:CF:12:98:AB:88",
    "A4:CF:12:98:AB:77",
    "A4:CF:12:98:AB:66",
    "A4:CF:12:98:AB:55",
    "A4:CF:12:98:AB:44",
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_simulation(api_url: str, images_dir: str, delay: float):
    images_path = Path(images_dir)

    if not images_path.exists():
        print(f"[Simulator] ❌ Folder not found: {images_path.resolve()}")
        return

    image_files = [
        f for f in sorted(images_path.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"[Simulator] ⚠️  No images found in '{images_path}'. "
              "Drop some .jpg / .png files into test_images/ and re-run.")
        return

    print(f"\n{'─'*60}")
    print(f"  🚁 Drone Simulator  |  {len(image_files)} image(s) to process")
    print(f"  API  → {api_url}")
    print(f"  MACs → {len(DRONE_MACS)} simulated drones")
    print(f"{'─'*60}\n")

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        health.raise_for_status()
        print("[Simulator] ✅ API is up.\n")
    except Exception as e:
        print(f"[Simulator] ❌ API not reachable: {e}")
        print("  → Make sure Flask is running:  python app.py\n")
        return

    # ── Loop through images ───────────────────────────────────────────────────
    for idx, img_path in enumerate(image_files, start=1):
        mac = random.choice(DRONE_MACS)

        print(f"[{idx}/{len(image_files)}] 📷 {img_path.name}  |  drone: {mac}")

        try:
            with open(img_path, "rb") as f:
                resp = requests.post(
                    f"{api_url}/detect",
                    files={"image": (img_path.name, f, "image/jpeg")},
                    data={"mac_address": mac},
                    timeout=60,
                )

            data = resp.json()

            if resp.status_code == 200:
                if data.get("fire_detected"):
                    best = max(data["detections"], key=lambda d: d["confidence"])
                    print(f"  🔥 FIRE DETECTED!")
                    print(f"     x={data['alert']['x']}, y={data['alert']['y']}")
                    print(f"     confidence : {best['confidence']:.0%}")
                    print(f"     image_url  : {data.get('image_url', 'N/A')}")
                else:
                    print(f"  ✅ No fire.")
            else:
                print(f"  ⚠️  Error {resp.status_code}: {data.get('error')}")

        except Exception as e:
            print(f"  ❌ Request failed: {e}")

        print()

        if delay > 0 and idx < len(image_files):
            time.sleep(delay)

    print("─" * 60)
    print("  Simulation complete.")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Drone Simulator")
    parser.add_argument(
        "--url",    default="http://localhost:5000",
        help="Flask API base URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--images", default="test_images",
        help="Folder containing test images (default: test_images)"
    )
    parser.add_argument(
        "--delay",  type=float, default=1.0,
        help="Seconds between requests (default: 1)"
    )
    args = parser.parse_args()
    run_simulation(args.url, args.images, args.delay)