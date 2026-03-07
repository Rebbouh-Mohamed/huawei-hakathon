"""
app.py
Flask API — Fire Detection Drone Simulator

Endpoints
---------
POST /detect
    Body (multipart/form-data):
        image       : image file
        mac_address : drone MAC address (string)

    Response (JSON):
        {
          "fire_detected": true,
          "detections": [...],
          "alert": { ...supabase row... }   ← only when fire detected
          "image_url": "https://..."        ← only when fire detected
        }

GET /health
    Returns 200 OK — used by simulate.py to verify the API is up.
"""

import os
import uuid
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from detector import load_model, detect_fire
from supabase_client import post_detection, confidence_to_level

load_dotenv()

app = Flask(__name__)

# ── Startup: load YOLO model ──────────────────────────────────────────────────
# Change "best.pt" to the path of your fire-detection weights file.
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "best.pt")
load_model(MODEL_WEIGHTS)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Main detection endpoint ───────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    # ── Validate inputs ───────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (key: 'image')"}), 400

    mac_address = request.form.get("mac_address", "00:00:00:00:00:00")
    image_file  = request.files["image"]

    # ── Save uploaded image to a temp file ───────────────────────────────────
    suffix    = Path(image_file.filename).suffix or ".jpg"
    tmp_path  = tempfile.mktemp(suffix=suffix)
    image_file.save(tmp_path)

    try:
        # ── Run YOLO detection ────────────────────────────────────────────────
        detections, annotated_path, fire_detected = detect_fire(tmp_path)

        if not fire_detected:
            return jsonify({
                "fire_detected": False,
                "mac_address":   mac_address,
                "detections":    [],
                "message":       "No fire detected.",
            }), 200

        # ── Best detection (highest confidence) ───────────────────────────────
        best = max(detections, key=lambda d: d["confidence"])

        # ── POST to Supabase Edge Function (handles bucket + DB) ──────────────
        # Sends: mac_address, x, y, confidence (HIGH/MED/LOW), image file
        supabase_response = post_detection(
            mac_address          = mac_address,
            x                    = best["x"],
            y                    = best["y"],
            confidence           = best["confidence"],
            annotated_image_path = annotated_path,
        )
        print("res:",supabase_response)
        return jsonify({
            "fire_detected": True,
            "mac_address":   mac_address,
            "detections":    detections,
            "alert": {
                "mac_address": mac_address,
                "x":           best["x"],
                "y":           best["y"],
                "status":      "FIRE_DETECTED",
                "confidence":  confidence_to_level(best["confidence"]),
                "confidence_raw": best["confidence"],
            },
            "supabase": supabase_response,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    print(f"\n🔥 Fire Detection API running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)