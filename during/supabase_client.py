"""
supabase_client.py
Posts detection results directly to the Supabase Edge Function.
The edge function handles DB insert + bucket upload internally.

Endpoint : POST https://ekybmuyqummcaqfpkzrt.supabase.co/functions/v1/ai-detection
Fields   : mac_address, status, confidence, x, y, image (file)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# supabase_client.py
EDGE_FUNCTION_URL = os.getenv("SUPABASE_EDGE_URL")
ANON_KEY          = os.getenv("SUPABASE_ANON_KEY")

def confidence_to_level(conf_float: float) -> str:
    """
    Map a 0–1 float confidence to a string level.
      >= 0.75  → HIGH
      >= 0.50  → MEDIUM
      <  0.50  → LOW
    """
    if conf_float >= 0.75:
        return "HIGH"
    elif conf_float >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def post_detection(mac_address: str, x: int, y: int,
                   confidence: float, annotated_image_path: str) -> dict:
    """
    POST multipart/form-data to the Supabase Edge Function.

    Parameters
    ----------
    mac_address          : drone MAC address string
    x, y                 : bounding-box centre coordinates (pixels)
    confidence           : float 0–1 from YOLO
    annotated_image_path : local path to the annotated .jpg

    Returns
    -------
    JSON response from the edge function as a dict.
    """
    headers = {
        "Authorization": f"Bearer {ANON_KEY}",
        # Do NOT manually set Content-Type — requests sets it automatically
        # for multipart/form-data and includes the correct boundary.
    }

    with open(annotated_image_path, "rb") as img_file:
        files = {
            "image": (
                os.path.basename(annotated_image_path),
                img_file,
                "image/jpeg",
            )
        }

        data = {
            "mac_address": mac_address,
            "status":      "FIRE_DETECTED",
            "confidence":  confidence_to_level(confidence),
            "x":           str(x),
            "y":           str(y),
        }

        print(f"[Supabase] Posting to edge function ...")
        print(f"           mac={mac_address} | x={x} y={y} | conf={confidence_to_level(confidence)}")

        resp = requests.post(
            EDGE_FUNCTION_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=30,
        )

    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Edge function error [{resp.status_code}]: {resp.text}"
        )

    result = resp.json()
    print(f"[Supabase] ✅ Response: {result}")
    return result