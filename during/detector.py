"""
detector.py
Loads a YOLOv5 model and runs fire detection on an image.
Returns bounding boxes, confidence scores, and an annotated image.
"""

import cv2
import torch
import numpy as np
from pathlib import Path


# ── Model singleton ───────────────────────────────────────────────────────────
_model = None


def _validate_pt_file(path: str) -> bool:
    """
    Check the file is a real PyTorch binary (starts with the pickle magic bytes)
    and not a Git LFS pointer or HTML error page.

    PyTorch .pt files start with \\x80\\x02 (pickle protocol 2) or PK (zip/torch zip).
    A Git LFS pointer starts with 'version https://' (plain text).
    """
    pt = Path(path)
    if not pt.exists():
        return False
    if pt.stat().st_size < 1024:          # real weights are always >> 1 KB
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    # Valid PyTorch files: pickle magic \x80\x02 … or PK zip header
    if header[:2] in (b"\x80\x02", b"\x80\x04", b"\x80\x05", b"PK"):
        return True
    # Anything else (text, HTML, LFS pointer '\nversion…') is invalid
    return False


def load_model(weights_path: str = "best.pt") -> None:
    """
    Load the YOLOv5 model once at startup.

    - If weights_path exists AND is a valid .pt binary  → load custom model
    - If it exists but is corrupted / LFS pointer        → print fix instructions, fallback
    - If it doesn't exist                                → fallback to yolov5s
    """
    global _model

    pt_path = Path(weights_path)

    if pt_path.exists():
        if _validate_pt_file(weights_path):
            print(f"[Detector] ✅ Loading custom weights: {weights_path}")
            _model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=weights_path,
                force_reload=True,   # force_reload=True avoids stale hub cache
            )
        else:
            # ── Likely a Git LFS pointer or bad download ──────────────────────
            size_kb = pt_path.stat().st_size // 1024
            with open(weights_path, "rb") as f:
                snippet = f.read(80)
            print("\n" + "─" * 60)
            print(f"[Detector] ❌ '{weights_path}' is NOT a valid PyTorch file!")
            print(f"   File size : {size_kb} KB  (real weights are usually > 5 MB)")
            print(f"   First bytes: {snippet}")
            print()
            print("   ── Most likely causes ──────────────────────────────")
            print("   1. Git LFS pointer — run:  git lfs pull")
            print("   2. Bad download    — re-download the .pt file")
            print("   3. Wrong file      — make sure it's the PyTorch weights")
            print("─" * 60 + "\n")
            print("[Detector] ⚠️  Falling back to yolov5s for now (no fire labels).")
            _model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    else:
        print(f"[Detector] ⚠️  '{weights_path}' not found — loading yolov5s as placeholder.")
        _model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    _model.eval()
    print("[Detector] Model ready.\n")


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


# ── Detection ─────────────────────────────────────────────────────────────────
def detect_fire(image_path: str, conf_threshold: float = 0.40):
    """
    Run inference on a single image.

    Returns
    -------
    detections : list of dicts
        Each dict: { x, y, width, height, confidence, label }
        x, y = centre of the bounding box in pixels
    annotated_path : str
        Path to the saved annotated image (bbox drawn in red).
    fire_detected : bool
    """
    model = get_model()

    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img_rgb)
    df = results.pandas().xyxy[0]          # DataFrame: xmin,ymin,xmax,ymax,confidence,class,name

    # Filter by confidence
    df = df[df["confidence"] >= conf_threshold]

    detections = []
    annotated = img_bgr.copy()

    for _, row in df.iterrows():
        xmin, ymin = int(row["xmin"]), int(row["ymin"])
        xmax, ymax = int(row["xmax"]), int(row["ymax"])
        conf       = float(row["confidence"])
        label      = str(row["name"])

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        detections.append({
            "x":          cx,
            "y":          cy,
            "width":      xmax - xmin,
            "height":     ymax - ymin,
            "confidence": round(conf, 4),
            "label":      label,
        })

        # Draw bounding box + label
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        text = f"{label} {conf:.0%}"
        cv2.putText(
            annotated, text,
            (xmin, max(ymin - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    # Save annotated image
    stem        = Path(image_path).stem
    out_dir     = Path("annotated")
    out_dir.mkdir(exist_ok=True)
    out_path    = str(out_dir / f"{stem}_annotated.jpg")
    cv2.imwrite(out_path, annotated)

    fire_detected = len(detections) > 0
    return detections, out_path, fire_detected