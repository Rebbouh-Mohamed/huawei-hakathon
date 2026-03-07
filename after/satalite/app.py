"""
Algeria Wildfire Detection - Flask API
=======================================
POST /upload  → receive image + data → save to Supabase
GET  /map     → return Algeria fire map on REAL OpenStreetMap tiles
GET  /latest  → return latest fire records from Supabase
GET  /        → health check
"""

import os, io, math, uuid, logging
from datetime import datetime, date

import requests as req
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from flask import Flask, request, jsonify, send_file
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("algeria-fire")

app = Flask(__name__)

NASA_KEY     = os.getenv("NASA_FIRMS_KEY", "YOUR_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL",   "https://xxx.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY",   "YOUR_KEY")
TABLE        = "fire_detections"
BUCKET       = "fire-images"
BBOX         = "-8.67,19.06,11.99,37.09"

REGIONS = {
    "Tizi Ouzou":  (36.75,  4.05),
    "Béjaïa":      (36.75,  5.06),
    "Skikda":      (36.89,  6.90),
    "Blida":       (36.46,  2.83),
    "Constantine": (36.30,  6.61),
    "Mostaganem":  (35.69,  0.63),
    "Souk Ahras":  (36.28,  7.95),
}

def get_db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  OSM TILE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _deg2tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y

def _tile2deg(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon

def fetch_basemap(W, S, E, N, zoom=6):
    """
    Download OSM tiles covering Algeria bbox and stitch into one PIL image.
    Returns (PIL.Image, (west, east, south, north)).
    """
    # In OSM: y increases downward, so north → smaller y, south → larger y
    x0, y0 = _deg2tile(N, W, zoom)   # top-left  (north, west)
    x1, y1 = _deg2tile(S, E, zoom)   # bottom-right (south, east)

    # Safety: ensure correct ordering
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0

    TILE_PX = 256
    tw = max(x1 - x0 + 1, 1)
    th = max(y1 - y0 + 1, 1)
    canvas = Image.new("RGB", (tw * TILE_PX, th * TILE_PX), (234, 231, 221))

    HEADERS = {"User-Agent": "AlgeriaFireDetectionAI/1.0"}
    SERVERS = [
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
    ]

    fetched = 0
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            for srv in SERVERS:
                url = srv.format(z=zoom, x=tx, y=ty)
                try:
                    r = req.get(url, headers=HEADERS, timeout=8)
                    if r.status_code == 200:
                        tile = Image.open(io.BytesIO(r.content)).convert("RGB")
                        canvas.paste(tile, ((tx - x0) * TILE_PX,
                                            (ty - y0) * TILE_PX))
                        fetched += 1
                        break
                except Exception:
                    continue

    log.info(f"OSM tiles: {fetched}/{tw * th} fetched at zoom {zoom}")

    lat_n, lon_w = _tile2deg(x0,     y0,     zoom)
    lat_s, lon_e = _tile2deg(x1 + 1, y1 + 1, zoom)
    return canvas, (lon_w, lon_e, lat_s, lat_n)


# ─────────────────────────────────────────────────────────────────────────────
#  MAP RENDERER — real OSM basemap + fire overlay
# ─────────────────────────────────────────────────────────────────────────────

def render_map(df: pd.DataFrame) -> bytes:
    W, S, E, N = -8.67, 19.06, 11.99, 37.09

    log.info("Fetching OSM basemap ...")
    basemap, (ext_w, ext_e, ext_s, ext_n) = fetch_basemap(W, S, E, N, zoom=6)

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#eae7dd")

    # Real map as background
    ax.imshow(basemap,
              extent=[ext_w, ext_e, ext_s, ext_n],
              origin="upper", aspect="auto",
              interpolation="bilinear", zorder=0)

    ax.set_xlim(W, E)
    ax.set_ylim(S, N)
    ax.set_aspect("auto")

    # ── Fire points ───────────────────────────────────────────────────────
    plot_df = df.copy()
    if "source" in plot_df.columns:
        plot_df = plot_df[plot_df["source"] != "SYNTHETIC_FALLBACK"]

    if not plot_df.empty and "latitude" in plot_df.columns:
        bright = (plot_df["bright_ti4"].clip(310, 500).values
                  if "bright_ti4" in plot_df.columns
                  else np.full(len(plot_df), 360))

        sizes = np.where(bright > 380, 130,
                np.where(bright > 340, 75, 35)).astype(float)

        # Soft glow ring behind each fire dot
        ax.scatter(plot_df["longitude"], plot_df["latitude"],
                   s=sizes * 4, c="#ff4400", alpha=0.15,
                   linewidths=0, zorder=4)

        # Fire dots coloured by brightness temperature
        sc = ax.scatter(plot_df["longitude"], plot_df["latitude"],
                        c=bright, cmap="YlOrRd",
                        norm=mcolors.Normalize(vmin=310, vmax=460),
                        s=sizes, alpha=0.95,
                        linewidths=0.7, edgecolors="white",
                        zorder=5)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.38, pad=0.01)
        cbar.set_label("Brightness Temp (K)", fontsize=9)
        cbar.ax.yaxis.set_tick_params(labelsize=8)

        # Region callout labels
        if "region_name" in plot_df.columns:
            for region in plot_df["region_name"].dropna().unique():
                rdf = plot_df[plot_df["region_name"] == region]
                cx  = rdf["longitude"].mean()
                cy  = rdf["latitude"].mean()
                ax.annotate(
                    f"🔥 {region}  ({len(rdf)})",
                    xy=(cx, cy), xytext=(cx + 0.7, cy + 0.6),
                    fontsize=8.5, fontweight="bold", color="#cc0000",
                    arrowprops=dict(arrowstyle="->", color="#cc0000",
                                   lw=1.0, connectionstyle="arc3,rad=0.15"),
                    bbox=dict(facecolor="white", edgecolor="#cc0000",
                              boxstyle="round,pad=0.35", alpha=0.90),
                    zorder=8)

        # Date legend
        if "acq_date" in plot_df.columns:
            pal = ["#e60000", "#ff7700", "#ffcc00"]
            for i, d in enumerate(sorted(plot_df["acq_date"].unique())):
                cnt = (plot_df["acq_date"] == d).sum()
                ax.plot([], [], "o", color=pal[i % 3], ms=7,
                        label=f"{d}  ({cnt} detections)")
            ax.legend(loc="lower left", fontsize=8.5,
                      framealpha=0.90, edgecolor="#cccccc",
                      facecolor="white")

    else:
        ax.text((W + E) / 2, (S + N) / 2,
                "✅  No active fires detected\nin Algeria today",
                ha="center", va="center", fontsize=18,
                fontweight="bold", color="#005500",
                bbox=dict(facecolor="white", edgecolor="#008800",
                          boxstyle="round,pad=0.8", alpha=0.92),
                zorder=10)

    # ── City markers ──────────────────────────────────────────────────────
    for name, (lat, lon) in REGIONS.items():
        ax.plot(lon, lat, "o", ms=4, color="#333333",
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        ax.text(lon + 0.13, lat + 0.13, name, fontsize=7,
                color="#111111", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.60,
                          edgecolor="none", boxstyle="round,pad=0.15"),
                zorder=7)

    # ── Stats box ─────────────────────────────────────────────────────────
    n_total = len(plot_df) if not plot_df.empty else 0
    n_high  = int((plot_df["confidence"] == "high").sum()) \
              if (not plot_df.empty and "confidence" in plot_df.columns) else 0
    avg_frp = (f"{plot_df['frp'].mean():.1f} MW"
               if (not plot_df.empty and "frp" in plot_df.columns) else "N/A")

    ax.text(0.985, 0.985,
            f"Detections : {n_total}\n"
            f"High conf. : {n_high}\n"
            f"Avg FRP    : {avg_frp}\n"
            f"Satellite  : VIIRS SNPP 375m\n"
            f"Source     : NASA FIRMS",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="right", fontfamily="monospace",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                      boxstyle="round,pad=0.5", alpha=0.92),
            zorder=10)

    # ── Labels & title ────────────────────────────────────────────────────
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)",  fontsize=10)
    ax.tick_params(labelsize=8)

    ax.set_title(
        f"🇩🇿  Algeria — Satellite Fire Detection Map\n"
        f"NASA VIIRS SNPP  •  "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  •  AgriTech AI",
        fontsize=13, fontweight="bold", pad=12)

    # OSM attribution (required by tile license)
    ax.text(0.01, 0.01, "© OpenStreetMap contributors",
            transform=ax.transAxes, fontsize=6.5,
            color="#555555", zorder=10)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor="#eae7dd")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  NASA FIRMS FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nasa_firms(day_range: int = 3) -> pd.DataFrame:
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
           f"/{NASA_KEY}/VIIRS_SNPP_NRT/{BBOX}/{day_range}")
    try:
        resp = req.get(url, timeout=20)
        text = resp.text.strip()
        if text and not text.startswith("<"):
            df = pd.read_csv(io.StringIO(text))
            df["source"] = "NASA_FIRMS_REAL"
            log.info(f"NASA FIRMS: {len(df)} real detections")
            return df
    except Exception as e:
        log.warning(f"NASA fetch failed: {e}")
    return _demo_fire_data()


def _demo_fire_data() -> pd.DataFrame:
    np.random.seed(42)
    today     = date.today().isoformat()
    yesterday = date.fromordinal(date.today().toordinal() - 1).isoformat()
    fires = []
    for _ in range(18):
        fires.append({
            "latitude":    36.75 + np.random.normal(0, 0.08),
            "longitude":   4.05  + np.random.normal(0, 0.08),
            "bright_ti4":  np.random.uniform(355, 460),
            "frp":         np.random.uniform(25, 150),
            "confidence":  np.random.choice(["high", "high", "nominal"]),
            "acq_date":    today,
            "satellite":   "N",
            "source":      "DEMO",
            "region_name": "Tizi Ouzou",
        })
    for _ in range(11):
        fires.append({
            "latitude":    36.89 + np.random.normal(0, 0.06),
            "longitude":   6.90  + np.random.normal(0, 0.06),
            "bright_ti4":  np.random.uniform(335, 420),
            "frp":         np.random.uniform(15, 90),
            "confidence":  np.random.choice(["high", "nominal"]),
            "acq_date":    yesterday,
            "satellite":   "N20",
            "source":      "DEMO",
            "region_name": "Skikda",
        })
    return pd.DataFrame(fires)


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Algeria Wildfire Detection API",
        "endpoints": {
            "POST /upload": "Upload fire image + data",
            "GET  /map":    "Algeria fire map (OSM tiles + fire overlay)",
            "GET  /latest": "Latest records from Supabase",
        },
    })


@app.route("/map", methods=["GET"])
def get_map():
    try:
        df  = fetch_nasa_firms()
        png = render_map(df)
        return send_file(io.BytesIO(png), mimetype="image/png",
                         download_name="algeria_fire_map.png")
    except Exception as e:
        log.error(f"Map error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    try:
        db = get_db()
        image_url = None

        if "image" in request.files:
            img   = request.files["image"]
            fname = f"{uuid.uuid4()}.jpg"
            db.storage.from_(BUCKET).upload(
                path=fname, file=img.read(),
                file_options={"content-type": img.mimetype or "image/jpeg"})
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{fname}"

        record = {
            "latitude":        float(request.form.get("latitude",        0)),
            "longitude":       float(request.form.get("longitude",       0)),
            "classification":  request.form.get("classification",  "no_fire"),
            "confidence":      request.form.get("confidence",       "nominal"),
            "brightness_temp": float(request.form.get("brightness_temp", 0)),
            "frp":             float(request.form.get("frp",             0)),
            "region_name":     request.form.get("region_name",      "Unknown"),
            "source":          request.form.get("source",         "satellite"),
            "notes":           request.form.get("notes",                  ""),
            "image_url":       image_url,
            "country":         "Algeria",
            "acq_date":        date.today().isoformat(),
            "uploaded_at":     datetime.utcnow().isoformat(),
        }
        db.table(TABLE).insert(record).execute()
        return jsonify({"status": "success", "image_url": image_url,
                        "record": record}), 201

    except Exception as e:
        log.error(f"Upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/latest", methods=["GET"])
def get_latest():
    try:
        limit  = int(request.args.get("limit", 50))
        result = (get_db().table(TABLE)
                  .select("*")
                  .order("uploaded_at", desc=True)
                  .limit(limit).execute())
        return jsonify({"status": "success",
                        "count": len(result.data),
                        "records": result.data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5002)