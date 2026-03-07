"""
test_fire_segmentation.py
=========================
Standalone test: simulates ONE active fire, renders the Algeria map,
and overlays a segmentation mask (glowing heatmap blob + bounding box
+ confidence badge) — exactly what you'd show on the frontend.

Run:
    python test_fire_segmentation.py

Output:
    test_fire_segmented.png   ← annotated map
    test_fire_raw.png         ← same map WITHOUT segmentation (baseline)
"""

import io, math, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import pandas as pd
from datetime import date

# ── Pull helpers from your existing app ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    from app import fetch_basemap, REGIONS
    print("✅  Imported fetch_basemap + REGIONS from app.py")
except ImportError:
    print("⚠️  app.py not found — using built-in stubs")
    REGIONS = {
        "Tizi Ouzou":  (36.75,  4.05),
        "Béjaïa":      (36.75,  5.06),
        "Skikda":      (36.89,  6.90),
        "Blida":       (36.46,  2.83),
        "Constantine": (36.30,  6.61),
        "Mostaganem":  (35.69,  0.63),
        "Souk Ahras":  (36.28,  7.95),
    }

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

    import requests as req
    def fetch_basemap(W, S, E, N, zoom=6):
        x0, y0 = _deg2tile(N, W, zoom)
        x1, y1 = _deg2tile(S, E, zoom)
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        TILE_PX = 256
        tw = max(x1 - x0 + 1, 1)
        th = max(y1 - y0 + 1, 1)
        canvas = Image.new("RGB", (tw * TILE_PX, th * TILE_PX), (234, 231, 221))
        HEADERS = {"User-Agent": "AlgeriaFireDetectionAI/1.0"}
        fetched = 0
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
                try:
                    r = req.get(url, headers=HEADERS, timeout=8)
                    if r.status_code == 200:
                        tile = Image.open(io.BytesIO(r.content)).convert("RGB")
                        canvas.paste(tile, ((tx - x0) * TILE_PX, (ty - y0) * TILE_PX))
                        fetched += 1
                except Exception:
                    continue
        lat_n, lon_w = _tile2deg(x0,     y0,     zoom)
        lat_s, lon_e = _tile2deg(x1 + 1, y1 + 1, zoom)
        return canvas, (lon_w, lon_e, lat_s, lat_n)


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHETIC FIRE EVENT — tweak these values to test different scenarios
# ─────────────────────────────────────────────────────────────────────────────

FIRE_EVENT = {
    # ── Location ──────────────────────────────────────────────────────────
    "region":       "Tizi Ouzou",
    "latitude":     36.78,
    "longitude":    4.08,

    # ── VIIRS pixel cluster (simulate a small burn cluster) ───────────────
    "cluster_radius_deg": 0.12,   # ~13 km radius
    "num_pixels":         22,     # how many VIIRS 375m pixels

    # ── Fire intensity ─────────────────────────────────────────────────────
    "peak_brightness_k":  435.0,  # K  (>380 = high-intensity active fire)
    "frp_mw":             87.3,   # MW fire radiative power
    "confidence":         "high",

    # ── Segmentation display options ───────────────────────────────────────
    "seg_color":    "#ff3300",
    "seg_alpha":    0.28,
    "bbox_padding": 0.15,         # degrees padding around cluster bbox
}


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE PIXEL CLUSTER  (realistic scatter around fire centroid)
# ─────────────────────────────────────────────────────────────────────────────

def make_cluster(event: dict) -> pd.DataFrame:
    np.random.seed(7)
    n   = event["num_pixels"]
    r   = event["cluster_radius_deg"]
    lat = event["latitude"]
    lon = event["longitude"]

    # Bivariate normal — fires tend to elongate along ridgelines
    lats = lat + np.random.normal(0, r * 0.55, n)
    lons = lon + np.random.normal(0, r * 0.70, n)

    peak = event["peak_brightness_k"]
    # Brightness decays from centre outward
    dists = np.sqrt((lats - lat)**2 + (lons - lon)**2) / r
    brights = peak - dists * np.random.uniform(30, 60, n)
    brights = np.clip(brights, 320, peak)

    frps = event["frp_mw"] * (1 - dists * 0.4) * np.random.uniform(0.7, 1.3, n)
    frps = np.clip(frps, 5, event["frp_mw"] * 1.4)

    return pd.DataFrame({
        "latitude":    lats,
        "longitude":   lons,
        "bright_ti4":  brights,
        "frp":         frps,
        "confidence":  event["confidence"],
        "acq_date":    date.today().isoformat(),
        "region_name": event["region"],
    })


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_heatmap_blob(ax, df, color="#ff3300", alpha=0.28):
    """
    Gaussian kernel density heatmap rendered as a semi-transparent
    filled contour — simulates a proper segmentation mask.
    """
    from scipy.ndimage import gaussian_filter

    # Build 2-D histogram on a fine grid
    W, E = df["longitude"].min() - 0.3, df["longitude"].max() + 0.3
    S, N = df["latitude"].min()  - 0.2, df["latitude"].max()  + 0.2
    res  = 200
    H, yedges, xedges = np.histogram2d(
        df["latitude"], df["longitude"],
        bins=res,
        range=[[S, N], [W, E]],
        weights=df["bright_ti4"] - 310   # weight by excess heat
    )

    # Smooth → contour → filled
    H_smooth = gaussian_filter(H, sigma=8)
    xs = (xedges[:-1] + xedges[1:]) / 2
    ys = (yedges[:-1] + yedges[1:]) / 2
    XX, YY = np.meshgrid(xs, ys)

    # Custom colormap: transparent → fire colour
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fire_seg",
        [(0, 0, 0, 0),           # fully transparent at low density
         (*mcolors.to_rgb(color), 0.15),
         (*mcolors.to_rgb(color), alpha),
         (*mcolors.to_rgb("#ffcc00"), alpha + 0.08)],  # hot core = yellow
        N=256
    )

    lvl_min = H_smooth.max() * 0.08
    lvl_max = H_smooth.max()
    levels  = np.linspace(lvl_min, lvl_max, 12)

    ax.contourf(XX, YY, H_smooth,
                levels=levels, cmap=cmap, zorder=6)

    # Bright segmentation border
    ax.contour(XX, YY, H_smooth,
               levels=[lvl_min * 2],
               colors=[color], linewidths=[1.8],
               linestyles=["--"], alpha=0.85, zorder=7)


def draw_bounding_box(ax, df, event, color="#ff3300"):
    """Tight bounding box + corner ticks (SAR-style annotation)."""
    pad = event["bbox_padding"]
    x0  = df["longitude"].min() - pad
    y0  = df["latitude"].min()  - pad
    w   = df["longitude"].max() - df["longitude"].min() + 2 * pad
    h   = df["latitude"].max()  - df["latitude"].min()  + 2 * pad

    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="square,pad=0",
        linewidth=2.2, edgecolor=color,
        facecolor="none", linestyle="-",
        zorder=9
    )
    ax.add_patch(rect)

    # Corner L-ticks (like a camera viewfinder)
    tick = 0.06
    for cx, cy in [(x0, y0), (x0+w, y0), (x0, y0+h), (x0+w, y0+h)]:
        dx = tick  if cx == x0   else -tick
        dy = tick  if cy == y0   else -tick
        ax.plot([cx, cx+dx], [cy, cy],      color=color, lw=2.5, zorder=10)
        ax.plot([cx, cx],    [cy, cy+dy],   color=color, lw=2.5, zorder=10)


def draw_confidence_badge(ax, df, event, color="#ff3300"):
    """Floating badge with fire stats anchored to top-right of bbox."""
    pad  = event["bbox_padding"]
    bx   = df["longitude"].max() + pad + 0.05
    by   = df["latitude"].max()  + pad

    text = (
        f"🔥 ACTIVE FIRE\n"
        f"Région : {event['region']}\n"
        f"Pixels : {event['num_pixels']}\n"
        f"Bright : {event['peak_brightness_k']:.0f} K\n"
        f"FRP    : {event['frp_mw']:.1f} MW\n"
        f"Conf.  : {event['confidence'].upper()}"
    )

    ax.annotate(
        text,
        xy=(df["longitude"].mean(), df["latitude"].mean()),
        xytext=(bx, by),
        fontsize=8.5, fontfamily="monospace",
        color="white",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=1.5,
            connectionstyle="arc3,rad=-0.2"
        ),
        bbox=dict(
            facecolor="#1a0000", edgecolor=color,
            boxstyle="round,pad=0.5", alpha=0.93,
            linewidth=1.8
        ),
        zorder=12
    )


def draw_pulse_rings(ax, event, color="#ff3300"):
    """Concentric semi-transparent rings — visual pulse / sonar effect."""
    cx, cy = event["longitude"], event["latitude"]
    for r, a in [(0.08, 0.55), (0.16, 0.30), (0.26, 0.14)]:
        circle = plt.Circle((cx, cy), r,
                             color=color, fill=False,
                             linewidth=1.2, alpha=a,
                             linestyle="-", zorder=8)
        ax.add_patch(circle)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_segmented_map(df: pd.DataFrame, event: dict,
                          with_segmentation: bool = True) -> bytes:
    W, S, E, N = -8.67, 19.06, 11.99, 37.09

    print("  🗺️  Fetching OSM basemap ...")
    basemap, (ext_w, ext_e, ext_s, ext_n) = fetch_basemap(W, S, E, N, zoom=6)

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#0d0d0d" if with_segmentation else "#eae7dd")

    ax.imshow(basemap,
              extent=[ext_w, ext_e, ext_s, ext_n],
              origin="upper", aspect="auto",
              interpolation="bilinear",
              zorder=0,
              alpha=0.80 if with_segmentation else 1.0)

    # Darken the map slightly in segmentation mode for contrast
    if with_segmentation:
        ax.imshow(
            np.zeros((10, 10, 4), dtype=np.float32) + [0, 0, 0, 0.35],
            extent=[ext_w, ext_e, ext_s, ext_n],
            origin="upper", aspect="auto", zorder=1
        )

    ax.set_xlim(W, E)
    ax.set_ylim(S, N)
    ax.set_aspect("auto")
    ax.set_facecolor("#0d0d0d" if with_segmentation else "#eae7dd")

    # ── Base fire dots ─────────────────────────────────────────────────────
    bright = df["bright_ti4"].values
    sizes  = np.where(bright > 380, 120,
             np.where(bright > 340, 65, 32)).astype(float)

    # Glow ring
    ax.scatter(df["longitude"], df["latitude"],
               s=sizes * 5, c=event["seg_color"], alpha=0.12,
               linewidths=0, zorder=4)

    sc = ax.scatter(df["longitude"], df["latitude"],
                    c=bright, cmap="YlOrRd",
                    norm=mcolors.Normalize(vmin=310, vmax=460),
                    s=sizes, alpha=0.97,
                    linewidths=0.6, edgecolors="white",
                    zorder=5)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.36, pad=0.01)
    cbar.set_label("Brightness Temp (K)", fontsize=9,
                   color="white" if with_segmentation else "black")
    cbar.ax.yaxis.set_tick_params(labelsize=8,
                                  colors="white" if with_segmentation else "black")
    if with_segmentation:
        cbar.ax.yaxis.label.set_color("white")

    # ── Segmentation layers ────────────────────────────────────────────────
    if with_segmentation:
        print("  🔬 Drawing segmentation overlay ...")
        try:
            draw_heatmap_blob(ax, df, event["seg_color"], event["seg_alpha"])
        except ImportError:
            print("     (scipy not installed — skipping heatmap blob)")

        draw_pulse_rings(ax, event, event["seg_color"])
        draw_bounding_box(ax, df, event, event["seg_color"])
        draw_confidence_badge(ax, df, event, event["seg_color"])

    # ── City dots ─────────────────────────────────────────────────────────
    txt_color = "white" if with_segmentation else "#111111"
    for name, (lat, lon) in REGIONS.items():
        ax.plot(lon, lat, "o", ms=4, color="#aaaaaa",
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        ax.text(lon + 0.13, lat + 0.13, name, fontsize=7,
                color=txt_color, fontweight="bold",
                bbox=dict(facecolor="black" if with_segmentation else "white",
                          alpha=0.55, edgecolor="none",
                          boxstyle="round,pad=0.15"),
                zorder=7)

    # ── Stats box ──────────────────────────────────────────────────────────
    box_bg  = "#0d0d0d" if with_segmentation else "white"
    box_txt = "white"   if with_segmentation else "black"
    ax.text(0.985, 0.985,
            f"TEST SCENARIO\n"
            f"Pixels     : {len(df)}\n"
            f"Max bright : {df['bright_ti4'].max():.0f} K\n"
            f"Max FRP    : {df['frp'].max():.1f} MW\n"
            f"Satellite  : VIIRS SNPP 375m\n"
            f"Segmented  : {'YES ✓' if with_segmentation else 'NO (baseline)'}",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="right", fontfamily="monospace",
            color=box_txt,
            bbox=dict(facecolor=box_bg, edgecolor="#ff3300" if with_segmentation else "#aaaaaa",
                      boxstyle="round,pad=0.5", alpha=0.92),
            zorder=11)

    # ── Title ──────────────────────────────────────────────────────────────
    title_color = "white" if with_segmentation else "black"
    seg_label   = " — SEGMENTATION TEST" if with_segmentation else " — BASELINE"
    ax.set_title(
        f"🇩🇿  Algeria Fire Detection Map{seg_label}\n"
        f"Synthetic event · {event['region']} · "
        f"({event['latitude']:.3f}°N, {event['longitude']:.3f}°E)",
        fontsize=13, fontweight="bold", pad=12, color=title_color
    )

    ax.set_xlabel("Longitude (°E)", fontsize=10,
                  color="white" if with_segmentation else "black")
    ax.set_ylabel("Latitude (°N)",  fontsize=10,
                  color="white" if with_segmentation else "black")
    ax.tick_params(labelsize=8,
                   colors="white" if with_segmentation else "black")
    for spine in ax.spines.values():
        spine.set_edgecolor("white" if with_segmentation else "#cccccc")

    ax.text(0.01, 0.01, "© OpenStreetMap contributors",
            transform=ax.transAxes, fontsize=6.5,
            color="#888888", zorder=10)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("🔥  FireWatch — Segmentation Test")
    print("=" * 55)
    print(f"\nFire event config:")
    for k, v in FIRE_EVENT.items():
        print(f"  {k:25s}: {v}")
    print()

    # 1. Build synthetic cluster
    print("📡 Generating synthetic fire cluster ...")
    df = make_cluster(FIRE_EVENT)
    print(f"   {len(df)} pixels generated")
    print(f"   Lat range  : {df['latitude'].min():.3f} → {df['latitude'].max():.3f}")
    print(f"   Lon range  : {df['longitude'].min():.3f} → {df['longitude'].max():.3f}")
    print(f"   Max bright : {df['bright_ti4'].max():.1f} K")
    print(f"   Max FRP    : {df['frp'].max():.1f} MW")
    print()

    # 2. Render baseline (no segmentation)
    print("🗺️  Rendering baseline map ...")
    raw_png = render_segmented_map(df, FIRE_EVENT, with_segmentation=False)
    with open("test_fire_raw.png", "wb") as f:
        f.write(raw_png)
    print(f"   ✅ Saved → test_fire_raw.png  ({len(raw_png)//1024} KB)")
    print()

    # 3. Render segmented map
    print("🔬 Rendering segmented map ...")
    seg_png = render_segmented_map(df, FIRE_EVENT, with_segmentation=True)
    with open("test_fire_segmented.png", "wb") as f:
        f.write(seg_png)
    print(f"   ✅ Saved → test_fire_segmented.png  ({len(seg_png)//1024} KB)")
    print()

    print("=" * 55)
    print("✅  Done!  Open test_fire_segmented.png to see the result.")
    print("=" * 55)
    print()
    print("💡 To test different fire scenarios, edit FIRE_EVENT at the top:")
    print("   • Change latitude/longitude to any point in Algeria")
    print("   • Increase num_pixels / peak_brightness_k for bigger fires")
    print("   • Change region to any of:", list(REGIONS.keys()))