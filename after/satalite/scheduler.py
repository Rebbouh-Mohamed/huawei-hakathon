"""
Daily Satellite Fire Scanner + AI Report Generator
====================================================
- Runs NASA FIRMS scan every day at DAILY_RUN_TIME (UTC).
- Generates a full FireWatch AI report via Gemini every 3 days at 07:00 UTC
  (runs AFTER the satellite scan so it includes fresh satellite data).
- Emergency report auto-triggered if active_fire detections spike (> threshold).
"""

import os
import io
import json
import logging
import schedule
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from datetime import datetime, date, timedelta
from dotenv import load_dotenv

import google.generativeai as genai

from app import fetch_nasa_firms, render_map

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("scheduler")

# ── Supabase Edge Function ────────────────────────────────────────────────────
EDGE_URL  = "https://ekybmuyqummcaqfpkzrt.supabase.co/functions/v1/ai-harmed-areas"
ANON_KEY  = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
MAC_ADDR  = os.getenv("MAC_ADDRESS", "A4:CF:12:98:AB:44")

SUPABASE_HEADERS = {
    "Authorization": f"Bearer {ANON_KEY}",
    "apikey":        ANON_KEY,
}

# ── Supabase REST (for reading back aggregated data) ─────────────────────────
SUPABASE_URL      = os.getenv("SUPABASE_URL", "https://ekybmuyqummcaqfpkzrt.supabase.co")
SUPABASE_REST_KEY = os.getenv("SUPABASE_KEY", "")   # service_role key preferred for reads

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"   # or "gemini-1.5-flash" for faster/cheaper

# ── Schedule config ───────────────────────────────────────────────────────────
DAILY_RUN_TIME   = os.getenv("DAILY_RUN_TIME",  "06:00")   # satellite scan
REPORT_RUN_TIME  = os.getenv("REPORT_RUN_TIME", "07:00")   # report (after scan)
REPORT_EVERY_N_DAYS = int(os.getenv("REPORT_EVERY_N_DAYS", "3"))

# Emergency threshold: if active_fire pixels exceed this in one scan → instant report
EMERGENCY_FIRE_THRESHOLD = int(os.getenv("EMERGENCY_FIRE_THRESHOLD", "15"))

# Monitored regions (same as app.py)
MONITORED_REGIONS = [
    {"name": "Tizi Ouzou",  "lat": 36.75, "lon": 4.05},
    {"name": "Béjaïa",      "lat": 36.75, "lon": 5.06},
    {"name": "Skikda",      "lat": 36.89, "lon": 6.90},
    {"name": "Blida",       "lat": 36.46, "lon": 2.83},
    {"name": "Constantine", "lat": 36.30, "lon": 6.61},
    {"name": "Mostaganem",  "lat": 35.69, "lon": 0.63},
    {"name": "Souk Ahras",  "lat": 36.28, "lon": 7.95},
]

# ─────────────────────────────────────────────────────────────────────────────
# SATELLITE SCAN (unchanged logic, same as original)
# ─────────────────────────────────────────────────────────────────────────────

def post_to_edge(record: dict, image_bytes: bytes, image_name: str) -> dict:
    """POST one fire record + map image to the Supabase Edge Function."""
    files = {"image": (image_name, image_bytes, "image/png")}
    data  = {
        "mac_address":     record.get("mac_address",     MAC_ADDR),
        "latitude":        str(record["latitude"]),
        "longitude":       str(record["longitude"]),
        "classification":  record["classification"],
        "confidence":      record["confidence"],
        "brightness_temp": str(record["brightness_temp"]),
        "frp":             str(record["frp"]),
        "region_name":     record["region_name"],
        "source":          record["source"],
        "notes":           record["notes"],
        "country":         "Algeria",
        "acq_date":        record["acq_date"],
    }

    resp = requests.post(EDGE_URL, headers=SUPABASE_HEADERS, data=data,
                         files=files, timeout=30)

    if resp.status_code in (200, 201):
        log.info(f"  ✅ {record['classification']} @ ({record['latitude']:.3f}, {record['longitude']:.3f})")
        return {"status": "success"}
    else:
        log.warning(f"  ⚠️  {resp.status_code}: {resp.text[:150]}")
        return {"status": "error", "code": resp.status_code}


def run_daily_scan() -> dict:
    """
    Fetch NASA FIRMS, render map, post all records to Supabase.
    Returns a summary dict so the report generator can use it directly
    without needing to re-query Supabase.
    """
    log.info("=" * 55)
    log.info(f"🛰️  Daily scan — {datetime.utcnow().isoformat()}")
    log.info("=" * 55)

    # Step 1 — NASA data
    log.info("Step 1 — Fetching NASA FIRMS data ...")
    df = fetch_nasa_firms(day_range=1)
    log.info(f"         {len(df)} detections")

    # Step 2 — Render map
    log.info("Step 2 — Rendering Algeria fire map ...")
    map_bytes = render_map(df)
    map_name  = f"algeria_map_{date.today().isoformat()}.png"

    # Step 3 — Build records + scan summary
    scan_summary = {
        "total_pixels":       len(df),
        "active_fire_pixels": 0,
        "old_burn_pixels":    0,
        "no_fire_pixels":     0,
        "hottest_pixel_k":    0.0,
        "max_frp":            0.0,
        "regions_affected":   [],
    }

    if df.empty:
        records = [{
            "mac_address":     MAC_ADDR,
            "latitude":        36.75,
            "longitude":       4.05,
            "classification":  "no_fire",
            "confidence":      "scan_complete",
            "brightness_temp": 0,
            "frp":             0,
            "region_name":     "Algeria (full scan)",
            "source":          "NASA_FIRMS_DAILY_SCAN",
            "notes":           "Daily scan complete — no active fires detected",
            "acq_date":        date.today().isoformat(),
        }]
    else:
        records = []
        region_stats = {}

        for _, row in df.iterrows():
            bt  = float(row.get("bright_ti4", 0))
            frp = float(row.get("frp", 0))
            classification = (
                "active_fire" if bt > 340 else
                "old_burn"    if bt > 315 else
                "no_fire"
            )

            # Update summary counters
            scan_summary[f"{classification}_pixels"] = \
                scan_summary.get(f"{classification}_pixels", 0) + 1
            scan_summary["hottest_pixel_k"] = max(scan_summary["hottest_pixel_k"], bt)
            scan_summary["max_frp"]         = max(scan_summary["max_frp"], frp)

            region = str(row.get("region_name", "Unknown"))
            if region not in region_stats:
                region_stats[region] = {"fire_pixels": 0, "max_brightness": 0.0}
            if classification in ("active_fire", "old_burn"):
                region_stats[region]["fire_pixels"] += 1
                region_stats[region]["max_brightness"] = max(
                    region_stats[region]["max_brightness"], bt)

            records.append({
                "mac_address":     MAC_ADDR,
                "latitude":        float(row.get("latitude",  0)),
                "longitude":       float(row.get("longitude", 0)),
                "classification":  classification,
                "confidence":      str(row.get("confidence", "nominal")),
                "brightness_temp": bt,
                "frp":             frp,
                "region_name":     region,
                "source":          "NASA_FIRMS_DAILY_SCAN",
                "notes":           f"Auto scan {date.today().isoformat()}",
                "acq_date":        str(row.get("acq_date", date.today().isoformat())),
            })

        scan_summary["regions_affected"] = [
            {"name": r, **stats}
            for r, stats in sorted(
                region_stats.items(),
                key=lambda x: x[1]["fire_pixels"],
                reverse=True
            )
        ]

    # Step 4 — POST to Edge Function
    log.info(f"Step 3 — Posting {len(records)} records to Edge Function ...")
    ok, fail = 0, 0
    for rec in records:
        res = post_to_edge(rec, map_bytes, map_name)
        if res["status"] == "success": ok += 1
        else: fail += 1

    log.info(f"         ✅ {ok} posted  |  ⚠️ {fail} failed")
    log.info(f"✅ Done — next scan at {DAILY_RUN_TIME} UTC tomorrow")
    log.info("=" * 55)

    # ── Emergency report trigger ───────────────────────────────────────────
    if scan_summary["active_fire_pixels"] >= EMERGENCY_FIRE_THRESHOLD:
        log.warning(
            f"🚨 EMERGENCY: {scan_summary['active_fire_pixels']} active fire pixels "
            f"≥ threshold ({EMERGENCY_FIRE_THRESHOLD}). Triggering emergency report..."
        )
        generate_report(period_days=1, emergency=True, latest_scan=scan_summary)

    return scan_summary


# ─────────────────────────────────────────────────────────────────────────────
# DATA AGGREGATION  (pulls from Supabase for the report window)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_supabase_table(table: str, from_dt: str, to_dt: str) -> list:
    """
    Simple helper: GET rows from a Supabase table filtered by created_at.
    Returns list of dicts, or [] on failure.
    Requires SUPABASE_URL + SUPABASE_KEY (service_role) in .env
    """
    if not SUPABASE_REST_KEY:
        log.warning(f"  SUPABASE_KEY not set — skipping {table} fetch")
        return []

    url = (
        f"{SUPABASE_URL}/rest/v1/{table}"
        f"?created_at=gte.{from_dt}&created_at=lte.{to_dt}"
        f"&order=created_at.desc&limit=1000"
    )
    headers = {
        "apikey":        SUPABASE_REST_KEY,
        "Authorization": f"Bearer {SUPABASE_REST_KEY}",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
        log.warning(f"  Supabase {table}: {r.status_code} {r.text[:100]}")
    except Exception as e:
        log.warning(f"  Supabase fetch error ({table}): {e}")
    return []


def build_report_context(period_days: int, latest_scan: dict = None) -> dict:
    """
    Aggregate data from all 3 layers over the past `period_days` days.
    `latest_scan` can be passed directly from run_daily_scan() to avoid
    re-querying Supabase for satellite data.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=period_days)
    from_dt = start.isoformat()
    to_dt   = end.isoformat()

    log.info(f"📊 Building report context: {from_dt} → {to_dt}")

    # ── Layer 1: Ground sensor detections ────────────────────────────────
    ground_rows = fetch_supabase_table("aidetection", from_dt, to_dt)
    alert_breakdown = {"SAFE": 0, "MONITOR": 0, "UNCERTAIN": 0,
                       "FIRE_WARNING": 0, "FIRE_CRITICAL": 0}
    critical_events = []
    temps, humidities, winds = [], [], []

    for row in ground_rows:
        status = row.get("status", "UNKNOWN")
        if status in alert_breakdown:
            alert_breakdown[status] += 1
        if status == "FIRE_CRITICAL":
            critical_events.append({
                "timestamp":       row.get("created_at", ""),
                "mac":             row.get("mac_address", ""),
                "temperature":     row.get("temperature"),
                "humidity":        row.get("humidity"),
                "wind":            row.get("wind"),
                "ensemble_score":  row.get("probability"),
                "confidence":      row.get("confidence_level"),
            })
        if row.get("temperature"): temps.append(float(row["temperature"]))
        if row.get("humidity"):    humidities.append(float(row["humidity"]))
        if row.get("wind"):        winds.append(float(row["wind"]))

    # Unique nodes (active vs silent)
    all_macs    = {r.get("mac_address") for r in ground_rows if r.get("mac_address")}
    known_macs  = set(os.getenv("KNOWN_SENSOR_MACS", "").split(",")) if os.getenv("KNOWN_SENSOR_MACS") else all_macs
    silent_macs = known_macs - all_macs

    # ── Layer 2: Drone detections ─────────────────────────────────────────
    drone_rows  = fetch_supabase_table("aidetection", from_dt, to_dt)
    fire_confirmed = sum(1 for r in drone_rows if r.get("fire_detected") is True)
    confidences    = [r["confidence"] for r in drone_rows if r.get("confidence")]

    # ── Layer 3: Satellite ────────────────────────────────────────────────
    if latest_scan:
        satellite_data = latest_scan  # use live scan data directly
        scans_completed = 1
    else:
        sat_rows        = fetch_supabase_table("aiharmed_areas", from_dt, to_dt)
        active_fire     = [r for r in sat_rows if r.get("classification") == "active_fire"]
        old_burn        = [r for r in sat_rows if r.get("classification") == "old_burn"]
        region_map: dict = {}
        for r in active_fire + old_burn:
            rn = r.get("region_name", "Unknown")
            if rn not in region_map:
                region_map[rn] = {"fire_pixels": 0, "max_brightness": 0.0}
            region_map[rn]["fire_pixels"] += 1
            bt = float(r.get("brightness_temp", 0))
            region_map[rn]["max_brightness"] = max(region_map[rn]["max_brightness"], bt)

        satellite_data = {
            "active_fire_pixels": len(active_fire),
            "old_burn_pixels":    len(old_burn),
            "hottest_pixel_k":    max((float(r.get("brightness_temp", 0)) for r in active_fire), default=0),
            "regions_affected":   [{"name": k, **v} for k, v in
                                   sorted(region_map.items(), key=lambda x: x[1]["fire_pixels"], reverse=True)],
        }
        # Rough scan count: 1 per day
        scans_completed = period_days

    # ── Assemble final context ────────────────────────────────────────────
    context = {
        "period": {
            "from":    from_dt,
            "to":      to_dt,
            "days":    period_days,
        },
        "ground_sensors": {
            "total_readings":  len(ground_rows),
            "nodes_active":    len(all_macs),
            "nodes_silent":    list(silent_macs),
            "alert_breakdown": alert_breakdown,
            "critical_events": critical_events[:10],   # cap at 10
            "avg_conditions": {
                "temperature": round(sum(temps) / len(temps), 1)     if temps       else None,
                "humidity":    round(sum(humidities) / len(humidities), 1) if humidities else None,
                "wind":        round(sum(winds) / len(winds), 1)      if winds       else None,
            },
        },
        "drone_detections": {
            "total_missions":  len(drone_rows),
            "fire_confirmed":  fire_confirmed,
            "avg_confidence":  confidences
        },
        "satellite": {
            "scans_completed":    scans_completed,
            "active_fire_pixels": satellite_data.get("active_fire_pixels", 0),
            "old_burn_pixels":    satellite_data.get("old_burn_pixels",    0),
            "hottest_pixel_k":    satellite_data.get("hottest_pixel_k",    0),
            "regions_affected":   satellite_data.get("regions_affected",   []),
        },
        "generated_at": end.isoformat(),
    }

    log.info(f"  Ground readings: {len(ground_rows)} | "
             f"Drone missions: {len(drone_rows)} | "
             f"Satellite active fire: {satellite_data.get('active_fire_pixels', 0)}")
    return context


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTIONS = """
You are FireWatch Intelligence — an expert wildfire analyst AI for the 
Algerian Forest Fire Prevention System. You generate structured, actionable 
surveillance reports for civil protection agencies, forest rangers, and 
regional emergency coordinators.

Your reports must be:
- Written in clear, professional French (primary) with English technical terms kept as-is
- Factual and data-driven — never speculate beyond what the data shows
- Prioritized by severity — lead with the most critical findings
- Actionable — every section ends with concrete recommended actions
- Concise — field responders read these under pressure

You understand the three detection layers:
  Layer 1 (Ground Sensors): Early environmental warning, pre-ignition
  Layer 2 (Drone Vision):   Visual fire confirmation, precise location
  Layer 3 (Satellite/NASA FIRMS): Wide-area burn mapping, post-fire extent

Alert severity scale: SAFE < MONITOR < UNCERTAIN < FIRE_WARNING < FIRE_CRITICAL
Satellite intensity: no_fire < old_burn < active_fire

Cross-layer confirmation: a FIRE_CRITICAL confirmed by drone = high-confidence incident.
When both ground and satellite agree on a region, flag it explicitly.
"""


def build_report_prompt(context: dict, emergency: bool = False) -> str:
    mode_note = (
        "\n⚠️  EMERGENCY REPORT — Un pic d'incendies actifs vient d'être détecté. "
        "Mettez en évidence les mesures immédiates en premier.\n"
        if emergency else ""
    )

    return f"""{mode_note}
Génère un rapport de surveillance des incendies de forêt pour la période suivante.
Voici toutes les données agrégées du système FireWatch :

{json.dumps(context, indent=2, ensure_ascii=False)}

Structure ton rapport EXACTEMENT ainsi (utilise ce formatage Markdown) :

---

# 🔥 RAPPORT DE SURVEILLANCE FIREWATCH
**Période :** {context['period']['from'][:10]} → {context['period']['to'][:10]}  
**Généré le :** {context['generated_at'][:19].replace('T', ' ')} UTC  
**Type :** {"🚨 RAPPORT D'URGENCE" if emergency else f"📋 Rapport périodique ({context['period']['days']} jours)"}  
**Niveau d'alerte global :** [détermine: 🟢 VERT / 🟡 JAUNE / 🟠 ORANGE / 🔴 ROUGE selon les données]

---

## 1. RÉSUMÉ EXÉCUTIF
Résume la situation globale en 3–4 phrases max. Mentionne l'événement le plus critique 
et l'action la plus urgente à prendre maintenant.

## 2. ÉVÉNEMENTS CRITIQUES
Liste chaque événement FIRE_CRITICAL ou FIRE_DETECTED. Pour chaque un :
- Timestamp et région
- Valeurs capteurs clés (temp, humidité, vent)
- Score ensemble (probability) et niveau de confiance
- Confirmation drone si disponible
- Confirmation satellite si disponible
Si aucun événement critique : indique explicitement "Aucun événement critique sur la période".

## 3. ANALYSE PAR COUCHE DE DÉTECTION

### 🌡️ Couche 1 — Capteurs Sol
- Tableau de répartition des alertes (SAFE / MONITOR / UNCERTAIN / FIRE_WARNING / FIRE_CRITICAL)
- Nœuds silencieux ou défaillants (si présents → action requise)
- Conditions environnementales moyennes (température, humidité, vent)
- Heure de pic de risque si identifiable

### 🚁 Couche 2 — Détection Drone
- Nombre de missions / feux visuellement confirmés
- Taux de confirmation (%)
- Confiance moyenne des détections

### 🛰️ Couche 3 — Satellite (NASA FIRMS)
- Pixels feu actif et ancienne brûlure
- Température de brillance max (K)
- Régions les plus touchées (classées par intensité décroissante)

## 4. CARTE DE RISQUE RÉGIONAL
Pour chacune des 7 régions surveillées, attribue un niveau de risque basé sur les données :
| Région | Niveau | Justification |
|--------|--------|---------------|
| Tizi Ouzou | 🟢/🟡/🟠/🔴 | ... |
| Béjaïa | ... | ... |
| Skikda | ... | ... |
| Blida | ... | ... |
| Constantine | ... | ... |
| Mostaganem | ... | ... |
| Souk Ahras | ... | ... |

## 5. RECOMMANDATIONS OPÉRATIONNELLES
Liste numérotée, ordonnée par priorité décroissante :
1. **[Immédiat — dans les 24h]** : ...
2. **[Court terme — dans les 72h]** : ...
3. **[Préventif — prochaine période]** : ...

## 6. ÉTAT DU SYSTÈME
- Nœuds actifs / silencieux
- Dernier scan satellite : {context['generated_at'][:10]}
- Santé globale : [NOMINAL si tout fonctionne / DÉGRADÉ si nœuds silencieux / CRITIQUE si couche entière hors ligne]

---
*FireWatch AI Report — Système de prévention des incendies de forêt, Algérie 🇩🇿*
*Propulsé par Google Gemini + NASA FIRMS + Réseau de capteurs IoT*
"""


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION  (Gemini API call)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(period_days: int = 3,
                    emergency: bool = False,
                    latest_scan: dict = None) -> str | None:
    """
    Aggregate data, call Gemini, save report to file + post to Supabase reports table.
    Returns the report text, or None on failure.
    """
    label = "🚨 EMERGENCY" if emergency else "📋 Periodic"
    log.info(f"{label} report — building context for last {period_days} day(s) ...")

    context = build_report_context(period_days, latest_scan=latest_scan)
    prompt  = build_report_prompt(context, emergency=emergency)

    log.info("  Calling Gemini API ...")
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_INSTRUCTIONS,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,          # low = more factual, less creative
                max_output_tokens=4096,
            ),
        )
        report_text = response.text
    except Exception as e:
        log.error(f"  ❌ Gemini API error: {e}")
        return None

    log.info(f"  ✅ Report generated ({len(report_text)} chars)")

    # ── Save report to local file ─────────────────────────────────────────
    ts          = datetime.utcnow().strftime("%Y%m%d_%H%M")
    report_type = "emergency" if emergency else "periodic"
    filename    = f"firewatch_report_{report_type}_{ts}.md"
    reports_dir = os.getenv("REPORTS_DIR", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"  💾 Saved locally: {filepath}")

    # ── Post report to Supabase (optional — requires a `reports` table) ───
    _post_report_to_supabase(report_text, context, report_type, ts)

    return report_text


def _post_report_to_supabase(report_text: str, context: dict,
                              report_type: str, timestamp: str):
    """
    Insert the generated report into a Supabase `reports` table.
    Table schema (create once in Supabase):
      id uuid default gen_random_uuid(),
      created_at timestamptz default now(),
      report_type text,
      period_days int,
      active_fire_pixels int,
      fire_critical_count int,
      report_markdown text,
      context_json jsonb
    """
    if not SUPABASE_REST_KEY:
        log.warning("  SUPABASE_KEY not set — skipping report DB insert")
        return

    url     = f"{SUPABASE_URL}/rest/v1/reports"
    headers = {
        "apikey":        SUPABASE_REST_KEY,
        "Authorization": f"Bearer {SUPABASE_REST_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }
    payload = {
        "report_type":         report_type,
        "period_days":         context["period"]["days"],
        "active_fire_pixels":  context["satellite"]["active_fire_pixels"],
        "fire_critical_count": context["ground_sensors"]["alert_breakdown"].get("FIRE_CRITICAL", 0),
        "report_markdown":     report_text,
        "context_json":        context,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code in (200, 201):
            log.info("  ✅ Report inserted into Supabase `firewatch_reports`")
        else:
            log.warning(f"  ⚠️  Supabase report insert: {r.status_code} {r.text[:100]}")
    except Exception as e:
        log.warning(f"  Supabase report insert error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("🚀 FireWatch Scheduler starting ...")
    log.info(f"   Satellite scan  : daily at {DAILY_RUN_TIME} UTC")
    log.info(f"   AI report       : every {REPORT_EVERY_N_DAYS} days at {REPORT_RUN_TIME} UTC")
    log.info(f"   Emergency trigger: active_fire pixels ≥ {EMERGENCY_FIRE_THRESHOLD}")
    log.info("")

    # Run immediately on startup
    log.info("▶️  Running first satellite scan now ...")
    run_daily_scan()

    log.info("▶️  Generating initial report now ...")
    generate_report(period_days=REPORT_EVERY_N_DAYS)

    # ── Schedule ──────────────────────────────────────────────────────────
    # Daily satellite scan
    schedule.every().day.at(DAILY_RUN_TIME).do(run_daily_scan)

    # Periodic AI report every N days (default: 3)
    # We use a day counter approach: schedule daily at REPORT_RUN_TIME,
    # but only generate the report every N calls.
    _report_day_counter = [0]

    def _maybe_generate_report():
        _report_day_counter[0] += 1
        if _report_day_counter[0] >= REPORT_EVERY_N_DAYS:
            _report_day_counter[0] = 0
            generate_report(period_days=REPORT_EVERY_N_DAYS)

    schedule.every().day.at(REPORT_RUN_TIME).do(_maybe_generate_report)

    log.info(f"✅ Scheduler running — next scan at {DAILY_RUN_TIME} UTC")
    while True:
        schedule.run_pending()
        time.sleep(60)