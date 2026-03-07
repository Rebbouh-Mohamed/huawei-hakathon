# 🔥 FireWatch AI — Forest Fire Prevention System
### AgriTech & Environmental Protection · AI, 5G & Digital Power

> **Core focus:** Multi-layer, real-time forest fire **early detection and prevention** in rural Algeria, combining IoT ground sensors, autonomous drones, satellite intelligence, and an AI-generated surveillance report engine.

---

## 🌍 Problem Statement

Algeria is one of the most wildfire-affected countries in the Mediterranean basin. Every summer, thousands of hectares of forest and farmland are devastated, threatening rural communities, biodiversity, and agricultural output. Traditional monitoring is **too slow** — fires are detected only after they've already spread.

Our solution introduces a **three-layer, AI-driven detection system** that monitors fire risk at every scale: ground sensors detecting pre-ignition conditions, drones visually confirming fire, and satellites mapping active burns across the entire country — all feeding into a **Gemini-powered intelligence report engine**.

---

## 🏗️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DETECTION LAYERS                                   │
│                                                                             │
│  🌡️  LAYER 1  (before/)        Ground Sensor Network                       │
│      IoT Sensors  →  MQTT  →  Ensemble AI (XGB+LGB+TabMLP+LR)  →  Supabase│
│                                                                             │
│  🚁  LAYER 2  (during/)        Drone Visual Detection                       │
│      Drone Camera  →  Flask API  →  YOLOv5  →  Supabase                    │
│                                                                             │
│  🛰️  LAYER 3  (after/satalite/)  Satellite + AI Intelligence               │
│      NASA FIRMS  →  Daily Scan  →  OSM Fire Map  →  Supabase               │
│                                        │                                    │
│                          ┌─────────────▼──────────────────┐                │
│                          │   Gemini AI Report Engine       │                │
│                          │   (periodic every 3 days        │                │
│                          │    OR emergency on spike)       │                │
│                          └─────────────┬──────────────────┘                │
└────────────────────────────────────────┼────────────────────────────────────┘
                                         │
                               ┌─────────▼─────────┐
                               │    Supabase DB     │
                               │  fire_detections   │
                               │  aiharmed_areas    │
                               │  reports           │
                               │  + Edge Functions  │
                               └─────────┬─────────┘
                                         │
                               ┌─────────▼─────────┐
                               │  Dashboard / App / │
                               │  Alert System      │
                               └───────────────────┘
```

---

## 🌡️ Layer 1 — Ground Sensor Network (`before/`)

> **Goal:** Detect fire-favourable environmental conditions *before* ignition using ambient sensor data.

### IoT Sensor Readings

Nodes deployed in forests continuously monitor:

| Sensor Field | Fire Relevance |
|---|---|
| `temperature` | High temp → increased ignition risk |
| `humidity` (RH) | Low humidity → dry vegetation |
| `gas_resistance` (FFMC proxy) | Fine fuel moisture content |
| `wind` | Spreads fire rapidly |
| `rain` | Reduces risk |
| `DC / DMC / ISI` | Canadian Forest Fire Weather Index |

Data arrives over **MQTT** (`fire/sensors/<MAC_ADDRESS>`, QoS-1) and is processed in real time.

### AI Ensemble Model (`predictor.py`)

A **stacking ensemble** of four models produces a fire probability score:

```
Raw Sensor Reading
      │
      ▼
Feature Engineering  →  35 derived features
  ├── Cyclic encoding (month/day sin/cos)
  ├── FWI interactions (temp×wind, DC/DMC, FFMC², ISI×DC/DMC…)
  ├── Boolean thresholds (RH<40 & temp>20, FFMC>90, wind>Q75…)
  └── Spatial features (√(X²+Y²), X×Y)
      │
      ├──► XGBoost  ─────────────────────────────────┐
      ├──► LightGBM ─────────────────────────────────┤
      └──► TabMLP (PyTorch · Residual blocks) ────────┤
                                                     ▼
                               Stacking Meta-Learner (Logistic Regression)
                                                     │
                                         Ensemble Probability Score
                                                     │
                          ┌──────────────────────────▼──────────────────────┐
                          │ SAFE · MONITOR · UNCERTAIN · FIRE_WARNING · FIRE_CRITICAL │
                          └─────────────────────────────────────────────────┘
```

**Confidence levels:** `HIGH` (prob > 0.8 or < 0.2) · `MEDIUM` (0.65–0.8) · `LOW` (otherwise)

### Data Flow

```
Sensor Node  ──(MQTT QoS-1)──►  FireWatch Server  ──►  Supabase Edge Function  ──►  Alert
                                 validate MAC                ai-fire-insert
                                 run ensemble AI             fire_detections table
                                 classify status
```

### Key Files

| File | Purpose |
|---|---|
| `app.py` | MQTT listener, message routing, Supabase push |
| `predictor.py` | Ensemble model loading and full inference pipeline |
| `config.py` | MQTT broker settings, sensor defaults, model path |
| `simulator.py` | Test harness — sends synthetic sensor readings |

---

## 🚁 Layer 2 — Drone Visual Detection (`during/`)

> **Goal:** Visually confirm fire presence using drone-mounted cameras and YOLOv5 computer vision.

### Detection Pipeline

```
Drone Camera Frame
      │  POST /detect (multipart: image + mac_address)
      ▼
Flask API (app.py)
      │
      ▼
YOLOv5 Inference (detector.py)
  ├── Confidence threshold: 40%
  ├── Bounding box extraction (x, y, width, height)
  └── Annotated image saved  annotated/<stem>_annotated.jpg
      │
      ▼  (fire detected)
Best Detection (highest confidence score)
      │
      ▼
Supabase Edge Function
  ├── Annotated image → Storage bucket
  └── Record: mac_address, x, y, confidence, image_url, status=FIRE_DETECTED
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect` | Submit drone image for fire detection |
| `GET` | `/health` | Verify service is running |

### Key Files

| File | Purpose |
|---|---|
| `app.py` | Flask API exposing `/detect` |
| `detector.py` | YOLOv5 model loader + inference + bounding box annotator |
| `supabase_client.py` | Posts detections to Supabase DB + storage bucket |
| `simulate.py` | Test harness — sends sample images to the API |

---

## 🛰️ Layer 3 — Satellite Intelligence + AI Reports (`after/satalite/`)

> **Goal:** Wide-area daily surveillance of all active wildfires across Algeria via NASA satellite, with an AI-generated intelligence report engine powered by Google Gemini.

This layer has two components:

### 3a — Satellite Scanner & Map (`app.py`)

Every day at **06:00 UTC**, the scheduler fetches fire detections from **NASA FIRMS** (VIIRS SNPP 375m sensor, 375m resolution). It then stitches an **OpenStreetMap basemap** and overlays fire hotspots:

```
NASA FIRMS API  (VIIRS_SNPP_NRT, Algeria BBOX)
      │
      ▼
Parse CSV detections → classify by brightness temperature
  bright_ti4 > 340 K  →  active_fire
  bright_ti4 > 315 K  →  old_burn
  else                →  no_fire
      │
      ├──►  Render Algeria Fire Map
      │       Download OSM tiles (zoom-6) + stitch basemap
      │       Overlay fire scatter (size & colour by brightness)
      │       Annotate city callouts + stats box
      │       Output: PNG  algeria_map_<date>.png
      │
      └──►  POST each record + map image
              Supabase Edge Function: ai-harmed-areas
              Table: aiharmed_areas
```

### API Endpoints (`app.py`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + endpoint list |
| `GET` | `/map` | Live Algeria fire map (PNG) |
| `GET` | `/latest` | Recent fire records from Supabase |
| `POST` | `/upload` | Manual fire image + data upload |

---

### 3b — Gemini AI Report Engine (`scheduler.py`) ⭐ New

This is the most significant new feature. The scheduler now also **automatically generates structured wildfire surveillance reports in French** using Google Gemini, acting as a field intelligence system for civil protection agencies.

#### Report Triggers

| Trigger | When | Period covered |
|---|---|---|
| **Periodic** | Every 3 days at 07:00 UTC (configurable) | Last N days |
| **Emergency 🚨** | Immediately when `active_fire_pixels ≥ 15` in any single scan | Last 24 h |

#### Multi-Layer Data Aggregation (`build_report_context`)

Before generating a report, the system pulls data from **all three detection layers** over the reporting window:

```
Report Window (e.g. last 3 days)
      │
      ├──► Layer 1 — Ground Sensors  (table: aidetection)
      │       Total readings · nodes active/silent
      │       Alert breakdown: SAFE/MONITOR/UNCERTAIN/FIRE_WARNING/FIRE_CRITICAL
      │       Critical events (temp, humidity, wind, ensemble score)
      │       Avg environmental conditions
      │
      ├──► Layer 2 — Drone Missions  (table: aidetection)
      │       Total missions · fire confirmed count
      │       Avg detection confidence
      │
      └──► Layer 3 — Satellite       (live scan or table: aiharmed_areas)
              Active fire pixels · old burn pixels
              Hottest pixel (brightness K) · max FRP (MW)
              Regions affected ranked by fire pixel count
```

#### Gemini Report Generation

```
Aggregated context JSON
      │
      ▼
Gemini API  (gemini-flash · temp=0.3 · max 4096 tokens)
  System: FireWatch Intelligence expert wildfire analyst
  Language: French (professional) + English technical terms
      │
      ▼
Structured Report (Markdown)
  1. Résumé Exécutif          ← 3-4 sentence global status
  2. Événements Critiques      ← per-event: timestamp, sensors, ensemble score, cross-layer confirm
  3. Analyse par Couche        ← Layer 1/2/3 breakdown tables
  4. Carte de Risque Régional  ← 7 regions: 🟢🟡🟠🔴 risk levels
  5. Recommandations           ← prioritised action list (24h / 72h / preventive)
  6. État du Système           ← node health, last scan, overall status
      │
      ├──►  Saved locally: reports/firewatch_report_<type>_<timestamp>.md
      └──►  Inserted into Supabase `reports` table (markdown + context_json)
```

#### Emergency Report Flow

```
run_daily_scan()
      │
      ▼
active_fire_pixels ≥ EMERGENCY_FIRE_THRESHOLD (default: 15)?
      │ YES
      ▼
generate_report(period_days=1, emergency=True)
      │
      ▼
Gemini generates 🚨 RAPPORT D'URGENCE
  → leads with immediate actions first
  → posted to Supabase + saved locally
```

### Full Scheduler Timeline

```
Startup
  │  run_daily_scan()   immediately
  │  generate_report()  immediately
  │
  ├── Every day at 06:00 UTC ──► run_daily_scan()
  │                                   └─ if fire spike ──► emergency report
  │
  └── Every day at 07:00 UTC ──► _maybe_generate_report()
                                      (fires every N days, default 3)
```

### Key Files

| File | Purpose |
|---|---|
| `app.py` | Flask API + NASA FIRMS fetcher + OSM tile map renderer |
| `scheduler.py` | Daily scan · Gemini report engine · emergency trigger · Supabase sync |

---

## 🧩 Technology Stack

| Category | Technologies |
|---|---|
| **AI / ML** | XGBoost, LightGBM, PyTorch (TabMLP residual), YOLOv5 |
| **Generative AI** | Google Gemini API (`gemini-flash`) |
| **Backend** | Python, Flask |
| **IoT / Messaging** | MQTT (paho-mqtt), QoS-1 |
| **Database** | Supabase (PostgreSQL + Storage + Edge Functions) |
| **Satellite Data** | NASA FIRMS, VIIRS SNPP 375 m |
| **Mapping** | OpenStreetMap tiles, Matplotlib |
| **Computer Vision** | OpenCV, YOLOv5 (PyTorch Hub) |
| **Scheduling** | Python `schedule` library |

---

## 🔔 Alert Status Taxonomy

| Status | Layer | Meaning |
|---|---|---|
| `SAFE` | Ground | No risk, high confidence |
| `MONITOR` | Ground | Borderline conditions |
| `UNCERTAIN` | Ground | Low-confidence reading |
| `FIRE_WARNING` | Ground | Fire probable, medium confidence |
| `FIRE_CRITICAL` | Ground | Fire confirmed, high confidence |
| `FIRE_DETECTED` | Drone | Visual fire confirmed |
| `active_fire` | Satellite | Brightness > 340 K |
| `old_burn` | Satellite | Residual heat 315–340 K |

**Cross-layer confirmation:** A `FIRE_CRITICAL` from a ground sensor that is also confirmed by drone **and** visible on the satellite map is flagged as a high-confidence incident in the AI report.

---

## 🚀 Running the System

### Layer 1 — Ground Sensor Server
```bash
cd before/
pip install flask paho-mqtt xgboost lightgbm torch joblib python-dotenv requests
cp .env.example .env   # SUPABASE_URL, SUPABASE_KEY, MQTT_BROKER
python app.py
# Health: GET http://localhost:5000/health
```

### Layer 2 — Drone Detection API
```bash
cd during/
pip install flask torch opencv-python python-dotenv supabase
# Place fire-trained weights as best.pt  (yolov5s.pt is the auto-fallback)
python app.py
# POST http://localhost:5000/detect  (multipart: image + mac_address)
```

### Layer 3 — Satellite Scanner + AI Reports
```bash
cd after/satalite/
pip install flask supabase matplotlib pandas numpy pillow requests schedule \
            google-generativeai python-dotenv
cp .env.example .env   # NASA_FIRMS_KEY, SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY
python scheduler.py
# Satellite scan: daily 06:00 UTC
# AI report:      every 3 days 07:00 UTC  (or instantly on fire spike)
# Map preview:    GET http://localhost:5000/map
```

### Environment Variables (Layer 3)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `NASA_FIRMS_KEY` | — | NASA FIRMS API key |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role key |
| `DAILY_RUN_TIME` | `06:00` | Satellite scan time (UTC) |
| `REPORT_RUN_TIME` | `07:00` | Report check time (UTC) |
| `REPORT_EVERY_N_DAYS` | `3` | Report generation frequency |
| `EMERGENCY_FIRE_THRESHOLD` | `15` | Active fire pixels to trigger emergency |
| `MAC_ADDRESS` | `A4:CF:12:98:AB:44` | Satellite scanner node identifier |

---

## 📊 Impact & Value

| Dimension | Impact |
|---|---|
| **Speed** | Ground sensors react in seconds vs. hours for traditional patrols |
| **Coverage** | Satellite covers all of Algeria daily at 375 m resolution |
| **Intelligence** | Gemini auto-reports fuse all 3 layers into actionable French briefings |
| **Accuracy** | Ensemble AI + cross-layer confirmation reduces false positives |
| **Scalability** | Each IoT node is independent — thousands can be deployed |
| **Cost** | Open satellite data (NASA FIRMS) + commodity IoT hardware |
| **Resilience** | Three independent layers — even if one fails, others continue |

---

## 🗺️ Monitored High-Risk Regions

| Region | Coordinates | Notes |
|---|---|---|
| Tizi Ouzou | 36.75°N, 4.05°E | Historically highest annual burn area |
| Béjaïa | 36.75°N, 5.06°E | Coastal Kabyle forests |
| Skikda | 36.89°N, 6.90°E | Dense pine forest zones |
| Blida | 36.46°N, 2.83°E | Atlas cedar forests |
| Constantine | 36.30°N, 6.61°E | Eastern highlands |
| Mostaganem | 35.69°N, 0.63°E | Western agricultural belt |
| Souk Ahras | 36.28°N, 7.95°E | Border forest zone |

---

## 🗺️ Future Roadmap

- [ ] **5G Integration** — Ultra-low-latency MQTT over 5G for remote nodes
- [ ] **Drone Swarm Dispatch** — Auto-scramble drones on FIRE_CRITICAL alert
- [ ] **FWI Trend Forecasting** — 7-day fire risk prediction from historical sensor data
- [ ] **SMS / Push Alerts** — Real-time notifications to local fire brigades
- [ ] **Mobile Dashboard** — Field responder app with live fire map
- [ ] **Smoke Segmentation** — Upgrade drone vision to per-pixel segmentation
- [ ] **Multi-language Reports** — Arabic version of the Gemini AI reports

---

*Built for the **AgriTech & Environmental Protection — AI, 5G & Digital Power** challenge.*  
*Focused on forest fire prevention and climate resilience in rural Algeria. 🇩🇿*  
*Powered by: Google Gemini · NASA FIRMS · YOLOv5 · IoT Sensor Network*
