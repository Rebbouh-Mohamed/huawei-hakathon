# 🔥 FireWatch AI — Forest Fire Prevention System
### AgriTech & Environmental Protection · AI, 5G & Digital Power

> **Core focus:** Multi-layer, real-time forest fire **early detection and prevention** in rural Algeria, combining IoT ground sensors, autonomous drones, satellite intelligence, and an AI-generated surveillance report engine — all deployed on **Huawei Cloud** using Docker containers orchestrated by Kubernetes.

---

## 🌍 Problem Statement

Algeria is one of the most wildfire-affected countries in the Mediterranean basin. Every summer, thousands of hectares of forest and farmland are devastated, threatening rural communities, biodiversity, and agricultural output. Traditional monitoring is **too slow** — fires are detected only after they've already spread.

Our solution introduces a **three-layer, AI-driven detection system** that monitors fire risk at every scale:

- **Layer 1 (before/)** — Ground IoT sensors detecting pre-ignition conditions
- **Layer 2 (during/)** — Drones visually confirming fire with computer vision
- **Layer 3 (after/)** — Satellites mapping active burns across the entire country

All three layers feed into a **Gemini-powered intelligence report engine**, deployed and scaled on **Huawei Cloud**.

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

## ☁️ Huawei Cloud Deployment Architecture

> FireWatch AI is fully containerised and deployed on **Huawei Cloud**, leveraging its managed Kubernetes service (CCE), container registry (SWR), object storage (OBS), message queuing (DMS), and AI inference services.

### Cloud Overview Diagram

```
                         ┌──────────────────────────────────────────────┐
                         │              HUAWEI CLOUD                    │
                         │                                              │
  IoT Sensors ──MQTT──►  │  ┌──────────┐    ┌────────────────────────┐ │
  Drone Cameras ────────► │  │  DMS     │    │   CCE Kubernetes       │ │
  NASA FIRMS ────────────►│  │ (Kafka)  │──► │   Cluster              │ │
                         │  └──────────┘    │                        │ │
                         │                  │  ┌──────────────────┐  │ │
                         │                  │  │  layer1-service   │  │ │
                         │                  │  │  (Pod × 3)        │  │ │
                         │                  │  ├──────────────────┤  │ │
                         │                  │  │  layer2-service   │  │ │
                         │                  │  │  (Pod × 2)        │  │ │
                         │                  │  ├──────────────────┤  │ │
                         │                  │  │  layer3-service   │  │ │
                         │                  │  │  (Pod × 2)        │  │ │
                         │                  │  ├──────────────────┤  │ │
                         │                  │  │  scheduler-svc    │  │ │
                         │                  │  │  (Pod × 1)        │  │ │
                         │                  │  └──────────────────┘  │ │
                         │                  └─────────┬──────────────┘ │
                         │                            │                 │
                         │  ┌─────────┐   ┌──────────▼──────────────┐  │
                         │  │  SWR    │   │  ELB (Load Balancer)    │  │
                         │  │(Docker  │   │  Routes external traffic │  │
                         │  │Registry)│   └──────────────────────────┘  │
                         │  └─────────┘                                  │
                         │                                              │
                         │  ┌──────────────────────────────────────────┐│
                         │  │  OBS (Object Storage)                    ││
                         │  │  - Annotated drone images                ││
                         │  │  - Satellite fire maps (PNG)             ││
                         │  │  - AI-generated reports (Markdown)       ││
                         │  └──────────────────────────────────────────┘│
                         └──────────────────────────────────────────────┘
```

---

### Huawei Cloud Services Used

| Service | Huawei Cloud Product | Role in FireWatch |
|---|---|---|
| **Kubernetes** | Cloud Container Engine (CCE) | Orchestrates all microservice pods |
| **Container Registry** | SoftWare Repository (SWR) | Stores and distributes Docker images |
| **Object Storage** | OBS (Object Storage Service) | Drone annotated images, satellite maps, AI reports |
| **Message Queue** | DMS for Kafka | MQTT-to-cloud bridge for IoT sensor data |
| **Load Balancer** | ELB (Elastic Load Balancer) | Routes traffic to Layer 2 & 3 Flask APIs |
| **Auto Scaling** | CCE Auto Scaler (HPA) | Scales pods on high fire event load |
| **Monitoring** | AOM (Application Operations Management) | Logs, metrics, alerts for all pods |
| **Secret Management** | DEW (Data Encryption Workshop) | Stores API keys (Gemini, NASA, Supabase) |
| **Virtual Network** | VPC + Security Groups | Isolates services, controls ingress/egress |

---

### Docker — Containerisation

Each layer is packaged as a **separate Docker image** for independent deployment and scaling.

#### Layer 1 — Ground Sensor Server (`before/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    flask paho-mqtt xgboost lightgbm torch torchvision \
    joblib python-dotenv requests supabase

COPY . .

EXPOSE 5000

# App listens on MQTT and exposes health endpoint
CMD ["python", "app.py"]
```

#### Layer 2 — Drone Detection API (`during/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# OpenCV requires libGL
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    flask torch torchvision opencv-python-headless \
    python-dotenv supabase

# Copy YOLOv5 weights
COPY best.pt .
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

#### Layer 3 — Satellite Scanner + AI Reports (`after/satalite/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Matplotlib requires a backend
ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir \
    flask supabase matplotlib pandas numpy \
    pillow requests schedule google-generativeai \
    python-dotenv

COPY . .

EXPOSE 5000

# Runs scheduler (satellite scan + AI report generation)
CMD ["python", "scheduler.py"]
```

#### Build & Push to Huawei SWR

```bash
# Configure Docker to use Huawei SWR
docker login -u <region>@<AK> -p <login-key> swr.<region>.myhuaweicloud.com

# Build all images
docker build -t swr.<region>.myhuaweicloud.com/firewatch/layer1-sensor:latest ./before/
docker build -t swr.<region>.myhuaweicloud.com/firewatch/layer2-drone:latest  ./during/
docker build -t swr.<region>.myhuaweicloud.com/firewatch/layer3-satellite:latest ./after/satalite/

# Push to SWR registry
docker push swr.<region>.myhuaweicloud.com/firewatch/layer1-sensor:latest
docker push swr.<region>.myhuaweicloud.com/firewatch/layer2-drone:latest
docker push swr.<region>.myhuaweicloud.com/firewatch/layer3-satellite:latest
```

---

### Kubernetes (CCE) — Orchestration

All services run inside a **CCE (Cloud Container Engine)** cluster on Huawei Cloud.

#### Cluster Topology

```
CCE Cluster: firewatch-cluster
  │
  ├── Namespace: firewatch-prod
  │     ├── Deployment: layer1-sensor      (replicas: 3)
  │     ├── Deployment: layer2-drone       (replicas: 2)
  │     ├── Deployment: layer3-satellite   (replicas: 2)
  │     ├── Deployment: scheduler          (replicas: 1)
  │     │
  │     ├── Service: layer2-svc            (ClusterIP → ELB)
  │     ├── Service: layer3-svc            (ClusterIP → ELB)
  │     │
  │     ├── ConfigMap: firewatch-config    (non-secret env vars)
  │     └── Secret: firewatch-secrets      (API keys via DEW)
  │
  └── Namespace: monitoring
        └── AOM agent DaemonSet
```

#### Kubernetes Manifests

##### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: firewatch-prod
```

##### Secret (API Keys via Huawei DEW)

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: firewatch-secrets
  namespace: firewatch-prod
type: Opaque
stringData:
  GEMINI_API_KEY: "<your-gemini-key>"
  NASA_FIRMS_KEY: "<your-nasa-key>"
  SUPABASE_URL: "<your-supabase-url>"
  SUPABASE_KEY: "<your-supabase-service-key>"
```

> 💡 In production, use **Huawei DEW (Data Encryption Workshop)** with the CSI Secrets Store driver to inject secrets as mounted files, instead of plain `stringData`.

##### Layer 1 — Ground Sensor Deployment

```yaml
# k8s/layer1-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: layer1-sensor
  namespace: firewatch-prod
  labels:
    app: layer1-sensor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: layer1-sensor
  template:
    metadata:
      labels:
        app: layer1-sensor
    spec:
      containers:
        - name: layer1-sensor
          image: swr.<region>.myhuaweicloud.com/firewatch/layer1-sensor:latest
          ports:
            - containerPort: 5000
          envFrom:
            - secretRef:
                name: firewatch-secrets
          env:
            - name: MQTT_BROKER
              value: "dms-kafka-endpoint.myhuaweicloud.com"
            - name: MQTT_PORT
              value: "9093"
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 30
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 10
```

##### Layer 2 — Drone Detection Deployment + Service

```yaml
# k8s/layer2-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: layer2-drone
  namespace: firewatch-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: layer2-drone
  template:
    metadata:
      labels:
        app: layer2-drone
    spec:
      containers:
        - name: layer2-drone
          image: swr.<region>.myhuaweicloud.com/firewatch/layer2-drone:latest
          ports:
            - containerPort: 5000
          envFrom:
            - secretRef:
                name: firewatch-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"   # YOLOv5 inference is CPU-intensive
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 60   # YOLOv5 model load time
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: layer2-svc
  namespace: firewatch-prod
spec:
  selector:
    app: layer2-drone
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: ClusterIP
```

##### Layer 3 — Satellite + Scheduler Deployment + Service

```yaml
# k8s/layer3-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: layer3-satellite
  namespace: firewatch-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: layer3-satellite
  template:
    metadata:
      labels:
        app: layer3-satellite
    spec:
      containers:
        - name: layer3-satellite
          image: swr.<region>.myhuaweicloud.com/firewatch/layer3-satellite:latest
          ports:
            - containerPort: 5000
          envFrom:
            - secretRef:
                name: firewatch-secrets
          env:
            - name: DAILY_RUN_TIME
              value: "06:00"
            - name: REPORT_RUN_TIME
              value: "07:00"
            - name: REPORT_EVERY_N_DAYS
              value: "3"
            - name: EMERGENCY_FIRE_THRESHOLD
              value: "15"
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: layer3-svc
  namespace: firewatch-prod
spec:
  selector:
    app: layer3-satellite
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: ClusterIP
```

##### Ingress — Huawei ELB (Elastic Load Balancer)

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: firewatch-ingress
  namespace: firewatch-prod
  annotations:
    kubernetes.io/ingress.class: "cce"
    kubernetes.io/elb.class: "union"           # Huawei shared ELB
    kubernetes.io/elb.autocreate: '{"type":"public","bandwidth_name":"firewatch-bw","bandwidth_size":5}'
spec:
  rules:
    - host: firewatch.myapp.com
      http:
        paths:
          - path: /detect
            pathType: Prefix
            backend:
              service:
                name: layer2-svc
                port:
                  number: 80
          - path: /map
            pathType: Prefix
            backend:
              service:
                name: layer3-svc
                port:
                  number: 80
          - path: /latest
            pathType: Prefix
            backend:
              service:
                name: layer3-svc
                port:
                  number: 80
```

##### Horizontal Pod Autoscaler (HPA)

```yaml
# k8s/hpa.yaml
# Scale Layer 2 when drone traffic spikes (wildfire incident)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: layer2-hpa
  namespace: firewatch-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: layer2-drone
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# Scale Layer 1 when many sensor nodes are active
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: layer1-hpa
  namespace: firewatch-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: layer1-sensor
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

---

### Deploy All to CCE

```bash
# Connect kubectl to your CCE cluster
# (Download kubeconfig from Huawei Cloud Console → CCE → Cluster → kubectl)

export KUBECONFIG=~/.kube/huawei-firewatch.yaml

# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/layer1-deployment.yaml
kubectl apply -f k8s/layer2-deployment.yaml
kubectl apply -f k8s/layer3-deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify all pods are Running
kubectl get pods -n firewatch-prod

# Check services
kubectl get svc -n firewatch-prod

# Check ingress + ELB IP
kubectl get ingress -n firewatch-prod
```

Expected output:
```
NAME                          READY   STATUS    RESTARTS   AGE
layer1-sensor-7d4b9-xxxx      1/1     Running   0          2m
layer1-sensor-7d4b9-yyyy      1/1     Running   0          2m
layer1-sensor-7d4b9-zzzz      1/1     Running   0          2m
layer2-drone-5f8c4-xxxx       1/1     Running   0          2m
layer2-drone-5f8c4-yyyy       1/1     Running   0          2m
layer3-satellite-6a2d1-xxxx   1/1     Running   0          2m
layer3-satellite-6a2d1-yyyy   1/1     Running   0          2m
```

---

### OBS — Object Storage for Fire Artifacts

Drone annotated images, satellite PNG maps, and AI-generated Markdown reports are stored in **Huawei OBS**:

```
OBS Bucket: firewatch-artifacts/
  ├── drone-images/
  │     └── annotated/<date>/<mac>_annotated.jpg
  ├── satellite-maps/
  │     └── maps/algeria_map_<date>.png
  └── reports/
        ├── firewatch_report_periodic_<timestamp>.md
        └── firewatch_report_emergency_<timestamp>.md
```

Configure in each service via environment variable:
```bash
OBS_ENDPOINT=https://obs.<region>.myhuaweicloud.com
OBS_BUCKET=firewatch-artifacts
OBS_AK=<your-access-key>
OBS_SK=<your-secret-key>
```

---

### DMS (Kafka) — IoT Message Bridge

Instead of connecting IoT sensors directly to a single MQTT broker, **Huawei DMS for Kafka** acts as a **scalable, fault-tolerant message bus**:

```
IoT Sensor (MQTT) ──► MQTT Bridge Pod ──► DMS Kafka Topic: fire/sensors
                                                    │
                        ┌───────────────────────────┤
                        │  Consumer: layer1-sensor  │
                        │  (all 3 replicas consume  │
                        │   partitioned by MAC addr)│
                        └───────────────────────────┘
```

**Kafka Topics:**

| Topic | Producers | Consumers | Purpose |
|---|---|---|---|
| `fire.sensors` | MQTT bridge pod | Layer 1 pods | Ground sensor readings |
| `fire.detections` | Layer 1, Layer 2 | Scheduler | Detected fire events |
| `fire.reports` | Scheduler | Dashboard | AI report notifications |

---

### AOM — Monitoring & Alerting

**Huawei AOM (Application Operations Management)** collects logs and metrics from all pods:

- **Metrics:** CPU/memory per pod, HTTP request rate, inference latency
- **Logs:** Structured JSON logs from all layers → searchable in AOM
- **Alerts:** Trigger SMS/email when:
  - Any pod CrashLoopBackOff
  - Layer 1 CPU > 85% (model inference overload)
  - Zero detections for > 24 h (sensor network failure)
  - Emergency AI report generated

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
  ├── Annotated image → OBS Storage bucket
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
| `supabase_client.py` | Posts detections to Supabase DB + OBS storage bucket |
| `simulate.py` | Test harness — sends sample images to the API |

---

## 🛰️ Layer 3 — Satellite Intelligence + AI Reports (`after/satalite/`)

> **Goal:** Wide-area daily surveillance of all active wildfires across Algeria via NASA satellite, with an AI-generated intelligence report engine powered by Google Gemini.

### 3a — Satellite Scanner & Map (`app.py`)

Every day at **06:00 UTC**, the scheduler fetches fire detections from **NASA FIRMS** (VIIRS SNPP 375m sensor). It then stitches an **OpenStreetMap basemap** and overlays fire hotspots:

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
      │       Output: PNG  algeria_map_<date>.png → OBS
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

### 3b — Gemini AI Report Engine (`scheduler.py`) ⭐

The scheduler **automatically generates structured wildfire surveillance reports in French** using Google Gemini.

#### Report Triggers

| Trigger | When | Period covered |
|---|---|---|
| **Periodic** | Every 3 days at 07:00 UTC (configurable) | Last N days |
| **Emergency 🚨** | Immediately when `active_fire_pixels ≥ 15` in any single scan | Last 24 h |

#### Multi-Layer Data Aggregation (`build_report_context`)

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
      ├──►  Saved to OBS: reports/firewatch_report_<type>_<timestamp>.md
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
  → posted to Supabase + saved to OBS
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
| **IoT / Messaging** | MQTT (paho-mqtt), QoS-1, Huawei DMS Kafka |
| **Database** | Supabase (PostgreSQL + Edge Functions) |
| **Object Storage** | Huawei OBS (drone images, maps, reports) |
| **Satellite Data** | NASA FIRMS, VIIRS SNPP 375 m |
| **Mapping** | OpenStreetMap tiles, Matplotlib |
| **Computer Vision** | OpenCV, YOLOv5 (PyTorch Hub) |
| **Scheduling** | Python `schedule` library |
| **Containerisation** | Docker |
| **Orchestration** | Kubernetes (Huawei CCE) |
| **Container Registry** | Huawei SWR |
| **Monitoring** | Huawei AOM |
| **Load Balancing** | Huawei ELB |
| **Security / Secrets** | Huawei DEW |

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

## 🚀 Running the System Locally

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

---

## 🐳 Running with Docker Compose (Local Dev)

```yaml
# docker-compose.yml
version: "3.9"

services:
  layer1-sensor:
    build: ./before
    ports:
      - "5001:5000"
    env_file: ./before/.env
    restart: unless-stopped

  layer2-drone:
    build: ./during
    ports:
      - "5002:5000"
    env_file: ./during/.env
    volumes:
      - ./during/best.pt:/app/best.pt
    restart: unless-stopped

  layer3-satellite:
    build: ./after/satalite
    ports:
      - "5003:5000"
    env_file: ./after/satalite/.env
    restart: unless-stopped
```

```bash
# Start all services locally
docker-compose up --build

# Access:
# Layer 1 health: http://localhost:5001/health
# Layer 2 detect: http://localhost:5002/detect
# Layer 3 map:    http://localhost:5003/map
```

---

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
| **Scalability** | CCE auto-scales pods on wildfire events; each IoT node is independent |
| **Cost** | Open satellite data (NASA FIRMS) + commodity IoT hardware + Huawei Cloud pay-as-you-go |
| **Resilience** | Three independent layers — even if one fails, others continue |
| **Cloud-Native** | Docker + Kubernetes on Huawei CCE enables zero-downtime deployments |

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

