"""
config.py — Central configuration for the Fire Detection System.
Edit this file to match your environment.
"""

CONFIG = {
    # ── Flask ──────────────────────────────────────────────────────
    "flask_port":  5000,
    "secret_key":  "change-this-secret-key-in-production",

    # ── MQTT ───────────────────────────────────────────────────────
    "mqtt_broker":   "localhost",       # Change to your broker IP
    "mqtt_port":     1883,
    "mqtt_topic":    "fire/sensors/data",
    "mqtt_username": None,              # Set if your broker requires auth
    "mqtt_password": None,

    # ── Model files ────────────────────────────────────────────────
    "model_dir": "./models",            # Folder with your saved model files
    #   Expected files:
    #     xgb_v2.json         → XGBoost model
    #     lgb_v2.txt          → LightGBM model
    #     tabmlp_v2.pt        → PyTorch TabMLP
    #     scaler_v2.pkl       → StandardScaler
    #     meta_lr_v2.pkl      → Stacking meta LogisticRegression

    # ── Prediction threshold ────────────────────────────────────────
    # Use the best_thresh value that was saved in your tabmlp_v2.pt
    # If None, predictor will load it from the .pt file
    "fire_threshold": None,

    # ── Sensor defaults (dataset medians) ──────────────────────────
    # If a sensor doesn't send a field, these values are used.
    # Values are medians from the forestfires.csv dataset.
    "sensor_defaults": {
        "X":     4,
        "Y":     4,
        "month": "aug",     # Most fire-prone month
        "day":   "fri",
        "FFMC":  91.6,
        "DMC":   108.3,
        "DC":    518.4,
        "ISI":   9.7,
        "temp":  18.9,
        "RH":    44.0,
        "wind":  4.0,
        "rain":  0.0,
    },

    # ── UI ─────────────────────────────────────────────────────────
    "site_name":     "FireWatch AI",
    "location_name": "Algeria",
    "alert_sound":   True,
}