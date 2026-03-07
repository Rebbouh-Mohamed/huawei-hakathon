"""
predictor.py — Loads the trained ensemble and runs the full inference pipeline.

Pipeline (mirrors the training notebook exactly):
  raw sensor dict
    → apply defaults
    → feature engineering  (same as training)
    → StandardScaler       (same scaler fitted during training)
    → XGBoost + LightGBM   (on unscaled features, as in training)
    → TabMLP               (on scaled features)
    → stacking meta-LR     (LogisticRegression on 3 model probas)
    → threshold decision   → {"is_fire": bool, "prob": float, ...}
"""

import os
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FirePredictor:
    # Dataset medians — used for cyclic encoding reference
    MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    DAY_MAP = {
        "mon": 1, "tue": 2, "wed": 3,
        "thu": 4, "fri": 5, "sat": 6, "sun": 7,
    }
    # Training dataset quantiles (precomputed — avoids needing full dataset at inference)
    TEMP_Q75  = 22.8
    RH_Q25    = 33.0
    WIND_Q75  = 5.4

    def __init__(self, model_dir: str):
        self.ready = False
        self._load_models(model_dir)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self, model_dir: str):
        import joblib
        import torch
        import xgboost as xgb
        import lightgbm as lgb

        try:
            # XGBoost
            xgb_path = os.path.join(model_dir, "xgb_v2.json")
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
            log.info(f"XGBoost loaded from {xgb_path}")

            # LightGBM
            lgb_path = os.path.join(model_dir, "lgb_v2.txt")
            self.lgb_model = lgb.Booster(model_file=lgb_path)
            log.info(f"LightGBM loaded from {lgb_path}")

            # Scaler
            scaler_path = os.path.join(model_dir, "scaler_v2.pkl")
            self.scaler = joblib.load(scaler_path)
            log.info(f"Scaler loaded from {scaler_path}")

            # Meta LR
            meta_path = os.path.join(model_dir, "meta_lr_v2.pkl")
            self.meta_lr = joblib.load(meta_path)
            log.info(f"Meta LR loaded from {meta_path}")

            # TabMLP (PyTorch)
            tab_path = os.path.join(model_dir, "tabmlp_v2.pt")
            checkpoint = torch.load(tab_path, map_location="cpu")
            self.threshold = checkpoint.get("best_threshold", 0.5)
            self._build_tabmlp(checkpoint)
            log.info(f"TabMLP loaded from {tab_path} | threshold={self.threshold:.4f}")

            self.ready = True
            log.info("All models loaded ✅")

        except FileNotFoundError as e:
            log.error(f"Model file not found: {e}")
            log.warning("Running in DEMO mode — returning random predictions")
            self._demo_mode = True
            self.threshold = 0.5
        except Exception as e:
            log.exception(f"Failed to load models: {e}")
            raise

    def _build_tabmlp(self, checkpoint):
        import torch
        import torch.nn as nn

        input_dim = checkpoint["input_dim"]
        hidden    = checkpoint["hidden"]
        n_blocks  = checkpoint["n_blocks"]
        dropout   = checkpoint["dropout"]

        class ResidualBlock(nn.Module):
            def __init__(self, dim, dp):
                super().__init__()
                self.block = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(dim * 2, dim), nn.Dropout(dp),
                )
            def forward(self, x): return x + self.block(x)

        class TabMLP(nn.Module):
            def __init__(self, inp, hid, nb, dp):
                super().__init__()
                self.embed = nn.Sequential(
                    nn.Linear(inp, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(dp))
                self.blocks = nn.Sequential(*[ResidualBlock(hid, dp) for _ in range(nb)])
                self.head = nn.Sequential(
                    nn.LayerNorm(hid), nn.Linear(hid, 64), nn.GELU(),
                    nn.Dropout(0.2), nn.Linear(64, 1))
            def forward(self, x):
                return self.head(self.blocks(self.embed(x))).squeeze(-1)

        self.tab_model = TabMLP(input_dim, hidden, n_blocks, dropout)
        self.tab_model.load_state_dict(checkpoint["model_state_dict"])
        self.tab_model.eval()

    # ── Feature engineering ───────────────────────────────────────────────────

    def _engineer_features(self, sensor: dict) -> np.ndarray:
        """Applies identical feature engineering as the training notebook."""
        month_num = self.MONTH_MAP.get(str(sensor["month"]).lower(), 8)
        day_num   = self.DAY_MAP.get(str(sensor["day"]).lower(), 5)

        # Season one-hot
        if month_num in [12, 1, 2]:   season = "winter"
        elif month_num in [3, 4, 5]:  season = "spring"
        elif month_num in [6, 7, 8]:  season = "summer"
        else:                          season = "autumn"

        s_autumn = int(season == "autumn")
        s_spring = int(season == "spring")
        s_summer = int(season == "summer")
        s_winter = int(season == "winter")

        temp  = float(sensor["temp"])
        RH    = float(sensor["RH"])
        wind  = float(sensor["wind"])
        rain  = float(sensor["rain"])
        FFMC  = float(sensor["FFMC"])
        DMC   = float(sensor["DMC"])
        DC    = float(sensor["DC"])
        ISI   = float(sensor["ISI"])
        X     = float(sensor["X"])
        Y     = float(sensor["Y"])

        features = [
            X, Y,
            FFMC, DMC, DC, ISI,
            temp, RH, wind,
            np.log1p(rain),
            int(rain > 0),
            # Cyclic
            np.sin(2 * np.pi * month_num / 12),
            np.cos(2 * np.pi * month_num / 12),
            np.sin(2 * np.pi * day_num / 7),
            np.cos(2 * np.pi * day_num / 7),
            # Season one-hot (s_autumn dropped to avoid multicollinearity, add all for simplicity)
            s_autumn, s_spring, s_summer, s_winter,
            # FWI interactions
            temp / (RH + 1),
            temp * wind,
            wind * FFMC,
            DC / (DMC + 1),
            FFMC * ISI / 100,
            DC + DMC,
            ISI * (DC / (DMC + 1)),
            FFMC ** 2,
            temp ** 2,
            int((RH < 40) and (temp > 20)),
            # Thresholds
            int(rain > 0),
            int(temp > self.TEMP_Q75),
            int(RH < self.RH_Q25),
            int(FFMC > 90),
            int(wind > self.WIND_Q75),
            # Spatial
            np.sqrt(X ** 2 + Y ** 2),
            X * Y,
        ]
        return np.array(features, dtype=np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, sensor: dict) -> dict:
        """
        Run full ensemble inference on one sensor reading.

        Returns:
            {
              "is_fire":    bool,
              "prob":       float  (0–1),
              "label":      "FIRE" | "NO FIRE",
              "confidence": "HIGH" | "MEDIUM" | "LOW",
            }
        """
        # Demo mode (models not found)
        if getattr(self, "_demo_mode", False):
            import random
            prob = random.uniform(0.1, 0.95)
            return self._format_result(prob)

        import torch

        features = self._engineer_features(sensor)
        features_2d = features.reshape(1, -1)

        # XGBoost & LightGBM use unscaled features
        xgb_prob = float(self.xgb_model.predict_proba(features_2d)[0, 1])
        lgb_prob = float(self.lgb_model.predict(features_2d)[0])

        # TabMLP uses scaled features
        features_scaled = self.scaler.transform(features_2d)
        with torch.no_grad():
            logit    = self.tab_model(torch.tensor(features_scaled, dtype=torch.float32))
            tab_prob = float(torch.sigmoid(logit).item())

        # Stacking meta-learner
        stack_input  = np.array([[xgb_prob, lgb_prob, tab_prob]])
        ensemble_prob = float(self.meta_lr.predict_proba(stack_input)[0, 1])

        log.debug(f"XGB={xgb_prob:.3f} LGB={lgb_prob:.3f} TAB={tab_prob:.3f} → ENS={ensemble_prob:.3f}")

        return self._format_result(ensemble_prob)

    def _format_result(self, prob: float) -> dict:
        is_fire = prob >= self.threshold
        # Confidence bands
        if prob > 0.8 or prob < 0.2:     confidence = "HIGH"
        elif prob > 0.65 or prob < 0.35: confidence = "MEDIUM"
        else:                             confidence = "LOW"

        return {
            "is_fire":    is_fire,
            "prob":       prob,
            "label":      "FIRE" if is_fire else "NO FIRE",
            "confidence": confidence,
        }