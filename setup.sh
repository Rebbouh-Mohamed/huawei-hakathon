#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  FireWatch AI — Ubuntu Setup Script
#  Run once: bash setup.sh
# ═══════════════════════════════════════════════════════════════════

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}🔥 FireWatch AI — Ubuntu Setup${NC}"
echo "════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────
echo -e "\n${YELLOW}[1/5] Installing system packages…${NC}"
sudo apt update -q
sudo apt install -y python3 python3-pip python3-venv \
                    mosquitto mosquitto-clients \
                    libgomp1 curl
echo -e "${GREEN}✅ System packages installed${NC}"

# ── 2. Start Mosquitto MQTT broker ───────────────────────────────
echo -e "\n${YELLOW}[2/5] Configuring & starting Mosquitto MQTT broker…${NC}"

# Allow anonymous connections on localhost (development config)
MOSQUITTO_CONF="/etc/mosquitto/conf.d/firewatch.conf"
sudo bash -c "cat > $MOSQUITTO_CONF" <<'MQCONF'
listener 1883 0.0.0.0
allow_anonymous true
MQCONF

sudo systemctl enable mosquitto
sudo systemctl restart mosquitto
sleep 1

# Test broker
if mosquitto_pub -h localhost -t test -m "ok" 2>/dev/null; then
  echo -e "${GREEN}✅ Mosquitto running on port 1883${NC}"
else
  echo -e "${RED}⚠️  Mosquitto test failed — check: sudo systemctl status mosquitto${NC}"
fi

# ── 3. Python virtual environment ────────────────────────────────
# echo -e "\n${YELLOW}[3/5] Creating Python virtual environment…${NC}"
# python3 -m venv venv
# source venv/bin/activate

# # ── 4. Python packages ────────────────────────────────────────────
# echo -e "\n${YELLOW}[4/5] Installing Python packages (this may take a few minutes)…${NC}"
# pip install --upgrade pip -q
# pip install -r requirements.txt -q
# echo -e "${GREEN}✅ Python packages installed${NC}"

# ── 5. Models folder ──────────────────────────────────────────────
echo -e "\n${YELLOW}[5/5] Creating models directory…${NC}"
mkdir -p models
echo -e "${GREEN}✅ models/ directory ready${NC}"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Copy your Kaggle model files into ./models/"
echo "     → xgb_v2.json, lgb_v2.txt, tabmlp_v2.pt, scaler_v2.pkl, meta_lr_v2.pkl"
echo ""
echo "  2. Activate venv and start the server:"
echo "     source venv/bin/activate"
echo "     python app.py"
echo ""
echo "  3. Open browser: http://localhost:5000"
echo ""
echo "  4. In a NEW terminal, run the sensor simulator:"
echo "     source venv/bin/activate"
echo "     python simulator.py"
echo ""
echo "  5. To simulate fire conditions:"
echo "     python simulator.py --fire"
echo "════════════════════════════════════════════════════════"