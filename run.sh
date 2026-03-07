#!/usr/bin/env bash
# =============================================================================
#  FireWatch AI — Master Launch Script
#  Usage:
#    ./run.sh 1    →  Start all three layers (up)
#    ./run.sh 0    →  Stop all three layers  (down)
# =============================================================================

set -euo pipefail

# ── Repo root (directory where this script lives) ─────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Layer directories ─────────────────────────────────────────────────────────
LAYER1_DIR="$ROOT/before"       # Ground sensor + MQTT ensemble AI
LAYER2_DIR="$ROOT/during"       # Drone YOLOv5 detection API
LAYER3_DIR="$ROOT/after/satalite"  # Satellite scanner + Gemini report engine

# ── PID file locations (stored in each layer's directory) ─────────────────────
L1_PID="$LAYER1_DIR/.firewatch_l1.pid"
L2_PID="$LAYER2_DIR/.firewatch_l2.pid"
L3_PID="$LAYER3_DIR/.firewatch_l3.pid"

# ── Log files ─────────────────────────────────────────────────────────────────
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

L1_LOG="$LOG_DIR/layer1_sensor.log"
L2_LOG="$LOG_DIR/layer2_drone.log"
L3_LOG="$LOG_DIR/layer3_satellite.log"

# ── Ports (for health-check display) ─────────────────────────────────────────
L1_PORT=5000
L2_PORT=5001
L3_PORT=5002

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'   # No Colour

# =============================================================================
#  Helpers
# =============================================================================

log() { echo -e "${BOLD}[FireWatch]${NC} $*"; }
ok()  { echo -e "${GREEN}  ✔${NC}  $*"; }
err() { echo -e "${RED}  ✘${NC}  $*"; }
inf() { echo -e "${CYAN}  →${NC}  $*"; }
warn(){ echo -e "${YELLOW}  ⚠${NC}  $*"; }

# Detect which Python to use
detect_python() {
    if command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        err "Python not found in PATH. Please install Python 3."
        exit 1
    fi
}

# Detect virtual environment or uv
detect_venv() {
    # Prefer the repo-level .venv if it exists
    if [ -f "$ROOT/.venv/bin/python" ]; then
        echo "$ROOT/.venv/bin/python"
    elif [ -f "$ROOT/.venv/bin/python3" ]; then
        echo "$ROOT/.venv/bin/python3"
    else
        detect_python
    fi
}

PYTHON="$(detect_venv)"
inf "Using Python: $PYTHON"

# Start a background process, save its PID, and return
start_process() {
    local name="$1"
    local dir="$2"
    local cmd="$3"
    local logfile="$4"
    local pidfile="$5"

    if [ -f "$pidfile" ]; then
        local old_pid
        old_pid=$(cat "$pidfile")
        if kill -0 "$old_pid" 2>/dev/null; then
            warn "$name is already running (PID $old_pid) — skipping."
            return 0
        else
            rm -f "$pidfile"
        fi
    fi

    inf "Starting $name ..."
    (
        cd "$dir"
        # shellcheck disable=SC2086
        eval "$cmd" >> "$logfile" 2>&1
    ) &
    local pid=$!
    echo "$pid" > "$pidfile"
    ok "$name started  (PID $pid)  →  tail: $logfile"
}

# Stop a background process by PID file
stop_process() {
    local name="$1"
    local pidfile="$2"

    if [ ! -f "$pidfile" ]; then
        warn "$name — no PID file found, already stopped?"
        return 0
    fi

    local pid
    pid=$(cat "$pidfile")

    if kill -0 "$pid" 2>/dev/null; then
        inf "Stopping $name (PID $pid) ..."
        kill "$pid" 2>/dev/null || true

        # Wait up to 6 seconds for graceful shutdown
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 6 ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            warn "$name didn't stop gracefully — sending SIGKILL ..."
            kill -9 "$pid" 2>/dev/null || true
        fi

        ok "$name stopped."
    else
        warn "$name (PID $pid) was not running."
    fi

    rm -f "$pidfile"
}

# =============================================================================
#  UP — Start all three layers
# =============================================================================

cmd_up() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║         🔥  FireWatch AI  —  STARTING UP         ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── Layer 1: Ground Sensor Server ─────────────────────────────────────────
    # Runs:  app.py  (MQTT listener + ensemble model + Flask /health)
    #        simulator.py  (sends synthetic sensor readings for testing)
    start_process \
        "Layer 1 — Ground Sensor (app.py)" \
        "$LAYER1_DIR" \
        "FLASK_PORT=$L1_PORT \"$PYTHON\" app.py" \
        "$L1_LOG" \
        "$L1_PID"

    # Give the model time to load before starting the simulator
    sleep 2

    # Simulator runs in the same layer dir (reads .env automatically)
    start_process \
        "Layer 1 — Simulator (simulator.py)" \
        "$LAYER1_DIR" \
        "\"$PYTHON\" simulator.py" \
        "$L1_LOG" \
        "$LAYER1_DIR/.firewatch_l1_sim.pid"

    echo ""

    # ── Layer 2: Drone Detection API ──────────────────────────────────────────
    # Runs:  app.py  (Flask /detect with YOLOv5)
    #        simulate.py  (sends test images to /detect)
    start_process \
        "Layer 2 — Drone Detection API (app.py)" \
        "$LAYER2_DIR" \
        "FLASK_PORT=$L2_PORT \"$PYTHON\" app.py" \
        "$L2_LOG" \
        "$L2_PID"

    sleep 2

    start_process \
        "Layer 2 — Drone Simulator (simulate.py)" \
        "$LAYER2_DIR" \
        "FLASK_PORT=$L2_PORT \"$PYTHON\" simulate.py" \
        "$L2_LOG" \
        "$LAYER2_DIR/.firewatch_l2_sim.pid"

    echo ""

    # ── Layer 3: Satellite Scanner + AI Reports ────────────────────────────────
    # Runs:  scheduler.py  (NASA FIRMS daily scan + Gemini report engine)
    #        app.py        (Flask API: /map, /latest, /upload)
    start_process \
        "Layer 3 — Satellite Scheduler (scheduler.py)" \
        "$LAYER3_DIR" \
        "FLASK_PORT=$L3_PORT \"$PYTHON\" scheduler.py" \
        "$L3_LOG" \
        "$LAYER3_DIR/.firewatch_l3_sched.pid"

    sleep 1

    start_process \
        "Layer 3 — Satellite API (app.py)" \
        "$LAYER3_DIR" \
        "FLASK_PORT=$L3_PORT \"$PYTHON\" app.py" \
        "$L3_LOG" \
        "$L3_PID"

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║              All Layers Started ✔                ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  Layer 1 health : http://localhost:${L1_PORT}/health   ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Layer 2 detect : http://localhost:${L2_PORT}/detect   ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Layer 3 map    : http://localhost:${L3_PORT}/map      ${BOLD}║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  Logs dir  : $LOG_DIR"
    echo -e "${BOLD}║${NC}  Stop all  : ./run.sh 0                          ${BOLD}║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
#  DOWN — Stop all three layers
# =============================================================================

cmd_down() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║         🔥  FireWatch AI  —  SHUTTING DOWN       ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""

    # Stop simulators first, then main services
    stop_process "Layer 1 Simulator"         "$LAYER1_DIR/.firewatch_l1_sim.pid"
    stop_process "Layer 1 Ground Sensor"     "$L1_PID"

    stop_process "Layer 2 Drone Simulator"   "$LAYER2_DIR/.firewatch_l2_sim.pid"
    stop_process "Layer 2 Drone API"         "$L2_PID"

    stop_process "Layer 3 Satellite Sched"   "$LAYER3_DIR/.firewatch_l3_sched.pid"
    stop_process "Layer 3 Satellite API"     "$L3_PID"

    echo ""
    ok "All FireWatch layers stopped."
    echo ""
}

# =============================================================================
#  Status — show what's running
# =============================================================================

cmd_status() {
    echo ""
    log "FireWatch AI — Service Status"
    echo ""

    show_status() {
        local name="$1"
        local pidfile="$2"
        if [ -f "$pidfile" ]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                ok "$name  (PID $pid)  RUNNING"
            else
                err "$name  (PID $pid)  DEAD  ← stale PID file"
            fi
        else
            err "$name  NOT RUNNING"
        fi
    }

    show_status "Layer 1 — Ground Sensor API"   "$L1_PID"
    show_status "Layer 1 — Simulator"           "$LAYER1_DIR/.firewatch_l1_sim.pid"
    show_status "Layer 2 — Drone Detection API" "$L2_PID"
    show_status "Layer 2 — Drone Simulator"     "$LAYER2_DIR/.firewatch_l2_sim.pid"
    show_status "Layer 3 — Satellite Scheduler" "$LAYER3_DIR/.firewatch_l3_sched.pid"
    show_status "Layer 3 — Satellite API"       "$L3_PID"
    echo ""
}

# =============================================================================
#  Entry point
# =============================================================================

case "${1:-}" in
    1|up|start)
        cmd_up
        ;;
    0|down|stop)
        cmd_down
        ;;
    status)
        cmd_status
        ;;
    *)
        echo ""
        echo -e "${BOLD}FireWatch AI — run.sh${NC}"
        echo ""
        echo "  Usage:  ./run.sh <command>"
        echo ""
        echo -e "  ${GREEN}./run.sh 1${NC}       Start all layers  (up)"
        echo -e "  ${RED}./run.sh 0${NC}       Stop  all layers  (down)"
        echo -e "  ${CYAN}./run.sh status${NC}  Show running status"
        echo ""
        exit 1
        ;;
esac
