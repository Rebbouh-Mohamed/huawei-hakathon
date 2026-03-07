#!/usr/bin/env bash
# =============================================================================
#  FireWatch AI — Master Launch Script
#
#  Usage:
#    ./run.sh 1        Start all three layers (up)
#    ./run.sh 0        Stop  all three layers (down)
#    ./run.sh status   Show what is running
#
#  Each layer starts app.py first, waits until /health responds,
#  then launches the additional script (simulator / scheduler).
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✔${NC}  $*"; }
err()  { echo -e "${RED}  ✘${NC}  $*"; }
inf()  { echo -e "${CYAN}  →${NC}  $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC}  $*"; }

# ── Python ────────────────────────────────────────────────────────────────────
if   [ -f "$ROOT/.venv/bin/python" ];  then PY="$ROOT/.venv/bin/python"
elif [ -f "$ROOT/.venv/bin/python3" ]; then PY="$ROOT/.venv/bin/python3"
elif command -v python3 &>/dev/null;   then PY="python3"
elif command -v python  &>/dev/null;   then PY="python"
else err "Python not found."; exit 1; fi
inf "Python: $PY"

# ── Logs ─────────────────────────────────────────────────────────────────────
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
#  Generic helpers
# =============================================================================

# Write PID to a file (absolute path supplied by caller)
save_pid() { echo "$1" > "$2"; }

# Kill a process tracked by a PID file
kill_pid_file() {
    local label="$1" pidfile="$2"
    [ -f "$pidfile" ] || { warn "$label — no PID file, skipping."; return 0; }
    local pid; pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        inf "Stopping $label (PID $pid)…"
        kill "$pid" 2>/dev/null || true
        local w=0
        while kill -0 "$pid" 2>/dev/null && [ $w -lt 6 ]; do sleep 1; w=$((w+1)); done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        ok "$label stopped."
    else
        warn "$label (PID $pid) was not running."
    fi
    rm -f "$pidfile"
}

# Wait until GET http://localhost:<port>/health returns HTTP 200
# Args: label port timeout_seconds
wait_healthy() {
    local label="$1" port="$2" timeout="${3:-30}"
    inf "Waiting for $label to be healthy on port $port…"
    local i=0
    while [ $i -lt $timeout ]; do
        if curl -sf "http://localhost:${port}/health" -o /dev/null 2>/dev/null; then
            ok "$label is up ✔"
            return 0
        fi
        sleep 1; i=$((i+1))
    done
    warn "$label did not respond within ${timeout}s — additional script will start anyway."
    return 0   # non-fatal: keep going even if health check times out
}

# Start a background process, saving its PID
# Args: label dir command logfile pidfile
start_bg() {
    local label="$1" dir="$2" cmd="$3" logfile="$4" pidfile="$5"
    if [ -f "$pidfile" ]; then
        local old; old=$(cat "$pidfile")
        if kill -0 "$old" 2>/dev/null; then
            warn "$label already running (PID $old), skipping."
            return 0
        fi
        rm -f "$pidfile"
    fi
    inf "Starting $label…"
    ( cd "$dir" && eval "$cmd" >> "$logfile" 2>&1 ) &
    save_pid $! "$pidfile"
    ok "$label started (PID $!)"
}

# =============================================================================
#  Layer definitions
#
#  supervisord.conf summary:
#    before/          flask-app = app.py        additional = simulator.py   port 5000
#    during/          flask-app = app.py        additional = simulate.py    port 5001
#    after/satalite/  flask-app = app.py        additional = scheduler.py   port 5002
# =============================================================================

L1_DIR="$ROOT/before"
L1_PORT=5000
L1_APP_PID="$L1_DIR/.fw_app.pid"
L1_SIM_PID="$L1_DIR/.fw_sim.pid"
L1_LOG="$LOG_DIR/layer1_sensor.log"

L2_DIR="$ROOT/during"
L2_PORT=5001
L2_APP_PID="$L2_DIR/.fw_app.pid"
L2_SIM_PID="$L2_DIR/.fw_sim.pid"
L2_LOG="$LOG_DIR/layer2_drone.log"

L3_DIR="$ROOT/after/satalite"
L3_PORT=5002
L3_APP_PID="$L3_DIR/.fw_app.pid"
L3_SIM_PID="$L3_DIR/.fw_sim.pid"
L3_LOG="$LOG_DIR/layer3_satellite.log"

# =============================================================================
#  UP
# =============================================================================

cmd_up() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║        🔥  FireWatch AI  —  STARTING UP          ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── Layer 1 — Ground Sensor ───────────────────────────────────────────────
    echo -e "${BOLD}[ Layer 1 — Ground Sensor ]${NC}"
    start_bg \
        "Layer 1 app.py" "$L1_DIR" \
        "FLASK_PORT=$L1_PORT \"$PY\" app.py" \
        "$L1_LOG" "$L1_APP_PID"

    wait_healthy "Layer 1 app.py" $L1_PORT 30

    start_bg \
        "Layer 1 simulator.py" "$L1_DIR" \
        "\"$PY\" simulator.py" \
        "$L1_LOG" "$L1_SIM_PID"
    echo ""

    # ── Layer 2 — Drone Detection ─────────────────────────────────────────────
    echo -e "${BOLD}[ Layer 2 — Drone Detection ]${NC}"
    start_bg \
        "Layer 2 app.py" "$L2_DIR" \
        "FLASK_PORT=$L2_PORT \"$PY\" app.py" \
        "$L2_LOG" "$L2_APP_PID"

    wait_healthy "Layer 2 app.py" $L2_PORT 40   # YOLOv5 model load takes ~30 s

    start_bg \
        "Layer 2 simulate.py" "$L2_DIR" \
        "FLASK_PORT=$L2_PORT \"$PY\" simulate.py" \
        "$L2_LOG" "$L2_SIM_PID"
    echo ""

    # ── Layer 3 — Satellite + AI Reports ─────────────────────────────────────
    echo -e "${BOLD}[ Layer 3 — Satellite + AI Reports ]${NC}"
    start_bg \
        "Layer 3 app.py" "$L3_DIR" \
        "FLASK_PORT=$L3_PORT \"$PY\" app.py" \
        "$L3_LOG" "$L3_APP_PID"

    wait_healthy "Layer 3 app.py" $L3_PORT 30

    start_bg \
        "Layer 3 scheduler.py" "$L3_DIR" \
        "FLASK_PORT=$L3_PORT \"$PY\" scheduler.py" \
        "$L3_LOG" "$L3_SIM_PID"
    echo ""

    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║             All Layers Running ✔                 ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
    printf "${BOLD}║${NC}  Layer 1 health : http://localhost:%s/health\n" "$L1_PORT"
    printf "${BOLD}║${NC}  Layer 2 detect : http://localhost:%s/detect\n" "$L2_PORT"
    printf "${BOLD}║${NC}  Layer 3 map    : http://localhost:%s/map\n"    "$L3_PORT"
    echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  Logs  →  $LOG_DIR"
    echo -e "${BOLD}║${NC}  Stop  →  ./run.sh 0"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
#  DOWN
# =============================================================================

cmd_down() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║       🔥  FireWatch AI  —  SHUTTING DOWN         ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""

    kill_pid_file "Layer 1 simulator.py" "$L1_SIM_PID"
    kill_pid_file "Layer 1 app.py"       "$L1_APP_PID"

    kill_pid_file "Layer 2 simulate.py"  "$L2_SIM_PID"
    kill_pid_file "Layer 2 app.py"       "$L2_APP_PID"

    kill_pid_file "Layer 3 scheduler.py" "$L3_SIM_PID"
    kill_pid_file "Layer 3 app.py"       "$L3_APP_PID"

    echo ""
    ok "All FireWatch layers stopped."
    echo ""
}

# =============================================================================
#  STATUS
# =============================================================================

cmd_status() {
    echo ""
    echo -e "${BOLD}FireWatch AI — Status${NC}"
    echo ""
    check() {
        local label="$1" pidfile="$2"
        if [ -f "$pidfile" ]; then
            local pid; pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then ok "$label  (PID $pid)  RUNNING"
            else err "$label  stale PID $pid — not running"; fi
        else err "$label  NOT RUNNING"; fi
    }
    check "Layer 1  app.py      (port $L1_PORT)" "$L1_APP_PID"
    check "Layer 1  simulator.py              "  "$L1_SIM_PID"
    check "Layer 2  app.py      (port $L2_PORT)" "$L2_APP_PID"
    check "Layer 2  simulate.py               "  "$L2_SIM_PID"
    check "Layer 3  app.py      (port $L3_PORT)" "$L3_APP_PID"
    check "Layer 3  scheduler.py              "  "$L3_SIM_PID"
    echo ""
}

# =============================================================================
#  Entry point
# =============================================================================

case "${1:-}" in
    1|up|start)   cmd_up     ;;
    0|down|stop)  cmd_down   ;;
    status)       cmd_status ;;
    *)
        echo ""
        echo -e "${BOLD}Usage:${NC}"
        echo -e "  ${GREEN}./run.sh 1${NC}       Start all layers"
        echo -e "  ${RED}./run.sh 0${NC}       Stop  all layers"
        echo -e "  ${CYAN}./run.sh status${NC}  Show status"
        echo ""
        exit 1 ;;
esac
