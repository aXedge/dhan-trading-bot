#!/usr/bin/env bash
#
# Dhan F&O Daily Runner
# =======================
# Automates the daily workflow on the GCP VM:
#   1. Fetch fresh trade history
#   2. Compute adaptive P&L limits
#   3. Start the P&L guard polling loop
#
# Designed to run inside tmux/screen or as a systemd service.
#
# Usage:
#   ./dhan_daily_runner.sh
#
# Environment variables (set in ~/.bashrc or a .env file):
#   DHAN_CLIENT_ID     — your Dhan client ID
#   DHAN_ACCESS_TOKEN  — current access token (refresh every 24h)
#
# Optional:
#   DHAN_SCRIPTS_DIR   — directory containing the fno scripts (default: ~/dhan-trading-bot/fno)
#   DHAN_DAYS          — trailing days of history (default: 90)
#   DHAN_TRADES_PER_DAY — expected F&O trades/day (default: 3)
#   DHAN_HALF_LIFE     — EWMA half-life (default: 50)
#   DHAN_POLL_INTERVAL — guard poll interval in seconds (default: 15)
#   DHAN_LOG_FILE      — log file path (default: ~/dhan-trading-bot/fno/logs/runner_YYYY-MM-DD.log)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPTS_DIR="${DHAN_SCRIPTS_DIR:-$HOME/dhan-trading-bot/fno}"
DAYS="${DHAN_DAYS:-90}"
TRADES_PER_DAY="${DHAN_TRADES_PER_DAY:-3}"
HALF_LIFE="${DHAN_HALF_LIFE:-50}"
POLL_INTERVAL="${DHAN_POLL_INTERVAL:-15}"
LOG_DIR="${SCRIPTS_DIR}/logs"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="${DHAN_LOG_FILE:-$LOG_DIR/runner_${TODAY}.log}"
DATA_FILE="${SCRIPTS_DIR}/fno_trades.csv"
STATE_FILE="${SCRIPTS_DIR}/.dhan_adaptive_guard_state.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

log "============================================================"
log "  DHAN F&O DAILY RUNNER"
log "  Date: $TODAY"
log "  Scripts dir: $SCRIPTS_DIR"
log "  Log file: $LOG_FILE"
log "============================================================"

# Check environment variables
if [ -z "${DHAN_CLIENT_ID:-}" ]; then
    error "DHAN_CLIENT_ID is not set. Export it first:"
    error "  export DHAN_CLIENT_ID='your-client-id'"
    exit 1
fi

if [ -z "${DHAN_ACCESS_TOKEN:-}" ]; then
    error "DHAN_ACCESS_TOKEN is not set. Generate a fresh token from"
    error "  web.dhan.co → My Profile → Access DhanHQ APIs → Generate Access Token"
    error "Then: export DHAN_ACCESS_TOKEN='your-token'"
    exit 1
fi

# Check scripts exist
for script in dhan_fno_avg_pnl.py dhan_fno_adaptive_guard.py; do
    if [ ! -f "$SCRIPTS_DIR/$script" ]; then
        error "Script not found: $SCRIPTS_DIR/$script"
        error "Copy the scripts to $SCRIPTS_DIR first."
        exit 1
    fi
done

# Check Python 3
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# Check requests module
if ! python3 -c "import requests" &>/dev/null; then
    log "Installing requests..."
    pip3 install requests || { error "Failed to install requests"; exit 1; }
fi

log "[OK] Pre-flight checks passed."
log "  Client ID: $DHAN_CLIENT_ID"
log "  Token length: ${#DHAN_ACCESS_TOKEN}"

# ---------------------------------------------------------------------------
# Step 1: Fetch trade history and save to CSV
# ---------------------------------------------------------------------------
log ""
log "========== STEP 1: FETCH TRADE HISTORY =========="
log "Fetching last $DAYS days of F&O trades..."

python3 "$SCRIPTS_DIR/dhan_fno_avg_pnl.py" \
    --days "$DAYS" \
    --data-file "$DATA_FILE" \
    2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$DATA_FILE" ]; then
    error "Trade data file was not created: $DATA_FILE"
    error "Check the log above for errors."
    exit 1
fi

TRADE_COUNT=$(wc -l < "$DATA_FILE")
log "[OK] Trade history saved to $DATA_FILE ($((TRADE_COUNT - 1)) trades)"

# ---------------------------------------------------------------------------
# Step 2: Compute adaptive limits (dry run first to log them)
# ---------------------------------------------------------------------------
log ""
log "========== STEP 2: COMPUTE ADAPTIVE LIMITS =========="
log "Computing EWMA-weighted limits (half-life=$HALF_LIFE, N=$TRADES_PER_DAY)..."

python3 "$SCRIPTS_DIR/dhan_fno_adaptive_guard.py" \
    --from-file "$DATA_FILE" \
    --dry-run \
    --trades-per-day "$TRADES_PER_DAY" \
    --half-life "$HALF_LIFE" \
    2>&1 | tee -a "$LOG_FILE"

log "[OK] Limits computed and saved to state file: $STATE_FILE"

# ---------------------------------------------------------------------------
# Step 3: Start the P&L guard
# ---------------------------------------------------------------------------
log ""
log "========== STEP 3: START P&L GUARD =========="
log "Starting adaptive guard with computed limits..."
log "Poll interval: ${POLL_INTERVAL}s"
log "Press Ctrl+C to stop."
log ""

python3 "$SCRIPTS_DIR/dhan_fno_adaptive_guard.py" \
    --from-file "$DATA_FILE" \
    --trades-per-day "$TRADES_PER_DAY" \
    --half-life "$HALF_LIFE" \
    --poll-interval "$POLL_INTERVAL" \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "============================================================"
log "  DAILY RUNNER COMPLETE"
log "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
log "  Log: $LOG_FILE"
log "============================================================"
