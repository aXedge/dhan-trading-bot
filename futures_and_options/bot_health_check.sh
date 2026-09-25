#!/bin/bash
# =============================================================================
# bot_health_check.sh — one command, one paste: the full system state
# =============================================================================
# Run on the VM (or anywhere with the repo + venv):
#     bash futures_and_options/bot_health_check.sh
#
# Designed for pasting the entire output into one message. Each section is
# marked with ==== so it can be parsed/analyzed in one pass.
#
# Read-only: runs reports and greps logs; places no orders, changes nothing.

set +e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD:$PWD/src"
export TRADING_MODE=PAPER
source venv/bin/activate 2>/dev/null

echo "################################################################"
echo "# BOT HEALTH CHECK — $(date '+%Y-%m-%d %H:%M:%S') IST"
echo "################################################################"

echo ""
echo "==== 1. HOST ===="
uptime
df -h / | tail -1

echo ""
echo "==== 2. REPO STATE ===="
git log --oneline -3 2>/dev/null
git status --short 2>/dev/null | head -10

echo ""
echo "==== 3. CRON (bot jobs) ===="
crontab -l 2>/dev/null | grep -v "^#" | grep -E "scanner|executor|monitor|guard|screener|refresher|eod" | head -15

echo ""
echo "==== 4. PAPER BOOK ===="
python src/live/performance_report.py 2>/dev/null | head -60

echo ""
echo "==== 5. CLOSED TRADES ===="
python3 - <<'PYEOF' 2>/dev/null
import json
d = json.load(open("data/paper_positions.json"))
print(f"open: {len(d.get('positions', []))} | closed: {len(d.get('closed_trades', []))}")
for t in d.get("closed_trades", []):
    print(t["symbol"], t["strategy"], "| exit:", t.get("exit_reason"),
          "@", t.get("exit_price"), "|", t.get("exit_date"), "| P&L Rs.",
          round(t.get("pnl_rs", 0)))
PYEOF

echo ""
echo "==== 6. MONITOR (this week) ===="
tail -3 logs/monitor.log 2>/dev/null
zgrep -h "exits today" logs/monitor.log-202609* logs/monitor.log 2>/dev/null \
    | sort | uniq -c | tail -5

echo ""
echo "==== 7. SCANNER (signal counts this week) ===="
zgrep -c "SIGNAL" logs/scanner.log-2026* logs/scanner.log 2>/dev/null | tail -8

echo ""
echo "==== 8. EXECUTOR (recent skips/entries) ===="
zgrep -h "SKIP\|OPENED" logs/executor.log-202609* logs/executor.log 2>/dev/null \
    | tail -8

echo ""
echo "==== 9. F&O GUARD (last cycle) ===="
tail -25 logs/fno_guard.log 2>/dev/null

echo ""
echo "==== 10. GUARD STATE FILE ===="
cat data/fno_guard_state.json 2>/dev/null || echo "(not created yet)"
echo ""

echo "==== 11. ERRORS (last 24h, all logs) ===="
grep -h -iE "error|failed|traceback" logs/*.log 2>/dev/null \
    | grep -v "Invalid TOTP" | tail -8
echo "(TOTP failures were filtered out of the count above; showing the total:)"
grep -h "Invalid TOTP" logs/*.log 2>/dev/null | wc -l

echo ""
echo "################################################################"
echo "# HEALTH CHECK COMPLETE"
echo "################################################################"
