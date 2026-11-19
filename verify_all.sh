#!/usr/bin/env bash
__SCRIPT_NAME="verify_all"
__SCRIPT_VERSION="v4"
__SCRIPT_BUILD="2026-11-17"
__SCRIPT_NOTES="reads the rotated log when the live one is empty"
#
# verify_all.sh — one command that checks everything, so nothing has to be
# hand-cross-checked. Prints a PASS/FAIL table; exits non-zero on any failure.
# Read-only: places no orders, changes no files.
#
#   ./verify_all.sh            # full
#   ./verify_all.sh --quiet    # failures only
#   ./verify_all.sh --version
#
# Expected checksums live in verify_expected.txt. Update that file when a new
# version is delivered — not this script.

set -uo pipefail
cd "$(dirname "$0")" || exit 2
REPO="$PWD"
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

if [ "${1:-}" = "--version" ]; then
    echo "$__SCRIPT_NAME $__SCRIPT_VERSION (build $__SCRIPT_BUILD) — $__SCRIPT_NOTES"
    exit 0
fi

# v2: never assume 'python' exists — on this VM it does not without the venv.
if [ -x venv/bin/python ]; then
    PY=venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "  FATAL no python interpreter (looked for venv/bin/python, python3, python)"
    exit 2
fi

PASS=0; FAIL=0; WARN=0
declare -a FAILED
ok()    { PASS=$((PASS+1)); [ $QUIET -eq 0 ] && printf "  PASS  %s\n" "$1"; return 0; }
bad()   { FAIL=$((FAIL+1)); FAILED+=("$1"); printf "  FAIL  %s\n" "$1"; return 0; }
warn()  { WARN=$((WARN+1)); [ $QUIET -eq 0 ] && printf "  WARN  %s\n" "$1"; return 0; }
head_() { [ $QUIET -eq 0 ] && printf "\n=== %s ===\n" "$1"; return 0; }

[ $QUIET -eq 0 ] && echo "  interpreter: $PY"

# ---------------------------------------------------------------- files
head_ "FILES  (version + checksum)"
EXP="verify_expected.txt"
if [ ! -f "$EXP" ]; then
    warn "no $EXP — skipping checksum checks"
else
    while IFS=$'\t' read -r path want_ver want_md5; do
        case "$path" in ''|\#*) continue;; esac
        if [ ! -f "$path" ]; then bad "$path MISSING"; continue; fi
        got_ver=$(grep -m1 '^__SCRIPT_VERSION' "$path" 2>/dev/null | cut -d'"' -f2)
        got_md5=$(md5sum "$path" | cut -d' ' -f1)
        if [ "$got_ver" != "$want_ver" ]; then
            bad "$path version $got_ver, expected $want_ver"
        elif [ -n "$want_md5" ] && [ "$got_md5" != "$want_md5" ]; then
            bad "$path checksum $got_md5, expected $want_md5"
        else
            ok "$(basename "$path") $got_ver $got_md5"
        fi
    done < "$EXP"
fi

# ---------------------------------------------------------------- selftests
head_ "SELFTESTS"
run_tests() {   # name, command, expected-count
    local name="$1" cmd="$2" want="$3" out n
    out=$(eval "$cmd" 2>&1)
    n=$(printf '%s\n' "$out" | grep -oE '[0-9]+ passed' | tail -1 | cut -d' ' -f1)
    if [ -z "$n" ]; then
        bad "$name produced no pass count"
        printf '%s\n' "$out" | tail -3 | sed 's/^/        | /'
    elif [ "$n" = "$want" ]; then
        ok "$name $n tests"
    else
        bad "$name $n tests, expected $want"
        printf '%s\n' "$out" | grep -E '^FAIL' | head -3 | sed 's/^/        | /'
    fi
}
run_tests "guard"      "$PY futures_and_options/test_guard_v3.py"       "127"
run_tests "stock scan" "$PY stock_fno/stock_fno_scan.py --selftest"     "16"
run_tests "stock mgr"  "$PY stock_fno/stock_fno_manager.py --selftest"  "34"
run_tests "value_rank" "$PY value_investing/value_rank.py --selftest"   "17"
run_tests "redteam"    "$PY value_investing/redteam.py --selftest"      "14"

# ---------------------------------------------------------------- cron
head_ "CRON"
CT=$(crontab -l 2>/dev/null)
if [ -z "$CT" ]; then
    bad "crontab unreadable"
else
    [ "$(printf '%s\n' "$CT" | grep -c 'strategy-simple')" = "2" ] \
        && ok "2 x --strategy-simple" || bad "expected 2 x --strategy-simple"
    for f in profit-floor "trail-arm 0" per-lot-profit; do
        if printf '%s\n' "$CT" | grep -q -- "--$f"; then bad "stale flag --$f present"
        else ok "no --$f"; fi
    done
    # check each line by CONTENT, not by count - counting broke the moment a
    # third stock_fno line (the 07:55 pre-run) was added, and reported a
    # healthy crontab as broken
    printf '%s\n' "$CT" | grep -q '^0 8 .*stock_fno_scan.py' \
        && ok "08:00 scan present" || bad "08:00 scan line missing"
    printf '%s\n' "$CT" | grep -q '^\*/15 9-15 .*stock_fno_manager.py' \
        && ok "*/15 manager present" || bad "*/15 manager line missing"
    if printf '%s\n' "$CT" | grep -q '^55 7 .*stock_fno_manager.py'; then
        ok "07:55 holdings pre-run present"
        _pre=$(printf '%s\n' "$CT" | grep '^55 7 ')
        printf '%s' "$_pre" | grep -q 'timeout' \
            && ok "07:55 pre-run has timeout" \
            || bad "07:55 pre-run has NO timeout - a hang blocks the 08:00 scan"
        printf '%s' "$_pre" | grep -q 'stock_fno_scan.lock' \
            && bad "07:55 pre-run holds the SCAN lock - a hang blocks the scan" \
            || ok "07:55 pre-run uses its own lock"
    else
        warn "no 07:55 pre-run - the 08:00 scan sees yesterday's holdings and assumes flat"
    fi
    printf '%s\n' "$CT" | grep 'stock_fno_manager' | grep -q 'flock' \
        && ok "manager has flock" || bad "manager has no flock"
    printf '%s\n' "$CT" | grep -q 'stock_fno_manager.*--execute' \
        && ok "manager runs --execute (acts, not just reports)" \
        || warn "manager is report-only - you are the exit mechanism"
    printf '%s\n' "$CT" | grep -q 'auth_health.py' \
        && ok "07:45 auth health check present" \
        || warn "no auth_health line - an auth failure kills the day silently"
    printf '%s\n' "$CT" | grep -q 'eod-exit 15:15' \
        && ok "eod-exit 15:15" || warn "eod-exit not 15:15 (10-minute window)"
    # /bin/sh only matters when the crontab does not set SHELL explicitly
    if printf '%s\n' "$CT" | grep -q '^SHELL='; then
        ok "cron shell: $(printf '%s\n' "$CT" | grep -m1 '^SHELL=' | cut -d= -f2)"
    elif [ "$(readlink -f /bin/sh)" = "/bin/bash" ]; then
        ok "cron shell: /bin/sh is bash"
    else
        bad "cron shell: no SHELL= and /bin/sh is $(readlink -f /bin/sh)"
    fi
    printf '%s\n' "$CT" | grep -q '^TELEGRAM_BOT_TOKEN=' \
        && ok "TELEGRAM_BOT_TOKEN set" || warn "no TELEGRAM_BOT_TOKEN (alerts silent)"
fi

# ---------------------------------------------------------------- broker
head_ "BROKER"
BAL=$("$PY" - <<'PYEOF' 2>/dev/null
import os, requests
from src.auth import get_access_token
h = {"access-token": get_access_token(),
     "client-id": os.getenv("DHAN_CLIENT_ID", "1000191656"),
     "Content-Type": "application/json", "Accept": "application/json"}
j = requests.get("https://api.dhan.co/v2/fundlimit", headers=h, timeout=30).json()
print(j.get("availabelBalance", "ERR"), j.get("withdrawableBalance", "ERR"))
PYEOF
)
if [ -z "$BAL" ] || printf '%s' "$BAL" | grep -q ERR; then
    bad "fundlimit unreadable (auth or network)"
else
    AVAIL=$(printf '%s' "$BAL" | cut -d' ' -f1)
    CASH=$(printf '%s' "$BAL" | cut -d' ' -f2)
    if "$PY" -c "import sys; sys.exit(0 if float('$AVAIL') >= 0 else 1)"; then
        ok "available balance Rs.$AVAIL"
    else
        bad "available balance NEGATIVE: Rs.$AVAIL"
    fi
    if "$PY" -c "import sys; sys.exit(0 if float('$CASH') >= 5000 else 1)"; then
        ok "cash Rs.$CASH (can buy back shorts)"
    else
        warn "cash Rs.$CASH — too little to buy back a short or fund a condor's wings"
    fi
fi

# ---------------------------------------------------------------- state
head_ "STATE"
[ -f stock_fno/stock_fno_holdings.json ] \
    && ok "holdings: $(tr -d '\n ' < stock_fno/stock_fno_holdings.json)" \
    || warn "no stock_fno_holdings.json (scan assumes flat)"
# the live log rotates at midnight, so between 00:00 and 09:00 it is empty and
# every check below would report a false alarm. Fall back to the newest rotated
# file (dateext naming: fno_guard.log-YYYYMMDD[.gz]) when the live one is empty.
GLOG=logs/fno_guard.log
if [ ! -f "$GLOG" ]; then
    bad "logs/fno_guard.log missing"
else
    _live=$(grep -c '' "$GLOG" 2>/dev/null || echo 0)
    if [ "$_live" -eq 0 ]; then
        _rot=$(ls -1t logs/fno_guard.log-* 2>/dev/null | head -1)
        if [ -n "$_rot" ]; then
            ok "live log empty (rotated at midnight); reading $(basename "$_rot")"
            GLOG="$_rot"
        else
            warn "live log empty and no rotated file found"
        fi
    else
        ok "guard log present ($_live lines)"
    fi
    if [ "$GLOG" != "logs/fno_guard.log" ] || [ "$_live" -gt 0 ]; then
        if zcat -f "$GLOG" 2>/dev/null | grep -q "Started at"; then
            ok "audit anchor present"
        else
            warn "no 'Started at' anchor in $(basename "$GLOG") — audit parses nothing"
        fi
        zcat -f "$GLOG" 2>/dev/null | grep -q "\[SL-CAP\]" \
            && bad "[SL-CAP] tripped — stop budget exhausted" || ok "no [SL-CAP] trips"
        zcat -f "$GLOG" 2>/dev/null | grep -q "FLATTEN-REFUSED\|FLATTEN-CAP" \
            && bad "flatten refused/capped — a position may not have closed" \
            || ok "no flatten refusals"
        if [ -f data/sl_events.csv ]; then
            ok "stop events recorded: $(( $(grep -c '' data/sl_events.csv) - 1 )) row(s)"
        else
            warn "no data/sl_events.csv yet (guard v29+ writes it)"
        fi
    fi
fi

# ---------------------------------------------------------------- summary
printf "\n==================== %d pass, %d fail, %d warn ====================\n" "$PASS" "$FAIL" "$WARN"
if [ "$FAIL" -gt 0 ]; then
    printf "FAILURES:\n"
    for f in "${FAILED[@]}"; do printf "  - %s\n" "$f"; done
    exit 1
fi
exit 0
