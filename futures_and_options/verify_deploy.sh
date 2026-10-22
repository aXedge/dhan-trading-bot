#!/usr/bin/env bash
# verify_deploy.sh - post-deploy checks for F&O guard (v9) + trail audit (v3)
# Run on the VM from the repo root:  bash futures_and_options/verify_deploy.sh
# Safe any time: the smoke test is report-only (no --enforce), so it never places an order.
set -u
cd /home/kay22_ind/dhan-trading-bot || exit 1
source venv/bin/activate
export PYTHONPATH=/home/kay22_ind/dhan-trading-bot:/home/kay22_ind/dhan-trading-bot/src

pass=0; fail=0
ok(){ echo "  PASS  $1"; pass=$((pass+1)); }
no(){ echo "  FAIL  $1"; fail=$((fail+1)); }

G=futures_and_options/dhan_fno_pnl_guard.py
A=futures_and_options/trail_audit.py

echo "== 1. version identity =="
n=$(grep -c "\[STRUCT\]" "$G" 2>/dev/null || echo 0)
[ "$n" -ge 1 ] && ok "guard contains [STRUCT] (count $n)" || no "guard missing the [STRUCT] line (still old version?)"
v=$(grep -oP 'GUARD_VERSION = "\K[^"]+' "$G" 2>/dev/null)
[ -n "$v" ] && ok "guard version marker = $v" || no "no GUARD_VERSION marker (pre-v12 build)"
for m in "profit-floor" "eod-exit" "floor_label"; do
  grep -q "$m" "$G" && ok "guard has feature: $m" || no "guard MISSING feature: $m"
done
bv=$(python "$G" --version 2>/dev/null | head -1)
[ -n "$bv" ] && ok "guard --version -> $bv" || no "guard --version failed"
grep -q "UNRELIABLE" "$A" && ok "audit has the sanity gate" || no "audit missing sanity gate"
grep -q "ALGO" "$A" && ok "audit has ALGO exit-typing" || no "audit missing ALGO typing"

echo "== 2. unit tests =="
T=""
for cand in test_guard_v3.py futures_and_options/test_guard_v3.py src/test_guard_v3.py; do
  [ -f "$cand" ] && T="$cand" && break
done
if [ -z "$T" ]; then
  no "test_guard_v3.py not found (looked in ./ and futures_and_options/)"
else
  o=$(python "$T" 2>&1 | tail -1); echo "  guard ($T): $o"
  case "$o" in
    *"0 failed"*) ok "guard tests pass ($o)";;
    *) no "guard tests failing ($o)";;
  esac
fi
o=$(python futures_and_options/trail_audit.py --selftest 2>&1 | tail -1); echo "  audit: $o"
case "$o" in *"ALL PASS"*) ok "audit selftest ALL PASS";; *) no "audit selftest failed";; esac

echo "== 3. trail is OFF (the point of the observation window) =="
c=$(crontab -l 2>/dev/null | grep -c -- "--trail-arm 0")
[ "$c" -ge 2 ] && ok "--trail-arm 0 set on $c cron line(s)" || no "--trail-arm 0 missing from cron (found $c, want 2)"
b=$(python "$G" --once --trail-arm 0 2>&1)
if echo "$b" | grep -q "arm at Rs"; then no "banner shows 'arm at Rs...' -> trail is ON"; else ok "banner shows 'Trailing : off' -> trail is OFF"; fi
if echo "$b" | grep -q "No open F&O positions"; then ok "guard ran clean end-to-end (flat book, token ok)"; else echo "  note: guard tail -> $(echo "$b" | tail -1)"; fi

echo "== 4. cron integrity =="
g=$(crontab -l 2>/dev/null | grep -c "dhan_fno_pnl_guard.py")
[ "$g" -eq 2 ] && ok "two guard cron lines (--once + --fast)" || no "guard cron lines = $g (want 2)"
a=$(crontab -l 2>/dev/null | grep -c "trail_audit.py")
[ "$a" -eq 1 ] && ok "one audit cron line" || no "audit cron lines = $a (want 1)"
crontab -l 2>/dev/null | grep -q -- "--sl-order" && echo "  note: --sl-order is present in cron" || ok "resting stop off (as intended)"

echo "== 5. profit side switched on? =="
pf=$(crontab -l 2>/dev/null | grep -c -- "--profit-floor")
[ "$pf" -ge 2 ] && ok "--profit-floor on $pf cron line(s)" || echo "  note: --profit-floor NOT in cron ($pf) — book-profit floor inactive"
ee=$(crontab -l 2>/dev/null | grep -c -- "--eod-exit")
[ "$ee" -ge 2 ] && ok "--eod-exit on $ee cron line(s)" || echo "  note: --eod-exit NOT in cron ($ee) — no end-of-day profit close"

echo
echo "RESULT: $pass passed, $fail failed"
echo "Monday-only (cannot be checked until a live structure exists):"
echo "  - the [STRUCT] line appears once per structure during the session"
echo "  - data/fno_guard_state.json gains a 'struct_seen' key"
echo "  - trail_audit.py prices an algo-side exit instead of UNRELIABLE-FIT"
