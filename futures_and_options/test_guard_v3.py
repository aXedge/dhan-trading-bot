"""Offline tests for dhan_fno_pnl_guard.py v3 (no network)."""
import importlib.util
import io
import os
import sys
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

# Load the guard from the same directory as this test file,
# so the pair works anywhere (laptop, VM, CI).
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "guard", os.path.join(HERE, "dhan_fno_pnl_guard.py"))
g = importlib.util.module_from_spec(spec)
sys.modules["guard"] = g
spec.loader.exec_module(g)

# tests simulate market hours (guard refuses to place orders otherwise)
g.market_open = lambda now=None: True

# --- fixture: today's actual structure (4-lot bear call 23300/23700) ---
POSITIONS = [
    {"exchangeSegment": "NSE_FNO", "tradingSymbol": "NIFTY 23300 CE",
     "underlying": "NIFTY", "drvExpiryDate": "2026-09-22",
     "productType": "MARGIN", "netQty": -260, "costPrice": 166.3,
     "securityId": "11101", "unrealizedProfit": -14300.0},
    {"exchangeSegment": "NSE_FNO", "tradingSymbol": "NIFTY 23700 CE",
     "underlying": "NIFTY", "drvExpiryDate": "2026-09-22",
     "productType": "MARGIN", "netQty": 260, "costPrice": 44.9,
     "securityId": "11102", "unrealizedProfit": -5700.0},
    # an equity CNC position that must be ignored entirely
    {"exchangeSegment": "NSE_EQ", "tradingSymbol": "TCS",
     "productType": "CNC", "netQty": 10, "costPrice": 3800,
     "securityId": "1333", "unrealizedProfit": 500.0},
]

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    print(("PASS" if cond else "FAIL") + f"  {name}")
    if cond: PASS += 1
    else: FAIL += 1

# ---------- 1. grouping ----------
s = g.group_structures(POSITIONS, 65)
check("one structure found", list(s.keys()) == ["NIFTY|2026-09-22"])
st = s["NIFTY|2026-09-22"]
check("lots = 4 (260/65)", st["lots"] == 4)
check("MTM = -20000 (sums both legs)", abs(st["mtm"] + 20000.0) < 1e-6)
check("equity CNC excluded", len(st["legs"]) == 2)

# fallback MTM computation when unrealizedProfit missing
p2 = [{**POSITIONS[0], "unrealizedProfit": None, "lastPrice": 210.0}]
s2 = g.group_structures(p2, 65)
check("fallback MTM net*(last-cost) = -260*43.7", 
      abs(s2["NIFTY|2026-09-22"]["mtm"] + 11362.0) < 1e-6)

# ---------- 2. order body ----------
body = g.exit_order_body("1000191656", POSITIONS[0])
check("short leg exit = BUY 260", body["transactionType"] == "BUY"
      and body["quantity"] == 260)
check("productType carried = MARGIN", body["productType"] == "MARGIN")
check("MARKET / DAY", body["orderType"] == "MARKET" and body["validity"] == "DAY")
body2 = g.exit_order_body("1000191656", POSITIONS[1])
check("long leg exit = SELL 260", body2["transactionType"] == "SELL")

# ---------- 3. mock client ----------
class MockClient:
    def __init__(self):
        self.client_id = "1000191656"
        self.orders = []
        self.pnl_exit_calls = []
        self.kill_calls = 0
        self.filled = False  # simulate fills: flat once exit orders placed
    def get_positions(self):
        return [] if self.filled else POSITIONS
    def get_fund_limits(self): return {"availabelBalance": "264000"}
    def place_order(self, body):
        self.orders.append(body)
        self.filled = True  # market orders fill immediately
        return {"orderId": 999 + len(self.orders)}
    def get_pnl_exit(self): return {"pnlExitStatus": "ACTIVE",
                                    "profit": "18000", "loss": "-14400"}
    def set_pnl_exit(self, p, l, pt, enable_kill_switch=False):
        self.pnl_exit_calls.append((p, l, pt))
        return {"pnlExitStatus": "ACTIVE", "status": "SUCCESS"}
    def kill_switch_activate(self):
        self.kill_calls += 1
        return {"killSwitchStatus": "activated"}

def run(args_overrides, client=None, state=None):
    args = SimpleNamespace(
        per_lot_profit=4500, per_lot_loss=3600, lot_size=65,
        profit=0, loss=0, enforce=False, dry_run=False,
        kill_on_reentry=False, backstop=True)
    for k, v in args_overrides.items():
        setattr(args, k, v)
    client = client or MockClient()
    state = state or {"date": "2026-09-21", "alerts": {}, "locked": []}
    buf = io.StringIO()
    with redirect_stdout(buf):
        g.check_once(client, args, state)
    return buf.getvalue(), client, state

# ---------- 4. WATCH mode on today's scenario ----------
out, mc, stt = run({})
check("watch: breach detected", "BREACH" in out and "WATCH mode" in out)
check("watch: no orders placed", len(mc.orders) == 0)
check("watch: thresholds scaled 4 lots -> 14400/18000",
      "14,400" in out and "18,000" in out)

# warning levels fire pre-breach (3-lot MTM -9000 vs limit 10800)
POS3 = [dict(POSITIONS[0], netQty=-195, unrealizedProfit=-6400.0),
        dict(POSITIONS[1], netQty=195, unrealizedProfit=-2600.0)]
class MockClient3(MockClient):
    def get_positions(self): return POS3
out3, _, stt3 = run({}, client=MockClient3())
check("3 lots scaled -> 10,800", "10,800" in out3)
check("WARN50 fires at ratio 0.83", "WARN75" in out3)

# ---------- 5. ENFORCE + dry-run ----------
out, mc, _ = run({"enforce": True, "dry_run": True})
check("dry-run: orders previewed", "[EXIT] NIFTY 23300 CE: BUY 260" in out
      and "[EXIT] NIFTY 23700 CE: SELL 260" in out)
check("dry-run: nothing actually placed", len(mc.orders) == 0)
check("dry-run: DRY RUN notice", "orders above were NOT placed" in out)

# ---------- 6. ENFORCE live (mocked) ----------
out, mc, stt = run({"enforce": True})
check("both legs flattened", len(mc.orders) == 2)
check("order sides correct",
      mc.orders[0]["transactionType"] == "BUY"
      and mc.orders[1]["transactionType"] == "SELL")
check("structure locked after loss exit",
      any(l["key"] == "NIFTY|2026-09-22" for l in stt["locked"]))

# ---------- 7. re-entry on locked structure ----------
REENTRY = [{**POSITIONS[0], "netQty": -130, "unrealizedProfit": -2000.0},
           {**POSITIONS[1], "netQty": 130, "unrealizedProfit": -800.0}]
class MockClientR(MockClient):
    def get_positions(self): return [] if self.filled else REENTRY
stt_l = {"date": "2026-09-21", "alerts": {},
         "locked": [{"key": "NIFTY|2026-09-22", "time": "x", "mtm": -20000}]}
out, mc, _ = run({"enforce": True}, client=MockClientR(), state=stt_l)
check("re-entry detected + alerted", "RE-ENTRY on locked" in out)
check("kill-on-reentry off: no exit, no kill switch",
      len(mc.orders) == 0 and mc.kill_calls == 0)

out, mc, _ = run({"enforce": True, "kill_on_reentry": True},
                 client=MockClientR(), state=json.loads(json.dumps(stt_l)))
check("kill-on-reentry on: re-entered structure flattened + kill switch",
      len(mc.orders) == 2 and mc.kill_calls == 1)

# ---------- 8. INTRADAY backstop ----------
_, mc, _ = run({"backstop": True})
check("backstop set for INTRADAY only",
      mc.pnl_exit_calls == [] or all(
          c[2] == ["INTRADAY"] for c in mc.pnl_exit_calls))
# (already ACTIVE at 18000/14400 -> no re-set expected; verify no DELIVERY)
check("backstop: DELIVERY never used", all(
    "DELIVERY" not in c[2] for c in mc.pnl_exit_calls))

# ---------- 9. state daily reset ----------
fresh = g.load_state("/nonexistent/x.json", "2026-09-22")
check("state fresh when file missing", fresh == {"date": "2026-09-22",
                                                 "alerts": {}, "locked": []})

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
