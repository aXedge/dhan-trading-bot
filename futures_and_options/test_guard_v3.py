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
import tempfile
TMP = tempfile.mkdtemp()
SF = os.path.join(TMP, "fno_guard_state.json")   # isolated state/hot-flag
spec = importlib.util.spec_from_file_location(
    "guard", os.path.join(HERE, "dhan_fno_pnl_guard.py"))
g = importlib.util.module_from_spec(spec)
sys.modules["guard"] = g
spec.loader.exec_module(g)

# tests simulate market hours (guard refuses to place orders otherwise)
g.market_open = lambda now=None: True

# eliminate sleeps inside the guard for test speed
# (only the guard module's reference is replaced; the real time module is
# untouched)
g.time = SimpleNamespace(sleep=lambda s: None)

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
        self.order_book = []  # pre-existing (e.g. pending) orders
    def get_order_list(self):
        return self.order_book
    def get_positions(self):
        return [] if self.filled else POSITIONS
    def get_fund_limits(self): return {"availabelBalance": "264000"}
    def place_order(self, body):
        self.orders.append(body)
        self.filled = True  # market orders fill immediately
        return {"orderId": 999 + len(self.orders)}
    def cancel_order(self, oid):
        return {"orderId": oid, "orderStatus": "CANCELLED"}
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
        kill_on_reentry=False, backstop=True,
        trail_arm=0, trail_lock=0.5,   # trailing OFF in harness default
        sl_order=False, sl_buffer_pct=2.0,
        fast_interval=30, fast_window=55,
        state_file=SF)
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
                                                 "alerts": {}, "locked": [],
                                                 "peaks": {}})

# ---------- 10. trailing stop (v3.1) ----------
class MockClientP(MockClient):
    """Fixed position book, independent of the fill flag."""
    def __init__(self, pos):
        super().__init__()
        self._pos = pos
    def get_positions(self):
        return [] if self.filled else self._pos

UP = [dict(POSITIONS[0], unrealizedProfit=1500.0),
      dict(POSITIONS[1], unrealizedProfit=1000.0)]    # 4 lots, MTM +2,500
FADE = [dict(POSITIONS[0], unrealizedProfit=500.0),
        dict(POSITIONS[1], unrealizedProfit=300.0)]   # same position, +800
SMALL = [dict(POSITIONS[0], unrealizedProfit=900.0),
         dict(POSITIONS[1], unrealizedProfit=600.0)]  # +1,500 (below arm)
TR = {"trail_arm": 2000, "trail_lock": 0.5}

# 10a: profit arms the trail, floor ratchets, nothing exits yet
out, mc, stt = run({"enforce": True, **TR}, client=MockClientP(UP))
check("trail arms at peak 2,500 -> floor 1,250",
      "floor Rs.1,250" in out and "TRAIL-ARMED" in out)
check("trail armed: no orders placed", len(mc.orders) == 0)
check("peak persisted in state",
      stt["peaks"].get("NIFTY|2026-09-22") == 2500)

# 10b: today's scenario — winner fades; exit at the floor, not at -8,000
out, mc, stt2 = run({"enforce": True, **TR}, client=MockClientP(FADE),
                    state=stt)
check("fade to +800 breaches floor 1,250 -> TRAIL-BREACH",
      "TRAIL-BREACH" in out)
check("trail exit placed orders (2 legs)", len(mc.orders) == 2)
check("peak held across cycles within the day",
      stt2["peaks"].get("NIFTY|2026-09-22") == 2500)
check("no lockout on profit-locking exit",
      all(l["key"] != "NIFTY|2026-09-22" for l in stt2["locked"]))

# 10c: below the arm threshold -> plain watching, base floor applies
out, mc, _ = run({"enforce": True, **TR}, client=MockClientP(SMALL))
check("below arm: base loss-limit applies, no trail",
      "loss-limit Rs.14,400" in out and "TRAIL" not in out)
check("below arm: no orders", len(mc.orders) == 0)

# 10d: peaks survive a day rollover, alerts reset
tmp = "/tmp/guard_state_test.json"
with open(tmp, "w") as f:
    json.dump({"date": "2026-09-25", "alerts": {"X": ["50"]},
               "locked": [{"key": "X"}],
               "peaks": {"NIFTY|2026-09-22": 2500}}, f)
st = g.load_state(tmp, "2026-09-26")
check("peaks survive day rollover",
      st["peaks"] == {"NIFTY|2026-09-22": 2500})
check("alerts/locks reset on new day",
      st["alerts"] == {} and st["locked"] == [])

# 10e: peaks pruned once the structure is gone
class Empty(MockClient):
    def get_positions(self):
        return []
stt3 = {"date": "2026-09-21", "alerts": {}, "locked": [],
         "peaks": {"NIFTY|2026-09-22": 2500}}
out, _, stt3 = run({}, client=Empty(), state=stt3)
check("flat -> peak memory pruned", stt3["peaks"] == {})

# 10f: zero-MTM sanity probe — legs open, P&L fields absent -> dump raw
NOUPL = [{"exchangeSegment": "NSE_FNO", "tradingSymbol": "NIFTY 23300 CE",
          "drvExpiryDate": "2026-09-29", "productType": "MARGIN",
          "netQty": -130, "costPrice": 100.0, "securityId": "9999"},
         {"exchangeSegment": "NSE_FNO", "tradingSymbol": "NIFTY 23700 CE",
          "drvExpiryDate": "2026-09-29", "productType": "MARGIN",
          "netQty": 130, "costPrice": 30.0, "securityId": "9998"}]
out, mc, stt_dbg = run({"enforce": True, **TR}, client=MockClientP(NOUPL))
check("zero-MTM debug probe fires and dumps raw leg",
      "[DEBUG]" in out and "raw first leg" in out
      and "zmtm-debug" in stt_dbg["alerts"]["NIFTY|2026-09-29"])

# ---------- 11. exit-race protections (2026-09-28 live-fire postmortem) ----------
# The 11:40 duplicate-order bug: exit orders accepted, positions book stale
# at +4s, retry re-fired the SAME stale legs and overshot one leg into an
# opposite position. Two fixes: fresh-positions retry + pending-order skip.

class MockRace(MockClient):
    """Exits accepted, but the positions book lags one fetch."""
    def __init__(self):
        super().__init__()
        self._post_fill = 0
    def get_positions(self):
        if not self.filled:
            return POSITIONS
        self._post_fill += 1
        return POSITIONS if self._post_fill == 1 else []

out, mc, stt_r = run({"enforce": True}, client=MockRace())
check("race: stale book does NOT duplicate exit orders", len(mc.orders) == 2)
check("race: lockout still recorded after clean exit",
      any(l["key"] == "NIFTY|2026-09-22" for l in stt_r["locked"]))

# pending exit order for the LONG leg (11102, exit side SELL) -> only the
# short leg (11101, BUY) gets an order; the pending one is skipped
pend = [{"securityId": "11102", "transactionType": "SELL",
         "orderStatus": "PENDING"}]
mp = MockClient()
mp.order_book = pend
out, mc, _ = run({"enforce": True}, client=mp)
check("pending exit is skipped, not stacked",
      len(mc.orders) == 1 and mc.orders[0]["transactionType"] == "BUY"
      and "[SKIP]" in out)

# a filled old order must NOT block a new exit (only PENDING states count)
filled_old = [{"securityId": "11102", "transactionType": "SELL",
               "orderStatus": "EXECUTED"}]
mf = MockClient()
mf.order_book = filled_old
out, mc, _ = run({"enforce": True}, client=mf)
check("EXECUTED (old) orders do not block exits", len(mc.orders) == 2)

# ---------- 12. adaptive polling (hot flag) ----------
# trail armed -> flag active so the every-minute poller runs full cycles
out, mc, stt = run({"trail_arm": 2000, "trail_lock": 0.5,
                    "state_file": SF},
                   state={"date": "2026-09-21", "alerts": {}, "locked": [],
                          "peaks": {"NIFTY|2026-09-22": 5000}})
check("armed trailing stop writes the hot flag", g.hot_flag_active(SF))

# flat -> flag cleared, poller idles
flat = MockClient(); flat.filled = True
out, mc, stt = run({"trail_arm": 2000, "state_file": SF}, client=flat)
check("flat market clears the hot flag", not g.hot_flag_active(SF))

# trailing off (peak below arm) -> not armed
out, mc, stt = run({"trail_arm": 2000, "state_file": SF},
                   state={"date": "2026-09-21", "alerts": {}, "locked": [],
                          "peaks": {"NIFTY|2026-09-22": 500}})
check("open position flags fast polling (no trail needed)",
      g.hot_flag_active(SF))

# stale flag (guard died while armed) -> ignored
p = g.hot_flag_path(SF)
with open(p, "w") as f:
    json.dump({"armed": True, "keys": ["NIFTY|2026-09-22"],
               "updated": "2020-01-01T09:00:00+05:30"}, f)
check("stale hot flag is ignored", not g.hot_flag_active(SF))

# fresh flag -> active
with open(p, "w") as f:
    json.dump({"armed": True, "keys": ["NIFTY|2026-09-22"],
               "updated": g.datetime.now(g.IST).isoformat()}, f)
check("fresh hot flag is active", g.hot_flag_active(SF))

# missing flag -> not active (fast mode must make no API call)
g.write_hot_flag(SF, [])
os.remove(p)
check("missing flag -> inactive", not g.hot_flag_active(SF))

# ---------- 13. resting stop-loss orders (SL-L) ----------
POS2 = [
    {"exchangeSegment": "NSE_FNO", "tradingSymbol": "NIFTY 23300 CE",
     "underlying": "NIFTY", "drvExpiryDate": "2026-09-22",
     "productType": "MARGIN", "netQty": -260, "costPrice": 100.0,
     "lastPrice": 80.0, "securityId": "22201"},
    {"exchangeSegment": "NSE_FNO", "tradingSymbol": "NIFTY 23700 CE",
     "underlying": "NIFTY", "drvExpiryDate": "2026-09-22",
     "productType": "MARGIN", "netQty": 260, "costPrice": 30.0,
     "lastPrice": 25.0, "securityId": "22202"},
]
st2 = g.group_structures(POS2, 65)["NIFTY|2026-09-22"]
# short leg -260, cost 100, last 80; other leg 260*(25-30) = -1300
# solve -260*(X-100) - 1300 = -7200  ->  X = 122.69
trig = g.stop_trigger_price(st2, st2["legs"][0], 7200)
check("stop trigger solves for the loss limit", abs(trig - 122.69) < 0.05)
check("long leg gets no stop",
      g.stop_trigger_price(st2, st2["legs"][1], 7200) is None)
sb = g.sl_order_body("1000191656", st2["legs"][0], trig, 2.0)
check("SL body: BUY STOP_LOSS with trigger + buffered limit",
      sb["transactionType"] == "BUY" and sb["orderType"] == "STOP_LOSS"
      and abs(sb["triggerPrice"] - 122.69) < 0.05
      and abs(sb["price"] - 125.14) < 0.05
      and sb["quantity"] == 260 and sb["productType"] == "MARGIN")

class MockSL(MockClient):
    def __init__(self, flat=False):
        super().__init__(); self.cancels = []; self.flat = flat
    def get_positions(self): return [] if self.flat else POS2
    def cancel_order(self, oid):
        self.cancels.append(oid)
        return {"orderId": oid, "orderStatus": "CANCELLED"}

mc = MockSL()
out, mc, stt = run({"sl_order": True, "state_file": SF}, client=mc)
check("resting stop placed on the short leg only",
      len(mc.orders) == 1 and mc.orders[0]["orderType"] == "STOP_LOSS"
      and mc.orders[0]["securityId"] == "22201")
check("SL order recorded in state",
      "22201" in stt.get("sl_orders", {}).get("NIFTY|2026-09-22", {}))

out2, mc, stt = run({"sl_order": True, "state_file": SF}, client=mc, state=stt)
check("stop is not re-placed on the next cycle", len(mc.orders) == 1)

mcflat = MockSL(flat=True)
out3, mcflat, stt = run({"sl_order": True, "state_file": SF},
                        client=mcflat, state=stt)
check("resting stops cancelled when the book goes flat",
      len(mcflat.cancels) == 1
      and "NIFTY|2026-09-22" not in stt.get("sl_orders", {}))

# pending-exit skip must ignore our resting stops but keep counting markets
mb = MockClient()
mb.order_book = [{"securityId": "22201", "transactionType": "BUY",
                  "orderStatus": "PENDING", "orderType": "STOP_LOSS"}]
check("pending-exit skip ignores resting stops",
      ("22201", "BUY") not in g._pending_exit_orders(mb))
mb2 = MockClient()
mb2.order_book = [{"securityId": "22201", "transactionType": "BUY",
                   "orderStatus": "PENDING", "orderType": "MARKET"}]
check("pending-exit skip still counts market exits",
      ("22201", "BUY") in g._pending_exit_orders(mb2))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
