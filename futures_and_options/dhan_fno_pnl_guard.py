#!/usr/bin/env python3
"""
Dhan F&O Structure Guard (v3)
=============================

WHY v3 EXISTS (post-mortem 2026-09-21)
--------------------------------------
Dhan's P&L-based exit (/pnlExit) only supports productType INTRADAY and
DELIVERY. Carry-forward F&O positions (productType MARGIN — i.e. anything
held overnight) are NOT covered by that API. On 2026-09-21 a 4-lot bear
call spread (MARGIN) ran ~Rs.31k underwater while a Rs.14,400 loss trigger
sat "configured" and never fired, because DELIVERY (CNC) cannot be attached
to a MARGIN position. The guard logged SUCCESS for four hours.

WHAT v3 DOES DIFFERENTLY
------------------------
1. Computes each structure's unrealized P&L directly from /positions
   (position-level, not day-level — it sees the full life of the position).
2. Enforces its own exits: when a structure crosses its per-lot threshold,
   the guard places the exit orders itself (market, per leg) and verifies
   the fills. It does not outsource protection to Dhan's pnlExit.
3. Same-day re-entry lockout: a structure exited at a loss is locked for
   the rest of the day; if a new position appears on that underlying+expiry,
   the guard alerts (and with --kill-on-reentry, exits it and activates the
   kill switch — Stratzy cannot "double down" unnoticed again).
4. Warning alerts at 50% / 75% of the loss threshold so the day never
   goes silent again.
5. TRAILING STOP: each structure's best MTM ("peak") is tracked in the
   state file. Once the peak reaches --trail-arm (default Rs.2,000), the
   effective loss floor ratchets up to --trail-lock x peak (default 50%)
   and never comes down — a structure that has shown real profit can no
   longer round-trip into a full loss. Peaks survive across days for
   carried positions and are forgotten when the structure closes.
6. Keeps a /pnlExit backstop for INTRADAY (MIS) F&O trades only — those
   ARE covered by Dhan's API. The DELIVERY product type is dropped: it
   guarded nothing.
7. TOTP retry: 3 attempts, 20s apart (5 auth failures on 2026-09-21 were
   5 blind cycles).

MODES
-----
  --watch    (default)  Monitor, log, alert. NO orders are ever placed.
  --enforce             Monitor + place exit orders when thresholds breach.
                        Re-run the script with --enforce on the command
                        line as the explicit go-ahead.
  --dry-run             Print the exact orders that WOULD be placed. No orders.

Usage:
    # WATCH (safe — start here, verify a few days of logs)
    python futures_and_options/dhan_fno_pnl_guard.py --watch --once

    # ENFORCE (live — places real exit orders on breach)
    python futures_and_options/dhan_fno_pnl_guard.py --enforce --once

    # Preview orders without placing them
    python futures_and_options/dhan_fno_pnl_guard.py --enforce --dry-run --once

    # Interactive loop (e.g. 60s poll while watching a position)
    python futures_and_options/dhan_fno_pnl_guard.py --watch

Cron (every 5 minutes during market hours):
    */5 9-15 * * 1-5 flock -n /tmp/fno_guard.lock bash -c \
      'cd /home/kay22_ind/dhan-trading-bot && source venv/bin/activate && \
       timeout 120 python futures_and_options/dhan_fno_pnl_guard.py \
       --enforce --once >> logs/fno_guard.log 2>&1'

State: data/fno_guard_state.json (alert levels sent + lockouts, resets daily)

Authentication: src/auth.py (PIN + TOTP). Telegram alerts use
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the environment (never hardcoded).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta

# Add project root (parent of futures_and_options/) to path for src.auth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("ERROR: requests package not installed. Run: pip install requests")
    sys.exit(1)

try:
    from src.auth import get_access_token
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    get_access_token = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "https://api.dhan.co/v2"
FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"}

DEFAULT_PER_LOT_PROFIT = 4500
DEFAULT_PER_LOT_LOSS = 3600
DEFAULT_LOT_SIZE = 65  # NIFTY

IST = timezone(timedelta(hours=5, minutes=30))

# Position payload keys, defensively tried (Dhan v2 field spellings)
UPL_KEYS = ["unrealizedProfit", "unRealizedProfit", "unrealized_profit"]
LAST_KEYS = ["lastPrice", "marketPrice", "ltp", "lastTradedPrice"]


# ---------------------------------------------------------------------------
# Auth with TOTP retry (2026-09-21: five Invalid-TOTP cycles = blind windows)
# ---------------------------------------------------------------------------
def get_token_with_retry(attempts=3, delay=20):
    last_err = None
    for i in range(attempts):
        try:
            return get_access_token()
        except Exception as e:
            last_err = e
            print(f"[AUTH] attempt {i + 1}/{attempts} failed: {e}")
            if i < attempts - 1:
                time.sleep(delay)  # TOTP windows roll every 30s
    return None


# ---------------------------------------------------------------------------
# Telegram (credentials only from env)
# ---------------------------------------------------------------------------
def telegram_send(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        return r.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"[TELEGRAM] send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Dhan client
# ---------------------------------------------------------------------------
class DhanClient:
    def __init__(self, access_token, client_id):
        self.access_token = access_token
        self.client_id = client_id
        self.base_url = API_BASE
        self.headers = {
            "access-token": access_token,
            "client-id": client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method, endpoint, json_body=None, params=None):
        url = f"{self.base_url}{endpoint}"
        try:
            r = requests.request(
                method, url, headers=self.headers,
                json=json_body, params=params, timeout=30,
            )
            if r.status_code in (401, 403):
                print(f"[AUTH ERROR] HTTP {r.status_code}: {r.text.strip()[:200]}")
                return {"errorCode": f"HTTP-{r.status_code}",
                        "errorMessage": r.text.strip()[:200]}
            try:
                return r.json()
            except ValueError:
                return {"errorCode": "NON-JSON",
                        "errorMessage": r.text[:200]}
        except requests.exceptions.RequestException as e:
            return {"errorCode": "NETWORK_ERROR", "errorMessage": str(e)}

    def get_positions(self):
        return self._request("GET", "/positions")

    def get_order_list(self):
        return self._request("GET", "/orders")

    def get_fund_limits(self):
        return self._request("GET", "/fundlimit")

    def place_order(self, body):
        return self._request("POST", "/orders", json_body=body)

    def set_pnl_exit(self, profit_value, loss_value, product_types,
                     enable_kill_switch=False):
        # Dhan rejects positive lossValue — always send negative.
        body = {
            "dhanClientId": self.client_id,
            "profitValue": str(abs(profit_value)),
            "lossValue": str(-abs(loss_value)),
            "productType": product_types,
            "enableKillSwitch": enable_kill_switch,
        }
        return self._request("POST", "/pnlExit", json_body=body)

    def get_pnl_exit(self):
        return self._request("GET", "/pnlExit")

    def kill_switch_activate(self):
        # NOTE: only succeeds when flat (no open positions / pending orders)
        return self._request(
            "POST", "/killswitch", params={"killSwitchStatus": "ACTIVATE"})

    def kill_switch_status(self):
        return self._request("GET", "/killswitch")


# ---------------------------------------------------------------------------
# Position -> structure grouping and MTM (pure functions, unit-testable)
# ---------------------------------------------------------------------------
def _underlying_of(position):
    for key in ("underlying", "underlyingScrip"):
        val = position.get(key)
        if val:
            return str(val)
    sym = str(position.get("tradingSymbol", "") or position.get("customSymbol", ""))
    m = re.match(r"^([A-Z&\-]+)", sym.upper())
    return m.group(1).rstrip("-") if m else sym or "UNKNOWN"


def _leg_mtm(position):
    """Unrealized P&L of one position leg (positive = profit)."""
    for key in UPL_KEYS:
        val = position.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                break
    # Fallback: netQty * (last - cost)
    try:
        net = int(position.get("netQty", 0) or 0)
        cost = float(position.get("costPrice", 0) or 0)
        last = None
        for k in LAST_KEYS:
            if position.get(k) is not None:
                last = float(position.get(k))
                break
        if last is not None:
            return net * (last - cost)
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def group_structures(positions, lot_size):
    """
    Group live F&O positions into structures (underlying + expiry).

    Returns {key: {"underlying", "expiry", "legs": [...], "lots": int,
                   "mtm": float}} — mtm is the summed unrealized P&L of all
    legs (whole-life of the position, not day-only).
    """
    if not isinstance(positions, list):
        return {}
    groups = {}
    for p in positions:
        if p.get("exchangeSegment", "") not in FNO_SEGMENTS:
            continue
        try:
            net = int(p.get("netQty", 0) or 0)
        except (TypeError, ValueError):
            continue
        if net == 0:
            continue
        expiry = str(p.get("drvExpiryDate", "UNKNOWN"))
        underlying = _underlying_of(p)
        key = f"{underlying}|{expiry}"
        g = groups.setdefault(
            key, {"underlying": underlying, "expiry": expiry,
                  "legs": [], "lots": 0, "mtm": 0.0})
        g["legs"].append(p)
        g["mtm"] += _leg_mtm(p)
    for g in groups.values():
        max_qty = max(abs(int(l.get("netQty", 0) or 0)) for l in g["legs"])
        g["lots"] = max_qty // lot_size if lot_size > 0 else 0
    return groups


def market_open(now=None):
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 30


# ---------------------------------------------------------------------------
# State (daily alert/lockout memory)
# ---------------------------------------------------------------------------
DAILY_KEYS = ("alerts", "locked")     # reset at the start of each day
PERSIST_KEYS = ("peaks",)             # survive days (trailing memory)


def state_path_default():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "fno_guard_state.json")


def load_state(path, today):
    state = {"date": today, "alerts": {}, "locked": [], "peaks": {}}
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                if saved.get("date") == today:
                    state.update({k: saved.get(k, state[k])
                                  for k in DAILY_KEYS + PERSIST_KEYS})
                else:
                    # new trading day: alert/lockout memory resets, but each
                    # structure's best MTM must survive so multi-day carried
                    # positions keep their trailing floor
                    state["peaks"] = saved.get("peaks", {}) or {}
        except (OSError, ValueError) as e:
            print(f"[STATE] could not read {path}: {e} — starting fresh")
    return state


def save_state(path, state):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        print(f"[STATE] could not save {path}: {e}")


# ---------------------------------------------------------------------------
# Exit orders
# ---------------------------------------------------------------------------
def exit_order_body(client_id, leg):
    """Build a market order that flattens one position leg."""
    net = int(leg.get("netQty", 0) or 0)
    side = "SELL" if net > 0 else "BUY"
    body = {
        "dhanClientId": client_id,
        "transactionType": side,
        "exchangeSegment": leg.get("exchangeSegment"),
        "productType": leg.get("productType", "MARGIN"),
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": str(leg.get("securityId", "")),
        "quantity": abs(net),
        "price": 0,
    }
    for opt in ("drvExpiryDate", "drvOptionType", "drvStrikePrice"):
        if leg.get(opt) is not None:
            body[opt] = leg.get(opt)
    return body


def flatten_structure(client, structure, dry_run=False):
    """
    Place market exit orders for every open leg of a structure.
    Returns list of (leg_symbol, order_response_or_None_if_dry).
    """
    results = []
    for leg in structure["legs"]:
        sym = leg.get("tradingSymbol", "?")
        body = exit_order_body(client.client_id, leg)
        print(f"    [EXIT] {sym}: {body['transactionType']} "
              f"{body['quantity']} MKT ({body['productType']})")
        if dry_run:
            results.append((sym, None))
            continue
        resp = client.place_order(body)
        ok = isinstance(resp, dict) and "orderId" in resp
        print(f"           -> {'OK orderId=' + str(resp['orderId']) if ok else 'FAILED: ' + json.dumps(resp)[:200]}")
        results.append((sym, resp))
    return results


def verify_flat(client, structure):
    """Re-fetch positions; True when all legs of the structure are flat."""
    resp = client.get_positions()
    positions = resp if isinstance(resp, list) else (
        resp.get("data", []) if isinstance(resp, dict) else [])
    for p in positions:
        if p.get("exchangeSegment", "") not in FNO_SEGMENTS:
            continue
        key = f"{_underlying_of(p)}|{str(p.get('drvExpiryDate', 'UNKNOWN'))}"
        try:
            if key == f"{structure['underlying']}|{structure['expiry']}" \
                    and int(p.get("netQty", 0) or 0) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# Core single-pass check
# ---------------------------------------------------------------------------
def check_once(client, args, state):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    today = state["date"]

    resp = client.get_positions()
    positions = resp if isinstance(resp, list) else (
        resp.get("data", []) if isinstance(resp, dict) else [])
    structures = group_structures(positions, args.lot_size)

    # forget peaks of structures that no longer exist (closed/expired):
    # a re-opened structure starts its trailing memory from scratch
    live_keys = set(structures.keys())
    for k in list(state.get("peaks", {}).keys()):
        if k not in live_keys:
            del state["peaks"][k]

    if not structures:
        print(f"[{ts}] No open F&O positions.")
        return True

    total_lots = sum(g["lots"] for g in structures.values())
    for key, g in sorted(structures.items()):
        if g["lots"] <= 0:
            print(f"[{ts}] {key}: open legs but lot count 0 (qty < lot size?) "
                  f"— watching MTM anyway")

        profit_limit = args.per_lot_profit * max(g["lots"], 1)
        loss_limit = args.per_lot_loss * max(g["lots"], 1)
        if args.profit > 0:
            profit_limit = args.profit
        if args.loss > 0:
            loss_limit = args.loss

        mtm = g["mtm"]

        # --- MTM sanity probe: open legs but MTM reads exactly 0 usually
        # means the positions payload's P&L fields didn't match our guesses
        # (unrealizedProfit / lastPrice / costPrice). Dump one raw leg,
        # once per structure per day, so field mapping can be fixed.
        if mtm == 0.0 and len(g["legs"]) > 0:
            sent_dbg = state["alerts"].setdefault(key, [])
            if "zmtm-debug" not in sent_dbg:
                sent_dbg.append("zmtm-debug")
                print(f"[DEBUG] {key}: MTM computed as 0 with "
                      f"{len(g['legs'])} open leg(s) — raw first leg:")
                print("        " + json.dumps(g["legs"][0])[:600])

        # --- trailing stop: remember the best MTM this structure has
        # shown; once it reaches --trail-arm, the loss floor ratchets up
        # to --trail-lock x peak and never comes down within the position.
        peaks = state.setdefault("peaks", {})
        peak = peaks.get(key)
        if peak is None or mtm > peak:
            peak = mtm
        peaks[key] = peak

        floor = -loss_limit
        trail_on = (args.trail_arm > 0 and peak >= args.trail_arm)
        if trail_on:
            floor = max(floor, args.trail_lock * peak)

        ratio = (abs(mtm) / loss_limit) if mtm < 0 and loss_limit else 0.0
        breach = mtm <= floor
        status = ("TRAIL-BREACH" if breach and floor > -loss_limit else
                  "BREACH" if breach else
                  "WARN75" if ratio >= 0.75 else
                  "WARN50" if ratio >= 0.50 else
                  "PROFIT-BREACH" if profit_limit and mtm >= profit_limit
                  else "TRAIL-ARMED" if trail_on else "ok")

        floor_str = (f"floor Rs.{floor:,.0f}" if trail_on
                     else f"loss-limit Rs.{-floor:,.0f}")
        print(f"[{ts}] {key}: {g['lots']} lot(s) | MTM Rs.{mtm:,.0f} | "
              f"peak Rs.{peak:,.0f} | {floor_str} "
              f"(profit-limit Rs.{profit_limit:,.0f}) -> {status}")

        sent = state["alerts"].setdefault(key, [])

        # one-shot notification when the trailing floor first arms
        if trail_on and "trail" not in sent:
            sent.append("trail")
            telegram_send(
                f"\U0001f512 Trailing stop armed: {key}\n"
                f"Peak Rs.{peak:,.0f} — loss floor is now Rs.{floor:,.0f}. "
                f"Profit is protected from here.")

        # --- warning alerts (once each per day) ---
        for level in ("50", "75"):
            if level not in sent and ratio >= float(level) / 100 \
                    and not breach:
                sent.append(level)
                telegram_send(
                    f"\u26a0 F&O guard: {key} at {level}% of loss limit\n"
                    f"MTM Rs.{mtm:,.0f} vs limit Rs.{loss_limit:,.0f}")

        # --- re-entry on a locked structure ---
        locked_keys = [lk.get("key") for lk in state["locked"]]
        if key in locked_keys:
            print(f"[{ts}] !!! RE-ENTRY on locked structure {key} !!!")
            telegram_send(
                f"\U0001f6d1 <b>RE-ENTRY on locked structure {key}</b>\n"
                f"This structure was exited at a loss today. "
                f"Current MTM Rs.{mtm:,.0f}.")
            if args.kill_on_reentry and args.enforce and market_open():
                flatten_structure(client, g, dry_run=False)
                time.sleep(4)
                if verify_flat(client, g):
                    ks = client.kill_switch_activate()
                    print(f"    [KILL SWITCH] {json.dumps(ks)[:200]}")
                    telegram_send(
                        "\U0001f6d1 Re-entered structure flattened and kill "
                        "switch activated. Manual re-arm required.")

        # --- threshold breach (base loss floor, trailing floor, profit) ---
        if breach:
            _handle_breach(client, args, state, key, g, mtm,
                           floor, ts, is_loss=True)
        elif profit_limit and mtm >= profit_limit:
            _handle_breach(client, args, state, key, g, mtm,
                           profit_limit, ts, is_loss=False)

    # --- INTRADAY backstop via Dhan's own pnlExit (MIS trades only) ---
    if args.backstop and total_lots > 0:
        profit_value = args.per_lot_profit * total_lots
        loss_value = args.per_lot_loss * total_lots
        active = client.get_pnl_exit()
        cur = (float(active.get("profit", 0) or 0),
               abs(float(active.get("loss", 0) or 0))) \
            if isinstance(active, dict) and active.get("pnlExitStatus") == "ACTIVE" \
            else None
        if cur is None or abs(cur[0] - profit_value) >= 1 \
                or abs(cur[1] - loss_value) >= 1:
            r = client.set_pnl_exit(profit_value, loss_value, ["INTRADAY"])
            ok = isinstance(r, dict) and "errorCode" not in r
            print(f"[{ts}] INTRADAY backstop (MIS only): "
                  f"{'set' if ok else 'FAILED'} "
                  f"profit Rs.{profit_value:,.0f} / loss Rs.{loss_value:,.0f}")

    return True


def _handle_breach(client, args, state, key, g, mtm, limit, ts, is_loss):
    label = "LOSS" if is_loss else "PROFIT"
    if not args.enforce:
        sent = state["alerts"].setdefault(key, [])
        if "breach" not in sent:
            sent.append("breach")
            print(f"[{ts}] *** {label} THRESHOLD BREACH ({key}: MTM "
                  f"Rs.{mtm:,.0f} vs floor Rs.{limit:,.0f}) — WATCH mode, "
                  f"no action ***")
            telegram_send(
                f"\U0001f534 <b>{label} BREACH</b> {key}\nMTM Rs.{mtm:,.0f} "
                f"vs floor Rs.{limit:,.0f}\nGuard is in WATCH mode — "
                f"no orders placed.")
        else:
            print(f"[{ts}] {label} breach continues ({key}: MTM "
                  f"Rs.{mtm:,.0f} vs floor Rs.{limit:,.0f}) — WATCH mode, "
                  f"already alerted.")
        return

    if not market_open():
        print(f"[{ts}] {label} breach outside market hours — cannot place "
              f"orders. Alerting.")
        telegram_send(
            f"\U0001f534 {label} BREACH {key} OUTSIDE MARKET HOURS — "
            f"manual action needed. MTM Rs.{mtm:,.0f}.")
        return

    print(f"[{ts}] *** {label} THRESHOLD BREACH — FLATTENING {key} ***")
    telegram_send(
        f"\U0001f534 <b>{label} BREACH — exiting {key}</b>\n"
        f"MTM Rs.{mtm:,.0f} vs floor Rs.{limit:,.0f}. "
        f"Placing exit orders.")
    flatten_structure(client, g, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[{ts}] DRY RUN — orders above were NOT placed.")
        return

    time.sleep(4)
    if not verify_flat(client, g):
        print(f"[{ts}] legs still open — retrying once")
        flatten_structure(client, g, dry_run=False)
        time.sleep(4)
        if not verify_flat(client, g):
            telegram_send(
                "\U0001f6a8 <b>GUARD FAILED TO FLATTEN "
                f"{key}</b> — manual intervention required!")
            return

    if is_loss and mtm < 0:
        state["locked"].append({
            "key": key, "time": datetime.now(IST).isoformat(), "mtm": mtm})
        telegram_send(
            f"\u2705 {key} flattened at Rs.{mtm:,.0f}. "
            f"Structure LOCKED for today (re-entry will alert"
            + (" + kill switch)." if args.kill_on_reentry else ")."))
    else:
        telegram_send(f"\u2705 {key} flattened at PROFIT Rs.{mtm:,.0f}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dhan F&O Structure Guard v3 — position-level MTM guard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true",
                      help="Monitor + alert only (default; no orders ever).")
    mode.add_argument("--enforce", action="store_true",
                      help="Place real exit orders on threshold breach.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --enforce: print orders, place nothing.")
    parser.add_argument("--per-lot-profit", type=float,
                        default=DEFAULT_PER_LOT_PROFIT)
    parser.add_argument("--per-lot-loss", type=float,
                        default=DEFAULT_PER_LOT_LOSS)
    parser.add_argument("--lot-size", type=int, default=DEFAULT_LOT_SIZE)
    parser.add_argument("--profit", type=float, default=0,
                        help="Optional absolute structure profit override.")
    parser.add_argument("--loss", type=float, default=0,
                        help="Optional absolute structure loss override.")
    parser.add_argument("--trail-arm", type=float, default=2000,
                        help="Arm the trailing floor once a structure's "
                             "peak MTM reaches this many rupees "
                             "(default 2000; 0 disables trailing).")
    parser.add_argument("--trail-lock", type=float, default=0.5,
                        help="Once armed, the loss floor rises to this "
                             "fraction of the structure's best peak MTM "
                             "(default 0.5 = lock half the peak).")
    parser.add_argument("--kill-on-reentry", action="store_true",
                        help="On re-entry into a locked structure: flatten "
                             "it and activate the kill switch.")
    parser.add_argument("--no-backstop", dest="backstop",
                        action="store_false",
                        help="Skip the INTRADAY pnlExit backstop (default on).")
    parser.add_argument("--interval", type=int, default=60,
                        help="Loop-mode poll interval seconds (default 60).")
    parser.add_argument("--once", action="store_true",
                        help="Single pass and exit (cron).")
    parser.add_argument("--state-file", default=state_path_default())
    args = parser.parse_args()

    client_id = os.environ.get("DHAN_CLIENT_ID")
    if not client_id:
        print("ERROR: DHAN_CLIENT_ID not set in .env or environment.")
        sys.exit(1)

    if get_access_token:
        token = get_token_with_retry()
        if not token:
            print("[ERROR] Could not generate access token after retries.")
            telegram_send("\U0001f6a8 F&O guard: auth failed after 3 "
                          "retries — guard is BLIND right now.")
            sys.exit(1)
    else:
        token = os.environ.get("DHAN_ACCESS_TOKEN")
        if not token:
            print("ERROR: DHAN_ACCESS_TOKEN not set and src/auth.py "
                  "not importable.")
            sys.exit(1)

    today = datetime.now(IST).strftime("%Y-%m-%d")
    state = load_state(args.state_file, today)

    print("=" * 60)
    print("  DHAN F&O STRUCTURE GUARD (v3)")
    print("=" * 60)
    print(f"  Client ID   : {client_id}")
    print(f"  Mode        : "
          f"{'ENFORCE' if args.enforce else 'WATCH'}"
          f"{' (DRY RUN)' if args.dry_run else ''}")
    print(f"  Sizing      : per-lot profit Rs.{args.per_lot_profit:,.0f} / "
          f"loss Rs.{args.per_lot_loss:,.0f} (lot {args.lot_size})")
    print(f"  Re-entry    : alert"
          + (" + flatten + kill switch" if args.kill_on_reentry else " only"))
    print(f"  Backstop    : INTRADAY pnlExit "
          f"({'on' if args.backstop else 'off'})")
    if args.trail_arm > 0:
        print(f"  Trailing    : arm at Rs.{args.trail_arm:,.0f}, "
              f"lock {args.trail_lock * 100:.0f}% of peak")
    else:
        print("  Trailing    : off")
    print(f"  Started at  : {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("=" * 60)

    client = DhanClient(token, client_id)
    funds = client.get_fund_limits()
    if isinstance(funds, dict) and "availabelBalance" in funds:
        print(f"[OK] Token valid. Available balance: "
              f"Rs.{funds.get('availabelBalance')}")
    else:
        print(f"[ERROR] Token validation failed: {json.dumps(funds)[:200]}")
        sys.exit(1)

    try:
        if args.once:
            ok = check_once(client, args, state)
            save_state(args.state_file, state)
            print("\n[DONE] Single-pass check complete, exiting.")
            sys.exit(0 if ok else 1)
        print(f"\n[LOOP] Polling every {args.interval}s. Ctrl+C to stop.\n")
        while True:
            check_once(client, args, state)
            save_state(args.state_file, state)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")


if __name__ == "__main__":
    main()
