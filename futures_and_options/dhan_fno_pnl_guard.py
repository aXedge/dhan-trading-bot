#!/usr/bin/env python3
"""
Dhan F&O P&L Guard (per-lot scaled)
===================================

Polls the Dhan order book and open positions. When F&O activity is detected,
configures a P&L-based auto-exit with profit and loss limits.

Two sizing modes:
  FIXED  : --profit / --loss absolute rupee values (original behaviour)
  PER-LOT: --per-lot — limits scale with the number of spread lots actually
           open (counted from live positions), so the guard is correctly
           sized whether the algos run 1, 2 or 4 lots.

Per-lot lot counting: positions are grouped by expiry; within each expiry,
lots = max(|netQty|) / lot-size. This is correct for equal-legged spreads:
  - 2-lot vertical (2 legs)  -> each leg qty = 2 x lot -> max = 2 lots
  - 2-lot iron fly (4 legs)  -> each leg qty = 2 x lot -> max = 2 lots
  - 4-lot vertical (doubled) -> each leg qty = 4 x lot -> max = 4 lots

If the algo adds lots intraday, the guard re-scales on the next cron cycle
(re-configures whenever computed levels differ from the active ones).

Kill switch is NOT activated — only position auto-exit on hitting thresholds.

Usage:
    # Fixed mode (cron)
    python dhan_fno_pnl_guard.py --profit 9000 --loss 7200 --once

    # Per-lot mode (recommended): 4500/3600 per lot, auto-scaled
    python dhan_fno_pnl_guard.py --per-lot --once

    # Per-lot with custom constants / lot size
    python dhan_fno_pnl_guard.py --per-lot --per-lot-profit 4500 \
        --per-lot-loss 3600 --lot-size 65 --once

    # Loop mode (interactive): poll every 15 seconds
    python dhan_fno_pnl_guard.py --per-lot

Cron usage (single-pass, every 10 minutes during market hours):
    */10 9-15 * * 1-5 cd /home/$USER/dhan-trading-bot && \
    flock -n /tmp/fno_guard.lock timeout 120 \
    venv/bin/python futures_and_options/dhan_fno_pnl_guard.py --per-lot \
    --once --products DELIVERY >> logs/fno_guard.log 2>&1

Authentication:
    Uses src/auth.py (PIN + TOTP). Requires DHAN_CLIENT_ID, DHAN_PIN,
    DHAN_TOTP_SECRET in .env.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

# Add project root (parent of futures_and_options/) to path so we can import src.auth
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
    # Fallback: allow standalone use with DHAN_ACCESS_TOKEN env var
    get_access_token = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "https://api.dhan.co/v2"

# F&O exchange segments (as used in Dhan order book / positions responses)
FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"}

# Default per-lot guard constants (tuned from the Sep 2026 trade history:
# current 9000/7200 fixed == 2 lots x 4500/3600; a 400pt NIFTY spread at
# lot 65 is Rs.26,000 max value per lot, so loss/lot = ~14% of structure max)
DEFAULT_PER_LOT_PROFIT = 4500
DEFAULT_PER_LOT_LOSS = 3600
DEFAULT_LOT_SIZE = 65  # NIFTY lot size — update if the exchange revises it


# ---------------------------------------------------------------------------
# Dhan client bootstrap (lightweight — no SDK helper dependency)
# ---------------------------------------------------------------------------
class DhanClient:
    """Thin wrapper around the DhanHQ v2 REST API using requests."""

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
                print(f"\n[AUTH ERROR] HTTP {r.status_code}: {r.text.strip()}")
                print("Your access token may have expired. Regenerate it from web.dhan.co.")
                sys.exit(1)
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"\n[NETWORK ERROR] {e}")
            return {"status": "failure", "remarks": str(e)}

    def get_order_list(self):
        return self._request("GET", "/orders")

    def get_positions(self):
        return self._request("GET", "/positions")

    def set_pnl_exit(self, profit_value, loss_value, product_types,
                     enable_kill_switch=False):
        # NOTE: Dhan rejects positive lossValue ("Loss Amount Cannot Be
        # Greater Than Zero") — always send the loss as a negative number.
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

    def stop_pnl_exit(self):
        return self._request("DELETE", "/pnlExit")

    def get_fund_limits(self):
        return self._request("GET", "/fundlimit")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def check_token_valid(client):
    """Verify the access token works before entering the poll loop."""
    resp = client.get_fund_limits()
    if isinstance(resp, dict) and (
        "dhanClientId" in resp or "availabelBalance" in resp
    ):
        balance = resp.get("availabelBalance", "N/A")
        print(f"[OK] Token valid. Available balance: Rs.{balance}")
        return True
    else:
        print(f"[ERROR] Token validation failed: {resp}")
        return False


def has_fno_order(order_list):
    """Check whether any F&O order exists in the current order book."""
    if not order_list:
        return False, None

    orders = order_list
    if isinstance(order_list, dict):
        orders = order_list.get("data", order_list)

    if not isinstance(orders, list):
        return False, None

    for order in orders:
        seg = order.get("exchangeSegment", "")
        if seg in FNO_SEGMENTS:
            symbol = order.get("tradingSymbol", order.get("securityId", "unknown"))
            status = order.get("orderStatus", "UNKNOWN")
            txn_type = order.get("transactionType", "")
            return True, {
                "symbol": symbol,
                "status": status,
                "transactionType": txn_type,
                "exchangeSegment": seg,
            }
    return False, None


def count_fno_lots(positions, lot_size):
    """
    Estimate open spread lots from the positions book.

    Groups F&O positions by expiry; within each expiry,
    lots = max(|netQty|) / lot_size.

    Returns (total_lots, {expiry: lots}).
    """
    if not isinstance(positions, list):
        return 0, {}

    by_expiry = {}
    for p in positions:
        seg = p.get("exchangeSegment", "")
        if seg not in FNO_SEGMENTS:
            continue
        net = p.get("netQty", 0)
        try:
            net = abs(int(net))
        except (TypeError, ValueError):
            continue
        if net == 0:
            continue
        expiry = str(p.get("drvExpiryDate", "UNKNOWN"))
        by_expiry.setdefault(expiry, []).append(net)

    total, detail = 0, {}
    for exp, qtys in by_expiry.items():
        lots = max(qtys) // lot_size
        detail[exp] = lots
        total += lots
    return total, detail


def configure_pnl_exit(client, profit_value, loss_value, product_types):
    """Set the P&L-based exit. Kill switch is always disabled."""
    print("\n" + "=" * 60)
    print("  CONFIGURING P&L-BASED AUTO-EXIT")
    print("=" * 60)
    print(f"  Max Profit : Rs.{profit_value:,.0f}")
    print(f"  Max Loss   : Rs.{loss_value:,.0f}")
    print(f"  Products   : {', '.join(product_types)}")
    print("  Kill Switch: DISABLED")
    print("=" * 60 + "\n")

    resp = client.set_pnl_exit(
        profit_value=profit_value,
        loss_value=loss_value,
        product_types=product_types,
        enable_kill_switch=False,
    )

    if isinstance(resp, dict) and "errorCode" not in resp:
        pnl_status = resp.get("pnlExitStatus", resp)
        print(f"[SUCCESS] P&L exit configured. Status: {pnl_status}")
        return True
    else:
        print("[FAILED] Could not configure P&L exit.")
        print(f"  Response: {json.dumps(resp, indent=2)}")
        return False


def active_exit_levels(client):
    """Return (profit, loss) of the currently ACTIVE pnlExit, or None."""
    resp = client.get_pnl_exit()
    if isinstance(resp, dict) and "errorCode" not in resp:
        if resp.get("pnlExitStatus") == "ACTIVE":
            try:
                return (abs(float(resp.get("profit", 0))),
                        abs(float(resp.get("loss", 0))))
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Single-pass check (for cron)
# ---------------------------------------------------------------------------
def check_once(client, args):
    """
    Single-pass check for cron. In per-lot mode, fetches open positions,
    computes scaled limits, and (re)configures the exit whenever the computed
    levels differ from what is currently active.
    """
    ts = datetime.now().strftime("%H:%M:%S")

    # ---------------- per-lot sizing ----------------
    if args.per_lot:
        try:
            pos_resp = client.get_positions()
        except Exception as e:
            print(f"[{ts}] ERROR fetching positions: {e}")
            return False

        positions = pos_resp if isinstance(pos_resp, list) else \
            pos_resp.get("data", []) if isinstance(pos_resp, dict) else []
        lots, detail = count_fno_lots(positions, args.lot_size)

        if lots <= 0:
            # fall back to the order-book check: maybe orders exist but
            # positions are already flat — nothing to guard either way
            print(f"[{ts}] No open F&O lots detected. Nothing to guard.")
            return False

        profit_value = args.per_lot_profit * lots
        loss_value = args.per_lot_loss * lots
        exp_str = ", ".join(f"{k}: {v} lot(s)" for k, v in sorted(detail.items()))
        print(f"[{ts}] OPEN F&O: {lots} spread lot(s) [{exp_str}]")
        print(f"[{ts}] Scaled limits: profit Rs.{profit_value:,.0f} / "
              f"loss Rs.{loss_value:,.0f} "
              f"(Rs.{args.per_lot_profit}/{args.per_lot_loss} per lot)")

        # re-scale if active levels differ (e.g. algo added lots intraday)
        active = active_exit_levels(client)
        if active and abs(active[0] - profit_value) < 1 \
                and abs(active[1] - loss_value) < 1:
            print(f"[{ts}] P&L exit already ACTIVE at correct scaled levels "
                  f"(profit Rs.{active[0]:,.0f} / loss Rs.{active[1]:,.0f}). "
                  f"Nothing to do.")
            return True

        success = configure_pnl_exit(client, profit_value, loss_value,
                                      args.products)
        if success:
            resp = client.get_pnl_exit()
            print(f"[VERIFY] Current config: "
                  f"{resp.get('pnlExitStatus')} | "
                  f"profit={resp.get('profit')} | loss={resp.get('loss')}")
        else:
            print(f"[{ts}] Will retry on next cron cycle.")
        return success

    # ---------------- fixed mode (original behaviour) ----------------
    # Check if P&L exit is already configured
    existing = client.get_pnl_exit()
    if isinstance(existing, dict) and "errorCode" not in existing:
        if existing.get("pnlExitStatus") == "ACTIVE":
            print(f"[{ts}] P&L exit already ACTIVE: "
                  f"Profit=Rs.{existing.get('profit', 'N/A')} "
                  f"Loss=Rs.{existing.get('loss', 'N/A')}")
            return True

    try:
        order_resp = client.get_order_list()
    except Exception as e:
        print(f"[{ts}] ERROR fetching orders: {e}")
        return False

    if isinstance(order_resp, dict) and "errorCode" in order_resp:
        print(f"[{ts}] Order book fetch failed: "
              f"{order_resp.get('errorMessage', 'unknown')}")
        return False

    orders = order_resp if isinstance(order_resp, list) else []
    found, order_info = has_fno_order(orders)

    if found:
        print(f"[{ts}] F&O ORDER DETECTED: {order_info['symbol']} "
              f"({order_info['status']})")
        success = configure_pnl_exit(client, args.profit, args.loss,
                                      args.products)
        if success:
            return True
        else:
            print(f"[{ts}] Will retry on next cron cycle.")
            return False
    else:
        print(f"[{ts}] No F&O orders found. P&L exit not needed.")
        return False


# ---------------------------------------------------------------------------
# Graceful shutdown (for loop mode)
# ---------------------------------------------------------------------------
_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n\n[STOP] Received interrupt signal. Shutting down gracefully...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Loop mode (interactive)
# ---------------------------------------------------------------------------
def run_guard_loop(client, args, poll_interval=15):
    """Poll for F&O positions/orders in a continuous loop (interactive only)."""
    print(f"\n[POLL] Watching for F&O activity every {poll_interval}s...")
    print("       Press Ctrl+C to stop.\n")

    while _running:
        check_once(client, args)
        slept = 0
        while _running and slept < poll_interval:
            time.sleep(1)
            slept += 1

    print("\n[DONE] F&O P&L Guard stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dhan F&O P&L Guard — auto-sets P&L exit limits, optionally\n"
            "scaled by the number of spread lots actually open.\n\n"
            "Modes:\n"
            "  --per-lot         Scale limits by open spread lots (recommended)\n"
            "  fixed (default)   Use absolute --profit/--loss values\n\n"
            "Examples:\n"
            "  python dhan_fno_pnl_guard.py --per-lot --once            # cron\n"
            "  python dhan_fno_pnl_guard.py --per-lot                    # interactive\n"
            "  python dhan_fno_pnl_guard.py --profit 9000 --loss 7200 --once\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profit", type=float,
        default=float(os.getenv("FNO_PROFIT_LIMIT", 0)),
        help="FIXED mode: max profit (Rs.). Default: FNO_PROFIT_LIMIT from .env.",
    )
    parser.add_argument(
        "--loss", type=float,
        default=float(os.getenv("FNO_LOSS_LIMIT", 0)),
        help="FIXED mode: max loss (Rs.). Default: FNO_LOSS_LIMIT from .env.",
    )
    parser.add_argument(
        "--per-lot", action="store_true",
        help="Scale limits by open spread lots (counted from live positions).",
    )
    parser.add_argument(
        "--per-lot-profit", type=float, default=DEFAULT_PER_LOT_PROFIT,
        help=f"Per-lot profit limit (default: {DEFAULT_PER_LOT_PROFIT}).",
    )
    parser.add_argument(
        "--per-lot-loss", type=float, default=DEFAULT_PER_LOT_LOSS,
        help=f"Per-lot loss limit (default: {DEFAULT_PER_LOT_LOSS}).",
    )
    parser.add_argument(
        "--lot-size", type=int, default=DEFAULT_LOT_SIZE,
        help=f"Futures/options lot size (default: {DEFAULT_LOT_SIZE}, NIFTY).",
    )
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Polling interval in seconds for loop mode (default: 15).",
    )
    parser.add_argument(
        "--products", nargs="+", default=["INTRADAY"],
        choices=["INTRADAY", "DELIVERY"],
        help="Product types to cover (default: INTRADAY). "
             "Pass 'INTRADAY DELIVERY' for both.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Single-pass mode: check once and exit (for cron).",
    )
    args = parser.parse_args()

    # Validate limits (per-lot mode doesn't need fixed values)
    if not args.per_lot and (args.profit <= 0 or args.loss <= 0):
        print("ERROR: Profit and loss limits must be positive numbers.")
        print("       Pass --profit and --loss (or use --per-lot), or set "
              "FNO_PROFIT_LIMIT and FNO_LOSS_LIMIT in .env")
        sys.exit(1)
    if args.per_lot and (args.per_lot_profit <= 0 or args.per_lot_loss <= 0
                         or args.lot_size <= 0):
        print("ERROR: --per-lot-profit, --per-lot-loss and --lot-size "
              "must be positive.")
        sys.exit(1)

    # --- Authenticate ---
    client_id = os.environ.get("DHAN_CLIENT_ID")
    if not client_id:
        print("ERROR: DHAN_CLIENT_ID not set in .env or environment.")
        sys.exit(1)

    if get_access_token:
        try:
            token = get_access_token()
        except Exception as e:
            print(f"[ERROR] Failed to generate access token: {e}")
            sys.exit(1)
    else:
        token = os.environ.get("DHAN_ACCESS_TOKEN")
        if not token:
            print("ERROR: DHAN_ACCESS_TOKEN not set and src/auth.py not available.")
            sys.exit(1)

    print("=" * 60)
    print("  DHAN F&O P&L GUARD")
    print("=" * 60)
    print(f"  Client ID    : {client_id}")
    print(f"  Mode         : {'SINGLE-PASS (cron)' if args.once else 'LOOP (interactive)'}")
    if args.per_lot:
        print(f"  Sizing       : PER-LOT "
              f"(profit Rs.{args.per_lot_profit:,.0f}/lot, "
              f"loss Rs.{args.per_lot_loss:,.0f}/lot, lot {args.lot_size})")
    else:
        print(f"  Profit limit : Rs.{args.profit:,.0f}")
        print(f"  Loss limit   : Rs.{args.loss:,.0f}")
    print(f"  Products     : {', '.join(args.products)}")
    print("  Kill switch  : DISABLED")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = DhanClient(token, client_id)

    if not check_token_valid(client):
        sys.exit(1)

    if args.once:
        check_once(client, args)
        print("\n[DONE] Single-pass check complete, exiting.")
    else:
        run_guard_loop(client, args, poll_interval=args.interval)


if __name__ == "__main__":
    main()
