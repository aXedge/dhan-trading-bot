#!/usr/bin/env python3
"""
Dhan F&O P&L Guard
==================

Polls the Dhan order book. When it detects that at least one F&O order has
been placed during the current trading session, it configures a P&L-based
auto-exit with the user's desired profit and loss limits.

Kill switch is NOT activated — only position auto-exit on hitting thresholds.

Usage:
    # Single-pass mode (for cron): check once and exit
    python dhan_fno_pnl_guard.py --profit 1500 --loss 500 --once

    # Loop mode (interactive): poll every 15 seconds
    python dhan_fno_pnl_guard.py --profit 1500 --loss 500

    # Custom poll interval (seconds)
    python dhan_fno_pnl_guard.py --profit 2000 --loss 800 --interval 30

    # Apply to both INTRADAY and DELIVERY (CNC) products
    python dhan_fno_pnl_guard.py --profit 1500 --loss 500 --products INTRADAY DELIVERY

    # Read limits from .env (FNO_PROFIT_LIMIT, FNO_LOSS_LIMIT)
    python dhan_fno_pnl_guard.py --once

    # Show help
    python dhan_fno_pnl_guard.py --help

Cron usage (single-pass, every 10 minutes during market hours):
    */10 9-15 * * 1-5 cd /home/$USER/dhan-trading-bot && source venv/bin/activate && \
    export PYTHONPATH=/home/$USER/dhan-trading-bot/src:$PYTHONPATH && \
    python futures_and_options/dhan_fno_pnl_guard.py --once >> logs/fno_guard.log 2>&1

Requirements:
    pip install dhanhq pyotp python-dotenv requests

Authentication:
    Uses src/auth.py from the dhan-trading-bot repo for automatic token
    generation via PIN + TOTP. No manual DHAN_ACCESS_TOKEN needed.
    Requires DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET in .env.

Configuration:
    Set FNO_PROFIT_LIMIT and FNO_LOSS_LIMIT in .env for cron usage,
    or pass --profit and --loss on the command line.
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

# F&O exchange segments (as used in Dhan order book responses)
FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"}


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

    def set_pnl_exit(self, profit_value, loss_value, product_types, enable_kill_switch=False):
        body = {
            "dhanClientId": self.client_id,
            "profitValue": str(profit_value),
            "lossValue": str(loss_value),
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
    """
    Check whether any F&O order exists in the current order book.
    Returns (found: bool, order_info: dict|None).
    """
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


def configure_pnl_exit(client, profit_value, loss_value, product_types):
    """Set the P&L-based exit. Kill switch is always disabled."""
    print("\n" + "=" * 60)
    print("  CONFIGURING P&L-BASED AUTO-EXIT")
    print("=" * 60)
    print(f"  Max Profit : Rs.{profit_value}")
    print(f"  Max Loss   : Rs.{loss_value}")
    print(f"  Products   : {', '.join(product_types)}")
    print(f"  Kill Switch: DISABLED")
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
        print(f"[FAILED] Could not configure P&L exit.")
        print(f"  Response: {json.dumps(resp, indent=2)}")
        return False


def verify_pnl_exit(client):
    """Fetch and display the currently active P&L exit config."""
    resp = client.get_pnl_exit()
    if isinstance(resp, dict) and "errorCode" not in resp:
        print("\n[VERIFICATION] Current P&L exit configuration:")
        print(f"  Status        : {resp.get('pnlExitStatus', 'N/A')}")
        print(f"  Profit limit  : Rs.{resp.get('profit', 'N/A')}")
        print(f"  Loss limit    : Rs.{resp.get('loss', 'N/A')}")
        print(f"  Product types : {resp.get('productType', 'N/A')}")
        print(f"  Kill switch   : {resp.get('enableKillSwitch', 'N/A')}")
        return True
    else:
        print(f"[WARNING] Could not verify P&L exit config: {resp}")
        return False


# ---------------------------------------------------------------------------
# Single-pass check (for cron)
# ---------------------------------------------------------------------------
def check_once(client, profit_value, loss_value, product_types):
    """
    Single-pass check: look for F&O orders once, configure P&L exit if found,
    then return. Does NOT loop. Perfect for cron.

    Returns:
        True if P&L exit was configured (or already active).
        False if no F&O orders found or configuration failed.
    """
    ts = datetime.now().strftime("%H:%M:%S")

    # Check if P&L exit is already configured
    existing = client.get_pnl_exit()
    if isinstance(existing, dict) and "errorCode" not in existing:
        if existing.get("pnlExitStatus") == "ACTIVE":
            print(f"[{ts}] P&L exit already ACTIVE: "
                  f"Profit=Rs.{existing.get('profit', 'N/A')} "
                  f"Loss=Rs.{existing.get('loss', 'N/A')}")
            return True

    # Check order book for F&O orders
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

        success = configure_pnl_exit(
            client,
            profit_value=profit_value,
            loss_value=loss_value,
            product_types=product_types,
        )

        if success:
            verify_pnl_exit(client)
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
def run_guard_loop(client, profit_limit, loss_limit, product_types, poll_interval=15):
    """Poll for F&O orders in a continuous loop (interactive mode only)."""
    pnl_configured = False
    poll_count = 0

    print(f"\n[POLL] Watching for F&O orders every {poll_interval}s...")
    print("       Press Ctrl+C to stop.\n")

    while _running:
        poll_count += 1
        ts = datetime.now().strftime("%H:%M:%S")

        try:
            order_resp = client.get_order_list()
        except Exception as e:
            print(f"[{ts}] ERROR fetching orders: {e}")
            time.sleep(poll_interval)
            continue

        if isinstance(order_resp, dict) and "errorCode" in order_resp:
            print(f"[{ts}] Order book fetch failed: "
                  f"{order_resp.get('errorMessage', 'unknown')}")
            time.sleep(poll_interval)
            continue

        orders = order_resp if isinstance(order_resp, list) else []
        found, order_info = has_fno_order(orders)

        if found and not pnl_configured:
            print(f"[{ts}] *** F&O ORDER DETECTED ***")
            print(f"       Symbol   : {order_info['symbol']}")
            print(f"       Segment  : {order_info['exchangeSegment']}")
            print(f"       Type     : {order_info['transactionType']}")
            print(f"       Status   : {order_info['status']}")

            success = configure_pnl_exit(
                client,
                profit_value=profit_limit,
                loss_value=loss_limit,
                product_types=product_types,
            )

            if success:
                verify_pnl_exit(client)
                pnl_configured = True
                print(f"\n[{ts}] P&L guard is now ACTIVE. "
                      "Continuing to poll for any new F&O orders...")
            else:
                print(f"[{ts}] Will retry on next poll cycle.")
        elif found and pnl_configured:
            print(f"[{ts}] F&O order present ({order_info['symbol']}, "
                  f"{order_info['status']}). P&L exit already configured.")
        else:
            print(f"[{ts}] Poll #{poll_count}: No F&O orders found.")

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
            "Dhan F&O P&L Guard — polls for F&O orders and auto-sets P&L exit limits.\n\n"
            "Two modes:\n"
            "  Default (loop):    Continuous polling, Ctrl+C to stop (interactive).\n"
            "  --once (single):   Check once and exit (for cron).\n\n"
            "Examples:\n"
            "  python dhan_fno_pnl_guard.py --profit 1500 --loss 500 --once  # cron mode\n"
            "  python dhan_fno_pnl_guard.py --profit 1500 --loss 500         # interactive\n"
            "  python dhan_fno_pnl_guard.py --once                           # use .env limits\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profit", type=float,
        default=float(os.getenv("FNO_PROFIT_LIMIT", 0)),
        help="Max profit (Rs.) at which positions are auto-exited. "
             "Default: FNO_PROFIT_LIMIT from .env.",
    )
    parser.add_argument(
        "--loss", type=float,
        default=float(os.getenv("FNO_LOSS_LIMIT", 0)),
        help="Max loss (Rs.) at which positions are auto-exited. "
             "Default: FNO_LOSS_LIMIT from .env.",
    )
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Polling interval in seconds for loop mode (default: 15).",
    )
    parser.add_argument(
        "--products", nargs="+",
        default=["INTRADAY"],
        choices=["INTRADAY", "DELIVERY"],
        help="Product types to cover (default: INTRADAY). "
             "Pass 'INTRADAY DELIVERY' for both.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Single-pass mode: check once and exit (for cron). "
             "Default is loop mode (interactive).",
    )
    args = parser.parse_args()

    # Validate limits
    if args.profit <= 0 or args.loss <= 0:
        print("ERROR: Profit and loss limits must be positive numbers.")
        print("       Pass --profit and --loss, or set FNO_PROFIT_LIMIT and "
              "FNO_LOSS_LIMIT in .env")
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
    if not args.once:
        print(f"  Poll interval: {args.interval}s")
    print(f"  Profit limit : Rs.{args.profit}")
    print(f"  Loss limit   : Rs.{args.loss}")
    print(f"  Products     : {', '.join(args.products)}")
    print(f"  Kill switch  : DISABLED")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = DhanClient(token, client_id)

    # --- Validate token ---
    if not check_token_valid(client):
        sys.exit(1)

    # --- Run ---
    if args.once:
        check_once(client, args.profit, args.loss, args.products)
        print("\n[DONE] Single-pass check complete, exiting.")
    else:
        # Check if P&L exit is already configured
        existing = client.get_pnl_exit()
        if isinstance(existing, dict) and "errorCode" not in existing:
            if existing.get("pnlExitStatus") == "ACTIVE":
                print(f"[INFO] P&L exit is already ACTIVE for today:")
                print(f"       Profit: Rs.{existing.get('profit', 'N/A')} | "
                      f"Loss: Rs.{existing.get('loss', 'N/A')}")
                print("       Skipping configuration. If you want to update it, "
                      "stop this script, delete the existing exit, and re-run.\n")

        run_guard_loop(
            client,
            profit_limit=args.profit,
            loss_limit=args.loss,
            product_types=args.products,
            poll_interval=args.interval,
        )


if __name__ == "__main__":
    main()
