#!/usr/bin/env python3
"""
Dhan F&O P&L Guard
==================
Polls the Dhan order book at a regular interval. As soon as it detects that
at least one F&O order has been placed (TRADED or PENDING) during the current
trading session, it configures a P&L-based auto-exit with the user's desired
profit and loss limits.

Kill switch is NOT activated — only position auto-exit on hitting the thresholds.

Usage:
    python dhan_fno_pnl_guard.py --profit 1500 --loss 500

    # Custom poll interval (seconds)
    python dhan_fno_pnl_guard.py --profit 2000 --loss 800 --interval 30

    # Apply to both INTRADAY and DELIVERY (CNC) products
    python dhan_fno_pnl_guard.py --profit 1500 --loss 500 --products INTRADAY DELIVERY

Requirements:
    pip install dhanhq pyotp python-dotenv requests

Authentication:
    Uses src/auth.py from the dhan-trading-bot repo for automatic token
    generation via PIN + TOTP. No manual DHAN_ACCESS_TOKEN needed.
    Requires DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET in .env.
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
                method,
                url,
                headers=self.headers,
                json=json_body,
                params=params,
                timeout=30,
            )
            # Dhan returns 200 on success; auth errors come back as 401/403
            if r.status_code in (401, 403):
                print(f"\n[AUTH ERROR] HTTP {r.status_code}: {r.text.strip()}")
                print("Your access token may have expired. Regenerate it from web.dhan.co.")
                sys.exit(1)
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"\n[NETWORK ERROR] {e}")
            return {"status": "failure", "remarks": str(e)}

    # -- Trader's Control endpoints ----------------------------------------

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

    # -- Helpers -----------------------------------------------------------

    def get_fund_limits(self):
        return self._request("GET", "/fundlimit")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def check_token_valid(client):
    """Verify the access token works before entering the poll loop."""
    resp = client.get_fund_limits()
    # Dhan returns fund limits as a flat dict (no status/data wrapper).
    # Presence of 'dhanClientId' or 'availabelBalance' means auth worked.
    if isinstance(resp, dict) and (
        "dhanClientId" in resp or "availabelBalance" in resp
    ):
        balance = resp.get("availabelBalance", "N/A")
        print(f"[OK] Token valid. Available balance: ₹{balance}")
        return True
    else:
        print(f"[ERROR] Token validation failed: {resp}")
        return False


def has_fno_order(order_list):
    """
    Check whether any F&O order exists in the current order book.

    The Dhan order book returns a list of order dicts. Each has an
    'exchangeSegment' field. We look for any F&O segment, regardless
    of order status (TRADED, PENDING, TRANSIT, etc.).
    """
    if not order_list:
        return False, None

    # The response may be a plain list or nested under a key
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
    print(f"  Max Profit : ₹{profit_value}")
    print(f"  Max Loss   : ₹{loss_value}")
    print(f"  Products   : {', '.join(product_types)}")
    print(f"  Kill Switch: DISABLED")
    print("=" * 60 + "\n")

    resp = client.set_pnl_exit(
        profit_value=profit_value,
        loss_value=loss_value,
        product_types=product_types,
        enable_kill_switch=False,  # NEVER activate kill switch
    )

    # Dhan returns the P&L exit config directly (no status/data wrapper).
    # An error response contains 'errorCode' / 'errorMessage'.
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
        print(f"  Profit limit  : ₹{resp.get('profit', 'N/A')}")
        print(f"  Loss limit    : ₹{resp.get('loss', 'N/A')}")
        print(f"  Product types : {resp.get('productType', 'N/A')}")
        print(f"  Kill switch   : {resp.get('enableKillSwitch', 'N/A')}")
        return True
    else:
        print(f"[WARNING] Could not verify P&L exit config: {resp}")
        return False


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n\n[STOP] Received interrupt signal. Shutting down gracefully...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Poll for F&O trades and auto-set P&L exit limits on Dhan."
    )
    parser.add_argument(
        "--profit",
        type=float,
        required=True,
        help="Max profit (₹) at which positions are auto-exited.",
    )
    parser.add_argument(
        "--loss",
        type=float,
        required=True,
        help="Max loss (₹) at which positions are auto-exited.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Polling interval in seconds (default: 15).",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["INTRADAY"],
        choices=["INTRADAY", "DELIVERY"],
        help="Product types to cover (default: INTRADAY). "
        "Pass 'INTRADAY DELIVERY' for both.",
    )
    args = parser.parse_args()

    # --- Authenticate ---
    client_id = os.environ.get("DHAN_CLIENT_ID")
    if not client_id:
        print("ERROR: DHAN_CLIENT_ID not set in .env or environment.")
        sys.exit(1)

    if get_access_token:
        # Use auto token generation from src/auth.py (PIN + TOTP)
        try:
            token = get_access_token()
        except Exception as e:
            print(f"[ERROR] Failed to generate access token: {e}")
            sys.exit(1)
    else:
        # Fallback: manual token from environment
        token = os.environ.get("DHAN_ACCESS_TOKEN")
        if not token:
            print("ERROR: DHAN_ACCESS_TOKEN not set and src/auth.py not available.")
            print("       Either install with src/auth.py or export DHAN_ACCESS_TOKEN manually.")
            sys.exit(1)

    print("=" * 60)
    print("  DHAN F&O P&L GUARD")
    print("=" * 60)
    print(f"  Client ID    : {client_id}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Profit limit : ₹{args.profit}")
    print(f"  Loss limit   : ₹{args.loss}")
    print(f"  Products     : {', '.join(args.products)}")
    print(f"  Kill switch  : DISABLED")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    client = DhanClient(token, client_id)

    # --- Validate token ---
    if not check_token_valid(client):
        sys.exit(1)

    # --- Check if P&L exit is already configured ---
    print("[CHECK] Checking if P&L exit is already active...")
    existing = client.get_pnl_exit()
    if isinstance(existing, dict) and "errorCode" not in existing:
        if existing.get("pnlExitStatus") == "ACTIVE":
            print(f"[INFO] P&L exit is already ACTIVE for today:")
            print(f"       Profit: ₹{existing.get('profit', 'N/A')} | "
                  f"Loss: ₹{existing.get('loss', 'N/A')}")
            print("       Skipping configuration. If you want to update it, "
                  "stop this script, delete the existing exit, and re-run.\n")

    # --- Poll loop ---
    print(f"[POLL] Watching for F&O orders every {args.interval}s...")
    print("       Press Ctrl+C to stop.\n")

    pnl_configured = False
    poll_count = 0

    while _running:
        poll_count += 1
        ts = datetime.now().strftime("%H:%M:%S")

        try:
            order_resp = client.get_order_list()
        except Exception as e:
            print(f"[{ts}] ERROR fetching orders: {e}")
            time.sleep(args.interval)
            continue

        # Dhan returns the order book as a list directly.
        # An error response is a dict with 'errorCode'.
        if isinstance(order_resp, dict) and "errorCode" in order_resp:
            print(f"[{ts}] Order book fetch failed: "
                  f"{order_resp.get('errorMessage', 'unknown error')}")
            time.sleep(args.interval)
            continue

        orders = order_resp if isinstance(order_resp, list) else []

        found, order_info = has_fno_order(orders)

        if found and not pnl_configured:
            print(f"[{ts}] *** F&O ORDER DETECTED ***")
            print(f"       Symbol   : {order_info['symbol']}")
            print(f"       Segment : {order_info['exchangeSegment']}")
            print(f"       Type     : {order_info['transactionType']}")
            print(f"       Status   : {order_info['status']}")

            success = configure_pnl_exit(
                client,
                profit_value=args.profit,
                loss_value=args.loss,
                product_types=args.products,
            )

            if success:
                verify_pnl_exit(client)
                pnl_configured = True
                print(f"\n[{ts}] P&L guard is now ACTIVE. "
                      "Continuing to poll for any new F&O orders...")
            else:
                print(f"[{ts}] Will retry on next poll cycle.")
        elif found and pnl_configured:
            # P&L already set; just note the order quietly
            print(f"[{ts}] F&O order present ({order_info['symbol']}, "
                  f"{order_info['status']}). P&L exit already configured.")
        else:
            print(f"[{ts}] Poll #{poll_count}: No F&O orders found.")

        # Sleep in small increments so Ctrl+C responds quickly
        slept = 0
        while _running and slept < args.interval:
            time.sleep(1)
            slept += 1

    print("\n[DONE] F&O P&L Guard stopped.")


if __name__ == "__main__":
    main()
