#!/usr/bin/env python3
"""
Dhan F&O Average Profit/Loss per Trade
======================================
Fetches your F&O trade history over the last N days (default 90),
groups trades by completed round-trip (BUY + SELL of the same instrument),
calculates net P&L per round-trip after all charges, and reports:

  - Average profit per winning trade
  - Average loss per losing trade
  - Overall average P&L per trade
  - Win rate, total trades, total P&L
  - A per-symbol breakdown

Usage:
    # Default: last 90 days (approx. 3 months), fetch from API
    python dhan_fno_avg_pnl.py

    # Last 30 days
    python dhan_fno_avg_pnl.py --days 30

    # Export round-trip results to CSV
    python dhan_fno_avg_pnl.py --csv fno_pnl_report.csv

    # Save raw fetched trades to CSV for reuse by other scripts
    python dhan_fno_avg_pnl.py --data-file fno_trades.csv

    # Load raw trades from a previously saved CSV (skip API call)
    python dhan_fno_avg_pnl.py --from-file fno_trades.csv

    # Enable debug logging
    python dhan_fno_avg_pnl.py --debug

    # Show help
    python dhan_fno_avg_pnl.py --help

Requirements:
    pip install requests pyotp python-dotenv dhanhq

Authentication:
    Uses src/auth.py from the dhan-trading-bot repo for automatic token
    generation via PIN + TOTP. No manual DHAN_ACCESS_TOKEN needed.
    Requires DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET in .env.
    When using --from-file, no authentication is needed.
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

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

# F&O exchange segments
FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"}

# Dhan's trade history is paginated — 500 records per page typically.
# We keep fetching until an empty page comes back.
MAX_PAGES = 50

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("dhan_fno_pnl")


def setup_logging(debug=False):
    """Configure the root logger. DEBUG level when --debug is passed."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if debug:
        # Reduce noise from third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    logger.debug("Logging initialised at %s level", logging.getLevelName(level))


# ---------------------------------------------------------------------------
# Dhan API client
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
        logger.debug("DhanClient initialised for client_id=%s", client_id)

    def _request(self, method, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        logger.debug("HTTP %s %s params=%s", method, url, params)
        try:
            r = requests.request(method, url, headers=self.headers, params=params, timeout=30)
            logger.debug("Response: HTTP %s, %d bytes", r.status_code, len(r.content))
            if r.status_code in (401, 403):
                logger.error("Auth error: HTTP %s — %s", r.status_code, r.text.strip())
                print(f"[AUTH ERROR] HTTP {r.status_code}: {r.text.strip()}")
                print("Your access token may have expired. Regenerate it from web.dhan.co.")
                sys.exit(1)
            if r.status_code == 200:
                data = r.json()
                logger.debug("Response body (first 500 chars): %s", json.dumps(data)[:500])
                return data
            else:
                logger.warning("Non-200 response: HTTP %s — %s", r.status_code, r.text[:200])
                return {"errorCode": f"HTTP-{r.status_code}", "errorMessage": r.text.strip()}
        except requests.exceptions.RequestException as e:
            logger.error("Network error: %s", e)
            return {"errorCode": "NETWORK_ERROR", "errorMessage": str(e)}

    def get_trade_history(self, from_date, to_date, page=0):
        """GET /v2/trades/{from-date}/{to-date}/{page} — paginated."""
        endpoint = f"/trades/{from_date}/{to_date}/{page}"
        logger.debug("Fetching trade history page %d for %s..%s", page, from_date, to_date)
        return self._request("GET", endpoint)

    def get_fund_limits(self):
        return self._request("GET", "/fundlimit")


# ---------------------------------------------------------------------------
# Fetch all F&O trades with pagination
# ---------------------------------------------------------------------------
def fetch_fno_trades(client, from_date, to_date):
    """Fetch all F&O trades across all pages for the date range."""
    all_trades = []
    page = 0

    print(f"\n[FETCH] Retrieving F&O trades from {from_date} to {to_date}...")
    logger.info("Fetching F&O trades from %s to %s", from_date, to_date)

    while page < MAX_PAGES:
        logger.debug("Requesting page %d", page)
        resp = client.get_trade_history(from_date, to_date, page)

        # Trade history returns a list of trade dicts directly
        if isinstance(resp, list):
            if len(resp) == 0:
                logger.debug("Page %d returned empty list — stopping pagination", page)
                break
            all_trades.extend(resp)
            logger.debug("Page %d: %d trades (running total: %d)", page, len(resp), len(all_trades))
            print(f"  Page {page}: {len(resp)} trades (total: {len(all_trades)})")
            page += 1
        elif isinstance(resp, dict) and "errorCode" in resp:
            logger.error("Page %d error: %s", page, resp.get("errorMessage", resp))
            print(f"  [ERROR] Page {page}: {resp.get('errorMessage', resp)}")
            break
        else:
            logger.warning("Page %d: unexpected response type %s: %s",
                           page, type(resp).__name__, str(resp)[:200])
            print(f"  [WARN] Page {page}: unexpected response: {str(resp)[:200]}")
            break

    logger.info("Total trades fetched: %d", len(all_trades))

    # Filter to F&O segments only
    fno_trades = [t for t in all_trades if t.get("exchangeSegment", "") in FNO_SEGMENTS]
    logger.info("F&O trades after filtering: %d (excluded %d non-F&O)",
                len(fno_trades), len(all_trades) - len(fno_trades))
    print(f"[FETCH] Total trades fetched: {len(all_trades)} | F&O trades: {len(fno_trades)}")

    # Log a sample trade for debugging
    if fno_trades:
        logger.debug("Sample trade (first): %s", json.dumps(fno_trades[0], default=str)[:500])

    return fno_trades


# ---------------------------------------------------------------------------
# Match buys and sells into round-trip trades and compute P&L
# ---------------------------------------------------------------------------
def compute_round_trips(trades):
    """
    Match BUY and SELL trades for the same instrument into round-trips
    and compute net P&L (after charges) for each completed round-trip.

    We use FIFO matching: the first BUY is matched against the first SELL
    of the same security_id.

    Returns a list of round-trip dicts:
      {
        "symbol": str,
        "securityId": str,
        "buyQty": int,
        "sellQty": int,
        "buyValue": float,      # gross buy value
        "sellValue": float,     # gross sell value
        "grossPnl": float,      # sellValue - buyValue
        "totalCharges": float,  # sum of all charges for both legs
        "netPnl": float,        # grossPnl - totalCharges
        "buyTime": str,
        "sellTime": str,
      }
    """
    logger.info("Computing round-trips from %d trades (FIFO matching)", len(trades))

    # Group trades by securityId
    by_security = defaultdict(list)
    for t in trades:
        sid = t.get("securityId", "")
        if not sid:
            sid = t.get("tradingSymbol", "") or t.get("customSymbol", "")
        by_security[sid].append(t)

    logger.debug("Grouped into %d unique securities", len(by_security))
    for sid, sec_trades in by_security.items():
        logger.debug("  %s: %d trades", sid, len(sec_trades))

    round_trips = []

    for sid, sec_trades in by_security.items():
        # Sort by exchange time for FIFO
        def sort_key(t):
            ts = t.get("exchangeTime", "") or t.get("updateTime", "") or ""
            return ts
        sec_trades.sort(key=sort_key)

        # Separate into buy and sell queues (FIFO)
        buy_queue = []  # list of {qty, price, charges, time, ...}
        sell_queue = []

        for t in sec_trades:
            txn = t.get("transactionType", "").upper()
            qty = int(t.get("tradedQuantity", 0))
            price = float(t.get("tradedPrice", 0))
            charges = (
                float(t.get("brokerageCharges", 0) or 0)
                + float(t.get("stt", 0) or 0)
                + float(t.get("exchangeTransactionCharges", 0) or 0)
                + float(t.get("sebiTax", 0) or 0)
                + float(t.get("serviceTax", 0) or 0)
                + float(t.get("stampDuty", 0) or 0)
            )
            symbol = t.get("tradingSymbol") or t.get("customSymbol") or sid
            time_str = t.get("exchangeTime", "") or t.get("updateTime", "")

            logger.debug("  %s %s qty=%d @₹%.2f charges=₹%.4f time=%s",
                         txn, symbol, qty, price, charges, time_str)

            if txn == "BUY":
                buy_queue.append({
                    "qty": qty, "price": price, "charges": charges,
                    "time": time_str, "symbol": symbol,
                })
            elif txn == "SELL":
                sell_queue.append({
                    "qty": qty, "price": price, "charges": charges,
                    "time": time_str, "symbol": symbol,
                })

        logger.debug("  %s: %d buys, %d sells", sid, len(buy_queue), len(sell_queue))

        # FIFO match: match sells against earliest buys
        si = 0  # sell index
        bi = 0  # buy index
        matched_count = 0

        while si < len(sell_queue) and bi < len(buy_queue):
            buy = buy_queue[bi]
            sell = sell_queue[si]

            matched_qty = min(buy["qty"], sell["qty"])
            if matched_qty == 0:
                if buy["qty"] == 0:
                    bi += 1
                if sell["qty"] == 0:
                    si += 1
                continue

            buy_value = matched_qty * buy["price"]
            sell_value = matched_qty * sell["price"]
            gross_pnl = sell_value - buy_value

            # Apportion charges by the matched fraction
            buy_charge_share = (matched_qty / buy["qty"]) * buy["charges"] if buy["qty"] else 0
            sell_charge_share = (matched_qty / sell["qty"]) * sell["charges"] if sell["qty"] else 0
            total_charges = buy_charge_share + sell_charge_share
            net_pnl = gross_pnl - total_charges

            round_trips.append({
                "symbol": buy["symbol"],
                "securityId": sid,
                "buyQty": matched_qty,
                "sellQty": matched_qty,
                "buyValue": round(buy_value, 2),
                "sellValue": round(sell_value, 2),
                "grossPnl": round(gross_pnl, 2),
                "totalCharges": round(total_charges, 2),
                "netPnl": round(net_pnl, 2),
                "buyTime": buy["time"],
                "sellTime": sell["time"],
            })

            logger.debug("    Match #%d: %s qty=%d buy@₹%.2f sell@₹%.2f gross=₹%.2f "
                         "charges=₹%.4f net=₹%.2f",
                         matched_count + 1, buy["symbol"], matched_qty,
                         buy["price"], sell["price"], gross_pnl, total_charges, net_pnl)

            matched_count += 1

            # Reduce quantities
            buy["qty"] -= matched_qty
            sell["qty"] -= matched_qty

            if buy["qty"] == 0:
                bi += 1
            if sell["qty"] == 0:
                si += 1

        # Log unmatched quantities
        remaining_buys = sum(b["qty"] for b in buy_queue[bi:])
        remaining_sells = sum(s["qty"] for s in sell_queue[si:])
        if remaining_buys:
            logger.debug("  %s: %d unmatched buy qty remaining (open position)", sid, remaining_buys)
        if remaining_sells:
            logger.debug("  %s: %d unmatched sell qty remaining", sid, remaining_sells)

    logger.info("Round-trip matching complete: %d round-trips from %d securities",
                len(round_trips), len(by_security))
    return round_trips


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary(round_trips):
    if not round_trips:
        print("\n" + "=" * 60)
        print("  No completed F&O round-trip trades found in this period.")
        print("=" * 60)
        return

    total_trades = len(round_trips)
    winners = [rt for rt in round_trips if rt["netPnl"] > 0]
    losers = [rt for rt in round_trips if rt["netPnl"] < 0]
    breakeven = [rt for rt in round_trips if rt["netPnl"] == 0]

    total_net_pnl = sum(rt["netPnl"] for rt in round_trips)
    total_gross_pnl = sum(rt["grossPnl"] for rt in round_trips)
    total_charges = sum(rt["totalCharges"] for rt in round_trips)

    avg_pnl = total_net_pnl / total_trades
    avg_profit = sum(rt["netPnl"] for rt in winners) / len(winners) if winners else 0
    avg_loss = sum(rt["netPnl"] for rt in losers) / len(losers) if losers else 0
    win_rate = (len(winners) / total_trades) * 100 if total_trades else 0

    # Per-symbol breakdown
    by_symbol = defaultdict(list)
    for rt in round_trips:
        by_symbol[rt["symbol"]].append(rt)

    logger.info("Summary: %d trades, %d wins, %d losses, %d breakeven, "
                "net P&L=₹%.2f, win rate=%.1f%%",
                total_trades, len(winners), len(losers), len(breakeven),
                total_net_pnl, win_rate)

    print("\n" + "=" * 70)
    print("  F&O TRADE PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Total round-trip trades : {total_trades}")
    print(f"  Winning trades          : {len(winners)}")
    print(f"  Losing trades           : {len(losers)}")
    print(f"  Breakeven trades        : {len(breakeven)}")
    print(f"  Win rate                : {win_rate:.1f}%")
    print("-" * 70)
    print(f"  Total gross P&L         : ₹{total_gross_pnl:,.2f}")
    print(f"  Total charges           : ₹{total_charges:,.2f}")
    print(f"  Total net P&L           : ₹{total_net_pnl:,.2f}")
    print("-" * 70)
    print(f"  Average P&L per trade   : ₹{avg_pnl:,.2f}")
    print(f"  Average profit (winners): ₹{avg_profit:,.2f}")
    print(f"  Average loss (losers)   : ₹{avg_loss:,.2f}")
    if avg_profit and avg_loss:
        risk_reward = abs(avg_profit / avg_loss)
        print(f"  Risk : Reward           : 1 : {risk_reward:.2f}")
    print("=" * 70)

    # Per-symbol table
    print("\n  PER-SYMBOL BREAKDOWN")
    print("-" * 70)
    print(f"  {'Symbol':<40} {'Trades':>6} {'Net P&L':>12} {'Avg P&L':>10}")
    print("-" * 70)

    sorted_symbols = sorted(by_symbol.items(), key=lambda x: sum(rt["netPnl"] for rt in x[1]), reverse=True)
    for sym, rts in sorted_symbols:
        sym_pnl = sum(rt["netPnl"] for rt in rts)
        sym_avg = sym_pnl / len(rts)
        sym_name = sym[:38] if len(sym) > 38 else sym
        print(f"  {sym_name:<40} {len(rts):>6} ₹{sym_pnl:>10,.2f} ₹{sym_avg:>8,.2f}")

    print("-" * 70)

    # Show individual round-trips if requested
    if len(round_trips) <= 30:
        print("\n  INDIVIDUAL ROUND-TRIPS")
        print("-" * 70)
        print(f"  {'#':>3} {'Symbol':<35} {'Qty':>5} {'Net P&L':>12}")
        print("-" * 70)
        for i, rt in enumerate(round_trips, 1):
            sym = rt["symbol"][:33]
            print(f"  {i:>3} {sym:<35} {rt['buyQty']:>5} ₹{rt['netPnl']:>10,.2f}")
        print("-" * 70)


# ---------------------------------------------------------------------------
# Raw trades CSV save/load (for inter-script data sharing)
# ---------------------------------------------------------------------------
# Fields we persist from each trade record. We store every field Dhan
# returns so downstream scripts (adaptive guard, etc.) have everything
# they need without re-fetching.
RAW_TRADE_FIELDS = [
    "dhanClientId", "orderId", "exchangeOrderId", "exchangeTradeId",
    "transactionType", "exchangeSegment", "productType", "orderType",
    "tradingSymbol", "customSymbol", "securityId", "tradedQuantity",
    "tradedPrice", "isin", "instrument",
    "sebiTax", "stt", "brokerageCharges", "serviceTax",
    "exchangeTransactionCharges", "stampDuty",
    "createTime", "updateTime", "exchangeTime",
    "drvExpiryDate", "drvOptionType", "drvStrikePrice",
]


def save_raw_trades(trades, filepath):
    """Save raw F&O trade records to a CSV file for reuse by other scripts."""
    if not trades:
        print("[DATA] No trades to save.")
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_TRADE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            writer.writerow(t)
    logger.info("Saved %d raw trades to %s", len(trades), filepath)
    print(f"\n[DATA] Saved {len(trades)} raw F&O trades to {filepath}")


def load_raw_trades(filepath):
    """Load raw F&O trade records from a CSV file saved by a previous run."""
    trades = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Type-cast numeric fields back from strings
            trade = dict(row)
            for num_field in ["tradedQuantity"]:
                if trade.get(num_field):
                    try:
                        trade[num_field] = int(trade[num_field])
                    except (ValueError, TypeError):
                        pass
            for num_field in ["tradedPrice", "drvStrikePrice", "sebiTax", "stt",
                              "brokerageCharges", "serviceTax",
                              "exchangeTransactionCharges", "stampDuty"]:
                if trade.get(num_field) and trade[num_field] != "NA":
                    try:
                        trade[num_field] = float(trade[num_field])
                    except (ValueError, TypeError):
                        pass
            trades.append(trade)
    logger.info("Loaded %d raw trades from %s", len(trades), filepath)
    print(f"\n[DATA] Loaded {len(trades)} F&O trades from {filepath}")
    return trades


# ---------------------------------------------------------------------------
# CSV export of round-trip results
# ---------------------------------------------------------------------------
def export_csv(round_trips, filepath):
    if not round_trips:
        print("[CSV] No trades to export.")
        return

    fields = [
        "symbol", "securityId", "buyQty", "sellQty",
        "buyValue", "sellValue", "grossPnl", "totalCharges", "netPnl",
        "buyTime", "sellTime",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(round_trips)
    logger.info("Exported %d round-trips to %s", len(round_trips), filepath)
    print(f"\n[CSV] Exported {len(round_trips)} round-trips to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate average profit and loss per F&O trade over the last N days.\n\n"
            "Fetches F&O trade history from Dhan, matches buys and sells into "
            "round-trips (FIFO), computes net P&L after all charges, and prints "
            "a performance summary with per-symbol breakdown.\n\n"
            "Environment variable DHAN_ACCESS_TOKEN must be set before running.\n\n"
            "Examples:\n"
            "  python dhan_fno_avg_pnl.py                  # last 90 days\n"
            "  python dhan_fno_avg_pnl.py --days 30        # last 30 days\n"
            "  python dhan_fno_avg_pnl.py --csv report.csv # export to CSV\n"
            "  python dhan_fno_avg_pnl.py --debug           # verbose debug logs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Trailing number of days to analyse (default: 90, approx. 3 months).",
    )
    parser.add_argument(
        "--csv", dest="csv_path",
        help="Export computed round-trip results to this CSV file.",
    )
    parser.add_argument(
        "--data-file", dest="data_file",
        help="Save raw fetched F&O trades to this CSV file. Other scripts "
             "(e.g. dhan_fno_adaptive_guard.py) can then load this file with "
             "--from-file to avoid re-fetching from the API.",
    )
    parser.add_argument(
        "--from-file", dest="from_file",
        help="Load raw F&O trades from a previously saved CSV file instead of "
             "fetching from the Dhan API. Use this when you've already run "
             "--data-file and want to re-analyse offline.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging for troubleshooting. Shows API requests, "
             "responses, trade matching details, and charge breakdowns.",
    )
    args = parser.parse_args()

    # --- Logging ---
    setup_logging(debug=args.debug)
    logger.debug("Arguments: %s", vars(args))

    # --- Dates ---
    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    to_date = today
    logger.debug("Date range: %s to %s (%d days)", from_date, to_date, args.days)

    # --- Authenticate (skip if loading from file) ---
    token = None
    client_id = os.environ.get("DHAN_CLIENT_ID")

    if not args.from_file:
        if not client_id:
            logger.error("DHAN_CLIENT_ID not set")
            print("ERROR: DHAN_CLIENT_ID not set in .env or environment.")
            sys.exit(1)

        if get_access_token:
            # Use auto token generation from src/auth.py (PIN + TOTP)
            try:
                token = get_access_token()
                logger.info("Access token generated via PIN + TOTP")
            except Exception as e:
                logger.error("Token generation failed: %s", e)
                print(f"[ERROR] Failed to generate access token: {e}")
                sys.exit(1)
        else:
            # Fallback: manual token from environment
            token = os.environ.get("DHAN_ACCESS_TOKEN")
            if not token:
                logger.error("DHAN_ACCESS_TOKEN not set and src/auth.py not available")
                print("ERROR: DHAN_ACCESS_TOKEN not set and src/auth.py not available.")
                print("       Either install with src/auth.py or export DHAN_ACCESS_TOKEN manually.")
                sys.exit(1)
            logger.debug("DHAN_ACCESS_TOKEN found in environment (length: %d)", len(token))

        logger.debug("DHAN_CLIENT_ID: %s", client_id)

    print("=" * 70)
    print("  DHAN F&O AVERAGE P&L ANALYSIS")
    print("=" * 70)
    if args.from_file:
        print(f"  Mode         : OFFLINE (loading from {args.from_file})")
    else:
        print(f"  Client ID    : {client_id}")
        print(f"  Date range   : {from_date} to {to_date}")
    print(f"  Segments     : {', '.join(sorted(FNO_SEGMENTS))}")
    print("=" * 70)

    # --- Fetch trades (from API or file) ---
    if args.from_file:
        if not os.path.exists(args.from_file):
            print(f"\n[ERROR] File not found: {args.from_file}")
            sys.exit(1)
        trades = load_raw_trades(args.from_file)
    else:
        client = DhanClient(token, client_id)

        # --- Validate token ---
        logger.info("Validating access token via /fundlimit")
        resp = client.get_fund_limits()
        if isinstance(resp, dict) and ("dhanClientId" in resp or "availabelBalance" in resp):
            print(f"[OK] Token valid. Available balance: ₹{resp.get('availabelBalance', 'N/A')}")
            logger.info("Token valid. Available balance: ₹%s", resp.get('availabelBalance', 'N/A'))
        else:
            logger.error("Token validation failed: %s", resp)
            print(f"[ERROR] Token validation failed: {resp}")
            sys.exit(1)

        trades = fetch_fno_trades(client, from_date, to_date)

        # --- Save raw trades for reuse by other scripts ---
        if args.data_file:
            save_raw_trades(trades, args.data_file)

    if not trades:
        print("\n[INFO] No F&O trades found in the specified period.")
        sys.exit(0)

    # --- Compute round-trips ---
    print("\n[MATCH] Computing round-trip P&L (FIFO matching)...")
    round_trips = compute_round_trips(trades)
    print(f"[MATCH] {len(round_trips)} completed round-trips identified.")

    # --- Summary ---
    print_summary(round_trips)

    # --- CSV export ---
    if args.csv_path:
        export_csv(round_trips, args.csv_path)


if __name__ == "__main__":
    main()
