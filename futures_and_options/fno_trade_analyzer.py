#!/usr/bin/env python3
"""
F&O Trade Analyzer — Reverse Engineer Stratzy Algos
=====================================================

Pulls historical F&O trade data from Dhan's API and analyzes patterns:
  - Entry timing (time of day, day of week)
  - Strike selection (ITM/ATM/OTM, distance from spot)
  - Spread structure (credit spread, naked, straddle?)
  - Holding period (intraday vs overnight)
  - Exit patterns (SL hit, target hit, time-based?)
  - P&L distribution
  - Position sizing

Requires: DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET in .env

Usage:
    cd ~/dhan-trading-bot
    source venv/bin/activate
    export PYTHONPATH=~/dhan-trading-bot:~/dhan-trading-bot/src
    python futures_and_options/fno_trade_analyzer.py --days 30

    # Analyze last 90 days
    python futures_and_options/fno_trade_analyzer.py --days 90

    # Export raw trades to CSV
    python futures_and_options/fno_trade_analyzer.py --days 60 --export
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

try:
    from src.auth import get_access_token
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    get_access_token = None

API_BASE = "https://api.dhan.co/v2"
FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"}


class DhanClient:
    def __init__(self, access_token, client_id):
        self.access_token = access_token
        self.client_id = client_id
        self.headers = {
            "access-token": access_token,
            "client-id": client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        r = requests.request(method, url, headers=self.headers, params=params, timeout=30)
        if r.status_code in (401, 403):
            print(f"[AUTH ERROR] HTTP {r.status_code}: token expired")
            sys.exit(1)
        return r.json()

    base_url = API_BASE

    def get_trade_history(self, from_date, to_date, page=0):
        """Fetch historical trades from Dhan API."""
        endpoint = f"/trades/{from_date}/{to_date}/{page}"
        return self._request("GET", endpoint)

    def get_all_trades(self, from_date, to_date):
        """Fetch all pages of trade history."""
        all_trades = []
        page = 0
        while True:
            resp = self.get_trade_history(from_date, to_date, page)
            if isinstance(resp, list):
                all_trades.extend(resp)
                if len(resp) < 500:  # Last page
                    break
                page += 1
            elif isinstance(resp, dict) and "errorCode" in resp:
                print(f"  API error: {resp.get('errorMessage', resp)}")
                break
            else:
                if isinstance(resp, list):
                    all_trades.extend(resp)
                break
        return all_trades

    def get_fund_limits(self):
        return self._request("GET", "/fundlimit")


def is_fno_trade(trade):
    """Check if trade is in F&O segment."""
    seg = trade.get("exchangeSegment", "")
    return seg in FNO_SEGMENTS


def analyze_trades(trades):
    """Analyze F&O trades and extract strategy patterns."""
    if not trades:
        print("No F&O trades found in the given period.")
        return

    print(f"\n{'='*70}")
    print(f"F&O TRADE ANALYSIS — {len(trades)} trades")
    print(f"{'='*70}")

    # Group trades by orderId to reconstruct positions
    order_groups = defaultdict(list)
    for t in trades:
        order_groups[t.get("orderId", "")].append(t)

    # Reconstruct round-trip trades (entry + exit pairs)
    # Group by trading symbol + date to match entries with exits
    symbol_date_groups = defaultdict(list)
    for t in trades:
        sym = t.get("tradingSymbol", t.get("customSymbol", "unknown"))
        exchange_time = t.get("exchangeTime", "")
        if exchange_time and exchange_time != "NA":
            trade_date = exchange_time.split(" ")[0]
        else:
            trade_date = "unknown"
        symbol_date_groups[(sym, trade_date)].append(t)

    # ========================================
    # 1. BASIC STATS
    # ========================================
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total F&O trades: {len(trades)}")
    print(f"   Unique symbols: {len(set(t.get('tradingSymbol', '') for t in trades))}")
    print(f"   Date range: {trades[0].get('exchangeTime', '?')} to {trades[-1].get('exchangeTime', '?')}")

    # Unique trading days
    trade_dates = set()
    for t in trades:
        et = t.get("exchangeTime", "")
        if et and et != "NA":
            trade_dates.add(et.split(" ")[0])
    print(f"   Trading days active: {len(trade_dates)}")
    if trade_dates:
        print(f"   Trades per day (avg): {len(trades) / len(trade_dates):.1f}")

    # ========================================
    # 2. INSTRUMENT BREAKDOWN
    # ========================================
    print(f"\n2. INSTRUMENT BREAKDOWN")
    instruments = Counter()
    option_types = Counter()
    for t in trades:
        sym = t.get("tradingSymbol", "unknown")
        instruments[sym] += 1
        opt_type = t.get("drvOptionType", "NA")
        if opt_type != "NA":
            option_types[opt_type] += 1

    print(f"   Top 15 traded instruments:")
    for sym, count in instruments.most_common(15):
        print(f"     {sym:40s}  {count:4d} trades")

    if option_types:
        print(f"\n   Option types:")
        for ot, count in option_types.most_common():
            print(f"     {ot:10s}  {count:4d} trades")

    # ========================================
    # 3. ENTRY TIMING ANALYSIS
    # ========================================
    print(f"\n3. ENTRY TIMING ANALYSIS")
    entry_times = []
    entry_hours = Counter()
    entry_days = Counter()
    for t in trades:
        et = t.get("exchangeTime", "")
        if et and et != "NA":
            try:
                dt = datetime.strptime(et, "%Y-%m-%d %H:%M:%S")
                entry_times.append(dt)
                entry_hours[dt.hour] += 1
                entry_days[dt.strftime("%A")] += 1
            except ValueError:
                pass

    if entry_times:
        print(f"   Entry time distribution (hour of day):")
        for hr in sorted(entry_hours.keys()):
            bar = "█" * (entry_hours[hr] // 2)
            print(f"     {hr:02d}:00  {entry_hours[hr]:4d}  {bar}")

        print(f"\n   Day of week distribution:")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in day_order:
            count = entry_days.get(day, 0)
            bar = "█" * (count // 2)
            print(f"     {day:10s}  {count:4d}  {bar}")

    # ========================================
    # 4. STRIKE PRICE ANALYSIS
    # ========================================
    print(f"\n4. STRIKE PRICE ANALYSIS")
    strikes = []
    for t in trades:
        strike = t.get("drvStrikePrice", 0)
        if strike and strike != 0:
            strikes.append(float(strike))

    if strikes:
        print(f"   Strike prices traded: {len(strikes)}")
        print(f"   Min: {min(strikes):.0f} | Max: {max(strikes):.0f} | Median: {sorted(strikes)[len(strikes)//2]:.0f}")
        unique_strikes = sorted(set(strikes))
        print(f"   Unique strikes: {len(unique_strikes)}")
        if len(unique_strikes) <= 20:
            print(f"   All strikes: {', '.join(f'{s:.0f}' for s in unique_strikes)}")

    # ========================================
    # 5. EXPIRY ANALYSIS
    # ========================================
    print(f"\n5. EXPIRY ANALYSIS")
    expiries = Counter()
    for t in trades:
        exp = t.get("drvExpiryDate", "NA")
        if exp and exp != "NA":
            expiries[exp] += 1

    if expiries:
        print(f"   Expiry dates used:")
        for exp, count in expiries.most_common(10):
            print(f"     {exp:25s}  {count:4d} trades")

    # ========================================
    # 6. TRANSACTION TYPE (BUY/SELL)
    # ========================================
    print(f"\n6. TRANSACTION TYPE ANALYSIS")
    txn_types = Counter()
    for t in trades:
        txn_types[t.get("transactionType", "UNKNOWN")] += 1

    for txn, count in txn_types.most_common():
        pct = count / len(trades) * 100
        print(f"   {txn:5s}  {count:4d} trades ({pct:.1f}%)")

    # ========================================
    # 7. PRODUCT TYPE
    # ========================================
    print(f"\n7. PRODUCT TYPE")
    products = Counter()
    for t in trades:
        products[t.get("productType", "UNKNOWN")] += 1
    for prod, count in products.most_common():
        print(f"   {prod:15s}  {count:4d} trades")

    # ========================================
    # 8. ORDER TYPE
    # ========================================
    print(f"\n8. ORDER TYPE")
    order_types = Counter()
    for t in trades:
        order_types[t.get("orderType", "UNKNOWN")] += 1
    for ot, count in order_types.most_common():
        print(f"   {ot:20s}  {count:4d} trades")

    # ========================================
    # 9. POSITION RECONSTRUCTION
    # ========================================
    print(f"\n9. POSITION RECONSTRUCTION")
    # Group trades by symbol and reconstruct buy/sell pairs
    positions = defaultdict(lambda: {"buys": [], "sells": []})
    for t in trades:
        sym = t.get("tradingSymbol", "unknown")
        txn = t.get("transactionType", "")
        qty = int(t.get("tradedQuantity", 0))
        price = float(t.get("tradedPrice", 0))
        et = t.get("exchangeTime", "")

        if txn == "BUY":
            positions[sym]["buys"].append({"qty": qty, "price": price, "time": et})
        elif txn == "SELL":
            positions[sym]["sells"].append({"qty": qty, "price": price, "time": et})

    # Match buys with sells to compute P&L
    completed_trades = []
    for sym, pos in positions.items():
        buys = sorted(pos["buys"], key=lambda x: x["time"])
        sells = sorted(pos["sells"], key=lambda x: x["time"])

        buy_idx = 0
        for sell in sells:
            remaining = sell["qty"]
            while remaining > 0 and buy_idx < len(buys):
                buy = buys[buy_idx]
                matched = min(remaining, buy["qty"])
                pnl = (sell["price"] - buy["price"]) * matched
                hold_seconds = 0
                try:
                    bt = datetime.strptime(buy["time"], "%Y-%m-%d %H:%M:%S")
                    st = datetime.strptime(sell["time"], "%Y-%m-%d %H:%M:%S")
                    hold_seconds = (st - bt).total_seconds()
                except (ValueError, TypeError):
                    pass

                completed_trades.append({
                    "symbol": sym,
                    "buy_price": buy["price"],
                    "sell_price": sell["price"],
                    "qty": matched,
                    "pnl": pnl,
                    "hold_seconds": hold_seconds,
                    "buy_time": buy["time"],
                    "sell_time": sell["time"],
                })

                buy["qty"] -= matched
                remaining -= matched
                if buy["qty"] <= 0:
                    buy_idx += 1

    # ========================================
    # 10. P&L ANALYSIS
    # ========================================
    print(f"\n10. P&L ANALYSIS")
    if completed_trades:
        pnls = [ct["pnl"] for ct in completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        print(f"   Completed round-trip trades: {len(completed_trades)}")
        print(f"   Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"   Win rate: {len(wins)/len(completed_trades)*100:.1f}%")
        print(f"   Total P&L: Rs.{sum(pnls):,.2f}")
        print(f"   Avg win:  Rs.{sum(wins)/len(wins):,.2f}" if wins else "   Avg win:  N/A")
        print(f"   Avg loss: Rs.{sum(losses)/len(losses):,.2f}" if losses else "   Avg loss: N/A")
        if wins and losses:
            pf = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
            print(f"   Profit factor: {pf:.2f}")
        print(f"   Best trade:  Rs.{max(pnls):,.2f}")
        print(f"   Worst trade: Rs.{min(pnls):,.2f}")
    else:
        print(f"   No completed round-trip trades found (open positions only)")

    # ========================================
    # 11. HOLDING PERIOD
    # ========================================
    print(f"\n11. HOLDING PERIOD ANALYSIS")
    if completed_trades:
        holds = [ct["hold_seconds"] for ct in completed_trades if ct["hold_seconds"] > 0]
        if holds:
            avg_hold = sum(holds) / len(holds)
            median_hold = sorted(holds)[len(holds)//2]

            # Categorize
            intraday = sum(1 for h in holds if h < 6 * 3600)  # < 6 hours
            overnight = sum(1 for h in holds if 6 * 3600 <= h < 48 * 3600)  # overnight to 2 days
            multi_day = sum(1 for h in holds if h >= 48 * 3600)  # > 2 days

            print(f"   Avg holding period: {avg_hold/3600:.1f} hours")
            print(f"   Median holding:     {median_hold/3600:.1f} hours")
            print(f"   Intraday (<6h):     {intraday} trades ({intraday/len(holds)*100:.1f}%)")
            print(f"   Overnight (6-48h):  {overnight} trades ({overnight/len(holds)*100:.1f}%)")
            print(f"   Multi-day (>48h):   {multi_day} trades ({multi_day/len(holds)*100:.1f}%)")

    # ========================================
    # 12. SPREAD DETECTION
    # ========================================
    print(f"\n12. SPREAD / STRATEGY DETECTION")
    # Group trades by time window (trades within 60 seconds = likely a spread)
    time_groups = defaultdict(list)
    for t in trades:
        et = t.get("exchangeTime", "")
        if et and et != "NA":
            try:
                dt = datetime.strptime(et, "%Y-%m-%d %H:%M:%S")
                # Round to nearest minute for grouping
                time_key = dt.strftime("%Y-%m-%d %H:%M")
                time_groups[time_key].append(t)
            except ValueError:
                pass

    spread_count = 0
    single_count = 0
    for time_key, group in time_groups.items():
        if len(group) >= 2:
            spread_count += 1
        else:
            single_count += 1

    print(f"   Single-leg entries: {single_count}")
    print(f"   Multi-leg entries (spreads): {spread_count}")

    # Analyze spread structure
    if spread_count > 0:
        print(f"\n   Spread structure analysis (first 10):")
        shown = 0
        for time_key, group in sorted(time_groups.items()):
            if len(group) >= 2 and shown < 10:
                legs = []
                for t in group:
                    sym = t.get("tradingSymbol", "?")
                    txn = t.get("transactionType", "?")
                    strike = t.get("drvStrikePrice", 0)
                    opt = t.get("drvOptionType", "NA")
                    price = t.get("tradedPrice", 0)
                    qty = t.get("tradedQuantity", 0)
                    legs.append(f"{txn} {sym} @{price} x{qty}")

                # Detect credit spread (sell higher premium, buy lower premium)
                total_credit = 0
                for t in group:
                    if t.get("transactionType") == "SELL":
                        total_credit += float(t.get("tradedPrice", 0))
                    else:
                        total_credit -= float(t.get("tradedPrice", 0))

                spread_type = "CREDIT" if total_credit > 0 else "DEBIT"
                print(f"     {time_key} [{spread_type} Rs.{total_credit:.2f}]")
                for leg in legs:
                    print(f"       {leg}")
                shown += 1

    # ========================================
    # 13. STRATEGY INFERENCE
    # ========================================
    print(f"\n13. STRATEGY INFERENCE")
    print(f"   Based on the above patterns:")

    if completed_trades:
        holds = [ct["hold_seconds"] for ct in completed_trades if ct["hold_seconds"] > 0]
        if holds:
            avg_hold_hours = sum(holds) / len(holds)
            if avg_hold_hours < 6:
                print(f"   → Intraday strategy (avg hold {avg_hold_hours:.1f}h)")
            elif avg_hold_hours < 48:
                print(f"   → Overnight strategy (avg hold {avg_hold_hours:.1f}h)")
            else:
                print(f"   → Multi-day positional strategy (avg hold {avg_hold_hours:.1f}h)")

    if spread_count > single_count:
        print(f"   → Spread strategy (multi-leg in {spread_count}/{spread_count+single_count} entries)")

    if option_types:
        calls = option_types.get("CALL", 0)
        puts = option_types.get("PUT", 0)
        if calls > puts * 2:
            print(f"   → Call-focused ({calls} calls vs {puts} puts) — bullish bias or call selling")
        elif puts > calls * 2:
            print(f"   → Put-focused ({puts} puts vs {calls} calls) — bearish bias or put selling")
        else:
            print(f"   → Balanced call/put ({calls} calls, {puts} puts) — possibly straddle/strangle")

    # Check for credit spread pattern (sell near, buy far)
    txn_types = Counter(t.get("transactionType", "") for t in trades)
    sells = txn_types.get("SELL", 0)
    buys = txn_types.get("BUY", 0)
    if sells > 0 and buys > 0:
        if sells >= buys * 0.8:
            print(f"   → Significant selling ({sells} sells vs {buys} buys) — likely credit spread / option selling")

    # ========================================
    # 14. DAILY P&L BREAKDOWN
    # ========================================
    print(f"\n14. DAILY P&L BREAKDOWN")
    if completed_trades:
        daily_pnl = defaultdict(float)
        daily_trades = defaultdict(int)
        for ct in completed_trades:
            try:
                sell_date = ct["sell_time"].split(" ")[0]
                daily_pnl[sell_date] += ct["pnl"]
                daily_trades[sell_date] += 1
            except (ValueError, IndexError):
                pass

        print(f"   {'Date':15s} {'Trades':>7s} {'P&L':>12s}")
        print(f"   {'-'*37}")
        for d in sorted(daily_pnl.keys()):
            pnl = daily_pnl[d]
            marker = "✓" if pnl > 0 else "✗"
            print(f"   {d:15s} {daily_trades[d]:7d} Rs.{pnl:>10,.2f} {marker}")

        total_pnl = sum(daily_pnl.values())
        print(f"   {'-'*37}")
        print(f"   {'TOTAL':15s} {sum(daily_trades.values()):7d} Rs.{total_pnl:>10,.2f}")

    print(f"\n{'='*70}")
    print(f"Analysis complete.")
    print(f"{'='*70}")


def export_trades(trades, filename):
    """Export trades to CSV."""
    import csv
    if not trades:
        print("No trades to export.")
        return

    fields = list(trades[0].keys())
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for t in trades:
            writer.writerow(t)
    print(f"\nExported {len(trades)} trades to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Analyze F&O trades to reverse engineer strategy")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    parser.add_argument("--export", action="store_true", help="Export raw trades to CSV")
    args = parser.parse_args()

    client_id = os.getenv("DHAN_CLIENT_ID")
    if not client_id:
        print("ERROR: DHAN_CLIENT_ID not set in .env")
        sys.exit(1)

    # Authenticate
    if get_access_token:
        print("Authenticating with Dhan...")
        token = get_access_token()
    else:
        token = os.getenv("DHAN_ACCESS_TOKEN")
        if not token:
            print("ERROR: No access token available")
            sys.exit(1)

    client = DhanClient(token, client_id)

    # Verify token
    resp = client.get_fund_limits()
    if isinstance(resp, dict) and "dhanClientId" in resp:
        balance = resp.get("availabelBalance", "N/A")
        print(f"Token valid. Available balance: Rs.{balance}")
    else:
        print(f"Token validation failed: {resp}")
        sys.exit(1)

    # Fetch trade history
    to_date = date.today().strftime("%Y-%m-%d")
    from_date = (date.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    print(f"\nFetching trade history from {from_date} to {to_date}...")

    all_trades = client.get_all_trades(from_date, to_date)
    print(f"Total trades fetched: {len(all_trades)}")

    # Filter F&O trades only
    fno_trades = [t for t in all_trades if is_fno_trade(t)]
    print(f"F&O trades: {len(fno_trades)}")

    if not fno_trades:
        print("\nNo F&O trades found in the given period.")
        print("Try a longer range: --days 90 or --days 180")
        return

    # Analyze
    analyze_trades(fno_trades)

    # Export if requested
    if args.export:
        export_file = os.path.join(REPO_ROOT, "data", "fno_trades_export.csv")
        os.makedirs(os.path.dirname(export_file), exist_ok=True)
        export_trades(fno_trades, export_file)


if __name__ == "__main__":
    main()
