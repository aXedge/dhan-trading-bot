#!/usr/bin/env python3
"""
F&O Trade Analyzer — Reverse Engineer Stratzy Algos
=====================================================

Pulls historical F&O trade data from Dhan's API and analyzes patterns.

Usage:
    python futures_and_options/fno_trade_analyzer.py --days 30
    python futures_and_options/fno_trade_analyzer.py --days 90 --export
"""

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict

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
    base_url = API_BASE

    def _request(self, method, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        r = requests.request(method, url, headers=self.headers, params=params, timeout=30)
        if r.status_code in (401, 403):
            print(f"[AUTH ERROR] HTTP {r.status_code}: token expired")
            sys.exit(1)
        return r.json()

    def get_trade_history(self, from_date, to_date, page=0):
        endpoint = f"/trades/{from_date}/{to_date}/{page}"
        return self._request("GET", endpoint)

    def get_all_trades(self, from_date, to_date):
        all_trades = []
        page = 0
        while True:
            resp = self.get_trade_history(from_date, to_date, page)
            if isinstance(resp, list):
                all_trades.extend(resp)
                if len(resp) < 500:
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


def get_symbol(trade):
    """Extract best available symbol from trade."""
    for field in ["tradingSymbol", "customSymbol", "securityId"]:
        val = trade.get(field)
        if val and val != "NA" and str(val).strip():
            return str(val)
    return "unknown"


def is_fno_trade(trade):
    return trade.get("exchangeSegment", "") in FNO_SEGMENTS


def safe_float(val, default=0.0):
    try:
        if val is None or val == "NA":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        if val is None or val == "NA":
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_str(val, default="NA"):
    if val is None or val == "":
        return default
    return str(val)


def parse_datetime(val):
    """Parse datetime string, return None on failure."""
    if not val or val == "NA":
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def analyze_trades(trades):
    if not trades:
        print("No F&O trades found in the given period.")
        return

    print(f"\n{'='*70}")
    print(f"F&O TRADE ANALYSIS — {len(trades)} trades")
    print(f"{'='*70}")

    # ========================================
    # 1. BASIC STATS
    # ========================================
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total F&O trades: {len(trades)}")
    symbols = set(get_symbol(t) for t in trades)
    print(f"   Unique symbols: {len(symbols)}")

    # Date range
    dates = []
    for t in trades:
        dt = parse_datetime(t.get("exchangeTime", ""))
        if dt:
            dates.append(dt)
    if dates:
        print(f"   Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        trade_dates = set(d.strftime("%Y-%m-%d") for d in dates)
        print(f"   Trading days active: {len(trade_dates)}")
        print(f"   Trades per day (avg): {len(trades) / len(trade_dates):.1f}")

    # ========================================
    # 2. INSTRUMENT BREAKDOWN
    # ========================================
    print(f"\n2. INSTRUMENT BREAKDOWN")
    instruments = Counter()
    for t in trades:
        instruments[get_symbol(t)] += 1

    print(f"   All traded instruments:")
    for sym, count in instruments.most_common(20):
        print(f"     {sym:45s}  {count:4d} trades")

    # Option types
    option_types = Counter()
    for t in trades:
        opt = safe_str(t.get("drvOptionType", "NA"))
        if opt != "NA":
            option_types[opt] += 1
    if option_types:
        print(f"\n   Option types:")
        for ot, count in option_types.most_common():
            print(f"     {ot:10s}  {count:4d} trades")

    # ========================================
    # 3. ENTRY TIMING
    # ========================================
    print(f"\n3. ENTRY TIMING ANALYSIS")
    entry_hours = Counter()
    entry_days = Counter()
    for t in trades:
        dt = parse_datetime(t.get("exchangeTime", ""))
        if dt:
            entry_hours[dt.hour] += 1
            entry_days[dt.strftime("%A")] += 1

    if entry_hours:
        print(f"   Entry time distribution (hour of day):")
        for hr in sorted(entry_hours.keys()):
            bar = "#" * (entry_hours[hr] * 3)
            print(f"     {hr:02d}:00  {entry_hours[hr]:4d}  {bar}")
        print(f"\n   Day of week distribution:")
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            count = entry_days.get(day, 0)
            bar = "#" * (count * 3)
            print(f"     {day:10s}  {count:4d}  {bar}")

    # ========================================
    # 4. STRIKE PRICE ANALYSIS
    # ========================================
    print(f"\n4. STRIKE PRICE ANALYSIS")
    strikes = []
    for t in trades:
        s = safe_float(t.get("drvStrikePrice"))
        if s > 0:
            strikes.append(s)

    if strikes:
        print(f"   Strike prices traded: {len(strikes)}")
        print(f"   Min: {min(strikes):.0f} | Max: {max(strikes):.0f} | Median: {sorted(strikes)[len(strikes)//2]:.0f}")
        unique_strikes = sorted(set(strikes))
        print(f"   Unique strikes: {len(unique_strikes)}")
        print(f"   All strikes: {', '.join(f'{s:.0f}' for s in unique_strikes)}")

        # Strike spread (max - min within same time window = spread width)
        print(f"   Strike range: {max(strikes) - min(strikes):.0f} points")

    # ========================================
    # 5. EXPIRY ANALYSIS
    # ========================================
    print(f"\n5. EXPIRY ANALYSIS")
    expiries = Counter()
    for t in trades:
        exp = safe_str(t.get("drvExpiryDate", "NA"))
        if exp != "NA":
            expiries[exp] += 1

    if expiries:
        print(f"   Expiry dates used:")
        for exp, count in expiries.most_common(10):
            # Check if expiry is a Monday (NIFTY weekly) or Thursday (monthly)
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dow = exp_dt.strftime("%A")
                print(f"     {exp:25s} ({dow:9s})  {count:4d} trades")
            except ValueError:
                print(f"     {exp:25s}  {count:4d} trades")

    # ========================================
    # 6. TRANSACTION TYPE
    # ========================================
    print(f"\n6. TRANSACTION TYPE ANALYSIS")
    txn_types = Counter()
    for t in trades:
        txn_types[safe_str(t.get("transactionType", "UNKNOWN"))] += 1
    for txn, count in txn_types.most_common():
        pct = count / len(trades) * 100
        print(f"   {txn:5s}  {count:4d} trades ({pct:.1f}%)")

    # ========================================
    # 7. PRODUCT TYPE
    # ========================================
    print(f"\n7. PRODUCT TYPE")
    products = Counter()
    for t in trades:
        products[safe_str(t.get("productType", "UNKNOWN"))] += 1
    for prod, count in products.most_common():
        print(f"   {prod:15s}  {count:4d} trades")

    # ========================================
    # 8. ORDER TYPE
    # ========================================
    print(f"\n8. ORDER TYPE")
    order_types = Counter()
    for t in trades:
        order_types[safe_str(t.get("orderType", "UNKNOWN"))] += 1
    for ot, count in order_types.most_common():
        print(f"   {ot:20s}  {count:4d} trades")

    # ========================================
    # 9. POSITION RECONSTRUCTION
    # ========================================
    print(f"\n9. POSITION RECONSTRUCTION")
    positions = defaultdict(lambda: {"buys": [], "sells": []})
    for t in trades:
        sym = get_symbol(t)
        txn = safe_str(t.get("transactionType", ""))
        qty = safe_int(t.get("tradedQuantity"))
        price = safe_float(t.get("tradedPrice"))
        et = safe_str(t.get("exchangeTime", ""))

        if txn == "BUY":
            positions[sym]["buys"].append({"qty": qty, "price": price, "time": et})
        elif txn == "SELL":
            positions[sym]["sells"].append({"qty": qty, "price": price, "time": et})

    print(f"   Symbols with buy activity: {sum(1 for p in positions.values() if p['buys'])}")
    print(f"   Symbols with sell activity: {sum(1 for p in positions.values() if p['sells'])}")

    # ========================================
    # 10. P&L ANALYSIS
    # ========================================
    print(f"\n10. P&L ANALYSIS")
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
                bt = parse_datetime(buy["time"])
                st = parse_datetime(sell["time"])
                if bt and st:
                    hold_seconds = (st - bt).total_seconds()

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

    if completed_trades:
        pnls = [ct["pnl"] for ct in completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        print(f"   Completed round-trip trades: {len(completed_trades)}")
        print(f"   Wins: {len(wins)} | Losses: {len(losses)}")
        if completed_trades:
            print(f"   Win rate: {len(wins)/len(completed_trades)*100:.1f}%")
        print(f"   Total P&L: Rs.{sum(pnls):,.2f}")
        if wins:
            print(f"   Avg win:  Rs.{sum(wins)/len(wins):,.2f}")
        if losses:
            print(f"   Avg loss: Rs.{sum(losses)/len(losses):,.2f}")
        if wins and losses and sum(losses) != 0:
            pf = abs(sum(wins) / sum(losses))
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
            intraday = sum(1 for h in holds if h < 6 * 3600)
            overnight = sum(1 for h in holds if 6 * 3600 <= h < 48 * 3600)
            multi_day = sum(1 for h in holds if h >= 48 * 3600)

            print(f"   Avg holding period: {avg_hold/3600:.1f} hours")
            print(f"   Median holding:     {median_hold/3600:.1f} hours")
            print(f"   Intraday (<6h):     {intraday} trades ({intraday/len(holds)*100:.1f}%)")
            print(f"   Overnight (6-48h):  {overnight} trades ({overnight/len(holds)*100:.1f}%)")
            print(f"   Multi-day (>48h):   {multi_day} trades ({multi_day/len(holds)*100:.1f}%)")

    # ========================================
    # 12. SPREAD DETECTION
    # ========================================
    print(f"\n12. SPREAD / STRATEGY DETECTION")
    time_groups = defaultdict(list)
    for t in trades:
        dt = parse_datetime(t.get("exchangeTime", ""))
        if dt:
            time_key = dt.strftime("%Y-%m-%d %H:%M")
            time_groups[time_key].append(t)

    spread_count = sum(1 for g in time_groups.values() if len(g) >= 2)
    single_count = sum(1 for g in time_groups.values() if len(g) == 1)
    print(f"   Single-leg entries: {single_count}")
    print(f"   Multi-leg entries (spreads): {spread_count}")

    if spread_count > 0:
        print(f"\n   Spread structure analysis (all entries):")
        for time_key in sorted(time_groups.keys()):
            group = time_groups[time_key]
            if len(group) >= 2:
                legs = []
                total_credit = 0
                for t in group:
                    sym = get_symbol(t)
                    txn = safe_str(t.get("transactionType", "?"))
                    strike = safe_float(t.get("drvStrikePrice"))
                    opt = safe_str(t.get("drvOptionType", "NA"))
                    price = safe_float(t.get("tradedPrice"))
                    qty = safe_int(t.get("tradedQuantity"))
                    legs.append(f"{txn} {sym} K={strike:.0f} {opt} @{price:.2f} x{qty}")
                    if txn == "SELL":
                        total_credit += price
                    else:
                        total_credit -= price

                spread_type = "CREDIT" if total_credit > 0 else "DEBIT"
                print(f"     {time_key} [{spread_type} Rs.{total_credit:.2f}]")
                for leg in legs:
                    print(f"       {leg}")

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
                print(f"   -> Intraday strategy (avg hold {avg_hold_hours:.1f}h)")
            elif avg_hold_hours < 48:
                print(f"   -> Overnight strategy (avg hold {avg_hold_hours:.1f}h)")
            else:
                print(f"   -> Multi-day positional strategy (avg hold {avg_hold_hours:.1f}h)")

    if spread_count > single_count:
        print(f"   -> Spread strategy ({spread_count} multi-leg entries vs {single_count} single)")

    if option_types:
        calls = option_types.get("CALL", 0)
        puts = option_types.get("PUT", 0)
        if calls > puts * 2:
            print(f"   -> Call-focused ({calls}C vs {puts}P) — bullish bias or call selling")
        elif puts > calls * 2:
            print(f"   -> Put-focused ({puts}P vs {calls}C) — bearish bias or put selling")
        else:
            print(f"   -> Balanced call/put ({calls}C, {puts}P)")

    sells = txn_types.get("SELL", 0)
    buys = txn_types.get("BUY", 0)
    if sells > 0 and buys > 0:
        if sells >= buys * 0.8:
            print(f"   -> Option selling strategy ({sells} sells vs {buys} buys)")

    # Expiry pattern
    if expiries:
        weekly_count = 0
        monthly_count = 0
        for exp in expiries:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                if exp_dt.weekday() == 0:  # Monday = NIFTY weekly
                    weekly_count += expiries[exp]
                elif exp_dt.weekday() == 3:  # Thursday = monthly
                    monthly_count += expiries[exp]
            except ValueError:
                pass
        if weekly_count > monthly_count:
            print(f"   -> Weekly expiry preferred ({weekly_count} weekly vs {monthly_count} monthly)")
        elif monthly_count > weekly_count:
            print(f"   -> Monthly expiry preferred ({monthly_count} monthly vs {weekly_count} weekly)")

    # ========================================
    # 14. DAILY P&L
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
            marker = "+" if pnl > 0 else "-" if pnl < 0 else " "
            print(f"   {d:15s} {daily_trades[d]:7d} Rs.{pnl:>10,.2f} {marker}")

        total_pnl = sum(daily_pnl.values())
        print(f"   {'-'*37}")
        print(f"   {'TOTAL':15s} {sum(daily_trades.values()):7d} Rs.{total_pnl:>10,.2f}")

    # ========================================
    # 15. INDIVIDUAL TRADE DETAILS
    # ========================================
    print(f"\n15. ALL TRADES (raw)")
    print(f"   {'#':>3} {'Time':20s} {'Txn':4s} {'Symbol':40s} {'Strike':>8s} {'Type':5s} {'Price':>8s} {'Qty':>4s} {'Expiry':12s}")
    print(f"   {'-'*110}")
    for i, t in enumerate(trades):
        et = safe_str(t.get("exchangeTime", ""), "?")
        txn = safe_str(t.get("transactionType", "?"))
        sym = get_symbol(t)
        strike = safe_float(t.get("drvStrikePrice"))
        opt = safe_str(t.get("drvOptionType", "NA"))
        price = safe_float(t.get("tradedPrice"))
        qty = safe_int(t.get("tradedQuantity"))
        exp = safe_str(t.get("drvExpiryDate", "NA"))
        print(f"   {i+1:3d} {et:20s} {txn:4s} {sym:40s} {strike:8.0f} {opt:5s} {price:8.2f} {qty:4d} {exp:12s}")

    print(f"\n{'='*70}")
    print(f"Analysis complete.")
    print(f"{'='*70}")


def export_trades(trades, filename):
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

    # Authenticate with retry for rate limiting
    if get_access_token:
        print("Authenticating with Dhan...")
        token = None
        for attempt in range(3):
            try:
                token = get_access_token()
                break
            except RuntimeError as e:
                if "once every 2 minutes" in str(e) and attempt < 2:
                    print(f"  Token rate-limited, waiting 120s before retry...")
                    _time.sleep(120)
                else:
                    raise
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

    fno_trades = [t for t in all_trades if is_fno_trade(t)]
    print(f"F&O trades: {len(fno_trades)}")

    if not fno_trades:
        print("\nNo F&O trades found in the given period.")
        print("Try a longer range: --days 90 or --days 180")
        return

    analyze_trades(fno_trades)

    if args.export:
        export_file = os.path.join(REPO_ROOT, "data", "fno_trades_export.csv")
        os.makedirs(os.path.dirname(export_file), exist_ok=True)
        export_trades(fno_trades, export_file)


if __name__ == "__main__":
    main()
