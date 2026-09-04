"""
Consensus Voting Backtest
=========================

Backtests the 5-strategy consensus voting system over 2 years of historical
data. Runs all 5 strategies on each trading day, applies the 3/5 majority
rule, and simulates trades with stop-loss and target.

Outputs a detailed report comparing Session V against the known results
of sessions A/B/C/T.

Usage:
    python consensus_backtest.py
    python consensus_backtest.py --stocks COLPAL OFSS HCLTECH
    python consensus_backtest.py --start 2024-01-01 --end 2025-09-01
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------------------------------------------------------
# Basket
# ---------------------------------------------------------------------------

DEFAULT_BASKET = [
    "COLPAL", "OFSS", "NATIONALUM", "HINDZINC", "IRCTC",
    "HCLTECH", "VBL", "HAL", "BOSCHLTD", "COALINDIA",
    "BRITANNIA", "SUNPHARMA", "ASTRAL", "WIPRO",
]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Compute all indicators needed by the 5 strategies."""
    df = df.copy()

    # EMAs
    for span in [10, 20, 50, 200]:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume average
    df["vol_avg"] = df["Volume"].rolling(20).mean()

    # Lookback high (excluding today)
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Supertrend
    df = compute_supertrend(df, period=10, multiplier=3)

    return df


def compute_supertrend(df, period=10, multiplier=3):
    """Compute Supertrend indicator."""
    df = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for i in range(1, len(df)):
        # Skip if current bands are NaN (ATR warmup)
        if pd.isna(upper_basic.iloc[i]) or pd.isna(lower_basic.iloc[i]):
            continue
        # If previous band is NaN (warmup), initialize from basic
        if pd.isna(upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
            lower_band.iloc[i] = lower_basic.iloc[i]
            continue
        if (upper_basic.iloc[i] < upper_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        if (lower_basic.iloc[i] > lower_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]):
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    # Find first valid index (where ATR is available)
    first_valid = None
    for i in range(len(df)):
        if not pd.isna(upper_band.iloc[i]) and not pd.isna(lower_band.iloc[i]):
            first_valid = i
            break

    if first_valid is not None:
        supertrend.iloc[first_valid] = upper_band.iloc[first_valid]
        direction.iloc[first_valid] = -1

        for i in range(first_valid + 1, len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
            close = df["Close"].iloc[i]
            prev_st = supertrend.iloc[i - 1]
            if pd.isna(prev_st):
                prev_st = upper_band.iloc[i]

            if close <= prev_st:
                supertrend.iloc[i] = min(upper_band.iloc[i], prev_st) if not pd.isna(upper_band.iloc[i]) else prev_st
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = max(lower_band.iloc[i], prev_st) if not pd.isna(lower_band.iloc[i]) else prev_st
                direction.iloc[i] = 1
                if direction.iloc[i - 1] == -1 or pd.isna(direction.iloc[i - 1]):
                    supertrend.iloc[i] = lower_band.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction
    return df


# ---------------------------------------------------------------------------
# 5 Strategies (identical to consensus_voter.py)
# ---------------------------------------------------------------------------

def strategy_breakout(df, config):
    """State-based: bullish when price is above recent high in an uptrend."""
    if len(df) < 200:
        return "HOLD"
    last = df.iloc[-1]
    lookback = config.get("lookback_days", 10)
    in_uptrend = last["ema50"] > last["ema200"]
    breakout = last["Close"] > last["high_lookback"]
    vol_good = last["Volume"] > last["vol_avg"] * 1.5  # relaxed from 2x
    rsi_ok = 30 < last["rsi"] < 75
    if breakout and vol_good and rsi_ok and in_uptrend:
        return "BUY"
    if last["Close"] < last["ema10"]:
        return "SELL"
    if last["rsi"] > 78:
        return "SELL"
    return "HOLD"


def strategy_pullback(df, config):
    """State-based: bullish when price is in a pullback zone near EMA20 in an uptrend."""
    if len(df) < 200:
        return "HOLD"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["ema50"] <= last["ema200"]:
        return "HOLD"
    # Relaxed: within 3% of EMA20 (was 1.5%), RSI 35-60 (was 40-55)
    dist_to_ema20 = abs(last["Close"] - last["ema20"]) / last["Close"]
    near_ema20 = dist_to_ema20 < 0.03
    rsi_zone = 35 <= last["rsi"] <= 60
    above_ema50 = last["Close"] > last["ema50"]
    rsi_rising = last["rsi"] > prev["rsi"]
    if near_ema20 and rsi_zone and above_ema50 and rsi_rising:
        return "BUY"
    if last["Close"] < last["ema50"]:
        return "SELL"
    if last["rsi"] > 72:
        return "SELL"
    return "HOLD"


def strategy_supertrend(df, config):
    """State-based: bullish while supertrend is green, bearish while red."""
    if len(df) < 15 or "supertrend_dir" not in df.columns:
        return "HOLD"
    last = df.iloc[-1]
    if last["supertrend_dir"] == 1:
        return "BUY"
    if last["supertrend_dir"] == -1:
        return "SELL"
    return "HOLD"


def strategy_macd(df, config):
    """State-based: bullish while MACD line > signal line and histogram positive."""
    if len(df) < 35 or "macd_line" not in df.columns:
        return "HOLD"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["macd_line"] > last["macd_signal"] and last["macd_hist"] > 0:
        return "BUY"
    if last["macd_line"] < last["macd_signal"] and last["macd_hist"] < 0:
        return "SELL"
    return "HOLD"


def strategy_rsi_mean_reversion(df, config):
    """State-based: bullish while RSI is recovering from oversold in an uptrend."""
    if len(df) < 200:
        return "HOLD"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["ema50"] <= last["ema200"]:
        return "HOLD"
    # RSI was oversold recently and is now recovering
    was_oversold = any(df["rsi"].iloc[i] < 45 for i in range(-5, 0))
    rsi_recovering = last["rsi"] > prev["rsi"]
    rsi_in_zone = 35 < last["rsi"] < 60
    if was_oversold and rsi_recovering and rsi_in_zone:
        return "BUY"
    if last["rsi"] > 70:
        return "SELL"
    if last["Close"] < last["ema50"]:
        return "SELL"
    return "HOLD"


STRATEGIES = {
    "breakout": strategy_breakout,
    "pullback": strategy_pullback,
    "supertrend": strategy_supertrend,
    "macd": strategy_macd,
    "rsi_mean_reversion": strategy_rsi_mean_reversion,
}

VOTE_THRESHOLD = 3


def run_consensus(df, config, in_position):
    """Run all 5 strategies and return consensus + vote breakdown."""
    votes = {}
    for name, func in STRATEGIES.items():
        try:
            votes[name] = func(df, config)
        except Exception:
            votes[name] = "HOLD"

    buy_votes = sum(1 for v in votes.values() if v == "BUY")
    sell_votes = sum(1 for v in votes.values() if v == "SELL")

    if in_position:
        consensus = "SELL" if sell_votes >= VOTE_THRESHOLD else "HOLD"
    else:
        consensus = "BUY" if buy_votes >= VOTE_THRESHOLD else "HOLD"

    return consensus, votes, buy_votes, sell_votes


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_stock(symbol, df, config):
    """
    Walk through the dataframe day by day, running the 5-strategy consensus
    on each day. Simulate trades with SL/target.

    Returns list of closed trades.
    """
    trades = []
    position = None
    warmup = 210  # need 200 EMA + buffer

    if len(df) < warmup:
        print(f"  {symbol}: only {len(df)} rows, need {warmup}")
        return trades

    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1]
        last = window.iloc[-1]
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])

        # If in position, check SL/target first (intraday)
        if position:
            # SL hit
            if last_low <= position["sl"]:
                position["exit_price"] = position["sl"]
                position["exit_date"] = last.name
                position["exit_reason"] = "SL hit"
                position["pnl_pct"] = (position["sl"] - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

            # Target hit
            if last_high >= position["target"]:
                position["exit_price"] = position["target"]
                position["exit_date"] = last.name
                position["exit_reason"] = "Target hit"
                position["pnl_pct"] = (position["target"] - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

        # Run consensus vote
        consensus, votes, buy_votes, sell_votes = run_consensus(
            window, config, in_position=(position is not None)
        )

        if consensus == "BUY" and position is None:
            sl_pct = config.get("stop_loss_pct", 0.03)
            target_pct = config.get("target_pct", 0.08)
            position = {
                "symbol": symbol,
                "entry": last_close,
                "entry_date": last.name,
                "sl": round(last_close * (1 - sl_pct), 2),
                "target": round(last_close * (1 + target_pct), 2),
                "votes": votes,
                "buy_votes": buy_votes,
            }

        elif consensus == "SELL" and position is not None:
            position["exit_price"] = last_close
            position["exit_date"] = last.name
            position["exit_reason"] = f"Consensus SELL ({sell_votes}/5)"
            position["exit_votes"] = votes
            position["sell_votes"] = sell_votes
            position["pnl_pct"] = (last_close - position["entry"]) / position["entry"]
            trades.append(position)
            position = None

    # Close any open position at last close
    if position:
        last = df.iloc[-1]
        position["exit_price"] = float(last["Close"])
        position["exit_date"] = last.name
        position["exit_reason"] = "End of data"
        position["pnl_pct"] = (position["exit_price"] - position["entry"]) / position["entry"]
        trades.append(position)

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_trades, config, start_date, end_date):
    """Print detailed backtest report."""

    total_trades = len(all_trades)
    if total_trades == 0:
        print("\n" + "=" * 70)
        print("CONSENSUS VOTING BACKTEST — NO TRADES")
        print("=" * 70)
        print("No trades were generated. Possible causes:")
        print("  - 3/5 majority threshold too strict")
        print("  - Trend filter (EMA50 > EMA200) filtered out most setups")
        print("  - Insufficient data")
        return

    wins = [t for t in all_trades if t["pnl_pct"] > 0]
    losses = [t for t in all_trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / total_trades * 100

    total_pnl_pct = sum(t["pnl_pct"] for t in all_trades)
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in all_trades:
        exit_reasons[t["exit_reason"]] += 1

    # Per-stock breakdown
    per_stock = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in all_trades:
        per_stock[t["symbol"]]["trades"] += 1
        if t["pnl_pct"] > 0:
            per_stock[t["symbol"]]["wins"] += 1
        per_stock[t["symbol"]]["pnl"] += t["pnl_pct"]

    # Vote pattern analysis
    vote_patterns = defaultdict(int)
    for t in all_trades:
        key = f"{t.get('buy_votes', 0)}/5 BUY"
        vote_patterns[key] += 1

    print("\n" + "=" * 70)
    print("CONSENSUS VOTING BACKTEST — Session V")
    print("=" * 70)
    print(f"Period:              {start_date} to {end_date}")
    print(f"Stocks:              {len(per_stock)}")
    print(f"Strategies:         {', '.join(STRATEGIES.keys())}")
    print(f"Vote threshold:     {VOTE_THRESHOLD}/5 for entry/exit")
    print(f"Stop loss:           {config.get('stop_loss_pct', 0.03)*100:.1f}%")
    print(f"Target:              {config.get('target_pct', 0.08)*100:.1f}%")
    print()
    print(f"Total trades:       {total_trades}")
    print(f"  Wins:              {len(wins)}")
    print(f"  Losses:            {len(losses)}")
    print(f"  Win rate:          {win_rate:.1f}%")
    print()
    print(f"Total P&L (sum %):  {total_pnl_pct*100:.2f}%")
    print(f"Avg win:            {avg_win*100:.2f}%")
    print(f"Avg loss:           {avg_loss*100:.2f}%")
    print(f"Profit factor:      {profit_factor:.2f}")
    print(f"Avg P&L per trade:  {total_pnl_pct/total_trades*100:.2f}%")
    print()
    print("Exit reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s} {count:3d} ({count/total_trades*100:.0f}%)")
    print()
    print("Vote patterns at entry:")
    for pattern, count in sorted(vote_patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern:15s} {count:3d} trades")
    print()
    print("Per-stock breakdown:")
    print(f"  {'Symbol':12s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
    for symbol in sorted(per_stock.keys()):
        s = per_stock[symbol]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  {symbol:12s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")
    print()
    print("Comparison with other sessions (from previous backtest):")
    print(f"  {'Session':12s} {'Trades':>6s} {'Win%':>6s} {'PF':>6s} {'P&L%':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    print(f"  {'A (cons)':10s} {'~60':>6s} {'36%':>6s} {'0.85':>6s} {'-3.2%':>8s}")
    print(f"  {'B (bal)':10s} {'~55':>6s} {'36%':>6s} {'0.88':>6s} {'-2.8%':>8s}")
    print(f"  {'C (aggr)':10s} {'~50':>6s} {'38%':>6s} {'0.92':>6s} {'-1.5%':>8s}")
    print(f"  {'T (tuned)':10s} {'21':>6s} {'52%':>6s} {'3.02':>6s} {'7.1%':>8s}")
    wr_str = f"{win_rate:.0f}%"
    pf_str = f"{profit_factor:.2f}"
    pnl_str = f"{total_pnl_pct*100:.1f}%"
    print(f"  {'V (vote)':10s} {str(total_trades):>6s} {wr_str:>6s} {pf_str:>6s} {pnl_str:>8s}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest the 5-strategy consensus voting system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stocks", nargs="*", default=DEFAULT_BASKET,
                        help="Stock symbols to backtest (default: 14-stock basket)")
    parser.add_argument("--start", default="2023-09-01",
                        help="Start date (default: 2023-09-01, ~2 years ago)")
    parser.add_argument("--end", default="2025-09-01",
                        help="End date (default: 2025-09-01)")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Lookback days for breakout (default: 10)")
    parser.add_argument("--vol-mult", type=float, default=2.0,
                        help="Volume multiplier (default: 2.0)")
    parser.add_argument("--rsi-min", type=float, default=35,
                        help="RSI minimum (default: 35)")
    parser.add_argument("--rsi-max", type=float, default=70,
                        help="RSI maximum (default: 70)")
    parser.add_argument("--sl", type=float, default=0.03,
                        help="Stop loss pct (default: 0.03 = 3%%)")
    parser.add_argument("--target", type=float, default=0.08,
                        help="Target pct (default: 0.08 = 8%%)")
    parser.add_argument("--capital", type=float, default=25000,
                        help="Capital per position (for P&L in Rs)")
    parser.add_argument("--save-trades", action="store_true",
                        help="Save detailed trades to JSON")
    args = parser.parse_args()

    config = {
        "lookback_days": args.lookback,
        "volume_multiplier": args.vol_mult,
        "rsi_min": args.rsi_min,
        "rsi_max": args.rsi_max,
        "stop_loss_pct": args.sl,
        "target_pct": args.target,
    }

    print(f"\nFetching {len(args.stocks)} stocks from yfinance...")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Config: lookback={args.lookback}, vol_mult={args.vol_mult}, "
          f"rsi={args.rsi_min}-{args.rsi_max}, SL={args.sl*100:.1f}%, "
          f"target={args.target*100:.1f}%")
    print()

    all_trades = []

    for symbol in args.stocks:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(start=args.start, end=args.end, auto_adjust=True)
            if hist.empty:
                print("NO DATA")
                continue
            print(f"{len(hist)} rows")

            df = hist.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            df.index = pd.DatetimeIndex(df.index)

            df = compute_indicators(df, lookback=args.lookback)

            trades = backtest_stock(symbol, df, config)
            all_trades.extend(trades)

            if trades:
                wins = sum(1 for t in trades if t["pnl_pct"] > 0)
                print(f"    -> {len(trades)} trades, {wins} wins")
            else:
                print(f"    -> 0 trades")

        except Exception as e:
            print(f"ERROR: {e}")

    # Print report
    print_report(all_trades, config, args.start, args.end)

    # Capital-based P&L
    if all_trades:
        total_pnl_rs = sum(t["pnl_pct"] * args.capital for t in all_trades)
        print(f"\nCapital-based P&L (₹{args.capital:,} per position):")
        print(f"  Total P&L:     ₹{total_pnl_rs:,.2f}")
        print(f"  Avg per trade: ₹{total_pnl_rs/len(all_trades):,.2f}")
        print(f"  Best trade:    ₹{max(t['pnl_pct'] for t in all_trades)*args.capital:,.2f}")
        print(f"  Worst trade:   ₹{min(t['pnl_pct'] for t in all_trades)*args.capital:,.2f}")

    # Save trades if requested
    if args.save_trades and all_trades:
        # Convert dates to strings for JSON
        for t in all_trades:
            if hasattr(t.get("entry_date"), "strftime"):
                t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
            if hasattr(t.get("exit_date"), "strftime"):
                t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")

        with open("consensus_trades.json", "w") as f:
            json.dump(all_trades, f, indent=2, default=str)
        print(f"\nTrades saved to consensus_trades.json")

    return all_trades


if __name__ == "__main__":
    trades = main()
