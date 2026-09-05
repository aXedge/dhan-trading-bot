"""
Regime-Switching Backtest — Session R
=====================================

Backtests the regime-switching strategy over 2 years of historical data.
Detects regime using ADX on each day, applies the appropriate strategy,
and simulates trades with SL/target.

Outputs a detailed report comparing Session R against T and V.

Usage:
    python regime_backtest.py
    python regime_backtest.py --stocks COLPAL OFSS HCLTECH
    python regime_backtest.py --adx-trending 25 --adx-choppy 20
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


DEFAULT_BASKET = [
    "COLPAL", "OFSS", "NATIONALUM", "HINDZINC", "IRCTC",
    "HCLTECH", "VBL", "HAL", "BOSCHLTD", "COALINDIA",
    "BRITANNIA", "SUNPHARMA", "ASTRAL", "WIPRO",
]


# ---------------------------------------------------------------------------
# Indicators (from regime_engine.py)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Compute all indicators needed by the regime engine."""
    df = df.copy()

    for span in [10, 20, 50, 200]:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["vol_max_20"] = df["Volume"].rolling(20).max()

    # Lookback high
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)

    # ADX
    df = compute_adx(df, period=14)

    # Bollinger Bands
    df["bb_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # 20-day returns
    df["ret_20d"] = df["Close"].pct_change(20)

    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX, +DI, -DI."""
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()

    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_regime(last, config):
    """Detect regime from last row of data."""
    adx = last["adx"]
    if pd.isna(adx):
        return -1  # TRANSITIONAL

    if adx >= config.get("adx_trending", 25):
        return 1  # TRENDING
    elif adx < config.get("adx_choppy", 20):
        return 0  # CHOPPY
    else:
        return -1  # TRANSITIONAL


# ---------------------------------------------------------------------------
# Strategy: Enhanced Breakout (Trending)
# ---------------------------------------------------------------------------

def entry_breakout(last, config):
    """Enhanced breakout entry — all conditions must be met."""
    # 1. Trend filter
    if last["ema50"] <= last["ema200"]:
        return False
    # 2. ADX strength
    if pd.isna(last["adx"]) or last["adx"] < config.get("adx_trending", 25):
        return False
    # 3. Directional indicator
    if last["plus_di"] <= last["minus_di"]:
        return False
    # 4. Breakout
    if last["Close"] <= last["high_lookback"]:
        return False
    # 5. Volume: above 1.2x average (relaxed — strong but not extreme)
    if last["Volume"] < last["vol_avg"] * 1.2:
        return False
    # 6. RSI range (relaxed)
    if not (35 <= last["rsi"] <= 75):
        return False
    # 7. Relative strength: positive 20-day return (relaxed — just not deeply negative)
    if pd.isna(last["ret_20d"]) or last["ret_20d"] <= -0.05:
        return False
    return True


def exit_breakout(last, config):
    """Exit rules for breakout positions."""
    if last["Close"] < last["ema10"]:
        return True
    if pd.notna(last["adx"]) and last["adx"] < 20:
        return True
    if last["rsi"] > 78:
        return True
    return False


# ---------------------------------------------------------------------------
# Strategy: Mean Reversion (Choppy)
# ---------------------------------------------------------------------------

def entry_mean_reversion(last, prev, config):
    """Mean reversion entry — buy oversold bounce in uptrend."""
    # 1. Broader uptrend
    if last["ema50"] <= last["ema200"]:
        return False
    # 2. Choppy market
    if pd.isna(last["adx"]) or last["adx"] >= 20:
        return False
    # 3. Oversold RSI (< 40 instead of 35)
    if last["rsi"] >= 40:
        return False
    # 4. Close below lower Bollinger Band OR RSI < 30
    if last["Close"] >= last["bb_lower"] and last["rsi"] >= 30:
        return False
    # 5. RSI turning up
    if last["rsi"] <= prev["rsi"]:
        return False
    return True


def exit_mean_reversion(last, config):
    """Exit rules for mean reversion positions."""
    if last["rsi"] > 60:
        return True
    if last["Close"] > last["bb_mid"]:
        return True
    if last["Close"] < last["ema50"]:
        return True
    return False


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_stock(symbol, df, config):
    """Walk through dataframe day by day, running the regime engine."""
    trades = []
    position = None
    warmup = 210

    if len(df) < warmup:
        print(f"  {symbol}: only {len(df)} rows, need {warmup}")
        return trades

    regime_counts = {1: 0, 0: 0, -1: 0}
    regime_names = {1: "TRENDING", 0: "CHOPPY", -1: "TRANSITIONAL"}

    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1]
        last = window.iloc[-1]
        prev = window.iloc[-2]
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])

        # Detect regime
        regime = detect_regime(last, config)
        regime_counts[regime] += 1

        # If in position, check SL/target first
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

        # Apply strategy based on regime
        action = "HOLD"
        strategy_name = "cash"

        if regime == 1:  # TRENDING
            strategy_name = "enhanced_breakout"
            if position:
                if exit_breakout(last, config):
                    action = "SELL"
            else:
                if entry_breakout(last, config):
                    action = "BUY"

        elif regime == 0:  # CHOPPY
            strategy_name = "mean_reversion"
            if position:
                if exit_mean_reversion(last, config):
                    action = "SELL"
            else:
                if entry_mean_reversion(last, prev, config):
                    action = "BUY"

        # TRANSITIONAL: stay in cash

        if action == "BUY" and position is None:
            # Set SL/target based on strategy
            if strategy_name == "mean_reversion":
                sl_pct = config.get("mr_stop_loss_pct", 0.05)
                target_pct = config.get("mr_target_pct", 0.06)
            else:
                sl_pct = config.get("stop_loss_pct", 0.03)
                target_pct = config.get("target_pct", 0.08)

            position = {
                "symbol": symbol,
                "entry": last_close,
                "entry_date": last.name,
                "sl": round(last_close * (1 - sl_pct), 2),
                "target": round(last_close * (1 + target_pct), 2),
                "strategy": strategy_name,
                "regime": regime_names[regime],
                "adx": float(last["adx"]) if pd.notna(last["adx"]) else 0.0,
                "rsi": float(last["rsi"]),
            }

        elif action == "SELL" and position is not None:
            position["exit_price"] = last_close
            position["exit_date"] = last.name
            position["exit_reason"] = f"{strategy_name} exit"
            position["pnl_pct"] = (last_close - position["entry"]) / position["entry"]
            trades.append(position)
            position = None

    # Close any open position
    if position:
        last = df.iloc[-1]
        position["exit_price"] = float(last["Close"])
        position["exit_date"] = last.name
        position["exit_reason"] = "End of data"
        position["pnl_pct"] = (position["exit_price"] - position["entry"]) / position["entry"]
        trades.append(position)

    # Log regime distribution
    total_days = sum(regime_counts.values())
    if total_days > 0:
        for r, count in regime_counts.items():
            pct = count / total_days * 100
            if count > 0:
                print(f"    {regime_names[r]:15s}: {count:4d} days ({pct:.1f}%)")

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_trades, config, start_date, end_date):
    """Print detailed backtest report."""
    total_trades = len(all_trades)

    if total_trades == 0:
        print("\n" + "=" * 70)
        print("REGIME SWITCHING BACKTEST — Session R — NO TRADES")
        print("=" * 70)
        print("No trades generated. The enhanced filters may be too strict.")
        print("Try: --adx-trending 22 --adx-choppy 18")
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

    # Strategy breakdown
    strategy_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in all_trades:
        s = t.get("strategy", "unknown")
        strategy_stats[s]["trades"] += 1
        if t["pnl_pct"] > 0:
            strategy_stats[s]["wins"] += 1
        strategy_stats[s]["pnl"] += t["pnl_pct"]

    # Per-stock breakdown
    per_stock = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in all_trades:
        per_stock[t["symbol"]]["trades"] += 1
        if t["pnl_pct"] > 0:
            per_stock[t["symbol"]]["wins"] += 1
        per_stock[t["symbol"]]["pnl"] += t["pnl_pct"]

    print("\n" + "=" * 70)
    print("REGIME SWITCHING BACKTEST — Session R")
    print("=" * 70)
    print(f"Period:              {start_date} to {end_date}")
    print(f"Stocks:              {len(per_stock)}")
    print(f"ADX Trending:        >= {config.get('adx_trending', 25)}")
    print(f"ADX Choppy:          < {config.get('adx_choppy', 20)}")
    print(f"Breakout SL/Target:  {config.get('stop_loss_pct', 0.03)*100:.1f}% / {config.get('target_pct', 0.08)*100:.1f}%")
    print(f"Mean Rev SL/Target:  {config.get('mr_stop_loss_pct', 0.05)*100:.1f}% / {config.get('mr_target_pct', 0.06)*100:.1f}%")
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
    print("Strategy breakdown:")
    print(f"  {'Strategy':25s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
    for s_name in sorted(strategy_stats.keys()):
        s = strategy_stats[s_name]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  {s_name:25s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")
    print()
    print("Exit reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s} {count:3d} ({count/total_trades*100:.0f}%)")
    print()
    print("Per-stock breakdown:")
    print(f"  {'Symbol':12s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
    for symbol in sorted(per_stock.keys()):
        s = per_stock[symbol]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  {symbol:12s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")
    print()
    print("Comparison with other sessions:")
    print(f"  {'Session':12s} {'Trades':>6s} {'Win%':>6s} {'PF':>6s} {'P&L%':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    print(f"  {'A (cons)':10s} {'~60':>6s} {'36%':>6s} {'0.85':>6s} {'-3.2%':>8s}")
    print(f"  {'B (bal)':10s} {'~55':>6s} {'36%':>6s} {'0.88':>6s} {'-2.8%':>8s}")
    print(f"  {'C (aggr)':10s} {'~50':>6s} {'38%':>6s} {'0.92':>6s} {'-1.5%':>8s}")
    print(f"  {'T (tuned)':10s} {'21':>6s} {'52%':>6s} {'3.02':>6s} {'7.1%':>8s}")
    print(f"  {'V (vote)':10s} {'99':>6s} {'28%':>6s} {'0.88':>6s} {'-19.9%':>8s}")
    wr_str = f"{win_rate:.0f}%"
    pf_str = f"{profit_factor:.2f}"
    pnl_str = f"{total_pnl_pct*100:.1f}%"
    print(f"  {'R (regime)':10s} {str(total_trades):>6s} {wr_str:>6s} {pf_str:>6s} {pnl_str:>8s}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest the regime-switching strategy (Session R).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stocks", nargs="*", default=DEFAULT_BASKET,
                        help="Stock symbols to backtest")
    parser.add_argument("--start", default="2023-09-01",
                        help="Start date (default: 2023-09-01)")
    parser.add_argument("--end", default="2025-09-01",
                        help="End date (default: 2025-09-01)")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Lookback days for breakout (default: 10)")
    parser.add_argument("--adx-trending", type=float, default=25,
                        help="ADX threshold for trending (default: 25)")
    parser.add_argument("--adx-choppy", type=float, default=20,
                        help="ADX threshold for choppy (default: 20)")
    parser.add_argument("--sl", type=float, default=0.03,
                        help="Breakout stop loss pct (default: 0.03)")
    parser.add_argument("--target", type=float, default=0.08,
                        help="Breakout target pct (default: 0.08)")
    parser.add_argument("--mr-sl", type=float, default=0.05,
                        help="Mean reversion stop loss pct (default: 0.05)")
    parser.add_argument("--mr-target", type=float, default=0.06,
                        help="Mean reversion target pct (default: 0.06)")
    parser.add_argument("--capital", type=float, default=25000,
                        help="Capital per position")
    parser.add_argument("--save-trades", action="store_true",
                        help="Save detailed trades to JSON")
    args = parser.parse_args()

    config = {
        "lookback_days": args.lookback,
        "adx_trending": args.adx_trending,
        "adx_choppy": args.adx_choppy,
        "stop_loss_pct": args.sl,
        "target_pct": args.target,
        "mr_stop_loss_pct": args.mr_sl,
        "mr_target_pct": args.mr_target,
    }

    print(f"\nFetching {len(args.stocks)} stocks from yfinance...")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  ADX Trending >= {args.adx_trending} | ADX Choppy < {args.adx_choppy}")
    print(f"  Breakout: SL={args.sl*100:.1f}% Target={args.target*100:.1f}%")
    print(f"  Mean Rev: SL={args.mr_sl*100:.1f}% Target={args.mr_target*100:.1f}%")
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
        print(f"\nCapital-based P&L (Rs {args.capital:,} per position):")
        print(f"  Total P&L:     Rs {total_pnl_rs:,.2f}")
        print(f"  Avg per trade: Rs {total_pnl_rs/len(all_trades):,.2f}")
        print(f"  Best trade:    Rs {max(t['pnl_pct'] for t in all_trades)*args.capital:,.2f}")
        print(f"  Worst trade:   Rs {min(t['pnl_pct'] for t in all_trades)*args.capital:,.2f}")

    # Save trades
    if args.save_trades and all_trades:
        for t in all_trades:
            if hasattr(t.get("entry_date"), "strftime"):
                t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
            if hasattr(t.get("exit_date"), "strftime"):
                t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")

        with open("regime_trades.json", "w") as f:
            json.dump(all_trades, f, indent=2, default=str)
        print(f"\nTrades saved to regime_trades.json")

    return all_trades


if __name__ == "__main__":
    trades = main()
