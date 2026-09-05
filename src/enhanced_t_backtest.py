"""
Enhanced Session T Backtest — Option 1
======================================

Takes Session T's profitable breakout strategy (PF 3.02) and adds
filters to eliminate the 48% losing trades:

  Original T filters:
    - EMA50 > EMA200 (trend filter)
    - Close > 10-day high (breakout)
    - Volume > 2x average
    - RSI 35-70

  Enhanced T additions:
    - ADX >= 20 (skip choppy markets where breakouts fail)
    - +DI > -DI (bullish directional movement)
    - Volume > 1.5x average (was 2x — slightly relaxed to get more trades)
    - 20-day return > 0 (relative strength: stock outperforming cash)

Backtests both original T and enhanced T side by side for comparison.

Usage:
    python enhanced_t_backtest.py
    python enhanced_t_backtest.py --save-trades
"""

import argparse
import json
import os
import sys
from datetime import datetime
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
# Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df, lookback=10):
    df = df.copy()

    for span in [10, 20, 50, 200]:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume
    df["vol_avg"] = df["Volume"].rolling(20).mean()

    # Lookback high
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)

    # ADX
    df = compute_adx(df, period=14)

    # 20-day return (relative strength)
    df["ret_20d"] = df["Close"].pct_change(20)

    return df


def compute_adx(df, period=14):
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]

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
    df["adx"] = dx.rolling(period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


# ---------------------------------------------------------------------------
# Original Session T entry (for baseline)
# ---------------------------------------------------------------------------

def entry_original_t(last, config):
    """Original Session T: breakout + 2x volume + RSI 35-70 + trend filter."""
    if last["ema50"] <= last["ema200"]:
        return False
    if last["Close"] <= last["high_lookback"]:
        return False
    vol_mult = config.get("volume_multiplier", 2.0)
    if last["Volume"] < last["vol_avg"] * vol_mult:
        return False
    rsi_min = config.get("rsi_min", 35)
    rsi_max = config.get("rsi_max", 70)
    if not (rsi_min <= last["rsi"] <= rsi_max):
        return False
    return True


def exit_original_t(last, config):
    """Original Session T exit."""
    if last["Close"] < last["ema10"]:
        return True
    if last["rsi"] > 75:
        return True
    return False


# ---------------------------------------------------------------------------
# Enhanced Session T entry (with ADX + DI + RS filters)
# ---------------------------------------------------------------------------

def entry_enhanced_t(last, config):
    """
    Enhanced Session T: original filters + ADX + +DI > -DI + relative strength.

    New filters:
      - ADX >= adx_min (default 20, not 25 — keep trade count reasonable)
      - +DI > -DI (bullish direction)
      - 20-day return > 0 (stock outperforming cash)
    """
    # Original T filters
    if not entry_original_t(last, config):
        return False

    # ADX filter: skip choppy markets
    adx_min = config.get("adx_min", 20)
    if pd.isna(last["adx"]) or last["adx"] < adx_min:
        return False

    # Directional indicator: bullish
    if last["plus_di"] <= last["minus_di"]:
        return False

    # Relative strength: positive 20-day return
    if pd.isna(last["ret_20d"]) or last["ret_20d"] <= 0:
        return False

    return True


def exit_enhanced_t(last, config):
    """
    Enhanced exit: original exits + ADX falling below 15 (trend dying).
    """
    if exit_original_t(last, config):
        return True
    # ADX dropped — trend is dying
    if pd.notna(last["adx"]) and last["adx"] < 15:
        return True
    return False


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_stock(symbol, df, config, use_enhanced=True):
    """Walk through dataframe, run strategy, simulate trades."""
    trades = []
    position = None
    warmup = 210

    if len(df) < warmup:
        return trades

    entry_fn = entry_enhanced_t if use_enhanced else entry_original_t
    exit_fn = exit_enhanced_t if use_enhanced else exit_original_t

    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1]
        last = window.iloc[-1]
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])

        # Check SL/target first
        if position:
            if last_low <= position["sl"]:
                position["exit_price"] = position["sl"]
                position["exit_date"] = last.name
                position["exit_reason"] = "SL hit"
                position["pnl_pct"] = (position["sl"] - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

            if last_high >= position["target"]:
                position["exit_price"] = position["target"]
                position["exit_date"] = last.name
                position["exit_reason"] = "Target hit"
                position["pnl_pct"] = (position["target"] - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

        # Check signal exit
        if position and exit_fn(last, config):
            position["exit_price"] = last_close
            position["exit_date"] = last.name
            position["exit_reason"] = "Signal exit"
            position["pnl_pct"] = (last_close - position["entry"]) / position["entry"]
            trades.append(position)
            position = None
            continue

        # Check entry
        if not position and entry_fn(last, config):
            sl_pct = config.get("stop_loss_pct", 0.03)
            target_pct = config.get("target_pct", 0.08)
            position = {
                "symbol": symbol,
                "entry": last_close,
                "entry_date": last.name,
                "sl": round(last_close * (1 - sl_pct), 2),
                "target": round(last_close * (1 + target_pct), 2),
                "adx": float(last["adx"]) if pd.notna(last["adx"]) else 0,
                "rsi": float(last["rsi"]),
            }

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

def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wins": 0, "win_rate": 0, "pf": 0, "pnl": 0,
                "avg_win": 0, "avg_loss": 0}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gp = sum(t["pnl_pct"] for t in wins)
    gl = abs(sum(t["pnl_pct"] for t in losses))

    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100,
        "pf": gp / gl if gl > 0 else float("inf"),
        "pnl": sum(t["pnl_pct"] for t in trades),
        "avg_win": np.mean([t["pnl_pct"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pnl_pct"] for t in losses]) if losses else 0,
    }


def print_comparison(original_trades, enhanced_trades, config, start, end):
    o = calc_metrics(original_trades)
    e = calc_metrics(enhanced_trades)

    print("\n" + "=" * 70)
    print("SESSION T: ORIGINAL vs ENHANCED")
    print("=" * 70)
    print(f"Period: {start} to {end}")
    print(f"SL: {config.get('stop_loss_pct', 0.03)*100:.1f}% | Target: {config.get('target_pct', 0.08)*100:.1f}%")
    print(f"ADX filter: >= {config.get('adx_min', 20)} | +DI > -DI: ON | RS > 0: ON")
    print()
    print(f"  {'Metric':25s} {'Original T':>12s} {'Enhanced T':>12s} {'Change':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Trades':25s} {o['trades']:12d} {e['trades']:12d} {e['trades']-o['trades']:+10d}")
    wr_o = f"{o['win_rate']:.1f}%"
    wr_e = f"{e['win_rate']:.1f}%"
    print(f"  {'Win rate':25s} {wr_o:>12s} {wr_e:>12s} {e['win_rate']-o['win_rate']:+9.1f}%")
    print(f"  {'Profit factor':25s} {o['pf']:12.2f} {e['pf']:12.2f} {e['pf']-o['pf']:+10.2f}")
    pnl_o = f"{o['pnl']*100:.1f}%"
    pnl_e = f"{e['pnl']*100:.1f}%"
    print(f"  {'Total P&L':25s} {pnl_o:>12s} {pnl_e:>12s} {(e['pnl']-o['pnl'])*100:+9.1f}%")
    aw_o = f"{o['avg_win']*100:.2f}%"
    aw_e = f"{e['avg_win']*100:.2f}%"
    print(f"  {'Avg win':25s} {aw_o:>12s} {aw_e:>12s}")
    al_o = f"{o['avg_loss']*100:.2f}%"
    al_e = f"{e['avg_loss']*100:.2f}%"
    print(f"  {'Avg loss':25s} {al_o:>12s} {al_e:>12s}")
    print()

    # Exit reason comparison
    for label, trades in [("Original", original_trades), ("Enhanced", enhanced_trades)]:
        reasons = defaultdict(int)
        for t in trades:
            reasons[t["exit_reason"]] += 1
        print(f"  {label} exit reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100 if trades else 0
            print(f"    {reason:25s} {count:3d} ({pct:.0f}%)")
        print()

    # Per-stock comparison
    print("  Per-stock breakdown (Enhanced T):")
    per_stock = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in enhanced_trades:
        per_stock[t["symbol"]]["trades"] += 1
        if t["pnl_pct"] > 0:
            per_stock[t["symbol"]]["wins"] += 1
        per_stock[t["symbol"]]["pnl"] += t["pnl_pct"]

    print(f"  {'Symbol':12s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
    for symbol in sorted(per_stock.keys()):
        s = per_stock[symbol]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  {symbol:12s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest Enhanced Session T (original T + ADX + DI + RS filters).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stocks", nargs="*", default=DEFAULT_BASKET)
    parser.add_argument("--start", default="2023-09-01")
    parser.add_argument("--end", default="2025-09-01")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--vol-mult", type=float, default=2.0)
    parser.add_argument("--rsi-min", type=float, default=35)
    parser.add_argument("--rsi-max", type=float, default=70)
    parser.add_argument("--sl", type=float, default=0.03)
    parser.add_argument("--target", type=float, default=0.08)
    parser.add_argument("--adx-min", type=float, default=20,
                        help="Minimum ADX for entry (default: 20)")
    parser.add_argument("--capital", type=float, default=25000)
    parser.add_argument("--save-trades", action="store_true")
    args = parser.parse_args()

    config = {
        "lookback_days": args.lookback,
        "volume_multiplier": args.vol_mult,
        "rsi_min": args.rsi_min,
        "rsi_max": args.rsi_max,
        "stop_loss_pct": args.sl,
        "target_pct": args.target,
        "adx_min": args.adx_min,
    }

    print(f"\nFetching {len(args.stocks)} stocks from yfinance...")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  ADX min: {args.adx_min}")
    print()

    original_trades = []
    enhanced_trades = []

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

            # Original T
            t_orig = backtest_stock(symbol, df, config, use_enhanced=False)
            original_trades.extend(t_orig)

            # Enhanced T
            t_enh = backtest_stock(symbol, df, config, use_enhanced=True)
            enhanced_trades.extend(t_enh)

            ow = sum(1 for t in t_orig if t["pnl_pct"] > 0)
            ew = sum(1 for t in t_enh if t["pnl_pct"] > 0)
            print(f"    Original: {len(t_orig)} trades, {ow} wins | Enhanced: {len(t_enh)} trades, {ew} wins")

        except Exception as e:
            print(f"ERROR: {e}")

    print_comparison(original_trades, enhanced_trades, config, args.start, args.end)

    # Capital P&L
    if enhanced_trades:
        total_pnl_rs = sum(t["pnl_pct"] * args.capital for t in enhanced_trades)
        print(f"\nEnhanced T — Capital-based P&L (Rs {args.capital:,} per position):")
        print(f"  Total P&L:     Rs {total_pnl_rs:,.2f}")
        print(f"  Avg per trade: Rs {total_pnl_rs/len(enhanced_trades):,.2f}")

    if args.save_trades and enhanced_trades:
        for t in enhanced_trades:
            if hasattr(t.get("entry_date"), "strftime"):
                t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
            if hasattr(t.get("exit_date"), "strftime"):
                t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")
        with open("enhanced_t_trades.json", "w") as f:
            json.dump(enhanced_trades, f, indent=2, default=str)
        print("\nTrades saved to enhanced_t_trades.json")

    return enhanced_trades


if __name__ == "__main__":
    main()
