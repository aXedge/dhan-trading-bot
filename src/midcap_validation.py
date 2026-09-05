"""
Midcap Validation — Test on 75 NIFTY MIDCAP stocks
====================================================

Tests multiple parameter sets on a large basket of 75 NIFTY MIDCAP
stocks to find configurations that generalize (not overfit).

Parameter sets tested:
  1. Original T:   trend filter + breakout + 2x vol + RSI 35-70 + SL 3% + T 8%
  2. Enhanced T:    + ADX >= 20 + +DI > -DI + RS > 0
  3. Relaxed:       ADX >= 15, vol 1.5x, SL 4%, T 10%
  4. Wide:          ADX >= 15, vol 1.0x, SL 5%, T 10%

Usage:
    python midcap_validation.py
    python midcap_validation.py --start 2023-09-01 --end 2025-09-01
"""

import argparse
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------------------------------------------------------
# Top 75 NIFTY MIDCAP stocks
# ---------------------------------------------------------------------------

MIDCAP_75 = [
    # Large midcaps
    "TATAPOWER", "PNB", "IOC", "VEDL", "BPCL",
    "SHRIRAMFIN", "BAJAJFINSV", "DMART", "JINDALSTEL", "SBILIFE",
    "PFC", "RECLTD", "NHPC", "TATAINVEST", "CHOLAFIN",
    "LICI", "TVSMOTOR", "MOTHERSON", "INDIGO", "BANKBARODA",
    "IDFCFIRSTB", "HINDPETRO", "TATAELXSI", "BANDHANBNK", "MAXHEALTH",
    # Mid midcaps
    "UPL", "GODREJPROP", "AMBUJACEM", "MRF", "PAGEIND",
    "DABUR", "COLPAL", "SIEMENS", "ABB", "BEL",
    "BHEL", "GAIL", "ONGC", "NMDC", "SAIL",
    "TATACHEM", "PIIND", "COROMANDEL", "CHAMBLFERT", "DEEPAKNTR",
    "SRF", "ASTRAL", "POLYCAB", "CROMPTON", "VOLTAS",
    # Smaller midcaps
    "CONCOR", "IRCTC", "RBLBANK", "YESBANK", "FEDERALBNK",
    "CANBK", "UNIONBANK", "INDIANB", "MAHABANK", "AUBANK",
    "JKCEMENT", "DALBHARAT", "ACC", "RAMCOCEM", "BERGEPAINT",
    "UNITEDSPIRITS", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "MARICO",
    "BIOCON", "LUPIN", "PIDILIT", "HAVELLS", "TRENT",
    "BATAINDIA", "ABFRL", "MCDOWELL-N", "SUPREMEIND", "GYANALAXMI",
]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df, lookback=10):
    df = df.copy()
    for span in [10, 20, 50, 200]:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)
    df = compute_adx(df, period=14)
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
# Strategy
# ---------------------------------------------------------------------------

def entry_signal(last, config):
    """Configurable entry — filters toggle on/off based on config."""
    # Trend filter (always on)
    if last["ema50"] <= last["ema200"]:
        return False
    # Breakout
    if last["Close"] <= last["high_lookback"]:
        return False
    # Volume
    if last["Volume"] < last["vol_avg"] * config["volume_multiplier"]:
        return False
    # RSI range
    if not (config["rsi_min"] <= last["rsi"] <= config["rsi_max"]):
        return False
    # ADX filter (optional)
    if config.get("use_adx", False):
        if pd.isna(last["adx"]) or last["adx"] < config["adx_min"]:
            return False
        if config.get("use_di", False) and last["plus_di"] <= last["minus_di"]:
            return False
    # Relative strength (optional)
    if config.get("use_rs", False):
        if pd.isna(last["ret_20d"]) or last["ret_20d"] <= 0:
            return False
    return True


def exit_signal(last, config):
    """Configurable exit."""
    if last["Close"] < last["ema10"]:
        return True
    if last["rsi"] > 75:
        return True
    if config.get("use_adx", False) and pd.notna(last["adx"]) and last["adx"] < 15:
        return True
    return False


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_stock(symbol, df, config):
    trades = []
    position = None
    warmup = 210
    if len(df) < warmup:
        return trades

    for i in range(warmup, len(df)):
        last = df.iloc[i]
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])

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
            if exit_signal(last, config):
                position["exit_price"] = last_close
                position["exit_date"] = last.name
                position["exit_reason"] = "Signal exit"
                position["pnl_pct"] = (last_close - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

        if not position and entry_signal(last, config):
            position = {
                "symbol": symbol,
                "entry": last_close,
                "entry_date": last.name,
                "sl": round(last_close * (1 - config["stop_loss_pct"]), 2),
                "target": round(last_close * (1 + config["target_pct"]), 2),
            }

    if position:
        last = df.iloc[-1]
        position["exit_price"] = float(last["Close"])
        position["exit_date"] = last.name
        position["exit_reason"] = "End of data"
        position["pnl_pct"] = (position["exit_price"] - position["entry"]) / position["entry"]
        trades.append(position)

    return trades


def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wins": 0, "win_rate": 0, "pf": 0, "pnl": 0}
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gp = sum(t["pnl_pct"] for t in wins)
    gl = abs(sum(t["pnl_pct"] for t in losses))
    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": gp / gl if gl > 0 else float("inf"),
        "pnl": sum(t["pnl_pct"] for t in trades),
    }


# ---------------------------------------------------------------------------
# Parameter sets to test
# ---------------------------------------------------------------------------

PARAM_SETS = {
    "Original T": {
        "volume_multiplier": 2.0,
        "rsi_min": 35,
        "rsi_max": 70,
        "stop_loss_pct": 0.03,
        "target_pct": 0.08,
        "lookback_days": 10,
        "use_adx": False,
        "use_di": False,
        "use_rs": False,
    },
    "Enhanced T (ADX 20)": {
        "volume_multiplier": 2.0,
        "rsi_min": 35,
        "rsi_max": 70,
        "stop_loss_pct": 0.03,
        "target_pct": 0.08,
        "lookback_days": 10,
        "use_adx": True,
        "adx_min": 20,
        "use_di": True,
        "use_rs": True,
    },
    "Enhanced T (ADX 25)": {
        "volume_multiplier": 1.5,
        "rsi_min": 30,
        "rsi_max": 75,
        "stop_loss_pct": 0.02,
        "target_pct": 0.08,
        "lookback_days": 15,
        "use_adx": True,
        "adx_min": 25,
        "use_di": True,
        "use_rs": True,
    },
    "Relaxed": {
        "volume_multiplier": 1.5,
        "rsi_min": 30,
        "rsi_max": 75,
        "stop_loss_pct": 0.04,
        "target_pct": 0.10,
        "lookback_days": 15,
        "use_adx": True,
        "adx_min": 15,
        "use_di": False,
        "use_rs": False,
    },
    "Wide net": {
        "volume_multiplier": 1.0,
        "rsi_min": 30,
        "rsi_max": 80,
        "stop_loss_pct": 0.05,
        "target_pct": 0.10,
        "lookback_days": 20,
        "use_adx": True,
        "adx_min": 15,
        "use_di": False,
        "use_rs": False,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate strategy on 75 NIFTY MIDCAP stocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default="2023-09-01")
    parser.add_argument("--end", default="2025-09-01")
    parser.add_argument("--capital", type=float, default=25000)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MIDCAP VALIDATION — {len(MIDCAP_75)} NIFTY MIDCAP stocks")
    print(f"{'='*70}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Parameter sets: {len(PARAM_SETS)}")
    for name, cfg in PARAM_SETS.items():
        adx_str = f"ADX>={cfg.get('adx_min', 'N/A')}" if cfg.get("use_adx") else "no ADX"
        print(f"    {name:25s} {adx_str:10s} vol={cfg['volume_multiplier']}x "
              f"SL={cfg['stop_loss_pct']*100:.0f}% T={cfg['target_pct']*100:.0f}% "
              f"LB={cfg['lookback_days']}")
    print()

    # Fetch data
    print("Fetching stock data from yfinance...")
    raw_data = {}
    failed = []
    for symbol in MIDCAP_75:
        yf_symbol = f"{symbol}.NS"
        print(f"  {symbol:15s}...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(start=args.start, end=args.end, auto_adjust=True)
            if hist.empty:
                print("NO DATA")
                failed.append(symbol)
                continue
            hist.index = pd.DatetimeIndex(hist.index)
            raw_data[symbol] = hist
            print(f"{len(hist)} rows")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(symbol)

    print(f"\n  {len(raw_data)} stocks loaded, {len(failed)} failed: {failed}")
    print()

    # Run each parameter set
    all_results = {}

    for set_name, config in PARAM_SETS.items():
        print(f"\n--- Testing: {set_name} ---")

        all_trades = []
        for symbol, hist in raw_data.items():
            df = hist.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            df = compute_indicators(df, lookback=config["lookback_days"])
            trades = backtest_stock(symbol, df, config)
            all_trades.extend(trades)

        m = calc_metrics(all_trades)
        all_results[set_name] = {"metrics": m, "trades": all_trades}

        print(f"  Trades: {m['trades']} | Wins: {m['wins']} | "
              f"Win rate: {m['win_rate']:.1f}% | PF: {m['pf']:.2f} | "
              f"P&L: {m['pnl']*100:.1f}%")

        # Exit reason breakdown
        if all_trades:
            reasons = defaultdict(int)
            for t in all_trades:
                reasons[t["exit_reason"]] += 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / len(all_trades) * 100
                print(f"    {reason:20s} {count:3d} ({pct:.0f}%)")

    # Summary comparison table
    print(f"\n{'='*70}")
    print("SUMMARY — All parameter sets on 75 NIFTY MIDCAP stocks")
    print(f"{'='*70}")
    print(f"  {'Config':25s} {'Trades':>7s} {'Win%':>6s} {'PF':>7s} {'P&L%':>8s} {'Rs P&L':>10s}")
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*10}")

    for set_name in PARAM_SETS:
        m = all_results[set_name]["metrics"]
        pnl_rs = m["pnl"] * args.capital if m["trades"] > 0 else 0
        print(f"  {set_name:25s} {m['trades']:7d} {m['win_rate']:5.1f}% "
              f"{m['pf']:7.2f} {m['pnl']*100:7.1f}% Rs {pnl_rs:,.0f}")

    # Best config per-stock breakdown
    best_set = max(all_results.keys(), key=lambda k: all_results[k]["metrics"]["pf"])
    best_trades = all_results[best_set]["trades"]

    if best_trades:
        print(f"\n  Best config: {best_set}")
        print(f"  Per-stock breakdown (top 15 by trade count):")
        per_stock = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        for t in best_trades:
            per_stock[t["symbol"]]["trades"] += 1
            if t["pnl_pct"] > 0:
                per_stock[t["symbol"]]["wins"] += 1
            per_stock[t["symbol"]]["pnl"] += t["pnl_pct"]

        sorted_stocks = sorted(per_stock.items(), key=lambda x: -x[1]["trades"])[:15]
        print(f"  {'Symbol':15s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
        print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
        for symbol, s in sorted_stocks:
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            print(f"  {symbol:15s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")

    # Save all trades
    for set_name, data in all_results.items():
        trades = data["trades"]
        if trades:
            safe_name = set_name.replace(" ", "_").replace("(", "").replace(")", "")
            for t in trades:
                if hasattr(t.get("entry_date"), "strftime"):
                    t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
                if hasattr(t.get("exit_date"), "strftime"):
                    t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")
            with open(f"midcap_{safe_name}_trades.json", "w") as f:
                json.dump(trades, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    best_m = all_results[best_set]["metrics"]
    if best_m["trades"] >= 30 and best_m["pf"] >= 1.3:
        print(f"  Best config: {best_set}")
        print(f"  PF: {best_m['pf']:.2f} | Win rate: {best_m['win_rate']:.1f}% | Trades: {best_m['trades']}")
        print(f"  With 30+ trades and PF >= 1.3, this is statistically meaningful.")
        print(f"  Recommend deploying this config for paper trading.")
    elif best_m["trades"] >= 30 and best_m["pf"] >= 1.0:
        print(f"  Best config: {best_set}")
        print(f"  PF: {best_m['pf']:.2f} | Win rate: {best_m['win_rate']:.1f}% | Trades: {best_m['trades']}")
        print(f"  PF >= 1.0 with 30+ trades — marginal but worth paper trading.")
    else:
        print(f"  No config achieved PF >= 1.0 with 30+ trades.")
        print(f"  The breakout strategy may not work on midcap stocks.")
        print(f"  Consider: different strategy (mean reversion), wider stops, or different basket.")
    print("=" * 70)


if __name__ == "__main__":
    main()
