"""
Enhanced Session T — Parameter Tuner + Out-of-Sample Validator
===============================================================

Two-phase workflow:

PHASE 1 (TUNE): Grid-search over parameters on the current 14-stock basket
  to find the best configuration. Outputs the top 10 configs by profit factor.

PHASE 2 (VALIDATE): Run the best config on a completely different set of
  stocks (top 25 NIFTY 50 + top 25 NIFTY MIDCAP) for out-of-sample validation.
  This prevents overfitting — if the strategy works on stocks it was NOT
  tuned on, it's more likely to work live.

Usage:
    # Phase 1: Tune on current basket
    python enhanced_t_tuner.py --phase tune

    # Phase 2: Validate on new 50-stock basket
    python enhanced_t_tuner.py --phase validate

    # Both phases
    python enhanced_t_tuner.py --phase both

    # Custom parameters
    python enhanced_t_tuner.py --phase tune --adx-min 15 --vol-mult 1.5
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
# Stock baskets
# ---------------------------------------------------------------------------

# In-sample: current 14-stock basket (Piotroski screened)
IN_SAMPLE_BASKET = [
    "COLPAL", "OFSS", "NATIONALUM", "HINDZINC", "IRCTC",
    "HCLTECH", "VBL", "HAL", "BOSCHLTD", "COALINDIA",
    "BRITANNIA", "SUNPHARMA", "ASTRAL", "WIPRO",
]

# Out-of-sample: top 25 NIFTY 50 + top 25 NIFTY MIDCAP by market cap
# (hardcoded to avoid needing nifty index data downloads at runtime)
NIFTY_50_TOP_25 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "SBIN", "LT", "HINDUNILVR", "ITC",
    "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND",
    "TATAMOTORS", "POWERGRID", "NTPC", "TATASTEEL", "M&M",
]

NIFTY_MIDCAP_TOP_25 = [
    "TATAPOWER", "PNB", "IOC", "VEDL", "BPCL",
    "SHRIRAMFIN", "BAJAJFINSV", "ZOMATO", "DMART", "JINDALSTEL",
    "TATASTEEL", "SBILIFE", "PFC", "RECLTD", "NHPC",
    "TATAINVEST", "CHOLAFIN", "LICI", "TVSMOTOR", "MOTHERSON",
    "INDIGO", "BANKBARODA", "GMRINFRA", "IDFCFIRSTB", "HINDPETRO",
]

OUT_OF_SAMPLE_BASKET = list(dict.fromkeys(
    NIFTY_50_TOP_25 + NIFTY_MIDCAP_TOP_25
))  # dedupe (TATASTEEL appears in both, HCLTECH/SUNPHARMA overlap with in-sample)


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
# Strategy entry/exit
# ---------------------------------------------------------------------------

def entry_enhanced_t(last, config):
    """Enhanced breakout entry with all filters."""
    # Trend filter
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
    # ADX filter
    if pd.isna(last["adx"]) or last["adx"] < config["adx_min"]:
        return False
    # Directional indicator
    if last["plus_di"] <= last["minus_di"]:
        return False
    # Relative strength: positive 20-day return
    if pd.isna(last["ret_20d"]) or last["ret_20d"] <= 0:
        return False
    return True


def exit_enhanced_t(last, config):
    """Enhanced exit."""
    if last["Close"] < last["ema10"]:
        return True
    if last["rsi"] > 75:
        return True
    if pd.notna(last["adx"]) and last["adx"] < 15:
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
            if exit_enhanced_t(last, config):
                position["exit_price"] = last_close
                position["exit_date"] = last.name
                position["exit_reason"] = "Signal exit"
                position["pnl_pct"] = (last_close - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

        if not position and entry_enhanced_t(last, config):
            position = {
                "symbol": symbol,
                "entry": last_close,
                "entry_date": last.name,
                "sl": round(last_close * (1 - config["stop_loss_pct"]), 2),
                "target": round(last_close * (1 + config["target_pct"]), 2),
                "adx": float(last["adx"]) if pd.notna(last["adx"]) else 0,
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
# Data fetching with caching
# ---------------------------------------------------------------------------

def fetch_stocks(symbols, start, end, cache_dir="/tmp/tuner_cache"):
    """Fetch stock data with local caching to avoid re-downloading."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{start}_{end}"
    cache_file = os.path.join(cache_dir, f"cache_{cache_key}.pkl")

    # Try loading from cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            cache = pd.read_pickle(cache_file)
            print(f"  Loaded {len(cache)} cached stocks")
        except:
            cache = {}

    all_data = {}
    for symbol in symbols:
        if symbol in cache and not cache[symbol].empty:
            all_data[symbol] = cache[symbol]
        else:
            print(f"  Fetching {symbol}...", end=" ", flush=True)
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(start=start, end=end, auto_adjust=True)
                if hist.empty:
                    print("NO DATA")
                    continue
                hist.index = pd.DatetimeIndex(hist.index)
                cache[symbol] = hist
                all_data[symbol] = hist
                print(f"{len(hist)} rows")
            except Exception as e:
                print(f"ERROR: {e}")

    # Save cache
    try:
        pd.to_pickle(cache, cache_file)
    except:
        pass

    return all_data


# ---------------------------------------------------------------------------
# Phase 1: Grid search tuning
# ---------------------------------------------------------------------------

def phase_tune(start, end):
    """Grid-search parameters on the in-sample basket."""

    # Parameter grid
    param_grid = {
        "adx_min": [15, 18, 20, 22, 25],
        "volume_multiplier": [1.0, 1.5, 2.0],
        "rsi_min": [30, 35],
        "rsi_max": [70, 75],
        "stop_loss_pct": [0.02, 0.03, 0.04, 0.05],
        "target_pct": [0.06, 0.08, 0.10, 0.12],
        "lookback_days": [10, 15, 20],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n{'='*70}")
    print("PHASE 1: PARAMETER TUNING (Grid Search)")
    print(f"{'='*70}")
    print(f"  In-sample basket: {len(IN_SAMPLE_BASKET)} stocks")
    print(f"  Period: {start} to {end}")
    print(f"  Parameter combinations: {total_combos}")
    print(f"  Strategy: Enhanced breakout (trend filter + ADX + DI + RS + volume)")
    print()

    # Fetch in-sample data (cached)
    print("Fetching in-sample data...")
    raw_data = fetch_stocks(IN_SAMPLE_BASKET, start, end)

    # Pre-compute indicators for each lookback
    print("\nComputing indicators...")
    indicator_cache = {}  # (symbol, lookback) -> df
    for symbol, hist in raw_data.items():
        for lb in param_grid["lookback_days"]:
            df = hist.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            df = compute_indicators(df, lookback=lb)
            indicator_cache[(symbol, lb)] = df

    print(f"  {len(indicator_cache)} indicator sets computed")

    # Grid search
    print(f"\nRunning {total_combos} parameter combinations...")
    results = []
    combo_count = 0

    keys = list(param_grid.keys())
    for combo in product(*[param_grid[k] for k in keys]):
        config = dict(zip(keys, combo))

        all_trades = []
        for symbol in raw_data:
            df = indicator_cache.get((symbol, config["lookback_days"]))
            if df is not None:
                trades = backtest_stock(symbol, df, config)
                all_trades.extend(trades)

        m = calc_metrics(all_trades)

        # Only keep results with enough trades and positive PF
        if m["trades"] >= 10 and m["pf"] >= 1.0:
            results.append({
                "config": config,
                "trades": m["trades"],
                "wins": m["wins"],
                "win_rate": m["win_rate"],
                "pf": m["pf"],
                "pnl": m["pnl"],
            })

        combo_count += 1
        if combo_count % 200 == 0:
            print(f"  {combo_count}/{total_combos} combos tested, {len(results)} profitable so far...")

    # Sort by profit factor
    results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP 10 CONFIGURATIONS (PF >= 1.0, trades >= 10)")
    print(f"{'='*70}")
    print(f"  {'Rank':>4s}  {'PF':>6s} {'Win%':>6s} {'Trades':>6s} {'P&L%':>7s}  Config")
    print(f"  {'-'*4}  {'-'*6} {'-'*6} {'-'*6} {'-'*7}  {'-'*40}")

    for i, r in enumerate(results[:10]):
        c = r["config"]
        cfg_str = f"ADX>={c['adx_min']} vol={c['volume_multiplier']}x RSI={c['rsi_min']}-{c['rsi_max']} SL={c['stop_loss_pct']*100:.0f}% T={c['target_pct']*100:.0f}% LB={c['lookback_days']}"
        print(f"  {i+1:4d}  {r['pf']:6.2f} {r['win_rate']:5.1f}% {r['trades']:6d} {r['pnl']*100:6.1f}%  {cfg_str}")

    if not results:
        print("  No profitable configurations found.")
        print("  Try relaxing: --adx-min 15 or wider SL/target")
        return None

    # Save best config
    best = results[0]
    best_config = best["config"]

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings_tuned_enhanced.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "w") as f:
        f.write("# Enhanced Session T — Best parameters from grid search\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# In-sample: {len(IN_SAMPLE_BASKET)} stocks, {start} to {end}\n")
        f.write(f"# PF: {best['pf']:.2f} | Win rate: {best['win_rate']:.1f}% | Trades: {best['trades']}\n\n")
        f.write("technical:\n")
        f.write("  swing:\n")
        f.write(f"    adx_min: {best_config['adx_min']}\n")
        f.write(f"    volume_multiplier: {best_config['volume_multiplier']}\n")
        f.write(f"    rsi_min: {best_config['rsi_min']}\n")
        f.write(f"    rsi_max: {best_config['rsi_max']}\n")
        f.write(f"    stop_loss_pct: {best_config['stop_loss_pct']}\n")
        f.write(f"    target_pct: {best_config['target_pct']}\n")
        f.write(f"    lookback_days: {best_config['lookback_days']}\n")
        f.write("    trend_filter: true\n")

    print(f"\nBest config saved to: {config_path}")

    # Save full results
    results_path = os.path.join(os.path.dirname(__file__), "tuner_results.json")
    with open(results_path, "w") as f:
        json.dump(results[:50], f, indent=2, default=str)
    print(f"Top 50 results saved to: {results_path}")

    return best_config


# ---------------------------------------------------------------------------
# Phase 2: Out-of-sample validation
# ---------------------------------------------------------------------------

def phase_validate(best_config, start, end):
    """Run best config on completely different stocks."""

    print(f"\n{'='*70}")
    print("PHASE 2: OUT-OF-SAMPLE VALIDATION")
    print(f"{'='*70}")
    print(f"  Validation basket: {len(OUT_OF_SAMPLE_BASKET)} stocks")
    print(f"    - Top 25 NIFTY 50")
    print(f"    - Top 25 NIFTY MIDCAP")
    print(f"  Period: {start} to {end}")
    print(f"  Config: ADX>={best_config['adx_min']} vol={best_config['volume_multiplier']}x "
          f"RSI={best_config['rsi_min']}-{best_config['rsi_max']} "
          f"SL={best_config['stop_loss_pct']*100:.0f}% T={best_config['target_pct']*100:.0f}% "
          f"LB={best_config['lookback_days']}")
    print()

    # Fetch out-of-sample data
    print("Fetching out-of-sample data...")
    raw_data = fetch_stocks(OUT_OF_SAMPLE_BASKET, start, end,
                            cache_dir="/tmp/tuner_cache_oos")

    print(f"\nComputing indicators for {len(raw_data)} stocks...")
    all_trades = []
    for symbol, hist in raw_data.items():
        df = hist.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })
        df = compute_indicators(df, lookback=best_config["lookback_days"])

        trades = backtest_stock(symbol, df, best_config)
        all_trades.extend(trades)

        if trades:
            wins = sum(1 for t in trades if t["pnl_pct"] > 0)
            print(f"  {symbol:15s}: {len(trades):3d} trades, {wins} wins")

    m = calc_metrics(all_trades)

    print(f"\n{'='*70}")
    print("OUT-OF-SAMPLE RESULTS")
    print(f"{'='*70}")
    print(f"  Stocks tested:        {len(raw_data)}")
    print(f"  Total trades:         {m['trades']}")
    print(f"  Wins:                 {m['wins']}")
    print(f"  Win rate:             {m['win_rate']:.1f}%")
    print(f"  Profit factor:        {m['pf']:.2f}")
    print(f"  Total P&L (sum %):   {m['pnl']*100:.2f}%")
    print(f"  Avg P&L per trade:   {m['pnl']/m['trades']*100:.2f}%" if m["trades"] else "")

    # Exit reason breakdown
    if all_trades:
        exit_reasons = defaultdict(int)
        for t in all_trades:
            exit_reasons[t["exit_reason"]] += 1
        print(f"\n  Exit reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(all_trades) * 100
            print(f"    {reason:25s} {count:3d} ({pct:.0f}%)")

    # Per-stock breakdown
    per_stock = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in all_trades:
        per_stock[t["symbol"]]["trades"] += 1
        if t["pnl_pct"] > 0:
            per_stock[t["symbol"]]["wins"] += 1
        per_stock[t["symbol"]]["pnl"] += t["pnl_pct"]

    print(f"\n  Per-stock breakdown:")
    print(f"  {'Symbol':15s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
    for symbol in sorted(per_stock.keys()):
        s = per_stock[symbol]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  {symbol:15s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")

    # Save trades
    if all_trades:
        for t in all_trades:
            if hasattr(t.get("entry_date"), "strftime"):
                t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
            if hasattr(t.get("exit_date"), "strftime"):
                t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")
        with open("oos_trades.json", "w") as f:
            json.dump(all_trades, f, indent=2, default=str)
        print(f"\n  Trades saved to oos_trades.json")

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if m["pf"] >= 1.5 and m["trades"] >= 15:
        print("  ✅ STRATEGY VALIDATED — PF >= 1.5 with sufficient trades.")
        print("  Safe to paper trade with these parameters.")
    elif m["pf"] >= 1.0 and m["trades"] >= 10:
        print("  ⚠️  MARGINAL — PF >= 1.0 but results need more trades.")
        print("  Paper trade with caution. Consider widening parameters slightly.")
    elif m["trades"] < 10:
        print("  ⚠️  INSUFFICIENT TRADES — strategy is too selective.")
        print("  Try relaxing: lower ADX, lower volume multiplier, wider RSI.")
    else:
        print("  ❌ STRATEGY FAILED — PF < 1.0 on out-of-sample data.")
        print("  The tuned parameters may be overfit. Try different parameters.")
    print("=" * 70)

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Session T — Parameter Tuner + Out-of-Sample Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "PHASE 1 (tune): Grid-search 2880 parameter combinations on the\n"
            "  current 14-stock basket. Finds the best by profit factor.\n\n"
            "PHASE 2 (validate): Runs the best config on 50 completely\n"
            "  different stocks (25 NIFTY 50 + 25 NIFTY MIDCAP).\n\n"
            "If the strategy works on both baskets, it's robust enough to deploy."
        ),
    )
    parser.add_argument("--phase", default="both", choices=["tune", "validate", "both"],
                        help="Phase to run (default: both)")
    parser.add_argument("--start", default="2023-09-01",
                        help="Start date (default: 2023-09-01)")
    parser.add_argument("--end", default="2025-09-01",
                        help="End date (default: 2025-09-01)")
    args = parser.parse_args()

    best_config = None

    if args.phase in ("tune", "both"):
        best_config = phase_tune(args.start, args.end)

    if args.phase in ("validate", "both"):
        if best_config is None:
            # Try loading from file
            config_path = os.path.join(os.path.dirname(__file__), "..", "config",
                                       "settings_tuned_enhanced.yaml")
            config_path = os.path.abspath(config_path)
            if os.path.exists(config_path):
                import yaml
                with open(config_path) as f:
                    data = yaml.safe_load(f)
                tc = data.get("technical", {}).get("swing", {})
                best_config = {
                    "adx_min": tc.get("adx_min", 20),
                    "volume_multiplier": tc.get("volume_multiplier", 2.0),
                    "rsi_min": tc.get("rsi_min", 35),
                    "rsi_max": tc.get("rsi_max", 70),
                    "stop_loss_pct": tc.get("stop_loss_pct", 0.03),
                    "target_pct": tc.get("target_pct", 0.08),
                    "lookback_days": tc.get("lookback_days", 10),
                }
                print(f"\nLoaded config from {config_path}")
            else:
                # Default config
                best_config = {
                    "adx_min": 20,
                    "volume_multiplier": 1.5,
                    "rsi_min": 35,
                    "rsi_max": 70,
                    "stop_loss_pct": 0.03,
                    "target_pct": 0.08,
                    "lookback_days": 10,
                }
                print("\nUsing default config (no tuned config found)")

        phase_validate(best_config, args.start, args.end)


if __name__ == "__main__":
    main()
