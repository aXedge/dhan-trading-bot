"""
Momentum Acceleration Parameter Tuner
======================================

Grid-searches v3 parameters on NIFTY 50 and Midcap baskets.
Tests combinations of:
  - Stop loss: 4%, 5%, 6%, 7%
  - Chandelier ATR multiplier: 2.5, 3.0, 3.5, 4.0
  - Chandelier lookback: 10, 15, 20
  - RSI exit: 70, 75, 80

Total: 4 x 4 x 3 x 3 = 144 combinations per basket.

Usage:
    python -m backtests.tune_momentum --basket nifty50
    python -m backtests.tune_momentum --basket midcap
"""

import argparse
import os
import sys
from itertools import product
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import fetch_yfinance, NIFTY_50, MIDCAP_75
from common.backtest import backtest_stock
from common.reporting import calc_metrics
import strategies.momentum_acceleration as ma


# Parameter grid
SL_VALUES = [0.04, 0.05, 0.06, 0.07]
ATR_MULTS = [2.5, 3.0, 3.5, 4.0]
LOOKBACKS = [10, 15, 20]
RSI_EXITS = [70, 75, 80]

BASKETS = {"nifty50": NIFTY_50, "midcap": MIDCAP_75}


def tune(basket_name, start="2023-09-01", end="2025-09-01"):
    symbols = list(dict.fromkeys(BASKETS[basket_name]))
    print(f"\nFetching {len(symbols)} stocks from yfinance...")
    raw_data = fetch_yfinance(symbols, start, end)
    print(f"  {len(raw_data)} stocks loaded")

    # Prepare data once with a superset of indicators
    prepared = {}
    for sym, df in raw_data.items():
        # Use default config for prepare (indicators don't depend on exit params)
        prepared[sym] = ma.prepare(df, ma.DEFAULT_CONFIG)

    results = []

    total = len(SL_VALUES) * len(ATR_MULTS) * len(LOOKBACKS) * len(RSI_EXITS)
    print(f"  Testing {total} parameter combinations...")

    for sl, atr_mult, lookback, rsi_exit in product(SL_VALUES, ATR_MULTS, LOOKBACKS, RSI_EXITS):
        config = ma.DEFAULT_CONFIG.copy()
        config["stop_loss_pct"] = sl
        config["trail_atr_mult"] = atr_mult
        config["chandelier_lookback"] = lookback
        config["rsi_exit"] = rsi_exit

        # Re-prepare only if lookback or atr_mult changed (affects chandelier_stop)
        # For efficiency, recompute chandelier columns on the fly
        all_trades = []
        for sym, df in prepared.items():
            df_copy = df.copy()
            df_copy["chandelier_max"] = df_copy["High"].rolling(lookback, min_periods=5).max()
            df_copy["chandelier_stop"] = df_copy["chandelier_max"] - atr_mult * df_copy["atr"]
            trades = backtest_stock(sym, df_copy, config, ma.should_enter, ma.should_exit)
            all_trades.extend(trades)

        m = calc_metrics(all_trades)
        results.append({
            "sl": sl,
            "atr_mult": atr_mult,
            "lookback": lookback,
            "rsi_exit": rsi_exit,
            "trades": m["trades"],
            "wins": m["wins"],
            "win_rate": m["win_rate"],
            "pf": m["pf"],
            "pnl": m["pnl"],
            "avg_win": m.get("avg_win", 0),
            "avg_loss": m.get("avg_loss", 0),
        })

    # Sort by PF
    results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n{'='*90}")
    print(f"TOP 20 PARAMETER COMBINATIONS — {basket_name.upper()}")
    print(f"{'='*90}")
    print(f"{'SL':>5} {'ATR':>5} {'LB':>4} {'RSI':>5} {'Trades':>7} {'Win%':>6} {'PF':>6} {'P&L%':>8} {'AvgW':>6} {'AvgL':>6}")
    print(f"{'-'*5} {'-'*5} {'-'*4} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")

    for r in results[:20]:
        print(f"{r['sl']*100:>4.0f}% {r['atr_mult']:>5.1f} {r['lookback']:>4} {r['rsi_exit']:>5} "
              f"{r['trades']:>7} {r['win_rate']:>5.1f}% {r['pf']:>6.2f} {r['pnl']*100:>7.1f}% "
              f"{r['avg_win']*100:>5.1f}% {r['avg_loss']*100:>5.1f}%")

    # Also show configs with >50 trades and PF > 1.0
    profitable = [r for r in results if r["pf"] >= 1.0 and r["trades"] >= 30]
    print(f"\n  Configurations with PF >= 1.0 and >= 30 trades: {len(profitable)}")
    if profitable:
        print(f"  Best: SL={profitable[0]['sl']*100:.0f}% ATR={profitable[0]['atr_mult']} "
              f"LB={profitable[0]['lookback']} RSI={profitable[0]['rsi_exit']} "
              f"PF={profitable[0]['pf']:.2f} Trades={profitable[0]['trades']} "
              f"Win={profitable[0]['win_rate']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Tune momentum acceleration parameters")
    parser.add_argument("--basket", default="nifty50", choices=list(BASKETS.keys()))
    parser.add_argument("--start", default="2023-09-01")
    parser.add_argument("--end", default="2025-09-01")
    args = parser.parse_args()

    results = tune(args.basket, args.start, args.end)

    # Save results
    import json
    with open(f"{args.basket}_momentum_tuning.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.basket}_momentum_tuning.json")


if __name__ == "__main__":
    main()
