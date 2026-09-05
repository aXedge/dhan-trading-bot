"""
Unified backtest runner — works with any strategy module.

Usage:
    python -m backtests.run --strategy reversal --basket nifty50
    python -m backtests.run --strategy bb_squeeze --basket midcap
    python -m backtests.run --compare --basket midcap
"""

import argparse
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import fetch_yfinance, NIFTY_50, MIDCAP_75
from common.backtest import backtest_stock
from common.reporting import (
    calc_metrics, print_summary, print_comparison_table, print_verdict,
    print_exit_reasons, print_per_stock
)

import strategies.breakout as _breakout
import strategies.enhanced_breakout as _enhanced_breakout
import strategies.positional_pullback as _positional_pullback
import strategies.momentum_acceleration as _momentum_acceleration
import strategies.supertrend as _supertrend
import strategies.reversal as _reversal
import strategies.bb_squeeze as _bb_squeeze

STRATEGIES = {
    "breakout": _breakout,
    "enhanced_breakout": _enhanced_breakout,
    "positional_pullback": _positional_pullback,
    "momentum_acceleration": _momentum_acceleration,
    "supertrend": _supertrend,
    "reversal": _reversal,
    "bb_squeeze": _bb_squeeze,
}

BASKETS = {
    "nifty50": NIFTY_50,
    "midcap": MIDCAP_75,
}


def run_strategy(strategy_name, basket_name, start, end, overrides=None):
    module = STRATEGIES[strategy_name]
    config = module.DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)

    symbols = list(dict.fromkeys(BASKETS[basket_name]))
    print(f"\nFetching {len(symbols)} stocks from yfinance...")
    raw_data = fetch_yfinance(symbols, start, end)
    print(f"  {len(raw_data)} stocks loaded")
    print(f"  Computing indicators and running backtest...")

    all_trades = []
    for symbol, hist in raw_data.items():
        df = hist.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })
        df = module.prepare(df, config)
        trades = backtest_stock(symbol, df, config, module.should_enter, module.should_exit)
        all_trades.extend(trades)

    m = calc_metrics(all_trades)
    return {"metrics": m, "trades": all_trades, "config": config}


def run_comparison(basket_name, start, end):
    symbols = list(dict.fromkeys(BASKETS[basket_name]))
    print(f"\n{'='*70}")
    print(f"STRATEGY COMPARISON — {basket_name.upper()} ({len(symbols)} stocks)")
    print(f"  Period: {start} to {end}")
    print(f"{'='*70}")
    print(f"\nFetching {len(symbols)} stocks from yfinance...")
    raw_data = fetch_yfinance(symbols, start, end)
    print(f"  {len(raw_data)} stocks loaded")

    results = {}
    for strategy_name, module in STRATEGIES.items():
        config = module.DEFAULT_CONFIG.copy()
        print(f"\n--- {strategy_name} ---")
        print(f"  Config: SL={config['stop_loss_pct']*100:.0f}% T={config['target_pct']*100:.0f}%")

        all_trades = []
        for symbol, hist in raw_data.items():
            df = hist.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            df = module.prepare(df, config)
            trades = backtest_stock(symbol, df, config, module.should_enter, module.should_exit)
            all_trades.extend(trades)

        m = calc_metrics(all_trades)
        results[strategy_name] = {"metrics": m, "trades": all_trades}
        print(f"  Trades: {m['trades']} | Wins: {m['wins']} | "
              f"Win rate: {m['win_rate']:.1f}% | PF: {m['pf']:.2f} | "
              f"P&L: {m['pnl']*100:.1f}%")
        print_exit_reasons(all_trades, indent="    ")

    print_comparison_table(results, f"COMPARISON — {basket_name.upper()}")
    best = max(results.keys(), key=lambda k: results[k]["metrics"]["pf"])
    best_m = results[best]["metrics"]
    print(f"\n  Best strategy: {best}")
    print(f"  PF: {best_m['pf']:.2f} | Win rate: {best_m['win_rate']:.1f}% | Trades: {best_m['trades']}")
    print_per_stock(results[best]["trades"])
    print_verdict(best_m["pf"], best_m["trades"], best_m["win_rate"])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified backtest runner for any strategy on any basket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()),
                        help="Strategy to run")
    parser.add_argument("--basket", default="nifty50", choices=list(BASKETS.keys()),
                        help="Stock basket (default: nifty50)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all strategies and compare")
    parser.add_argument("--start", default="2023-09-01")
    parser.add_argument("--end", default="2025-09-01")
    parser.add_argument("--capital", type=float, default=25000)
    parser.add_argument("--sl", type=float, help="Override stop loss pct")
    parser.add_argument("--target", type=float, help="Override target pct")
    parser.add_argument("--adx-min", type=float, help="Override ADX minimum")
    parser.add_argument("--vol-mult", type=float, help="Override volume multiplier")
    args = parser.parse_args()

    overrides = {}
    if args.sl: overrides["stop_loss_pct"] = args.sl
    if args.target: overrides["target_pct"] = args.target
    if args.adx_min: overrides["adx_min"] = args.adx_min
    if args.vol_mult: overrides["volume_multiplier"] = args.vol_mult

    if args.compare:
        results = run_comparison(args.basket, args.start, args.end)
    elif args.strategy:
        result = run_strategy(args.strategy, args.basket, args.start, args.end, overrides)
        config = result["config"]
        trades = result["trades"]
        m = result["metrics"]
        title = f"{args.strategy.upper()} on {args.basket.upper()}"
        print_summary(title, trades, config)
        if trades:
            pnl_rs = m["pnl"] * args.capital
            print(f"\nCapital-based P&L (Rs {args.capital:,} per position):")
            print(f"  Total P&L:     Rs {pnl_rs:,.2f}")
            print(f"  Avg per trade: Rs {pnl_rs/len(trades):,.2f}")
            print(f"  Best trade:    Rs {max(t['pnl_pct'] for t in trades)*args.capital:,.2f}")
            print(f"  Worst trade:   Rs {min(t['pnl_pct'] for t in trades)*args.capital:,.2f}")
        print_verdict(m["pf"], m["trades"], m["win_rate"])
        if trades:
            for t in trades:
                if hasattr(t.get("entry_date"), "strftime"):
                    t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
                if hasattr(t.get("exit_date"), "strftime"):
                    t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")
            with open(f"{args.basket}_{args.strategy}_trades.json", "w") as f:
                json.dump(trades, f, indent=2, default=str)
            print(f"\nTrades saved to {args.basket}_{args.strategy}_trades.json")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
