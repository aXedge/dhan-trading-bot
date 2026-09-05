"""
Unified backtest runner — works with any strategy module.

Tests a strategy on any stock basket and prints results using the
common reporting module. Supports running multiple strategies or
multiple parameter sets side-by-side.

Usage:
    # Run positional pullback on NIFTY 50
    python -m backtests.run --strategy positional_pullback --basket nifty50

    # Run breakout on midcaps
    python -m backtests.run --strategy breakout --basket midcap

    # Run enhanced_breakout with custom params
    python -m backtests.run --strategy enhanced_breakout --basket nifty50 \\
        --adx-min 25 --sl 0.05 --target 0.10

    # Compare all 3 strategies on the same basket
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

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import fetch_yfinance, NIFTY_50, MIDCAP_75
from common.backtest import backtest_stock
from common.reporting import (
    calc_metrics, print_summary, print_comparison_table, print_verdict,
    print_exit_reasons, print_per_stock
)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {}


def register_strategy(name):
    """Decorator to register a strategy."""
    def decorator(module):
        STRATEGIES[name] = module
        return module
    return decorator


# Import and register all strategies
import strategies.breakout as _breakout
import strategies.enhanced_breakout as _enhanced_breakout
import strategies.positional_pullback as _positional_pullback

STRATEGIES["breakout"] = _breakout
STRATEGIES["enhanced_breakout"] = _enhanced_breakout
STRATEGIES["positional_pullback"] = _positional_pullback


# ---------------------------------------------------------------------------
# Basket registry
# ---------------------------------------------------------------------------

BASKETS = {
    "nifty50": NIFTY_50,
    "midcap": MIDCAP_75,
}


# ---------------------------------------------------------------------------
# Run a single strategy on a basket
# ---------------------------------------------------------------------------

def run_strategy(strategy_name: str, basket_name: str, start: str, end: str,
                 overrides: dict = None) -> dict:
    """
    Run a strategy on a basket.

    Returns: {"metrics": {...}, "trades": [...]}
    """
    module = STRATEGIES[strategy_name]
    config = module.DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)

    symbols = BASKETS[basket_name]
    symbols = list(dict.fromkeys(symbols))  # dedupe

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


# ---------------------------------------------------------------------------
# Compare all strategies on the same basket
# ---------------------------------------------------------------------------

def run_comparison(basket_name: str, start: str, end: str):
    """Run all strategies on the same basket and compare."""

    print(f"\n{'='*70}")
    print(f"STRATEGY COMPARISON — {basket_name.upper()} ({len(BASKETS[basket_name])} stocks)")
    print(f"  Period: {start} to {end}")
    print(f"{'='*70}")

    # Fetch data once for all strategies
    symbols = list(dict.fromkeys(BASKETS[basket_name]))
    print(f"\nFetching {len(symbols)} stocks from yfinance...")
    raw_data = fetch_yfinance(symbols, start, end)
    print(f"  {len(raw_data)} stocks loaded")

    results = {}

    for strategy_name, module in STRATEGIES.items():
        config = module.DEFAULT_CONFIG.copy()
        print(f"\n--- {strategy_name} ---")
        print(f"  Config: SL={config['stop_loss_pct']*100:.0f}% "
              f"T={config['target_pct']*100:.0f}%")

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

    # Comparison table
    print_comparison_table(results, f"COMPARISON — {basket_name.upper()}")

    # Best strategy details
    best = max(results.keys(), key=lambda k: results[k]["metrics"]["pf"])
    best_m = results[best]["metrics"]
    print(f"\n  Best strategy: {best}")
    print(f"  PF: {best_m['pf']:.2f} | Win rate: {best_m['win_rate']:.1f}% | "
          f"Trades: {best_m['trades']}")
    print_per_stock(results[best]["trades"])

    # Verdict
    print_verdict(best_m["pf"], best_m["trades"], best_m["win_rate"])

    # Save trades
    for name, data in results.items():
        trades = data["trades"]
        if trades:
            for t in trades:
                if hasattr(t.get("entry_date"), "strftime"):
                    t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
                if hasattr(t.get("exit_date"), "strftime"):
                    t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")
            safe = name.replace(" ", "_")
            with open(f"{basket_name}_{safe}_trades.json", "w") as f:
                json.dump(trades, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified backtest runner for any strategy on any basket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Strategies:\n"
            "  breakout             — Original Session T (trend + breakout + volume)\n"
            "  enhanced_breakout    — Session T + ADX + DI + RS filters\n"
            "  positional_pullback  — NEW: weekly trend + daily pullback (1-3 month holds)\n\n"
            "Baskets:\n"
            "  nifty50              — 48 NIFTY 50 stocks\n"
            "  midcap               — 75 NIFTY MIDCAP stocks\n\n"
            "Examples:\n"
            "  python -m backtests.run --strategy positional_pullback --basket nifty50\n"
            "  python -m backtests.run --compare --basket midcap\n"
            "  python -m backtests.run --strategy breakout --basket nifty50 --sl 0.05"
        ),
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

        # Save trades
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
