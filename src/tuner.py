"""
Strategy Tuner — Parameter Optimization via Grid Search
========================================================

Runs on your laptop (not the VM). Tests hundreds of parameter combinations
against historical data to find the configuration with the best risk-adjusted
returns (profit factor).

Fetches 2 years of OHLCV data once via yfinance, then tests each parameter
combination against all 14 basket stocks. Uses the same indicator logic
as technicals.py but with configurable parameters.

Output:
  - Console: top 10 parameter sets ranked by profit factor
  - data/tuner_results.json: all tested combinations
  - config/settings_tuned.yaml: the best config, ready to deploy

Usage:
    # Full grid search (default: ~100-500 combinations)
    python -m src.tuner

    # Quick search (fewer combinations, ~30)
    python -m src.tuner --quick

    # Custom period
    python -m src.tuner --days 365

    # Export full results to CSV
    python -m src.tuner --csv tuner_results.csv

    python -m src.tuner --help

No Dhan API access needed — runs entirely on yfinance data.
Takes 5-15 minutes depending on grid size.
"""

import argparse
import os
import sys
import json
import csv
import itertools
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json, save_json, setup_logger, now_iso
import yaml

logger = setup_logger(__name__, "tuner.log")


# ---------------------------------------------------------------------------
# Price data fetching (fetch once, reuse for all combinations)
# ---------------------------------------------------------------------------

def fetch_all_prices(basket: list, days: int) -> dict:
    """Fetch price history for all basket stocks. Returns {symbol: candles list}."""
    import yfinance as yf
    import pandas as pd

    prices = {}
    for i, stock in enumerate(basket, 1):
        symbol = stock["symbol"]
        logger.info(f"[{i}/{len(basket)}] Fetching {symbol}...")
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=f"{days}d")
            if len(df) < 200:
                logger.warning(f"  {symbol}: only {len(df)} days")
                continue
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            elif "datetime" in df.columns:
                df["date"] = df["datetime"].astype(str)
                df = df.drop(columns=["datetime"])
            prices[symbol] = df.to_dict(orient="records")
            logger.info(f"  {symbol}: {len(prices[symbol])} candles")
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
    return prices


# ---------------------------------------------------------------------------
# Indicator computation (optimized for backtest speed)
# ---------------------------------------------------------------------------

def precompute_indicators(candles: list, params: dict) -> list:
    """
    Pre-compute indicators for all candles with the given parameters.
    Returns a list of indicator dicts, one per candle.
    """
    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    volumes = [float(c["volume"]) for c in candles]
    n = len(closes)

    rsi_period = params.get("rsi_period", 14)
    vol_period = params.get("volume_avg_period", 20)
    ema_fast = params.get("ema_fast", 50)
    ema_slow = params.get("ema_slow", 200)
    lookback = params.get("lookback_days", 20)

    # EMAs (compute full series)
    ema10 = _ema_series(closes, 10)
    ema20 = _ema_series(closes, 20)
    ema50 = _ema_series(closes, ema_fast)
    ema200 = _ema_series(closes, ema_slow) if n >= ema_slow else [0] * n

    # RSI (compute full series)
    rsi = _rsi_series(closes, rsi_period)

    # Volume average
    vol_avg = [0] * n
    for i in range(vol_period - 1, n):
        vol_avg[i] = sum(volumes[i - vol_period + 1:i + 1]) / vol_period

    # Lookback highs (excluding today)
    high_lookback = [0] * n
    for i in range(lookback, n):
        high_lookback[i] = max(highs[i - lookback:i])

    high_50d = [0] * n
    for i in range(50, n):
        high_50d[i] = max(highs[i - 50:i])

    # Build indicator list
    indicators = []
    for i in range(n):
        indicators.append({
            "close": closes[i],
            "ema10": ema10[i],
            "ema20": ema20[i],
            "ema50": ema50[i],
            "ema200": ema200[i],
            "rsi": rsi[i],
            "vol_avg": vol_avg[i],
            "volume": volumes[i],
            "high_lookback": high_lookback[i],
            "high_50d": high_50d[i],
        })

    return indicators


def _ema_series(values: list, period: int) -> list:
    """Compute EMA for entire series."""
    if not values:
        return []
    if len(values) < period:
        return [values[0]] * len(values)

    multiplier = 2 / (period + 1)
    ema = [0] * len(values)
    ema[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1]
    # Fill early values
    for i in range(period - 1):
        ema[i] = values[i]
    return ema


def _rsi_series(closes: list, period: int) -> list:
    """Compute RSI for entire series."""
    n = len(closes)
    rsi = [50.0] * n

    if n < period + 1:
        return rsi

    gains = [0] * n
    losses = [0] * n
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        gains[i] = max(diff, 0)
        losses[i] = max(-diff, 0)

    # First RSI value
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


# ---------------------------------------------------------------------------
# Signal generation (configurable)
# ---------------------------------------------------------------------------

def check_entry(ind: dict, p: dict, trend_filter: bool) -> tuple:
    """
    Check entry conditions with given parameters.
    Returns (strategy, sl_price, target_price) or (None, 0, 0).
    """
    # Trend filter: only buy when EMA50 > EMA200 (uptrend)
    if trend_filter and ind["ema200"] and ind["ema50"] < ind["ema200"]:
        return None, 0, 0

    # Swing entry
    breakout = ind["close"] > ind["high_lookback"]
    vol_confirm = ind["volume"] > ind["vol_avg"] * p["volume_multiplier"]
    not_overbought = p["rsi_min"] < ind["rsi"] < p["rsi_max"]

    if breakout and vol_confirm and not_overbought:
        sl = ind["close"] * (1 - p["stop_loss_pct"])
        target = ind["close"] * (1 + p["target_pct"])
        return "swing", sl, target

    return None, 0, 0


def check_exit(ind: dict, held: dict, p: dict) -> str | None:
    """Check exit conditions. Returns reason string or None."""
    strategy = held["strategy"]

    # SL hit
    if ind["close"] <= held["sl"]:
        return "SL hit"

    # Target hit
    if ind["close"] >= held["target"]:
        return "Target hit"

    if strategy == "swing":
        if ind["close"] < ind["ema10"]:
            return "Close < EMA10"
        if ind["rsi"] > 75:
            return "RSI > 75"
    else:
        if ind["ema200"] and ind["ema50"] < ind["ema200"]:
            return "Death cross"
        if ind["ema200"] and ind["close"] < ind["ema200"]:
            return "Close < EMA200"

    return None


# ---------------------------------------------------------------------------
# Backtest for a single parameter set
# ---------------------------------------------------------------------------

def backtest_params(prices: dict, params: dict, capital_per_stock: int = 25000) -> dict:
    """
    Run a full backtest with the given parameters across all stocks.
    Returns performance stats.
    """
    max_positions = params.get("max_positions", 8)
    trend_filter = params.get("trend_filter", True)

    closed_trades = []
    open_positions = {}  # {symbol: position}

    for symbol, candles in prices.items():
        if len(candles) < 200:
            continue

        indicators = precompute_indicators(candles, params)
        held = None

        for i in range(200, len(indicators)):
            ind = indicators[i]
            date = candles[i].get("date", "")

            if held:
                reason = check_exit(ind, held, params)
                if reason:
                    exit_price = ind["close"]
                    pnl = (exit_price - held["entry_price"]) * held["qty"]
                    pnl_pct = (exit_price - held["entry_price"]) / held["entry_price"] * 100
                    closed_trades.append({
                        "symbol": symbol,
                        "entry_date": held["entry_date"],
                        "exit_date": date,
                        "entry_price": held["entry_price"],
                        "exit_price": round(exit_price, 2),
                        "qty": held["qty"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "exit_reason": reason,
                    })
                    held = None
            else:
                if len(open_positions) >= max_positions:
                    continue
                strategy, sl, target = check_entry(ind, params, trend_filter)
                if strategy:
                    entry_price = ind["close"]
                    qty = int(capital_per_stock / entry_price)
                    if qty > 0:
                        held = {
                            "symbol": symbol,
                            "strategy": strategy,
                            "entry_price": entry_price,
                            "entry_date": date,
                            "qty": qty,
                            "sl": sl,
                            "target": target,
                        }
                        open_positions[symbol] = held

    # Compute stats
    if not closed_trades:
        return {
            "trades": 0, "win_rate": 0, "total_pnl": 0,
            "profit_factor": 0, "avg_profit": 0, "avg_loss": 0,
            "avg_hold": 0, "max_profit": 0, "max_loss": 0,
            "return_pct": 0, "params": params,
        }

    winners = [t for t in closed_trades if t["pnl"] > 0]
    losers = [t for t in closed_trades if t["pnl"] < 0]
    total_pnl = sum(t["pnl"] for t in closed_trades)
    gross_profit = sum(t["pnl"] for t in winners)
    gross_loss = abs(sum(t["pnl"] for t in losers))

    # Hold days
    total_hold = 0
    for t in closed_trades:
        try:
            d1 = datetime.fromisoformat(t["entry_date"].split(" ")[0])
            d2 = datetime.fromisoformat(t["exit_date"].split(" ")[0])
            total_hold += (d2 - d1).days
        except (ValueError, TypeError):
            pass

    total_capital = max_positions * capital_per_stock

    return {
        "trades": len(closed_trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(closed_trades) * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_profit": round(gross_profit / len(winners), 2) if winners else 0,
        "avg_loss": round(gross_loss / len(losers), 2) if losers else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0,
        "avg_hold_days": round(total_hold / len(closed_trades), 1) if closed_trades else 0,
        "max_profit": round(max(t["pnl"] for t in closed_trades), 2),
        "max_loss": round(min(t["pnl"] for t in closed_trades), 2),
        "return_pct": round(total_pnl / total_capital * 100, 2),
        "params": params,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def build_grid(quick: bool = False) -> list:
    """
    Build the parameter grid to search.

    Each combination is a dict with all tunable parameters.
    """
    if quick:
        grid = {
            "rsi_min": [40, 45],
            "rsi_max": [60, 65],
            "volume_multiplier": [1.5, 2.0],
            "stop_loss_pct": [0.02, 0.03],
            "target_pct": [0.04, 0.06],
            "lookback_days": [15, 20],
            "trend_filter": [True],
            "max_positions": [8],
        }
    else:
        grid = {
            "rsi_min": [35, 40, 45],
            "rsi_max": [60, 65, 70],
            "volume_multiplier": [1.2, 1.5, 2.0],
            "stop_loss_pct": [0.015, 0.02, 0.03],
            "target_pct": [0.04, 0.06, 0.08],
            "lookback_days": [10, 15, 20],
            "trend_filter": [True, False],
            "max_positions": [5, 8],
        }

    keys = list(grid.keys())
    combinations = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        # Skip invalid combos
        if combo["rsi_min"] >= combo["rsi_max"]:
            continue
        # Skip R:R < 1.0 (not worth testing)
        rr = combo["target_pct"] / combo["stop_loss_pct"]
        if rr < 1.0:
            continue
        combinations.append(combo)

    return combinations


def run_grid_search(prices: dict, combinations: list, capital_per_stock: int = 25000) -> list:
    """Run backtest for each parameter combination."""
    results = []
    total = len(combinations)

    for i, params in enumerate(combinations, 1):
        start = time.time()
        stats = backtest_params(prices, params, capital_per_stock)
        elapsed = time.time() - start

        results.append(stats)

        pf = stats["profit_factor"]
        wr = stats["win_rate"]
        pnl = stats["total_pnl"]
        trades = stats["trades"]

        logger.info(
            f"[{i}/{total}] PF:{pf:.2f} WR:{wr:.1f}% P&L:Rs.{pnl:,.0f} "
            f"Trades:{trades} ({elapsed:.1f}s) "
            f"RSI:{params['rsi_min']}-{params['rsi_max']} "
            f"Vol:{params['volume_multiplier']}x "
            f"SL:{params['stop_loss_pct']} T:{params['target_pct']} "
            f"LB:{params['lookback_days']} TF:{params['trend_filter']}"
        )

        # Progress print every 10
        if i % 10 == 0:
            best_so_far = max(results, key=lambda r: r["profit_factor"])
            logger.info(
                f"  --- Progress: {i}/{total} | "
                f"Best PF: {best_so_far['profit_factor']:.2f} "
                f"(WR: {best_so_far['win_rate']:.1f}%, "
                f"P&L: Rs.{best_so_far['total_pnl']:,.0f}) ---"
            )

    return results


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def print_top_results(results: list, top_n: int = 10):
    """Print top N parameter sets ranked by profit factor."""
    # Filter: must have at least 20 trades for statistical significance
    significant = [r for r in results if r["trades"] >= 20]

    # Sort by profit factor (descending)
    sorted_results = sorted(significant, key=lambda r: r["profit_factor"], reverse=True)

    print("\n" + "=" * 100)
    print(f"  TOP {min(top_n, len(sorted_results))} PARAMETER SETS (ranked by Profit Factor)")
    print(f"  Only showing results with >= 20 trades for statistical significance")
    print("=" * 100)

    headers = [
        "Rank", "PF", "WR%", "Trades", "Total P&L",
        "Avg Win", "Avg Loss", "MaxDD", "Ret%", "Hold(d)",
        "RSI", "Vol", "SL%", "Tgt%", "LB", "TF", "Pos"
    ]

    print(f"  {'':<4} " + "  ".join(f"{h:>8}" for h in headers[1:]))
    print("  " + "-" * 96)

    for rank, r in enumerate(sorted_results[:top_n], 1):
        p = r["params"]
        print(
            f"  {rank:<4} "
            f"{r['profit_factor']:>8.2f}  "
            f"{r['win_rate']:>8.1f}  "
            f"{r['trades']:>8}  "
            f"Rs.{r['total_pnl']:>8,.0f}  "
            f"Rs.{r['avg_profit']:>8,.0f}  "
            f"Rs.{r['avg_loss']:>8,.0f}  "
            f"Rs.{r['max_loss']:>8,.0f}  "
            f"{r['return_pct']:>8.1f}  "
            f"{r['avg_hold_days']:>8.1f}  "
            f"{p['rsi_min']:>3}-{p['rsi_max']:<3}  "
            f"{p['volume_multiplier']:>5.1f}x  "
            f"{p['stop_loss_pct']*100:>5.1f}  "
            f"{p['target_pct']*100:>5.1f}  "
            f"{p['lookback_days']:>3}  "
            f"{str(p['trend_filter']):>5}  "
            f"{p['max_positions']:>3}"
        )

    print("=" * 100)

    if sorted_results:
        best = sorted_results[0]
        print(f"\n  BEST PARAMETER SET:")
        print(f"    Profit Factor : {best['profit_factor']:.2f}")
        print(f"    Win Rate      : {best['win_rate']:.1f}%")
        print(f"    Total P&L     : Rs.{best['total_pnl']:,.0f}")
        print(f"    Total Trades  : {best['trades']}")
        print(f"    Return        : {best['return_pct']:.1f}%")
        print(f"    Avg Hold      : {best['avg_hold_days']:.1f} days")
        print(f"\n    Parameters:")
        for k, v in best["params"].items():
            print(f"      {k}: {v}")
    else:
        print("\n  No parameter sets with >= 20 trades found. Try expanding the grid.")

    print()

    return sorted_results[:top_n] if sorted_results else []


def generate_tuned_config(best_params: dict, output_path: str):
    """Generate a settings_tuned.yaml config file from the best parameters."""
    config = {
        "fundamental": {
            "universe_indices": ["CNXMIDCAP", "CNX100"],
            "min_f_score": 7,
            "min_roce": 18.0,
            "max_debt_equity": 0.5,
            "min_promoter_holding": 50.0,
            "max_pledged": 5.0,
            "weights": {
                "piotroski": 25, "roce": 20, "debt_equity": 15,
                "revenue_growth": 20, "promoter_holding": 10, "pledge": 10,
            },
            "basket_size": 15,
            "output_file": "data/basket.json",
            "request_delay": 2,
        },
        "technical": {
            "swing": {
                "lookback_days": best_params["lookback_days"],
                "volume_multiplier": best_params["volume_multiplier"],
                "rsi_min": best_params["rsi_min"],
                "rsi_max": best_params["rsi_max"],
                "stop_loss_pct": best_params["stop_loss_pct"],
                "target_pct": best_params["target_pct"],
            },
            "positional": {
                "ema_fast": 50,
                "ema_slow": 200,
                "stop_loss_pct": 0.05,
                "target_pct": 0.15,
            },
            "rsi_period": 14,
            "volume_avg_period": 20,
            "price_history_days": 365,
            "trend_filter": best_params.get("trend_filter", True),
            "output_file": "data/signals_tuned.json",
        },
        "execution": {
            "max_positions": best_params.get("max_positions", 8),
            "capital_per_stock": 25000,
            "product_type": "CNC",
            "order_type": "MARKET",
            "sl_check_interval_sec": 600,
            "market_open": "09:15",
            "market_close": "15:30",
            "positions_file": "data/positions_tuned.json",
        },
        "schedule": {
            "technical_scan": "20 9 * * 1-5",
            "execute_signals": "25 9 * * 1-5",
            "sl_monitor_start": "15 9 * * 1-5",
            "token_refresh": "45 8 * * 1-5",
        },
        "dhan": {
            "base_url": "https://api.dhan.co/v2",
            "exchange_segment": "NSE_EQ",
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Tuned config saved to {output_path}")
    print(f"\n  Tuned config saved to: {output_path}")


def export_csv(results: list, filepath: str):
    """Export all results to CSV."""
    fields = [
        "profit_factor", "win_rate", "trades", "winners", "losers",
        "total_pnl", "avg_profit", "avg_loss", "max_profit", "max_loss",
        "return_pct", "avg_hold_days",
        "rsi_min", "rsi_max", "volume_multiplier", "stop_loss_pct",
        "target_pct", "lookback_days", "trend_filter", "max_positions",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != "params"}
            row.update(r["params"])
            writer.writerow(row)

    print(f"\n  Full results exported to: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(days: int = 730, quick: bool = False, csv_path: str = None):
    """Main entry point for the strategy tuner."""
    logger.info("=" * 60)
    logger.info("Strategy Tuner — Grid Search Parameter Optimization")
    logger.info("=" * 60)

    # Load basket
    basket = load_json("basket.json")
    if not basket:
        logger.error("No basket found. Run screener.py first.")
        return

    logger.info(f"Basket: {len(basket)} stocks")

    # Fetch price data (once)
    logger.info(f"Fetching {days} days of price data...")
    prices = fetch_all_prices(basket, days)
    logger.info(f"Price data ready for {len(prices)} stocks")

    # Build parameter grid
    combinations = build_grid(quick=quick)
    logger.info(f"Parameter grid: {len(combinations)} combinations to test")

    # Run grid search
    logger.info("Starting grid search...")
    start_time = time.time()

    results = run_grid_search(prices, combinations)

    elapsed = time.time() - start_time
    logger.info(f"Grid search complete: {len(results)} combinations tested in {elapsed:.0f}s")

    # Save all results
    save_json(results, "tuner_results.json")
    logger.info("Results saved to data/tuner_results.json")

    # Print top results
    top_results = print_top_results(results)

    # Generate tuned config from best params
    if top_results:
        best = top_results[0]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config/settings_tuned.yaml")
        generate_tuned_config(best["params"], config_path)

    # CSV export
    if csv_path:
        export_csv(results, csv_path)

    logger.info("=" * 60)
    logger.info("Strategy Tuner — complete")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Strategy Tuner — finds the best parameters via grid search.\n\n"
            "Fetches historical data once, tests hundreds of parameter\n"
            "combinations, and outputs the top-performing configuration.\n\n"
            "Output:\n"
            "  - Console: top 10 parameter sets\n"
            "  - data/tuner_results.json: all results\n"
            "  - config/settings_tuned.yaml: best config, ready to deploy\n\n"
            "Examples:\n"
            "  python -m src.tuner                   # full grid (~200+ combos)\n"
            "  python -m src.tuner --quick           # quick grid (~30 combos)\n"
            "  python -m src.tuner --days 365        # 1 year\n"
            "  python -m src.tuner --csv results.csv  # export all results"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=730,
        help="Historical data period in days (default: 730 = 2 years).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer parameter combinations (faster but less thorough).",
    )
    parser.add_argument(
        "--csv", dest="csv_path",
        help="Export all results to this CSV file.",
    )
    args = parser.parse_args()
    run(days=args.days, quick=args.quick, csv_path=args.csv_path)
