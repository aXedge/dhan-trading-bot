"""
Generic backtest engine — strategy-agnostic.

The engine walks through historical data day by day and calls
strategy-provided entry/exit functions. It handles:
  - Position tracking (one position per stock at a time)
  - Stop-loss and target execution (intraday simulation)
  - Signal-based exits
  - Cooldown after losses
  - P&L calculation

Strategies provide two functions:
  - should_enter(last_row, config) -> bool
  - should_exit(last_row, config) -> bool
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Callable, List, Dict
from collections import defaultdict


def backtest_stock(
    symbol: str,
    df: pd.DataFrame,
    config: dict,
    entry_fn: Callable,
    exit_fn: Callable,
    warmup: int = 210,
) -> List[dict]:
    """
    Backtest a single stock.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV + indicators (must be pre-computed)
        config: Strategy config dict (contains SL, target, cooldown, etc.)
        entry_fn: Function(last_row, config) -> bool
        exit_fn: Function(last_row, config) -> bool
        warmup: Number of initial rows to skip (need 200 for EMA200)

    Returns:
        List of trade dicts with entry/exit dates, prices, P&L, reasons
    """
    trades = []
    position = None
    cooldown_until = None

    if len(df) < warmup:
        return trades

    sl_pct = config.get("stop_loss_pct", 0.03)
    target_pct = config.get("target_pct", 0.08)
    cooldown_days = config.get("cooldown_days", 0)

    for i in range(warmup, len(df)):
        last = df.iloc[i]
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])
        current_date = last.name

        # Check cooldown
        in_cooldown = False
        if cooldown_until and current_date <= cooldown_until:
            in_cooldown = True

        # If in position, check SL/target first (intraday simulation)
        if position:
            # SL hit
            if last_low <= position["sl"]:
                position["exit_price"] = position["sl"]
                position["exit_date"] = current_date
                position["exit_reason"] = "SL hit"
                position["pnl_pct"] = (position["sl"] - position["entry"]) / position["entry"]
                trades.append(position)

                # Set cooldown if it was a loss
                if position["pnl_pct"] < 0 and cooldown_days > 0:
                    cooldown_until = current_date + timedelta(days=cooldown_days)

                position = None
                continue

            # Target hit
            if last_high >= position["target"]:
                position["exit_price"] = position["target"]
                position["exit_date"] = current_date
                position["exit_reason"] = "Target hit"
                position["pnl_pct"] = (position["target"] - position["entry"]) / position["entry"]
                trades.append(position)
                position = None
                continue

            # Signal-based exit
            if exit_fn(last, config):
                position["exit_price"] = last_close
                position["exit_date"] = current_date
                position["exit_reason"] = "Signal exit"
                position["pnl_pct"] = (last_close - position["entry"]) / position["entry"]
                trades.append(position)

                if position["pnl_pct"] < 0 and cooldown_days > 0:
                    cooldown_until = current_date + timedelta(days=cooldown_days)

                position = None
                continue

        # Check entry
        if not position and not in_cooldown:
            if entry_fn(last, config):
                position = {
                    "symbol": symbol,
                    "entry": last_close,
                    "entry_date": current_date,
                    "sl": round(last_close * (1 - sl_pct), 2),
                    "target": round(last_close * (1 + target_pct), 2),
                }

    # Close any open position at end of data
    if position:
        last = df.iloc[-1]
        position["exit_price"] = float(last["Close"])
        position["exit_date"] = last.name
        position["exit_reason"] = "End of data"
        position["pnl_pct"] = (position["exit_price"] - position["entry"]) / position["entry"]
        trades.append(position)

    return trades


def backtest_multiple(
    data: Dict[str, pd.DataFrame],
    config: dict,
    entry_fn: Callable,
    exit_fn: Callable,
    prepare_fn: Callable = None,
    warmup: int = 210,
) -> List[dict]:
    """
    Backtest multiple stocks.

    Args:
        data: Dict mapping {symbol: raw DataFrame}
        config: Strategy config
        entry_fn: Entry function
        exit_fn: Exit function
        prepare_fn: Optional function to compute indicators on each DataFrame
        warmup: Warmup period

    Returns:
        List of all trades across all stocks
    """
    all_trades = []

    for symbol, df in data.items():
        if prepare_fn:
            df = prepare_fn(df, config)

        trades = backtest_stock(symbol, df, config, entry_fn, exit_fn, warmup)
        all_trades.extend(trades)

    return all_trades
