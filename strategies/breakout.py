"""
Breakout Strategy — original Session T (refactored).

Entry: Close > N-day high + volume confirmation + trend filter (EMA50 > EMA200)
Exit:  Close < EMA10 or RSI > 75

This module provides entry_fn and exit_fn that plug into the common backtest engine.
"""

import pandas as pd
from common.indicators import (
    add_emas, add_rsi, add_volume_indicators, add_lookback_highs
)


def prepare(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute indicators needed by this strategy."""
    lookback = config.get("lookback_days", 10)
    rsi_period = config.get("rsi_period", 14)
    df = add_emas(df, spans=[10, 20, 50, 200])
    df = add_rsi(df, rsi_period)
    df = add_volume_indicators(df, config.get("volume_avg_period", 20))
    df = add_lookback_highs(df, lookback)
    return df


def should_enter(last, config: dict) -> bool:
    """
    Original Session T entry:
    1. EMA50 > EMA200 (uptrend)
    2. Close > N-day high (breakout)
    3. Volume > 2x average
    4. RSI 35-70
    """
    # Trend filter
    if last["ema50"] <= last["ema200"]:
        return False
    # Breakout
    if last["Close"] <= last["high_lookback"]:
        return False
    # Volume confirmation
    vol_mult = config.get("volume_multiplier", 2.0)
    if last["Volume"] < last["vol_avg"] * vol_mult:
        return False
    # RSI range
    rsi_min = config.get("rsi_min", 35)
    rsi_max = config.get("rsi_max", 70)
    if not (rsi_min <= last["rsi"] <= rsi_max):
        return False
    return True


def should_exit(last, config: dict) -> bool:
    """Original Session T exit."""
    if last["Close"] < last["ema10"]:
        return True
    if last["rsi"] > 75:
        return True
    return False


# Default config
DEFAULT_CONFIG = {
    "lookback_days": 10,
    "volume_multiplier": 2.0,
    "rsi_min": 35,
    "rsi_max": 70,
    "rsi_period": 14,
    "volume_avg_period": 20,
    "stop_loss_pct": 0.03,
    "target_pct": 0.08,
}
