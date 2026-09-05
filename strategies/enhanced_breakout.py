"""
Enhanced Breakout Strategy — original T + ADX + DI + RS filters.

Adds three filters to the base breakout strategy:
  - ADX >= threshold (skip choppy markets)
  - +DI > -DI (bullish direction)
  - 20-day return > 0 (relative strength)

Also adds ADX-based exit (trend dying).
"""

import pandas as pd
from common.indicators import (
    add_emas, add_rsi, add_volume_indicators, add_lookback_highs,
    add_adx, add_returns
)


def prepare(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute indicators needed by this strategy."""
    lookback = config.get("lookback_days", 10)
    rsi_period = config.get("rsi_period", 14)
    adx_period = config.get("adx_period", 14)
    df = add_emas(df, spans=[10, 20, 50, 200])
    df = add_rsi(df, rsi_period)
    df = add_adx(df, adx_period)
    df = add_volume_indicators(df, config.get("volume_avg_period", 20))
    df = add_lookback_highs(df, lookback)
    df = add_returns(df)
    return df


def should_enter(last, config: dict) -> bool:
    """
    Enhanced breakout entry:
    1. EMA50 > EMA200 (uptrend)
    2. Close > N-day high (breakout)
    3. Volume > multiplier x average
    4. RSI in range
    5. ADX >= adx_min (strong trend)
    6. +DI > -DI (bullish direction)
    7. 20-day return > 0 (relative strength)
    """
    # Base breakout filters
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

    # Enhanced filters
    adx_min = config.get("adx_min", 20)
    if pd.isna(last["adx"]) or last["adx"] < adx_min:
        return False
    if config.get("use_di", True) and last["plus_di"] <= last["minus_di"]:
        return False
    if config.get("use_rs", True):
        if pd.isna(last["ret_20d"]) or last["ret_20d"] <= 0:
            return False
    return True


def should_exit(last, config: dict, prev=None) -> bool:
    """Enhanced exit: base exits + ADX < 15 (trend dying)."""
    if last["Close"] < last["ema10"]:
        return True
    if last["rsi"] > 75:
        return True
    if pd.notna(last["adx"]) and last["adx"] < 15:
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
    "adx_period": 14,
    "adx_min": 20,
    "use_di": True,
    "use_rs": True,
    "stop_loss_pct": 0.03,
    "target_pct": 0.08,
}
