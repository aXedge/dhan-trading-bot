"""
Positional Pullback Strategy — 1-3 month holding period.

Designed for swing/positional trades with:
  - Entry frequency: 1-2 trades per week across a 40-50 stock basket
  - Holding period: 1-3 months
  - Risk: 6-8% stop loss (positional, survives multi-week noise)
  - Target: 15-20% (matches 1-3 month horizon)
  - Reward-risk ratio: ~2.5:1

Entry conditions (all must be met):
  1. Weekly trend: EMA20 > EMA50 on weekly chart (multi-week uptrend)
  2. Daily trend: EMA50 > EMA200 (broader uptrend intact)
  3. Pullback: Close within 3% of EMA20 (bought the dip, not the breakout)
  4. RSI 40-60 (not overbought, not oversold — mid-pullback zone)
  5. RSI turning up: today's RSI > yesterday's RSI (bounce confirmed)
  6. Volume above average (some institutional participation)
  7. ADX >= 15 (enough trend to sustain a multi-week move)

Exit conditions (any one triggers):
  - Close below EMA50 (daily trend broken)
  - RSI > 72 (overbought — take profits)
  - Weekly close below EMA20 (approximated by daily close < EMA20 for 3+ days)
  - Trailing SL based on EMA20 (SL trails up as price rises)
"""

import pandas as pd
import numpy as np
from common.indicators import (
    add_emas, add_rsi, add_adx, add_volume_indicators,
    add_lookback_highs, add_atr, add_returns, add_macd
)


def prepare(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute indicators needed by the positional pullback strategy.

    Uses both daily and weekly indicators. Weekly is computed by
    resampling the daily data.
    """
    lookback = config.get("lookback_days", 20)
    rsi_period = config.get("rsi_period", 14)
    adx_period = config.get("adx_period", 14)

    # Daily indicators
    df = add_emas(df, spans=[10, 20, 50, 200])
    df = add_rsi(df, rsi_period)
    df = add_adx(df, adx_period)
    df = add_volume_indicators(df, config.get("volume_avg_period", 20))
    df = add_atr(df, 14)
    df = add_returns(df)
    df = add_macd(df)
    df = add_lookback_highs(df, 250)  # 52-week high

    # Weekly indicators (resample daily to weekly, compute EMA20/EMA50)
    if len(df) >= 10:
        weekly = df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
        weekly["w_ema20"] = weekly["Close"].ewm(span=20, adjust=False).mean()
        weekly["w_ema50"] = weekly["Close"].ewm(span=50, adjust=False).mean()

        # Merge weekly trend back to daily (forward fill — weekly signal
        # applies to all days in that week)
        weekly_trend = weekly[["w_ema20", "w_ema50"]].reindex(
            df.index, method="ffill"
        )
        df["w_ema20"] = weekly_trend["w_ema20"]
        df["w_ema50"] = weekly_trend["w_ema50"]

    return df


def should_enter(last, config: dict) -> bool:
    """
    Positional pullback entry.

    Buys quality stocks in an uptrend when they pull back to EMA20
    with a rising RSI. Targets 1-3 month holding period.
    """
    # 1. Weekly trend: EMA20 > EMA50 on weekly chart
    if pd.isna(last.get("w_ema20")) or pd.isna(last.get("w_ema50")):
        return False
    if last["w_ema20"] <= last["w_ema50"]:
        return False

    # 2. Daily trend: EMA50 > EMA200
    if last["ema50"] <= last["ema200"]:
        return False

    # 3. Pullback: Close within 3% of EMA20
    dist_to_ema20 = abs(last["Close"] - last["ema20"]) / last["Close"]
    if dist_to_ema20 > config.get("pullback_tolerance", 0.03):
        return False

    # 4. Price above EMA50 (pullback, not a crash)
    if last["Close"] <= last["ema50"]:
        return False

    # 5. RSI in mid-zone (40-60)
    rsi_min = config.get("rsi_min", 40)
    rsi_max = config.get("rsi_max", 60)
    if not (rsi_min <= last["rsi"] <= rsi_max):
        return False

    # 6. Volume above average (institutional participation)
    vol_mult = config.get("volume_multiplier", 1.0)
    if last["Volume"] < last["vol_avg"] * vol_mult:
        return False

    # 7. ADX >= 15 (enough trend to sustain a multi-week move)
    adx_min = config.get("adx_min", 15)
    if pd.isna(last["adx"]) or last["adx"] < adx_min:
        return False

    # 8. MACD histogram > 0 (momentum still positive despite pullback)
    if config.get("use_macd", True):
        if pd.isna(last.get("macd_hist")) or last["macd_hist"] <= 0:
            return False

    # 9. Within 15% of 52-week high (not in a deep correction)
    if config.get("use_52w_high", True):
        if pd.isna(last.get("high_lookback")):
            return False
        pct_from_high = (last["high_lookback"] - last["Close"]) / last["high_lookback"]
        if pct_from_high > config.get("max_pct_from_52w_high", 0.15):
            return False

    return True


def should_exit(last, config: dict, prev=None) -> bool:
    """
    Positional pullback exit.

    Exits when the trend shows signs of reversing — not on
    short-term noise. Designed for 1-3 month holds.

    Uses 2-consecutive-day confirmation to avoid false exits.
    """
    # Close below EMA50 for 2 consecutive days (real trend break, not noise)
    if prev is not None:
        if last["Close"] < last["ema50"] and prev["Close"] < prev["ema50"]:
            return True
    else:
        if last["Close"] < last["ema50"]:
            return True

    # RSI overbought (take profits) — raised to 75 for positional holds
    rsi_exit = config.get("rsi_exit", 75)
    if last["rsi"] > rsi_exit:
        return True

    # Weekly trend broken (EMA20 < EMA50 on weekly)
    if pd.notna(last.get("w_ema20")) and pd.notna(last.get("w_ema50")):
        if last["w_ema20"] < last["w_ema50"]:
            return True

    return False


def get_stop_loss(entry_price: float, config: dict) -> float:
    """Compute stop loss for positional strategy."""
    sl_pct = config.get("stop_loss_pct", 0.07)
    return round(entry_price * (1 - sl_pct), 2)


def get_target(entry_price: float, config: dict) -> float:
    """Compute target for positional strategy."""
    target_pct = config.get("target_pct", 0.15)
    return round(entry_price * (1 + target_pct), 2)


# Default config — tuned for 1-3 month positional trades
DEFAULT_CONFIG = {
    "lookback_days": 20,
    "rsi_period": 14,
    "adx_period": 14,
    "volume_avg_period": 20,
    "pullback_tolerance": 0.03,   # within 3% of EMA20
    "rsi_min": 40,
    "rsi_max": 60,
    "rsi_exit": 75,
    "volume_multiplier": 1.0,     # above average (not 2x — positional is less volume-dependent)
    "adx_min": 15,
    "stop_loss_pct": 0.07,        # 7% SL (positional — survives multi-week noise)
    "target_pct": 0.15,           # 15% target (1-3 month horizon)
    "use_macd": True,             # require positive MACD histogram
    "use_52w_high": True,         # require stock near 52-week high
    "max_pct_from_52w_high": 0.15, # within 15% of 52-week high
    "cooldown_days": 0,
}
