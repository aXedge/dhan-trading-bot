"""
Momentum Acceleration Strategy v4
==================================

Based on the user\'s personal trading methodology:
- Entry: MA gap acceleration + MACD confirmation + volume confirmation + strong trend
- Exit: Wide Chandelier trailing stop (3.5x ATR, 20-bar) + RSI overbought
- Risk: Initial 6% stop loss (backstop), 15% target

v4 changes:
- Removed support proximity condition (contradictory with gap widening)
- Added volume filter: volume > 1.5x 20-day average (momentum needs volume)
- Increased ADX minimum from 15 to 20 (stronger trend filter)
- Entry is now: trend + gap acceleration + MACD + volume + ADX
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.06,
    "target_pct": 0.15,

    # Exit parameters
    "trail_atr_mult": 3.5,
    "chandelier_lookback": 20,
    "rsi_exit": 75,

    # Entry parameters
    "ema_fast": 5,
    "ema_slow": 20,
    "ema_trend": 50,
    "ema_long": 200,
    "gap_min_pct": 0.002,
    "rsi_entry_max": 70,
    "adx_min": 20,                  # raised from 15 to 20
    "volume_mult": 1.5,             # volume must be > 1.5x 20-day average
    "volume_avg_period": 20,

    # Indicator parameters
    "rsi_period": 14,
    "atr_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "swing_window": 10,
}


def prepare(df, config):
    """Compute all indicators on the DataFrame."""
    df = df.copy()

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # EMAs
    df["ema5"] = df["Close"].ewm(span=config.get("ema_fast", 5)).mean()
    df["ema20"] = df["Close"].ewm(span=config.get("ema_slow", 20)).mean()
    df["ema50"] = df["Close"].ewm(span=config.get("ema_trend", 50)).mean()
    df["ema200"] = df["Close"].ewm(span=config.get("ema_long", 200)).mean()

    # MACD
    macd_fast = df["Close"].ewm(span=config.get("macd_fast", 12)).mean()
    macd_slow = df["Close"].ewm(span=config.get("macd_slow", 26)).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=config.get("macd_signal", 9)).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (Wilder)
    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (Wilder)
    atr_period = config.get("atr_period", 14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    # ADX
    adx_period = config.get("atr_period", 14)
    plus_dm_raw = df["High"].diff()
    minus_dm_raw = -df["Low"].diff()
    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / adx_period, min_periods=adx_period).mean() / df["atr"])
    minus_di = 100 * (minus_dm.ewm(alpha=1 / adx_period, min_periods=adx_period).mean() / df["atr"])
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(alpha=1 / adx_period, min_periods=adx_period).mean()

    # MA gap
    df["ma_gap"] = df["ema5"] - df["ema20"]

    # Volume average
    vol_period = config.get("volume_avg_period", 20)
    df["vol_avg"] = df["Volume"].rolling(vol_period, min_periods=5).mean()

    # Chandelier trailing stop (wide: 3.5x ATR, 20-bar lookback)
    lookback = config.get("chandelier_lookback", 20)
    df["chandelier_max"] = df["High"].rolling(lookback, min_periods=5).max()
    df["chandelier_stop"] = df["chandelier_max"] - config.get("trail_atr_mult", 3.5) * df["atr"]

    # Precomputed entry signal
    df["gap_widening"] = df["ma_gap"] > df["ma_gap"].shift(1)

    return df


def should_enter(last, config, prev=None):
    """
    Entry: MA gap accelerating + MACD + volume + strong trend.

    1. Price above EMA200 (long-term uptrend)
    2. EMA5 > EMA20 (short-term uptrend)
    3. Gap widening (current gap > prev gap)
    4. Gap is meaningfully positive (> gap_min_pct of price)
    5. MACD line > signal line, histogram > 0
    6. Volume > 1.5x 20-day average (momentum needs volume)
    7. RSI not overbought (< rsi_entry_max)
    8. ADX >= 20 (stronger trend filter)
    """
    if pd.isna(last.get("ema200")) or pd.isna(last.get("atr")) or pd.isna(last.get("adx")):
        return False

    close = last["Close"]

    # 1. Long-term trend filter
    if close < last["ema200"]:
        return False

    # 2. Short-term trend
    if last["ema5"] <= last["ema20"]:
        return False

    # 3 & 4. MA gap acceleration
    gap = last["ma_gap"]
    gap_min = close * config.get("gap_min_pct", 0.002)
    if gap < gap_min:
        return False
    if not last.get("gap_widening", False):
        return False

    # 5. MACD confirmation
    if pd.isna(last.get("macd")) or last["macd"] <= last["macd_signal"]:
        return False
    if last.get("macd_hist", 0) <= 0:
        return False

    # 6. Volume confirmation — momentum needs volume
    vol_mult = config.get("volume_mult", 1.5)
    vol_avg = last.get("vol_avg", 0)
    if vol_avg <= 0 or last.get("Volume", 0) < vol_avg * vol_mult:
        return False

    # 7. RSI not overbought
    if last.get("rsi", 50) > config.get("rsi_entry_max", 70):
        return False

    # 8. ADX trend strength (raised to 20)
    if last.get("adx", 0) < config.get("adx_min", 20):
        return False

    return True


def should_exit(last, config, prev=None):
    """
    Exit logic (v4 — same as v3):
    1. Chandelier trailing stop (3.5x ATR, 20-bar)
    2. RSI overbought (> 75)
    """
    if pd.isna(last.get("atr")) or pd.isna(last.get("chandelier_stop")):
        return False

    # 1. Chandelier trailing stop
    if last["Close"] < last["chandelier_stop"]:
        return True

    # 2. RSI overbought
    if last.get("rsi", 50) > config.get("rsi_exit", 75):
        return True

    return False
