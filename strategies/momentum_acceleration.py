"""
Momentum Acceleration Strategy v3 (best performing version)
==================================

Based on the user\'s personal trading methodology:
- Entry: MA gap acceleration + MACD confirmation + strong trend (ADX >= 15)
- Exit: Wide Chandelier trailing stop (3.5x ATR, 20-bar) + RSI overbought
- Risk: Initial 6% stop loss (backstop), 15% target

Best results so far:
  NIFTY 50: PF 0.91, 182 trades, 36% win rate, avg win 7.33%
  Midcap: PF 0.80, 259 trades, 30% win rate, avg win 9.58%
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    "stop_loss_pct": 0.06,
    "target_pct": 0.15,

    "trail_atr_mult": 3.5,
    "chandelier_lookback": 20,
    "rsi_exit": 75,

    "ema_fast": 5,
    "ema_slow": 20,
    "ema_trend": 50,
    "ema_long": 200,
    "gap_min_pct": 0.002,
    "rsi_entry_max": 70,
    "adx_min": 15,

    "rsi_period": 14,
    "atr_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "swing_window": 10,
}


def prepare(df, config):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["ema5"] = df["Close"].ewm(span=config.get("ema_fast", 5)).mean()
    df["ema20"] = df["Close"].ewm(span=config.get("ema_slow", 20)).mean()
    df["ema50"] = df["Close"].ewm(span=config.get("ema_trend", 50)).mean()
    df["ema200"] = df["Close"].ewm(span=config.get("ema_long", 200)).mean()

    macd_fast = df["Close"].ewm(span=config.get("macd_fast", 12)).mean()
    macd_slow = df["Close"].ewm(span=config.get("macd_slow", 26)).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=config.get("macd_signal", 9)).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    atr_period = config.get("atr_period", 14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    adx_period = config.get("atr_period", 14)
    plus_dm_raw = df["High"].diff()
    minus_dm_raw = -df["Low"].diff()
    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / adx_period, min_periods=adx_period).mean() / df["atr"])
    minus_di = 100 * (minus_dm.ewm(alpha=1 / adx_period, min_periods=adx_period).mean() / df["atr"])
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(alpha=1 / adx_period, min_periods=adx_period).mean()

    df["ma_gap"] = df["ema5"] - df["ema20"]

    swing_window = config.get("swing_window", 10)
    df["swing_low"] = df["Low"].rolling(swing_window, min_periods=3).min()
    swing_high = df["High"].rolling(20, min_periods=5).max()
    df["fib_50"] = (swing_high + df["swing_low"]) / 2

    lookback = config.get("chandelier_lookback", 20)
    df["chandelier_max"] = df["High"].rolling(lookback, min_periods=5).max()
    df["chandelier_stop"] = df["chandelier_max"] - config.get("trail_atr_mult", 3.5) * df["atr"]

    df["gap_widening"] = df["ma_gap"] > df["ma_gap"].shift(1)

    return df


def should_enter(last, config, prev=None):
    if pd.isna(last.get("ema200")) or pd.isna(last.get("atr")) or pd.isna(last.get("adx")):
        return False

    close = last["Close"]

    if close < last["ema200"]:
        return False
    if last["ema5"] <= last["ema20"]:
        return False

    gap = last["ma_gap"]
    gap_min = close * config.get("gap_min_pct", 0.002)
    if gap < gap_min:
        return False
    if not last.get("gap_widening", False):
        return False

    if pd.isna(last.get("macd")) or last["macd"] <= last["macd_signal"]:
        return False
    if last.get("macd_hist", 0) <= 0:
        return False

    if last.get("rsi", 50) > config.get("rsi_entry_max", 70):
        return False

    if last.get("adx", 0) < config.get("adx_min", 15):
        return False

    return True


def should_exit(last, config, prev=None):
    if pd.isna(last.get("atr")) or pd.isna(last.get("chandelier_stop")):
        return False

    if last["Close"] < last["chandelier_stop"]:
        return True

    if last.get("rsi", 50) > config.get("rsi_exit", 75):
        return True

    return False
