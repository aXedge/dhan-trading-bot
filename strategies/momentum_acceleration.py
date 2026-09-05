"""
Momentum Acceleration Strategy v2.1
====================================

Based on the user\'s personal trading methodology:
- Entry: MA gap acceleration (EMA5 > EMA20, gap widening) + MACD confirmation + support proximity
- Exit: Chandelier trailing stop (stateless) + RSI overbought + MACD crossover + gap collapse
- Risk: Initial 6% stop loss (backstop), 15% target

v2.1 fixes:
- Removed module-level _state (was causing 0 trades — incompatible with backtest engine)
- Replaced stateful trailing stop with Chandelier exit (rolling max high - ATR multiplier)
- All exit logic is now stateless (only depends on last/prev row data)
- No minimum holding period needed — Chandelier stop is loose at entry (uses 10-bar lookback)
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.06,          # 6% initial fixed stop (backstop)
    "target_pct": 0.15,             # 15% target

    # Exit parameters
    "trail_atr_mult": 2.5,          # Chandelier: rolling_max(High, 10) - 2.5 * ATR
    "chandelier_lookback": 10,      # bars to look back for max high
    "rsi_exit": 75,                 # exit on RSI overbought
    "gap_collapse_pct": 0.003,      # gap must collapse below 0.3% of price

    # Entry parameters
    "ema_fast": 5,
    "ema_slow": 20,
    "ema_trend": 50,
    "ema_long": 200,
    "gap_min_pct": 0.002,          # minimum gap as % of price (0.2%)
    "support_tol_pct": 0.02,        # within 2% of support level
    "rsi_entry_max": 70,            # max RSI for entry
    "adx_min": 15,                  # minimum ADX for entry

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

    # RSI (Wilder\'s method)
    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (Wilder\'s method)
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

    # Swing low (for support proximity)
    swing_window = config.get("swing_window", 10)
    df["swing_low"] = df["Low"].rolling(swing_window, min_periods=3).min()

    # Fib 50% approx: midpoint of recent range
    swing_high = df["High"].rolling(20, min_periods=5).max()
    df["fib_50"] = (swing_high + df["swing_low"]) / 2

    # Chandelier exit: rolling max high for trailing stop
    lookback = config.get("chandelier_lookback", 10)
    df["chandelier_max"] = df["High"].rolling(lookback, min_periods=3).max()
    df["chandelier_stop"] = df["chandelier_max"] - config.get("trail_atr_mult", 2.5) * df["atr"]

    return df


def should_enter(last, config, prev=None):
    """
    Entry: MA gap accelerating + MACD confirmation + support proximity.

    1. Price above EMA200 (long-term uptrend)
    2. EMA5 > EMA20 (short-term uptrend)
    3. Gap widening (current gap > prev gap)
    4. Gap is meaningfully positive (> gap_min_pct of price)
    5. MACD line > signal line
    6. MACD histogram > 0
    7. Near support (EMA20, EMA50, swing low, or fib 50%)
    8. RSI not overbought (< rsi_entry_max)
    9. ADX >= adx_min (trend strength)
    """
    if prev is None:
        return False

    if pd.isna(last.get("ema200")) or pd.isna(last.get("atr")) or pd.isna(last.get("adx")):
        return False

    close = last["Close"]

    # 1. Long-term trend filter
    if close < last["ema200"]:
        return False

    # 2. Short-term trend
    if last["ema5"] <= last["ema20"]:
        return False

    # 3 & 4. MA gap acceleration — gap must be positive AND widening
    gap = last["ema5"] - last["ema20"]
    prev_gap = prev["ema5"] - prev["ema20"]
    gap_min = close * config.get("gap_min_pct", 0.002)
    if gap < gap_min:
        return False
    if gap <= prev_gap:
        return False

    # 5 & 6. MACD confirmation
    if pd.isna(last.get("macd")) or last["macd"] <= last["macd_signal"]:
        return False
    if last.get("macd_hist", 0) <= 0:
        return False

    # 7. Support proximity — near EMA20, EMA50, swing low, or fib 50%
    support_tol = close * config.get("support_tol_pct", 0.02)
    near_ema20 = abs(close - last["ema20"]) < support_tol
    near_ema50 = abs(close - last["ema50"]) < support_tol * 1.5
    near_swing = abs(close - last.get("swing_low", close)) < support_tol * 1.5
    near_fib = abs(close - last.get("fib_50", close)) < support_tol * 1.5
    if not (near_ema20 or near_ema50 or near_swing or near_fib):
        return False

    # 8. RSI not overbought
    if last.get("rsi", 50) > config.get("rsi_entry_max", 70):
        return False

    # 9. ADX trend strength
    if last.get("adx", 0) < config.get("adx_min", 15):
        return False

    return True


def should_exit(last, config, prev=None):
    """
    Exit logic (v2.1 — fully stateless):

    1. Chandelier trailing stop: close < chandelier_stop (primary exit)
    2. RSI overbought: RSI > rsi_exit
    3. MACD bearish crossover: MACD crosses below signal
    4. Gap collapse: MA gap narrows below gap_collapse_pct of price
    """
    if pd.isna(last.get("atr")) or pd.isna(last.get("chandelier_stop")):
        return False

    # 1. Chandelier trailing stop — primary exit, lets winners run
    if last["Close"] < last["chandelier_stop"]:
        return True

    # 2. RSI overbought — take profit
    if last.get("rsi", 50) > config.get("rsi_exit", 75):
        return True

    # 3. MACD bearish crossover
    if prev is not None:
        if last.get("macd", 0) < last.get("macd_signal", 0) and \
           prev.get("macd", 0) >= prev.get("macd_signal", 0):
            return True

    # 4. MA gap collapse — momentum has dissipated
    gap = last.get("ma_gap", 0)
    gap_threshold = last["Close"] * config.get("gap_collapse_pct", 0.003)
    if gap < gap_threshold:
        if prev is not None:
            prev_gap = prev.get("ma_gap", gap * 2)
            if prev_gap > gap_threshold:
                return True

    return False
