"""
Technical indicators — strategy-agnostic.

All strategies import indicators from here. This ensures consistency
across strategies and avoids duplicate implementations.

Available indicators:
  - EMA (10, 20, 50, 200 or custom spans)
  - RSI (14 or custom period)
  - ADX / +DI / -DI (14 or custom period)
  - MACD (12, 26, 9 or custom)
  - Supertrend (10, 3 or custom)
  - Bollinger Bands (20, 2σ or custom)
  - ATR (14 or custom)
  - Volume averages and highs
  - Lookback highs
  - Returns (1d, 5d, 20d)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def add_emas(df: pd.DataFrame, spans: list = [10, 20, 50, 200]) -> pd.DataFrame:
    """Add EMA columns (ema10, ema20, etc.) to the DataFrame."""
    df = df.copy()
    for span in spans:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# ---------------------------------------------------------------------------
# ADX / +DI / -DI
# ---------------------------------------------------------------------------

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add ADX, +DI, -DI columns.

    ADX >= 25 = strong trend
    ADX < 20 = choppy/range-bound
    +DI > -DI = bullish direction
    """
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)

    # Wilder's smoothing
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.rolling(period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    return df


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram."""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    return df


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.DataFrame:
    """
    Add Supertrend indicator with NaN-safe warmup handling.

    supertrend_dir: 1 = green (uptrend), -1 = red (downtrend)
    """
    df = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for i in range(1, len(df)):
        if pd.isna(upper_basic.iloc[i]) or pd.isna(lower_basic.iloc[i]):
            continue
        if pd.isna(upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
            lower_band.iloc[i] = lower_basic.iloc[i]
            continue
        if (upper_basic.iloc[i] < upper_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        if (lower_basic.iloc[i] > lower_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]):
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    first_valid = None
    for i in range(len(df)):
        if not pd.isna(upper_band.iloc[i]) and not pd.isna(lower_band.iloc[i]):
            first_valid = i
            break

    if first_valid is not None:
        supertrend.iloc[first_valid] = upper_band.iloc[first_valid]
        direction.iloc[first_valid] = -1

        for i in range(first_valid + 1, len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
            close = df["Close"].iloc[i]
            prev_st = supertrend.iloc[i - 1]
            if pd.isna(prev_st):
                prev_st = upper_band.iloc[i]

            if close <= prev_st:
                supertrend.iloc[i] = min(upper_band.iloc[i], prev_st) if not pd.isna(upper_band.iloc[i]) else prev_st
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = max(lower_band.iloc[i], prev_st) if not pd.isna(lower_band.iloc[i]) else prev_st
                direction.iloc[i] = 1
                if direction.iloc[i - 1] == -1 or pd.isna(direction.iloc[i - 1]):
                    supertrend.iloc[i] = lower_band.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def add_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """Add Bollinger Bands (upper, mid, lower)."""
    df = df.copy()
    df["bb_mid"] = df["Close"].rolling(period).mean()
    bb_std = df["Close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + std_dev * bb_std
    df["bb_lower"] = df["bb_mid"] - std_dev * bb_std
    return df


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range."""
    df = df.copy()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    df["atr_pct"] = df["atr"] / df["Close"]  # ATR as % of price
    return df


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def add_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume average and max."""
    df = df.copy()
    df["vol_avg"] = df["Volume"].rolling(period).mean()
    df["vol_max"] = df["Volume"].rolling(period).max()
    return df


# ---------------------------------------------------------------------------
# Lookback highs
# ---------------------------------------------------------------------------

def add_lookback_highs(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Add N-day high (excluding today)."""
    df = df.copy()
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)
    return df


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-day, 5-day, and 20-day returns."""
    df = df.copy()
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    return df


# ---------------------------------------------------------------------------
# Convenience: compute everything
# ---------------------------------------------------------------------------

def compute_all_indicators(
    df: pd.DataFrame,
    lookback: int = 10,
    rsi_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Compute all indicators in one call. Used by strategies that need everything."""
    df = add_emas(df)
    df = add_rsi(df, rsi_period)
    df = add_adx(df, adx_period)
    df = add_macd(df)
    df = add_supertrend(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_indicators(df)
    df = add_lookback_highs(df, lookback)
    df = add_returns(df)
    return df
