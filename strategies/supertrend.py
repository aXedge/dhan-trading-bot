"""
Multi-Timeframe Supertrend Strategy v3
======================================

Based on documented backtest results on Indian stocks (PF 2.40, 44.9% win rate,
127 trades, avg win 9.41%, avg loss -3.19%, max DD -22.4%).

v3 = Fixed ST computation (from v2) + ATR trailing exit (from documented strategy):
  - Fixed Supertrend band computation (correct Pine Script logic)
  - Entry: Daily ST flip bullish + Weekly ST bullish
  - Exit: 2.5x ATR trailing stop (PRIMARY) + Daily ST flip bearish + Weekly ST bearish
  - The ATR trailing stop is the key — it exits before the trend fully reverses

v1 had wrong ST computation (2645 trades) + Chandelier (too tight lookback=10) → PF 0.93
v2 had fixed ST (83 trades) + pure ST exit (no trailing) → PF 0.44
v3 = fixed ST (83 trades) + ATR trailing (20-bar lookback) → should be the sweet spot
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.15,          # 15% fixed stop (backstop)
    "target_pct": 0.30,            # 30% target (backstop)

    # Supertrend parameters
    "st_period": 10,
    "st_multiplier": 3.0,
    "weekly_st_period": 10,
    "weekly_st_multiplier": 3.0,

    # Exit: ATR trailing stop (this is the key exit from the documented strategy)
    "trail_atr_mult": 2.5,          # 2.5x ATR trailing stop
    "trail_lookback": 20,           # 20-bar lookback for max high
    "use_weekly_exit": True,

    # Indicator parameters
    "atr_period": 14,
}


def compute_atr(df, period=14):
    """Compute ATR (Wilder's method)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def compute_supertrend(df, atr, period, multiplier):
    """Compute Supertrend using standard Pine Script logic."""
    hl2 = (df["High"] + df["Low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(df)
    final_upper = pd.Series(index=df.index, dtype=float)
    final_lower = pd.Series(index=df.index, dtype=float)
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    first_valid = atr.first_valid_index()
    if first_valid is None:
        return supertrend, direction

    first_idx = df.index.get_loc(first_valid)
    final_upper.iloc[first_idx] = basic_upper.iloc[first_idx]
    final_lower.iloc[first_idx] = basic_lower.iloc[first_idx]
    direction.iloc[first_idx] = 1 if df["Close"].iloc[first_idx] > final_upper.iloc[first_idx] else -1
    supertrend.iloc[first_idx] = final_lower.iloc[first_idx] if direction.iloc[first_idx] == 1 else final_upper.iloc[first_idx]

    for i in range(first_idx + 1, n):
        prev_close = df["Close"].iloc[i - 1]
        prev_fu = final_upper.iloc[i - 1]
        prev_fl = final_lower.iloc[i - 1]

        if pd.isna(prev_fu):
            final_upper.iloc[i] = basic_upper.iloc[i]
        elif basic_upper.iloc[i] < prev_fu or prev_close > prev_fu:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_fu

        if pd.isna(prev_fl):
            final_lower.iloc[i] = basic_lower.iloc[i]
        elif basic_lower.iloc[i] > prev_fl or prev_close < prev_fl:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_fl

        prev_dir = direction.iloc[i - 1]
        close = df["Close"].iloc[i]

        if prev_dir == 1:
            if close < final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if close > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]

    return supertrend, direction


def resample_weekly(df):
    """Resample daily OHLCV to weekly bars."""
    return df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def prepare(df, config):
    """Compute all indicators on the DataFrame."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Daily ATR and Supertrend
    atr_period = config.get("atr_period", 14)
    df["atr"] = compute_atr(df, atr_period)

    st_period = config.get("st_period", 10)
    st_mult = config.get("st_multiplier", 3.0)
    daily_st, daily_dir = compute_supertrend(df, df["atr"], st_period, st_mult)
    df["st_value"] = daily_st
    df["st_dir"] = daily_dir
    df["st_flip_bull"] = (daily_dir == 1) & (daily_dir.shift(1) == -1)
    df["st_flip_bear"] = (daily_dir == -1) & (daily_dir.shift(1) == 1)

    # Weekly Supertrend
    weekly = resample_weekly(df)
    weekly["atr"] = compute_atr(weekly, atr_period)

    w_st_period = config.get("weekly_st_period", 10)
    w_st_mult = config.get("weekly_st_multiplier", 3.0)
    weekly_st, weekly_dir = compute_supertrend(weekly, weekly["atr"], w_st_period, w_st_mult)
    weekly_bullish = (weekly_dir == 1)

    df["weekly_bullish"] = weekly_bullish.reindex(df.index, method="ffill")

    weekly_bear_flip = (weekly_dir == -1) & (weekly_dir.shift(1) == 1)
    df["weekly_flip_bear"] = weekly_bear_flip.reindex(df.index, method="ffill").fillna(False)

    # ATR trailing stop (2.5x ATR, 20-bar lookback)
    trail_mult = config.get("trail_atr_mult", 2.5)
    lookback = config.get("trail_lookback", 20)
    df["trail_max"] = df["High"].rolling(lookback, min_periods=5).max()
    df["trail_stop"] = df["trail_max"] - trail_mult * df["atr"]

    return df


def should_enter(last, config, prev=None):
    """Entry: Daily ST flip bullish + Weekly ST bullish."""
    if pd.isna(last.get("st_dir")) or pd.isna(last.get("weekly_bullish")):
        return False
    if not last.get("st_flip_bull", False):
        return False
    if not last.get("weekly_bullish", False):
        return False
    return True


def should_exit(last, config, prev=None):
    """
    Exit (3 layers):
    1. ATR trailing stop (PRIMARY — 2.5x ATR below 20-bar max high)
    2. Daily Supertrend flips bearish
    3. Weekly Supertrend turns bearish
    """
    if pd.isna(last.get("atr")) or pd.isna(last.get("trail_stop")):
        return False

    # 1. ATR trailing stop — primary exit, cuts losses early
    if last["Close"] < last["trail_stop"]:
        return True

    # 2. Daily ST flips bearish
    if last.get("st_flip_bear", False):
        return True

    # 3. Weekly ST turns bearish
    if config.get("use_weekly_exit", True) and last.get("weekly_flip_bear", False):
        return True

    return False
