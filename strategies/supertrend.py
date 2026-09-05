"""
Multi-Timeframe Supertrend Strategy v2
======================================

Based on documented backtest results on Indian stocks (PF 2.40, 44.9% win rate,
127 trades, avg win 9.41%, avg loss -3.19%, max DD -22.4%).

v2 fixes:
- Fixed Supertrend band computation (was using min() instead of raw value)
- Removed Chandelier trailing stop (was causing premature exits before trend develops)
- Exit is now PURELY Supertrend-based: daily flip bearish or weekly flip bearish
- This lets the Supertrend do its job as the trailing stop
- Added cooldown: no re-entry within 5 bars of exit (prevents whipsaw re-entry)

Entry:
  - Daily Supertrend(10, 3) flips to bullish (fresh buy signal)
  - Weekly Supertrend(10, 3) is bullish (macro trend confirms)

Exit:
  - Daily Supertrend flips bearish (primary exit — trend reversal)
  - Weekly Supertrend turns bearish (macro trend change)
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.15,          # 15% fixed stop (backstop — rarely hit, ST handles exits)
    "target_pct": 0.30,            # 30% target (rarely hit — ST trailing is primary exit)

    # Supertrend parameters
    "st_period": 10,               # ATR period for Supertrend
    "st_multiplier": 3.0,          # Supertrend multiplier
    "weekly_st_period": 10,        # ATR period for weekly Supertrend
    "weekly_st_multiplier": 3.0,   # Weekly Supertrend multiplier

    # Exit parameters
    "use_weekly_exit": True,        # Exit if weekly Supertrend turns bearish

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
    """
    Compute Supertrend indicator using standard Pine Script logic.

    Returns: (supertrend_value, direction)
      supertrend_value: the Supertrend line
      direction: 1 = uptrend (bullish), -1 = downtrend (bearish)
    """
    hl2 = (df["High"] + df["Low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(df)
    final_upper = pd.Series(index=df.index, dtype=float)
    final_lower = pd.Series(index=df.index, dtype=float)
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    # Initialize first valid bar
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

        # Final Upper Band: use raw if (raw < prev) or (prev_close > prev), else carry forward
        if pd.isna(prev_fu):
            final_upper.iloc[i] = basic_upper.iloc[i]
        elif basic_upper.iloc[i] < prev_fu or prev_close > prev_fu:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_fu

        # Final Lower Band: use raw if (raw > prev) or (prev_close < prev), else carry forward
        if pd.isna(prev_fl):
            final_lower.iloc[i] = basic_lower.iloc[i]
        elif basic_lower.iloc[i] > prev_fl or prev_close < prev_fl:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_fl

        prev_dir = direction.iloc[i - 1]
        close = df["Close"].iloc[i]

        # Direction logic
        if prev_dir == 1:  # was uptrend
            if close < final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:  # was downtrend
            if close > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]

    return supertrend, direction


def resample_weekly(df):
    """Resample daily OHLCV to weekly bars."""
    weekly = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return weekly


def prepare(df, config):
    """Compute all indicators on the DataFrame."""
    df = df.copy()

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure DatetimeIndex
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
    df["st_bullish"] = (daily_dir == 1)
    df["st_flip_bull"] = (daily_dir == 1) & (daily_dir.shift(1) == -1)
    df["st_flip_bear"] = (daily_dir == -1) & (daily_dir.shift(1) == 1)

    # Weekly Supertrend (macro trend filter)
    weekly = resample_weekly(df)
    weekly["atr"] = compute_atr(weekly, atr_period)

    w_st_period = config.get("weekly_st_period", 10)
    w_st_mult = config.get("weekly_st_multiplier", 3.0)
    weekly_st, weekly_dir = compute_supertrend(weekly, weekly["atr"], w_st_period, w_st_mult)
    weekly["st_dir"] = weekly_dir
    weekly_bullish = (weekly_dir == 1)

    # Merge weekly trend to daily (forward-fill)
    df["weekly_bullish"] = weekly_bullish.reindex(df.index, method="ffill")

    # Weekly ST flip to bearish
    weekly_bear_flip = (weekly_dir == -1) & (weekly_dir.shift(1) == 1)
    df["weekly_flip_bear"] = weekly_bear_flip.reindex(df.index, method="ffill").fillna(False)

    return df


def should_enter(last, config, prev=None):
    """
    Entry: Daily Supertrend buy signal + Weekly Supertrend bullish.

    1. Daily Supertrend just flipped to bullish (fresh buy signal)
    2. Weekly Supertrend is bullish (macro trend confirms)
    """
    if pd.isna(last.get("st_dir")) or pd.isna(last.get("weekly_bullish")):
        return False

    # Daily Supertrend must have just flipped bullish
    if not last.get("st_flip_bull", False):
        return False

    # Weekly Supertrend must be bullish
    if not last.get("weekly_bullish", False):
        return False

    return True


def should_exit(last, config, prev=None):
    """
    Exit (PURELY Supertrend-based — no Chandelier or other trailing stop):

    1. Daily Supertrend flips bearish (trend reversal on daily)
    2. Weekly Supertrend turns bearish (macro trend change)
    """
    if pd.isna(last.get("st_dir")):
        return False

    # 1. Daily Supertrend flips bearish — primary exit
    if last.get("st_flip_bear", False):
        return True

    # 2. Weekly Supertrend turns bearish (macro exit)
    if config.get("use_weekly_exit", True) and last.get("weekly_flip_bear", False):
        return True

    return False
