"""
Multi-Timeframe Supertrend Strategy
=====================================

Based on documented backtest results on Indian stocks (PF 2.40, 44.9% win rate,
127 trades, avg win 9.41%, avg loss -3.19%, max DD -22.4%).

The key insight: a weekly Supertrend filter on daily Supertrend signals
cuts drawdown from -40% to -22% while maintaining profit factor.

Strategy:
  - Weekly filter: Supertrend(10, 3) on weekly bars must be bullish (green)
  - Daily entry:   Supertrend(10, 3) on daily bars flips to bullish (green)
  - Exit:          Daily Supertrend flips bearish (red), OR
                   2.5x ATR trailing stop from highest high since entry, OR
                   Weekly Supertrend turns bearish (macro trend change)

The Supertrend indicator:
  - Uses ATR for volatility-adaptive bands
  - Upper band = (High + Low)/2 + multiplier * ATR
  - Lower band = (High + Low)/2 - multiplier * ATR
  - When close > upper band, trend = UP (lower band becomes support)
  - When close < lower band, trend = DOWN (upper band becomes resistance)
  - Supertrend line = lower band when uptrend, upper band when downtrend
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.07,          # 7% fixed stop (backstop — rarely hit with trailing)
    "target_pct": 0.20,            # 20% target (rarely hit — trailing is primary exit)

    # Supertrend parameters
    "st_period": 10,               # ATR period for Supertrend
    "st_multiplier": 3.0,          # Supertrend multiplier (standard: 3.0)
    "weekly_st_period": 10,        # ATR period for weekly Supertrend
    "weekly_st_multiplier": 3.0,   # Weekly Supertrend multiplier

    # Exit parameters
    "trail_atr_mult": 2.5,         # Trailing stop: max_high - 2.5 * ATR
    "chandelier_lookback": 20,     # Bars to look back for max high
    "use_weekly_exit": True,        # Exit if weekly Supertrend turns bearish

    # Indicator parameters
    "atr_period": 14,
}


def compute_supertrend(df, period, multiplier):
    """
    Compute Supertrend indicator.

    Returns: (supertrend_value, supertrend_dir)
      supertrend_value: the Supertrend line (support in uptrend, resistance in downtrend)
      supertrend_dir: 1 = uptrend (bullish/green), -1 = downtrend (bearish/red)
    """
    hl2 = (df["High"] + df["Low"]) / 2
    atr = df["atr"]  # ATR must already be computed

    # Initial bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Final bands (carry forward logic)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    # Initialize
    direction.iloc[0] = 1 if df["Close"].iloc[0] > final_upper.iloc[0] else -1

    for i in range(1, len(df)):
        # Update final bands
        if final_upper.iloc[i] > final_upper.iloc[i-1] or df["Close"].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]

        if final_lower.iloc[i] < final_lower.iloc[i-1] or df["Close"].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]

        # Determine direction
        if direction.iloc[i-1] == 1:  # was uptrend
            if df["Close"].iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1  # flip to downtrend
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:  # was downtrend
            if df["Close"].iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1  # flip to uptrend
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


def compute_atr(df, period=14):
    """Compute ATR (Wilder's method)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def prepare(df, config):
    """Compute all indicators on the DataFrame."""
    df = df.copy()

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Daily ATR
    atr_period = config.get("atr_period", 14)
    df["atr"] = compute_atr(df, atr_period)

    # Daily Supertrend
    st_period = config.get("st_period", 10)
    st_mult = config.get("st_multiplier", 3.0)
    daily_st, daily_dir = compute_supertrend(df, st_period, st_mult)
    df["st_value"] = daily_st
    df["st_dir"] = daily_dir
    df["st_bullish"] = daily_dir == 1
    df["st_flip_bull"] = (daily_dir == 1) & (daily_dir.shift(1) == -1)  # fresh buy signal
    df["st_flip_bear"] = (daily_dir == -1) & (daily_dir.shift(1) == 1)  # fresh sell signal

    # Weekly Supertrend (macro trend filter)
    weekly = resample_weekly(df)
    weekly["atr"] = compute_atr(weekly, atr_period)

    w_st_period = config.get("weekly_st_period", 10)
    w_st_mult = config.get("weekly_st_multiplier", 3.0)
    weekly_st, weekly_dir = compute_supertrend(weekly, w_st_period, w_st_mult)
    weekly["st_value"] = weekly_st
    weekly["st_dir"] = weekly_dir
    weekly["weekly_bullish"] = weekly_dir == 1

    # Merge weekly trend back to daily (forward-fill)
    df["weekly_bullish"] = weekly["weekly_bullish"].reindex(df.index, method="ffill")

    # Chandelier trailing stop
    lookback = config.get("chandelier_lookback", 20)
    df["chandelier_max"] = df["High"].rolling(lookback, min_periods=5).max()
    df["chandelier_stop"] = df["chandelier_max"] - config.get("trail_atr_mult", 2.5) * df["atr"]

    # Weekly ST flip to bearish (for macro exit)
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
    Exit:
    1. Daily Supertrend flips bearish (trend reversal on daily)
    2. Chandelier trailing stop hit (2.5x ATR below 20-bar max high)
    3. Weekly Supertrend turns bearish (macro trend change) — if enabled
    """
    if pd.isna(last.get("atr")) or pd.isna(last.get("chandelier_stop")):
        return False

    # 1. Daily Supertrend flips bearish
    if last.get("st_flip_bear", False):
        return True

    # 2. Chandelier trailing stop
    if last["Close"] < last["chandelier_stop"]:
        return True

    # 3. Weekly Supertrend turns bearish (macro exit)
    if config.get("use_weekly_exit", True) and last.get("weekly_flip_bear", False):
        return True

    return False
