"""
Bollinger Band Squeeze Breakout Strategy
==========================================

Based on academic research on Nifty 50 (2011-2020):
  - BB outperformed both MACD and RSI strategies
  - BB was profitable in 80% of years vs 50% for MACD/RSI
  - BB squeeze specifically works because Indian stocks have strong
    volatility-cycle patterns — low volatility reliably precedes big moves

Strategy:
  - Entry: BB bandwidth at 6-month low (squeeze), then close above upper band
    with volume > 1.5x average
  - Exit: Close back below middle band (20 SMA) or 3x ATR target
  - Risk: 7% stop loss, 15% target

The squeeze identifies compression. The breakout above upper band confirms direction.
Volume confirms institutional participation.
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.07,          # 7% stop loss
    "target_pct": 0.15,            # 15% target

    # Bollinger Band parameters
    "bb_period": 20,               # BB period (standard: 20)
    "bb_std": 2.0,                 # BB standard deviations (standard: 2.0)
    "squeeze_lookback": 125,        # 6-month lookback for squeeze detection (~125 trading days)
    "min_squeeze_days": 3,          # bandwidth must be at min for >= 3 consecutive days

    # Volume confirmation
    "volume_mult": 1.5,             # breakout volume > 1.5x average
    "volume_avg_period": 20,

    # Exit parameters
    "use_midband_exit": True,       # exit if close < middle band
    "rsi_exit": 75,                 # RSI overbought exit

    # Indicator parameters
    "rsi_period": 14,
    "atr_period": 14,
}


def prepare(df, config):
    """Compute all indicators on the DataFrame."""
    df = df.copy()

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Bollinger Bands
    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    df["bb_mid"] = df["Close"].rolling(bb_period).mean()
    rolling_std = df["Close"].rolling(bb_period).std()
    df["bb_upper"] = df["bb_mid"] + bb_std * rolling_std
    df["bb_lower"] = df["bb_mid"] - bb_std * rolling_std

    # Bandwidth: (upper - lower) / mid * 100
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100

    # Squeeze detection: bandwidth at 6-month low
    squeeze_lookback = config.get("squeeze_lookback", 125)
    df["bb_width_min"] = df["bb_width"].rolling(squeeze_lookback, min_periods=20).min()
    df["is_squeeze"] = df["bb_width"] <= df["bb_width_min"]

    # Consecutive squeeze days (must be >= min_squeeze_days)
    min_sq = config.get("min_squeeze_days", 3)
    # Count consecutive True values
    squeeze_run = df["is_squeeze"].astype(int)
    squeeze_run = squeeze_run.groupby((squeeze_run != squeeze_run.shift()).cumsum()).cumsum()
    df["squeeze_days"] = squeeze_run.where(df["is_squeeze"], 0)
    df["squeeze_confirmed"] = df["squeeze_days"] >= min_sq

    # Track if we were in squeeze recently (within last 10 bars)
    df["recent_squeeze"] = df["squeeze_confirmed"].rolling(10, min_periods=1).max().astype(bool)

    # Breakout above upper band
    df["bb_breakout_up"] = df["Close"] > df["bb_upper"]

    # Volume average
    vol_period = config.get("volume_avg_period", 20)
    df["vol_avg"] = df["Volume"].rolling(vol_period, min_periods=5).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_avg"]

    # RSI
    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    atr_period = config.get("atr_period", 14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    # EMA200 for trend context
    df["ema200"] = df["Close"].ewm(span=200).mean()

    # Precomputed: close below mid band (for exit)
    df["below_midband"] = df["Close"] < df["bb_mid"]

    return df


def should_enter(last, config, prev=None):
    """
    Entry: BB squeeze confirmed + breakout above upper band + volume confirmation.

    1. Recent squeeze (within last 10 bars, bandwidth was at 6-month low for >= 3 days)
    2. Close above upper BB (breakout)
    3. Volume > 1.5x average (institutional participation)
    4. Price above EMA200 (trade in direction of long-term trend)
    """
    if pd.isna(last.get("bb_upper")) or pd.isna(last.get("ema200")) or pd.isna(last.get("vol_ratio")):
        return False

    close = last["Close"]

    # 1. Recent squeeze
    if not last.get("recent_squeeze", False):
        return False

    # 2. Breakout above upper band
    if not last.get("bb_breakout_up", False):
        return False

    # 3. Volume confirmation
    vol_mult = config.get("volume_mult", 1.5)
    if last.get("vol_ratio", 0) < vol_mult:
        return False

    # 4. Long-term trend filter
    if close < last["ema200"]:
        return False

    return True


def should_exit(last, config, prev=None):
    """
    Exit:
    1. Close below middle band (20 SMA) — breakout failed
    2. RSI overbought (> 75) — take profit
    """
    if pd.isna(last.get("bb_mid")):
        return False

    # 1. Close below middle band — breakout failed
    if config.get("use_midband_exit", True) and last.get("below_midband", False):
        return True

    # 2. RSI overbought
    if last.get("rsi", 50) > config.get("rsi_exit", 75):
        return True

    return False
