"""
Short-Term Reversal Strategy
==============================

Based on documented research on Indian equities:
  - Stocks that fall the most over 5-21 days tend to bounce back within 5 days
  - This is a structural behavioral effect — driven by retail overreaction
  - India has 35-45% retail participation (vs ~15% in US)
  - Documented: 54-58% win rate, +40% return over 5.4 years, IC +0.020-0.025

Adapted for per-stock backtest (original is cross-sectional):
  - Entry: 5-day return < -5%, RSI < 35, above EMA200, volume spike (forced selling)
  - Also check: stock didn't gap down >10% today (panic continuation risk)
  - Exit: RSI > 50 (mean reversion complete) or 10-day time stop
  - Risk: 8% stop loss, 10% target

Key insight: This is the OPPOSITE of trend following. We're buying panic, not momentum.
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.08,          # 8% stop loss
    "target_pct": 0.10,            # 10% target

    # Entry parameters
    "min_5d_return": -0.05,         # stock must be down at least 5% in 5 days
    "max_rsi_entry": 35,           # RSI must be oversold (< 35)
    "ema_long": 200,               # must be above EMA200 (long-term uptrend intact)
    "max_gap_down": -0.10,         # skip if gap down > 10% (panic continuation)
    "volume_spike_mult": 1.3,      # volume > 1.3x average (forced selling signature)
    "volume_avg_period": 20,

    # Exit parameters
    "rsi_exit": 50,                # exit when RSI reverts to 50 (mean reversion complete)
    "max_hold_bars": 10,           # time stop: exit after 10 days regardless

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

    # EMA200 (long-term trend filter)
    df["ema200"] = df["Close"].ewm(span=config.get("ema_long", 200)).mean()

    # RSI (Wilder)
    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 5-day return
    df["ret_5d"] = df["Close"].pct_change(5)

    # Volume average
    vol_period = config.get("volume_avg_period", 20)
    df["vol_avg"] = df["Volume"].rolling(vol_period, min_periods=5).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_avg"]

    # Gap (today's open vs yesterday's close)
    df["gap"] = df["Open"] / df["Close"].shift(1) - 1

    # ATR (for context, not critical)
    atr_period = config.get("atr_period", 14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    # Bar index for time-stop tracking (precomputed as rolling count)
    df["bar_num"] = np.arange(len(df))

    return df


def should_enter(last, config, prev=None):
    """
    Entry: Stock has fallen sharply, RSI oversold, but long-term trend intact.

    1. 5-day return < -5% (sharp decline)
    2. RSI < 35 (oversold)
    3. Close > EMA200 (long-term uptrend still intact — we're buying the dip, not catching a falling knife)
    4. No gap down > 10% today (avoid panic continuation)
    5. Volume > 1.3x average (forced selling signature)
    """
    if pd.isna(last.get("ema200")) or pd.isna(last.get("rsi")) or pd.isna(last.get("ret_5d")):
        return False

    close = last["Close"]

    # 1. Sharp 5-day decline
    if last["ret_5d"] > config.get("min_5d_return", -0.05):
        return False

    # 2. RSI oversold
    if last["rsi"] > config.get("max_rsi_entry", 35):
        return False

    # 3. Long-term trend intact
    if close < last["ema200"]:
        return False

    # 4. No extreme gap down (panic continuation risk)
    if last.get("gap", 0) < config.get("max_gap_down", -0.10):
        return False

    # 5. Volume spike (forced selling)
    vol_mult = config.get("volume_spike_mult", 1.3)
    if last.get("vol_ratio", 0) < vol_mult:
        return False

    return True


def should_exit(last, config, prev=None):
    """
    Exit: RSI reverts to 50 (mean reversion complete) or time stop.

    Note: The backtest engine handles SL/target. We only handle signal exits.
    The time stop is tricky in stateless mode — we use RSI > 50 as the primary exit.
    """
    if pd.isna(last.get("rsi")):
        return False

    # RSI has reverted to neutral — mean reversion complete
    if last["rsi"] > config.get("rsi_exit", 50):
        return True

    return False
