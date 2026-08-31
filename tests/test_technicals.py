"""
Tests for the technical signal generator (Layer 2).

Run: pytest tests/test_technicals.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestComputeIndicators:
    """Test technical indicator computation."""

    def _make_config(self):
        return {
            "rsi_period": 14,
            "volume_avg_period": 20,
            "swing": {"lookback_days": 20},
            "positional": {"ema_fast": 50, "ema_slow": 200},
        }

    def _make_price_data(self, days=250, trend="up"):
        """Generate synthetic price data with a clear trend."""
        dates = pd.date_range(start="2025-01-01", periods=days, freq="B")

        if trend == "up":
            prices = 100 + np.cumsum(np.random.uniform(0.1, 1.0, days))
        elif trend == "down":
            prices = 300 - np.cumsum(np.random.uniform(0.1, 1.0, days))
        else:  # sideways
            prices = 200 + np.random.uniform(-10, 10, days)

        df = pd.DataFrame({
            "Open": prices,
            "High": prices + np.random.uniform(0.5, 3, days),
            "Low": prices - np.random.uniform(0.5, 3, days),
            "Close": prices,
            "Volume": np.random.randint(100000, 500000, days),
        }, index=dates)

        return df

    def test_indicators_added(self):
        """All indicator columns should be present after computation."""
        from src.technicals import compute_indicators

        df = self._make_price_data(days=250)
        config = self._make_config()

        result = compute_indicators(df, config)

        assert "ema10" in result.columns
        assert "ema20" in result.columns
        assert "ema50" in result.columns
        assert "ema200" in result.columns
        assert "rsi" in result.columns
        assert "vol_avg" in result.columns
        assert "high_lookback" in result.columns

    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        from src.technicals import compute_indicators

        df = self._make_price_data(days=250)
        config = self._make_config()

        result = compute_indicators(df, config)

        rsi_values = result["rsi"].dropna()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100


class TestSwingEntry:
    """Test swing entry signal detection."""

    def _make_config(self):
        return {
            "swing": {
                "lookback_days": 20,
                "volume_multiplier": 1.5,
                "rsi_min": 40,
                "rsi_max": 65,
                "stop_loss_pct": 0.02,
            }
        }

    def test_returns_false_for_insufficient_data(self):
        """Should return False if not enough data."""
        from src.technicals import check_swing_entry

        df = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1000, 1100, 1200],
                           "rsi": [50, 50, 50], "high_lookback": [99, 100, 101],
                           "vol_avg": [1000, 1000, 1000]})
        config = self._make_config()
        assert check_swing_entry(df, config) is False


class TestCheckExit:
    """Test exit signal detection."""

    def test_swing_exit_below_ema10(self):
        """Should detect exit when price closes below EMA10."""
        from src.technicals import check_exit

        config = {"swing": {"stop_loss_pct": 0.02}, "positional": {"stop_loss_pct": 0.05}}

        df = pd.DataFrame({
            "Close": [105, 104, 103],
            "ema10": [104, 104, 104],
            "rsi": [50, 50, 50],
            "ema50": [100, 100, 100],
            "ema200": [90, 90, 90],
        })

        assert check_exit(df, "swing", config) is True

    def test_swing_no_exit_when_above_ema(self):
        """Should NOT trigger exit when price is above EMA10 and RSI is normal."""
        from src.technicals import check_exit

        config = {"swing": {"stop_loss_pct": 0.02}, "positional": {"stop_loss_pct": 0.05}}

        df = pd.DataFrame({
            "Close": [110, 111, 112],
            "ema10": [105, 105, 105],
            "rsi": [55, 55, 55],
            "ema50": [100, 100, 100],
            "ema200": [90, 90, 90],
        })

        assert check_exit(df, "swing", config) is False
