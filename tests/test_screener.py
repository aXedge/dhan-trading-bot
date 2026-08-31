"""
Tests for the fundamental screener (Layer 1).

Run: pytest tests/test_screener.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPiotroskiFscore:
    """Test the Piotroski F-score computation."""

    def _make_statements(self, net_income=100, cfo=120, roce=15, prev_roce=12,
                         borrowings=50, equity=200, prev_borrowings=60,
                         curr_assets=300, curr_liab=150,
                         shares=1000, prev_shares=1000,
                         opm=20, prev_opm=18,
                         sales=1000, prev_sales=900):
        """Helper: build mock financial statement lists."""
        pl = [
            {"Net Profit": net_income * 0.8, "ROCE": prev_roce, "OPM": prev_opm, "Sales": prev_sales},
            {"Net Profit": net_income, "ROCE": roce, "OPM": opm, "Sales": sales},
        ]
        bs = [
            {"Borrowings": prev_borrowings, "Total Equity": equity,
             "Current Assets": curr_assets, "Current Liabilities": curr_liab,
             "Shares Outstanding": prev_shares},
            {"Borrowings": borrowings, "Total Equity": equity,
             "Current Assets": curr_assets, "Current Liabilities": curr_liab,
             "Shares Outstanding": shares},
        ]
        cf = [
            {"Operating Cash Flow": cfo * 0.8},
            {"Operating Cash Flow": cfo},
        ]
        return pl, bs, cf

    def test_all_positive(self):
        """A stock improving on all metrics should score 9."""
        from src.screener import compute_piotroski_fscore

        pl, bs, cf = self._make_statements(
            net_income=100, cfo=120, roce=15, prev_roce=12,
            borrowings=50, prev_borrowings=60,
            shares=1000, prev_shares=1000,
            opm=20, prev_opm=18,
            sales=1000, prev_sales=900,
        )
        score = compute_piotroski_fscore(pl, bs, cf)
        assert score == 9, f"Expected 9, got {score}"

    def test_negative_income(self):
        """Negative net income should lose 1 point (and possibly CFO > NI point)."""
        from src.screener import compute_piotroski_fscore

        pl, bs, cf = self._make_statements(net_income=-50, cfo=-30)
        score = compute_piotroski_fscore(pl, bs, cf)
        assert score <= 7, f"Expected <= 7 with negative income, got {score}"

    def test_insufficient_data(self):
        """Insufficient data (only 1 year) should return 0."""
        from src.screener import compute_piotroski_fscore

        pl = [{"Net Profit": 100, "ROCE": 15, "OPM": 20, "Sales": 1000}]
        bs = [{"Borrowings": 50, "Total Equity": 200}]
        cf = [{"Operating Cash Flow": 120}]

        score = compute_piotroski_fscore(pl, bs, cf)
        assert score == 0, f"Expected 0 with insufficient data, got {score}"


class TestRevenueCAGR:
    """Test revenue CAGR computation."""

    def test_positive_growth(self):
        from src.screener import _compute_revenue_cagr

        pl = [
            {"Sales": 800},   # 3 years ago
            {"Sales": 900},
            {"Sales": 1000},
            {"Sales": 1100},  # current
        ]
        cagr = _compute_revenue_cagr(pl, 3)
        assert cagr > 0
        # (1100/800)^(1/3) - 1 = ~11.1%
        assert 10 < cagr < 12

    def test_zero_past_sales(self):
        from src.screener import _compute_revenue_cagr

        pl = [{"Sales": 0}, {"Sales": 500}, {"Sales": 600}, {"Sales": 700}]
        cagr = _compute_revenue_cagr(pl, 3)
        assert cagr == 0.0


class TestSafeDiv:
    """Test the safe division helper."""

    def test_normal_division(self):
        from src.screener import _safe_div
        assert _safe_div(10, 2) == 5.0

    def test_zero_denominator(self):
        from src.screener import _safe_div
        assert _safe_div(10, 0) == 0.0
