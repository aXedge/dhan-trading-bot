"""
Layer 1: Fundamental Screener
==============================

Scans the Nifty Midcap 150 + Nifty 100 universe and scores each stock on
Piotroski F-score, ROCE, debt/equity, promoter holding, and pledging.

Output: data/basket.json — top 10-15 stocks by composite score.

Runs weekly (Saturday 10 AM IST) — can be run on your laptop.
No Dhan API access needed.
"""

import time
from utils import load_config, save_json, load_json, setup_logger, now_iso

logger = setup_logger(__name__, "screener.log")


def compute_piotroski_fscore(pl: list, bs: list, cf: list, ratios: dict) -> int:
    """
    Compute Piotroski F-score (0-9) from financial statements.

    Uses the latest year vs previous year comparisons.
    Expects lists of dicts from openscreener, ordered oldest → newest.

    Args:
        pl: Profit & Loss rows (list of dicts)
        bs: Balance Sheet rows
        cf: Cash Flow rows
        ratios: Ratios dict from openscreener (contains roce_percent)

    Returns:
        F-score integer (0-9)
    """
    if len(pl) < 2 or len(bs) < 2 or len(cf) < 2:
        return 0

    score = 0
    reasons = []

    # Helper: safely get a numeric field from a dict
    def num(d, key, default=0):
        v = d.get(key, default)
        try:
            return float(v) if v is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    # --- Profitability (4 points) ---

    # 1. Positive net income
    net_income = num(pl[-1], "net_profit")
    if net_income > 0:
        score += 1
        reasons.append("1.NetIncome+")
    else:
        reasons.append("1.NetIncome-")

    # 2. Positive operating cash flow (CFO)
    cfo = num(cf[-1], "operating_cash_flow")
    if cfo > 0:
        score += 1
        reasons.append("2.CFO+")
    else:
        reasons.append("2.CFO-")

    # 3. CFO > Net Income (earnings quality)
    if cfo > net_income:
        score += 1
        reasons.append("3.CFO>NI")
    else:
        reasons.append("3.CFO<NI")

    # 4. ROCE improving (current vs previous year)
    curr_roce = num(ratios, "roce_percent")
    # For previous year ROCE, use the ratios from the P&L if available,
    # otherwise compute a rough proxy: operating_profit / (equity_capital + reserves + borrowings)
    prev_roce = _compute_roce(
        num(pl[-2], "operating_profit"),
        num(bs[-2], "equity_capital") + num(bs[-2], "reserves") + num(bs[-2], "borrowings"),
    )
    if curr_roce > prev_roce:
        score += 1
        reasons.append(f"4.ROCE↑({curr_roce:.1f}>{prev_roce:.1f})")
    else:
        reasons.append(f"4.ROCE↓({curr_roce:.1f}<{prev_roce:.1f})")

    # --- Leverage, Liquidity, Source of Funds (3 points) ---

    # 5. Lower leverage (debt/equity decreasing)
    curr_de = _safe_div(
        num(bs[-1], "borrowings"),
        num(bs[-1], "equity_capital") + num(bs[-1], "reserves"),
    )
    prev_de = _safe_div(
        num(bs[-2], "borrowings"),
        num(bs[-2], "equity_capital") + num(bs[-2], "reserves"),
    )
    if curr_de < prev_de:
        score += 1
        reasons.append(f"5.D/E↓({curr_de:.2f}<{prev_de:.2f})")
    else:
        reasons.append(f"5.D/E↑({curr_de:.2f}>{prev_de:.2f})")

    # 6. Current ratio > 1 (liquidity)
    # openscreener doesn't provide current assets/liabilities separately.
    # Substitute: positive free cash flow as a liquidity proxy
    fcf = num(cf[-1], "free_cash_flow")
    if fcf > 0:
        score += 1
        reasons.append("6.FCF+")
    else:
        reasons.append("6.FCF-")

    # 7. No share dilution (equity capital same or lower)
    curr_equity_cap = num(bs[-1], "equity_capital")
    prev_equity_cap = num(bs[-2], "equity_capital")
    if curr_equity_cap <= prev_equity_cap:
        score += 1
        reasons.append("7.NoDilution")
    else:
        reasons.append("7.Diluted")

    # --- Operating Efficiency (2 points) ---

    # 8. Operating margin (OPM) expanding
    curr_opm = num(pl[-1], "operating_margin_percent")
    prev_opm = num(pl[-2], "operating_margin_percent")
    if curr_opm > prev_opm:
        score += 1
        reasons.append(f"8.OPM↑({curr_opm:.1f}>{prev_opm:.1f})")
    else:
        reasons.append(f"8.OPM↓({curr_opm:.1f}<{prev_opm:.1f})")

    # 9. Higher revenue (sales growth)
    curr_sales = num(pl[-1], "sales")
    prev_sales = num(pl[-2], "sales")
    if curr_sales > prev_sales:
        score += 1
        reasons.append("9.Sales↑")
    else:
        reasons.append("9.Sales↓")

    logger.info(f"    Piotroski F-score: {score}/9 [{', '.join(reasons)}]")
    return min(score, 9)


def _compute_roce(operating_profit: float, total_capital: float) -> float:
    """Compute ROCE = Operating Profit / (Equity + Borrowings)."""
    if total_capital <= 0:
        return 0.0
    return (operating_profit / total_capital) * 100


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division — returns 0 if denominator is 0."""
    return numerator / denominator if denominator else 0.0


def compute_revenue_cagr(pl: list, years: int) -> float:
    """Compute compounded annual growth rate of revenue over N years."""
    if len(pl) < years + 1:
        return 0.0

    latest_sales = _num(pl[-1], "sales")
    past_sales = _num(pl[-(years + 1)], "sales")

    if past_sales <= 0:
        return 0.0

    cagr = ((latest_sales / past_sales) ** (1 / years) - 1) * 100
    return max(cagr, 0.0)


def _num(d: dict, key: str, default=0) -> float:
    """Safely extract a numeric value from a dict."""
    v = d.get(key, default)
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def score_stock(symbol: str, config: dict) -> dict | None:
    """
    Fetch fundamentals for a stock and compute a composite score.

    Applies hard filters first — returns None if the stock fails any filter.

    Args:
        symbol: NSE ticker symbol (e.g. "RELIANCE")
        config: The 'fundamental' section of settings.yaml

    Returns:
        Dict with scores and metrics, or None if filtered out
    """
    try:
        from openscreener import Stock

        stock = Stock(symbol, consolidated=True)
        summary = stock.summary()
        ratios = stock.ratios()
        pl = stock.profit_loss()
        bs = stock.balance_sheet()
        cf = stock.cash_flow()
        sh = stock.shareholding_quarterly()

        if not pl or not bs or not cf:
            logger.debug(f"{symbol}: insufficient financial data, skipping")
            return None

        logger.info(f"  {symbol} ({summary.get('company_name', '?')[:40]}):")

        # --- Print raw metrics for transparency ---
        net_income = _num(pl[-1], "net_profit")
        cfo = _num(cf[-1], "operating_cash_flow")
        fcf = _num(cf[-1], "free_cash_flow")
        sales = _num(pl[-1], "sales")
        opm = _num(pl[-1], "operating_margin_percent")
        borrowings = _num(bs[-1], "borrowings")
        equity = _num(bs[-1], "equity_capital") + _num(bs[-1], "reserves")

        logger.info(f"    Net Profit: ₹{net_income:,.0f} | CFO: ₹{cfo:,.0f} | FCF: ₹{fcf:,.0f}")
        logger.info(f"    Sales: ₹{sales:,.0f} | OPM: {opm:.1f}%")
        logger.info(f"    Borrowings: ₹{borrowings:,.0f} | Equity: ₹{equity:,.0f}")

        # --- Piotroski F-score ---
        f_score = compute_piotroski_fscore(pl, bs, cf, ratios)

        if f_score < config["min_f_score"]:
            logger.info(f"    ✗ FILTERED: F-score {f_score} < {config['min_f_score']}")
            return None

        # --- Quality metrics ---
        roce = _num(ratios, "roce_percent")
        logger.info(f"    ROCE: {roce:.1f}% (min: {config['min_roce']}%)")
        if roce < config["min_roce"]:
            logger.info(f"    ✗ FILTERED: ROCE {roce:.1f}% < {config['min_roce']}%")
            return None

        debt_eq = _safe_div(borrowings, equity)
        logger.info(f"    Debt/Equity: {debt_eq:.2f} (max: {config['max_debt_equity']})")
        if debt_eq > config["max_debt_equity"]:
            logger.info(f"    ✗ FILTERED: D/E {debt_eq:.2f} > {config['max_debt_equity']}")
            return None

        # --- Management quality ---
        latest_sh = sh[-1] if sh else {}
        promoter = _num(latest_sh, "promoters")
        logger.info(f"    Promoter: {promoter:.1f}% (min: {config['min_promoter_holding']}%)")
        if promoter < config["min_promoter_holding"]:
            logger.info(f"    ✗ FILTERED: Promoter {promoter:.1f}% < {config['min_promoter_holding']}%")
            return None

        # Pledged % — openscreener doesn't provide this directly
        # Set to 0 (best case) if not available
        pledged = 0.0
        logger.info(f"    Pledged: {pledged:.1f}% (assumed 0 — not available via openscreener)")

        # --- Revenue growth (3Y CAGR) ---
        revenue_growth = compute_revenue_cagr(pl, 3)
        logger.info(f"    Revenue CAGR (3Y): {revenue_growth:.1f}%")

        # --- Composite score (0-100) ---
        w = config["weights"]
        composite = (
            (f_score / 9) * w["piotroski"]
            + min(roce / 30, 1) * w["roce"]
            + max(1 - debt_eq, 0) * w["debt_equity"]
            + min(revenue_growth / 20, 1) * w["revenue_growth"]
            + min(promoter / 75, 1) * w["promoter_holding"]
            + max(1 - pledged / 10, 0) * w["pledge"]
        )

        market_cap = _num(summary.get("ratios", {}), "market_cap") if isinstance(summary.get("ratios"), dict) else 0

        result = {
            "symbol": symbol,
            "name": summary.get("company_name", symbol),
            "f_score": f_score,
            "roce": round(roce, 2),
            "debt_equity": round(debt_eq, 2),
            "revenue_growth_3y": round(revenue_growth, 2),
            "promoter_holding": round(promoter, 2),
            "pledged_pct": round(pledged, 2),
            "market_cap": market_cap,
            "net_profit": round(net_income, 2),
            "operating_cash_flow": round(cfo, 2),
            "free_cash_flow": round(fcf, 2),
            "sales": round(sales, 2),
            "opm": round(opm, 2),
            "composite_score": round(composite, 2),
            "screened_at": now_iso(),
        }

        logger.info(f"    ✓ PASSED — Composite Score: {composite:.2f}")
        return result

    except Exception as e:
        logger.warning(f"{symbol}: error during screening: {e}")
        return None


def fetch_universe(indices: list[str]) -> list[str]:
    """
    Fetch the full list of stock symbols from NSE indices.

    Args:
        indices: List of index symbols (e.g. ["CNXMIDCAP", "CNX100"])

    Returns:
        Deduplicated list of stock symbols
    """
    all_symbols = []
    for idx_symbol in indices:
        try:
            from openscreener import Index

            index = Index(idx_symbol)
            result = index.constituents()
            companies = result.get("companies", [])
            symbols = [c.get("symbol", "") for c in companies]
            all_symbols.extend(symbols)
            total_pages = int(result.get("total_pages", 1))
            current_page = int(result.get("page", 1))
            logger.info(f"Fetched {len(symbols)} constituents from {idx_symbol} (page {current_page}/{total_pages})")
        except Exception as e:
            logger.error(f"Failed to fetch index {idx_symbol}: {e}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in all_symbols:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    logger.info(f"Universe: {len(unique)} unique stocks from {len(indices)} indices")
    return unique


def run():
    """
    Main entry point for the fundamental screener.

    Fetches the stock universe, scores each stock, saves the top N to basket.json.
    """
    config = load_config()["fundamental"]
    logger.info("=" * 60)
    logger.info("Layer 1: Fundamental Screener — starting")
    logger.info(f"Universe: {config['universe_indices']}")
    logger.info(f"Basket size: {config['basket_size']}")
    logger.info(f"Filters: F-score>={config['min_f_score']}, ROCE>={config['min_roce']}%, "
                f"D/E<={config['max_debt_equity']}, Promoter>={config['min_promoter_holding']}%")
    logger.info("=" * 60)

    # Fetch universe
    symbols = fetch_universe(config["universe_indices"])
    logger.info(f"Screening {len(symbols)} stocks...")

    # Score each stock
    results = []
    passed = 0
    failed = 0
    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Screening {sym}...")
        score = score_stock(sym, config)
        if score:
            results.append(score)
            passed += 1
            logger.info(
                f"  ✓ {sym}: F:{score['f_score']} ROCE:{score['roce']}% "
                f"Score:{score['composite_score']}"
            )
        else:
            failed += 1
        time.sleep(config["request_delay"])

    # Sort by composite score, take top N
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    basket = results[: config["basket_size"]]

    # Save
    save_json(basket, "basket.json")
    logger.info("=" * 60)
    logger.info(f"Screening complete. {passed} passed filters, {failed} filtered out.")
    logger.info(f"Basket: {len(basket)} stocks saved to data/basket.json")
    for s in basket:
        logger.info(
            f"  {s['symbol']:15s} F:{s['f_score']} ROCE:{s['roce']:5.1f}% "
            f"D/E:{s['debt_equity']:.2f} CAGR:{s['revenue_growth_3y']:.1f}% "
            f"Prom:{s['promoter_holding']:.1f}% Score:{s['composite_score']}"
        )
    logger.info("=" * 60)

    return basket


if __name__ == "__main__":
    run()
