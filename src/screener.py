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


def compute_piotroski_fscore(pl: list, bs: list, cf: list) -> int:
    """
    Compute Piotroski F-score (0-9) from financial statements.

    Uses the latest year vs previous year comparisons.
    Expects lists of dicts from openscreener, ordered oldest → newest.

    Args:
        pl: Profit & Loss rows (list of dicts)
        bs: Balance Sheet rows
        cf: Cash Flow rows

    Returns:
        F-score integer (0-9)
    """
    if len(pl) < 2 or len(bs) < 2 or len(cf) < 2:
        return 0

    score = 0

    # --- Profitability (4 points) ---
    # 1. Positive net income
    if float(pl[-1].get("Profit after tax", 0) or pl[-1].get("Net Profit", 0)) > 0:
        score += 1

    # 2. Positive operating cash flow (CFO)
    cfo = float(cf[-1].get("Cash from Operating Activity", 0) or cf[-1].get("Operating Cash Flow", 0))
    if cfo > 0:
        score += 1

    # 3. CFO > Net Income (earnings quality)
    net_income = float(pl[-1].get("Profit after tax", 0) or pl[-1].get("Net Profit", 0))
    if cfo > net_income:
        score += 1

    # 4. ROCE improving (current vs previous year)
    curr_roce = float(pl[-1].get("ROCE", 0) or 0)
    prev_roce = float(pl[-2].get("ROCE", 0) or 0)
    if curr_roce > prev_roce:
        score += 1

    # --- Leverage, Liquidity, Source of Funds (3 points) ---
    # 5. Lower leverage (debt/equity decreasing)
    curr_de = _safe_div(
        float(bs[-1].get("Borrowings", 0) or 0),
        float(bs[-1].get("Total Equity", 0) or bs[-1].get("Shareholders Equity", 0) or 1),
    )
    prev_de = _safe_div(
        float(bs[-2].get("Borrowings", 0) or 0),
        float(bs[-2].get("Total Equity", 0) or bs[-2].get("Shareholders Equity", 0) or 1),
    )
    if curr_de < prev_de:
        score += 1

    # 6. Current ratio > 1 (liquidity)
    curr_assets = float(bs[-1].get("Current Assets", 0) or bs[-1].get("Other Current Assets", 0) or 0)
    curr_liab = float(bs[-1].get("Current Liabilities", 0) or bs[-1].get("Other Current Liabilities", 0) or 0)
    if curr_assets > curr_liab:
        score += 1

    # 7. No share dilution (shares outstanding same or lower)
    curr_shares = float(bs[-1].get("Shares Outstanding", 0) or bs[-1].get("Number of Shares", 0) or 0)
    prev_shares = float(bs[-2].get("Shares Outstanding", 0) or bs[-2].get("Number of Shares", 0) or 0)
    if curr_shares <= prev_shares:
        score += 1

    # --- Operating Efficiency (2 points) ---
    # 8. Operating margin (OPM) expanding
    curr_opm = float(pl[-1].get("OPM", 0) or pl[-1].get("Operating Profit Margin", 0) or 0)
    prev_opm = float(pl[-2].get("OPM", 0) or pl[-2].get("Operating Profit Margin", 0) or 0)
    if curr_opm > prev_opm:
        score += 1

    # 9. Higher revenue (asset turnover improving)
    curr_sales = float(pl[-1].get("Sales", 0) or pl[-1].get("Revenue", 0) or 0)
    prev_sales = float(pl[-2].get("Sales", 0) or pl[-2].get("Revenue", 0) or 0)
    if curr_sales > prev_sales:
        score += 1

    return min(score, 9)


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division — returns 0 if denominator is 0."""
    return numerator / denominator if denominator else 0.0


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

        # --- Piotroski F-score ---
        f_score = compute_piotroski_fscore(pl, bs, cf)
        if f_score < config["min_f_score"]:
            logger.debug(f"{symbol}: F-score {f_score} < {config['min_f_score']}, skipping")
            return None

        # --- Quality metrics ---
        roce = float(ratios.get("roce_percent", 0) or 0)
        if roce < config["min_roce"]:
            logger.debug(f"{symbol}: ROCE {roce:.1f}% < {config['min_roce']}%, skipping")
            return None

        debt_eq = _safe_div(
            float(bs[-1].get("Borrowings", 0) or 0),
            float(bs[-1].get("Total Equity", 0) or bs[-1].get("Shareholders Equity", 0) or 1),
        )
        if debt_eq > config["max_debt_equity"]:
            logger.debug(f"{symbol}: D/E {debt_eq:.2f} > {config['max_debt_equity']}, skipping")
            return None

        # --- Management quality ---
        latest_sh = sh[-1] if sh else {}
        promoter = float(latest_sh.get("Promoters", 0) or 0)
        if promoter < config["min_promoter_holding"]:
            logger.debug(f"{symbol}: Promoter {promoter:.1f}% < {config['min_promoter_holding']}%, skipping")
            return None

        pledged = float(latest_sh.get("Pledged", 0) or 0)
        if pledged > config["max_pledged"]:
            logger.debug(f"{symbol}: Pledged {pledged:.1f}% > {config['max_pledged']}%, skipping")
            return None

        # --- Revenue growth (3Y CAGR) ---
        revenue_growth = _compute_revenue_cagr(pl, 3)

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

        return {
            "symbol": symbol,
            "name": summary.get("company_name", symbol),
            "f_score": f_score,
            "roce": round(roce, 2),
            "debt_equity": round(debt_eq, 2),
            "revenue_growth_3y": round(revenue_growth, 2),
            "promoter_holding": round(promoter, 2),
            "pledged_pct": round(pledged, 2),
            "market_cap": summary.get("ratios", {}).get("market_cap"),
            "composite_score": round(composite, 2),
            "screened_at": now_iso(),
        }

    except Exception as e:
        logger.warning(f"{symbol}: error during screening: {e}")
        return None


def _compute_revenue_cagr(pl: list, years: int) -> float:
    """Compute compounded annual growth rate of revenue over N years."""
    if len(pl) < years + 1:
        return 0.0

    latest_sales = float(pl[-1].get("Sales", 0) or pl[-1].get("Revenue", 0) or 0)
    past_sales = float(pl[-(years + 1)].get("Sales", 0) or pl[-(years + 1)].get("Revenue", 0) or 0)

    if past_sales <= 0:
        return 0.0

    cagr = ((latest_sales / past_sales) ** (1 / years) - 1) * 100
    return max(cagr, 0.0)


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
            constituents = index.constituents()
            symbols = [c.get("symbol", c.get("name", "")) for c in constituents]
            all_symbols.extend(symbols)
            logger.info(f"Fetched {len(symbols)} constituents from {idx_symbol}")
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
    logger.info("=" * 60)

    # Fetch universe
    symbols = fetch_universe(config["universe_indices"])
    logger.info(f"Screening {len(symbols)} stocks...")

    # Score each stock
    results = []
    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Screening {sym}...")
        score = score_stock(sym, config)
        if score:
            results.append(score)
            logger.info(
                f"  ✓ {sym}: F:{score['f_score']} ROCE:{score['roce']}% "
                f"Score:{score['composite_score']}"
            )
        time.sleep(config["request_delay"])

    # Sort by composite score, take top N
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    basket = results[: config["basket_size"]]

    # Save
    save_json(basket, "basket.json")
    logger.info("=" * 60)
    logger.info(f"Screening complete. Basket: {len(basket)} stocks saved to data/basket.json")
    for s in basket:
        logger.info(
            f"  {s['symbol']:15s} F:{s['f_score']} ROCE:{s['roce']:5.1f}% "
            f"D/E:{s['debt_equity']:.2f} Score:{s['composite_score']}"
        )
    logger.info("=" * 60)

    return basket


if __name__ == "__main__":
    run()
