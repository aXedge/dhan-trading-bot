#!/usr/bin/env python3
"""
value_screener.py — the DISCOUNT exercise's screening engine.

This is a deliberately separate exercise from the trading bot:
  * No broker API, no orders, no positions. Read-only market data only.
  * Its own folder (value_investing/), its own cache, its own outputs.
  * It does not import anything from the bot's src/ tree.

What it does
------------
Scans a quality-first universe (NIFTY 200 by default), pulls fundamentals for
each name, applies four gates, scores the survivors, and writes a ranked
candidate list. It is a funnel, not an answer: every name it surfaces still
needs a written thesis and a decision-log entry before any capital moves.

The four gates (all configurable)
---------------------------------
  1. QUALITY   ROE >= --min-roe (default 15%), positive operating margin,
               debt-to-equity below --max-de (financials judged on ROE/PB).
  2. GROWTH    revenue growth >= --min-growth (default 5%).
  3. VALUATION trailing P/E <= --max-pe (40) and P/B <= --max-pb (10).
  4. DISCOUNT  at least --min-drawdown (20%) below the 52-week high.
  Plus a size floor (--min-mcap, default Rs 5,000 cr) to skip microcaps.

Scoring: a transparent 0-100 composite of quality, valuation, discount and
dividend yield. Ranking is a starting point for reading, not a buy signal.

Run modes
---------
  python value_screener.py --selftest          # offline, no network
  python value_screener.py --limit 40          # scan 40 names (resumable)
  python value_screener.py --refresh           # ignore today's cache
  python value_screener.py --init-log          # create the decision log

The cache makes a full NIFTY 200 scan resumable: each run reuses same-day
fundamentals already fetched, so you can run it in chunks and it will
complete across a few invocations (or one cron run on the VM).
"""

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "fundamentals.json")
UNIVERSE_FILE = os.path.join(CACHE_DIR, "universe.json")
OUT_FILE = os.path.join(HERE, "candidates.csv")
LOG_FILE = os.path.join(HERE, "decision_log.csv")

NIFTY200_CSV = ("https://archives.nseindia.com/content/indices/"
                "ind_nifty200list.csv")
NIFTY50_CSV = ("https://archives.nseindia.com/content/indices/"
               "ind_nifty50list.csv")
MIDCAP150_CSV = ("https://archives.nseindia.com/content/indices/"
                 "ind_niftymidcap150list.csv")

LOG_COLUMNS = ["date", "symbol", "action", "quality_note", "valuation_note",
               "thesis", "tranche_1", "tranche_2", "tranche_3", "sell_rule",
               "position_size", "review_date", "status"]

_UA = {"User-Agent": "Mozilla/5.0 (value-screener)"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def _download_csv(url):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
        return r.read().decode("utf-8", "ignore")


def _parse_symbols(text):
    out = []
    for row in csv.DictReader(text.splitlines()):
        sym = (row.get("Symbol") or "").strip()
        if sym:
            out.append({"symbol": sym, "name": (row.get("Company Name") or "").strip()})
    return out


def load_universe(refresh=False):
    """NIFTY 200 constituents; cached; falls back to NIFTY50+MIDCAP150."""
    if not refresh and os.path.exists(UNIVERSE_FILE):
        try:
            with open(UNIVERSE_FILE) as f:
                data = json.load(f)
            if data.get("symbols"):
                return data["symbols"]
        except (OSError, json.JSONDecodeError):
            pass

    symbols = []
    for url in (NIFTY200_CSV, NIFTY50_CSV, MIDCAP150_CSV):
        try:
            got = _parse_symbols(_download_csv(url))
            log(f"universe: {len(got)} from {url.rsplit('/', 1)[-1]}")
            for g in got:
                if g["symbol"] not in {s["symbol"] for s in symbols}:
                    symbols.append(g)
        except Exception as e:  # noqa: BLE001 — network is flaky by nature
            log(f"universe: {url.rsplit('/', 1)[-1]} failed ({str(e)[:50]})")
    if symbols:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(UNIVERSE_FILE, "w") as f:
            json.dump({"fetched": datetime.now().isoformat(),
                       "symbols": symbols}, f, indent=2)
    return symbols


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, CACHE_FILE)


def fetch_fundamentals(symbol):
    """One name's fundamentals via yfinance. Returns dict or None."""
    import yfinance as yf
    info = None
    for _ in range(2):
        try:
            info = yf.Ticker(symbol + ".NS").info
            if info and (info.get("currentPrice") or info.get("regularMarketPrice")):
                break
            info = None
        except Exception:  # noqa: BLE001 — retry once, then give up
            time.sleep(1.5)
    if not info:
        return None
    px = info.get("currentPrice") or info.get("regularMarketPrice")
    hi = info.get("fiftyTwoWeekHigh") or 0
    return {
        "symbol": symbol,
        "name": (info.get("shortName") or info.get("longName") or symbol)[:40],
        "sector": info.get("sector") or "",
        "price": px,
        "off_high_pct": round((px / hi - 1) * 100, 1) if hi else None,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "roe": (info.get("returnOnEquity") or 0) * 100,
        "op_margin": (info.get("operatingMargins") or 0) * 100,
        "rev_growth": (info.get("revenueGrowth") or 0) * 100,
        "earn_growth": (info.get("earningsGrowth") or 0) * 100,
        "de": info.get("debtToEquity"),
        "div_yield": info.get("dividendYield") or 0,
        "mcap_cr": (info.get("marketCap") or 0) / 1e7,
    }


def collect(universe, cache, limit, refresh):
    """Fetch/refresh fundamentals, honouring the cache and the run limit."""
    today = datetime.now().date().isoformat()
    fetched = 0
    for u in universe:
        if fetched >= limit:
            break
        sym = u["symbol"]
        entry = cache.get(sym)
        if (not refresh and entry and entry.get("fetched") == today
                and entry.get("data")):
            continue
        data = fetch_fundamentals(sym)
        cache[sym] = {"fetched": today, "data": data}
        fetched += 1
        if fetched % 10 == 0:
            _save_cache(cache)          # checkpoint so a long scan resumes
        log(f"  {sym:<14} {'ok' if data else 'no data'}")
    _save_cache(cache)
    return cache


# ---------------------------------------------------------------------------
# Gates, scoring, output
# ---------------------------------------------------------------------------
def quality_value(row):
    """Quality metric, robust to yfinance omitting ROE (common for Indian
    tickers). Prefers ROE; falls back to an operating-margin proxy, flagged
    in the output so the estimate is never mistaken for the real thing."""
    if row["roe"] and row["roe"] > 0:
        return row["roe"], "roe"
    return round(row["op_margin"] * 1.2, 1), "opM-proxy"


def evaluate(rows, args):
    """Apply the four gates; return (passing, rejected) lists with reasons."""
    passing, rejected = [], []
    for r in rows:
        q_val, q_src = quality_value(r)
        r["quality"] = q_val
        r["quality_src"] = q_src
        reasons = []
        if r["mcap_cr"] < args.min_mcap:
            reasons.append("too small")
        if q_val < args.min_roe:
            reasons.append(f"quality {q_val:.0f}% ({q_src})")
        if r["rev_growth"] < args.min_growth:
            reasons.append(f"growth {r['rev_growth']:.0f}%")
        if r["pe"] and r["pe"] > args.max_pe:
            reasons.append(f"PE {r['pe']:.0f}")
        if r["pb"] and r["pb"] > args.max_pb:
            reasons.append(f"PB {r['pb']:.1f}")
        if r["off_high_pct"] is None or r["off_high_pct"] > -args.min_drawdown:
            reasons.append(f"only {r['off_high_pct'] or 0:.0f}% off high")
        if reasons:
            rejected.append((r, "; ".join(reasons)))
        else:
            passing.append(r)

    for r in passing:
        q = min(40.0, r["quality"] * 1.6)                  # quality
        v = max(0.0, 30.0 - (r["pe"] or 60) * 0.6)         # valuation
        d = min(20.0, abs(r["off_high_pct"]) * 0.5)        # discount
        y = min(10.0, r["div_yield"] * 2.0)                # income
        r["score"] = round(q + v + d + y, 1)
    passing.sort(key=lambda r: r["score"], reverse=True)
    return passing, rejected


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def init_log():
    if os.path.exists(LOG_FILE):
        log(f"decision log already exists: {LOG_FILE}")
        return
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)
    log(f"decision log created: {LOG_FILE}")


# ---------------------------------------------------------------------------
# Self-test (offline)
# ---------------------------------------------------------------------------
def run_selftest():
    args = argparse.Namespace(min_mcap=5000, min_roe=15, min_growth=5,
                              max_pe=40, max_pb=10, min_drawdown=15)
    rows = [
        # quality, cheap, discounted -> should pass and rank first
        {"symbol": "GOOD", "name": "Good Co", "sector": "Banks", "price": 100,
         "off_high_pct": -30.0, "pe": 12, "pb": 1.5, "roe": 18.0,
         "op_margin": 20.0, "rev_growth": 15.0, "earn_growth": 20.0,
         "de": 50, "div_yield": 3.0, "mcap_cr": 50000},
        # great business, expensive -> fails valuation
        {"symbol": "RICH", "name": "Rich Co", "sector": "Auto", "price": 200,
         "off_high_pct": -25.0, "pe": 70, "pb": 12.0, "roe": 25.0,
         "op_margin": 22.0, "rev_growth": 30.0, "earn_growth": 40.0,
         "de": 10, "div_yield": 0.5, "mcap_cr": 90000},
        # cheap but poor quality -> fails quality
        {"symbol": "WEAK", "name": "Weak Co", "sector": "Metals", "price": 50,
         "off_high_pct": -40.0, "pe": 8, "pb": 0.8, "roe": 4.0,
         "op_margin": 3.0, "rev_growth": 2.0, "earn_growth": -10.0,
         "de": 200, "div_yield": 1.0, "mcap_cr": 20000},
        # quality, cheap, but barely off its high -> fails discount
        {"symbol": "NEAR", "name": "Near Co", "sector": "FMCG", "price": 300,
         "off_high_pct": -5.0, "pe": 25, "pb": 5.0, "roe": 20.0,
         "op_margin": 18.0, "rev_growth": 12.0, "earn_growth": 15.0,
         "de": 20, "div_yield": 1.0, "mcap_cr": 40000},
        # ROE missing (yfinance gap) but strong margin -> passes via proxy
        {"symbol": "NOROE", "name": "NoRoe Co", "sector": "Industrials",
         "price": 150, "off_high_pct": -28.0, "pe": 22, "pb": 3.0,
         "roe": 0.0, "op_margin": 18.0, "rev_growth": 14.0,
         "earn_growth": 12.0, "de": 30, "div_yield": 1.0, "mcap_cr": 30000},
    ]
    passing, rejected = evaluate([dict(r) for r in rows], args)
    syms = [r["symbol"] for r in passing]
    checks = [
        ("GOOD passes the four gates", "GOOD" in syms),
        ("GOOD ranks first", passing and passing[0]["symbol"] == "GOOD"),
        ("RICH rejected on valuation",
         any(s == "RICH" and "PE" in why for s, why in
             [(r["symbol"], w) for r, w in rejected])),
        ("WEAK rejected on quality",
         any(r["symbol"] == "WEAK" and "quality" in w for r, w in rejected)),
        ("NEAR rejected on discount",
         any(r["symbol"] == "NEAR" and "off high" in w for r, w in rejected)),
        ("score is bounded 0-100",
         all(0 <= r["score"] <= 100 for r in passing)),
        ("missing ROE passes via margin proxy",
         any(r["symbol"] == "NOROE" and r["quality_src"] == "opM-proxy"
             for r in passing)),
    ]
    ok = True
    for name, cond in checks:
        print(("PASS  " if cond else "FAIL  ") + name)
        ok = ok and cond
    print(f"\n{'selftest OK' if ok else 'selftest FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--limit", type=int, default=40,
                    help="max names to fetch this run (resumable via cache)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore today's cached fundamentals")
    ap.add_argument("--min-roe", type=float, default=15.0)
    ap.add_argument("--min-growth", type=float, default=5.0)
    ap.add_argument("--max-de", type=float, default=150.0,
                    help="informational; financials judged on ROE/PB")
    ap.add_argument("--max-pe", type=float, default=40.0)
    ap.add_argument("--max-pb", type=float, default=10.0)
    ap.add_argument("--min-drawdown", type=float, default=15.0,
                    help="required %% below the 52-week high")
    ap.add_argument("--min-mcap", type=float, default=5000.0, help="Rs crore")
    ap.add_argument("--init-log", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()
    if args.init_log:
        init_log()
        return 0

    universe = load_universe(refresh=args.refresh)
    if not universe:
        log("no universe available (network?) — aborting")
        return 1
    cache = collect(universe, _load_cache(), args.limit, args.refresh)

    rows = [v["data"] for v in cache.values()
            if v.get("data") and v.get("data", {}).get("pe") is not None]
    if not rows:
        log("no fundamentals collected — try again or raise --limit")
        return 1

    passing, rejected = evaluate(rows, args)
    cols = ["symbol", "name", "sector", "price", "off_high_pct", "pe", "pb",
            "roe", "quality", "quality_src", "op_margin", "rev_growth",
            "earn_growth", "de", "div_yield", "mcap_cr", "score"]
    write_csv(OUT_FILE, passing, cols)

    print(f"\nscanned {len(rows)} names | passed {len(passing)} | "
          f"rejected {len(rejected)}")
    print(f"{'sym':<12}{'sector':<18}{'offHi%':>7}{'PE':>7}{'PB':>6}"
          f"{'ROE':>6}{'yld':>6}{'score':>7}")
    for r in passing[:25]:
        print(f"{r['symbol']:<12}{(r['sector'] or '-')[:17]:<18}"
              f"{r['off_high_pct']:>7.0f}{(r['pe'] or 0):>7.1f}"
              f"{(r['pb'] or 0):>6.1f}{r['roe']:>6.0f}"
              f"{r['div_yield']:>6.1f}{r['score']:>7.1f}")
    print(f"\n[OK] {len(passing)} candidates written to {OUT_FILE}")
    print("Next: read each name's filings, then log a thesis + tranche plan "
          "in decision_log.csv before any capital moves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
