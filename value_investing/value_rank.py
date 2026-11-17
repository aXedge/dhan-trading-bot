#!/usr/bin/env python3
"""
value_rank.py — rank holdings by "is it fairly priced, and is this a sane level?"

Three independent, factual inputs. No forecasting:

  1. VALUATION PERCENTILE — where today's P/E and P/B sit inside the stock's
     own 5-year history. "Cheaper than 91% of its own history" is a statement
     about a distribution, and it is auditable. This is the confidence you can
     actually have.
  2. LEVEL POSITION — distance to the nearest >=N-level confluence cluster and
     where price sits versus its volume value area (from holdings_levels).
  3. FUNDAMENTAL DIRECTION — 3-year revenue and earnings CAGR, plus the latest
     year's change. Is the business getting better or worse?

The composite is deliberately simple and printed alongside its parts so you can
disagree with the weighting. A percentile is NOT a probability of future return:
the 2-year backtest showed these signals have no predictive edge. What a good
entry buys you is a defined downside, not a better expected return.

Usage
  python value_rank.py
  python value_rank.py --symbols TITAN,LT --min-levels 3
  python value_rank.py --selftest
"""

__SCRIPT_NAME = "value_rank"
__SCRIPT_VERSION = "v1"
__SCRIPT_BUILD = "2026-11-17"
__SCRIPT_NOTES = "initial release"

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import holdings_levels as hl                       # noqa: E402

REPORT_COLUMNS = [
    "symbol", "price", "pe", "pe_pct", "pb", "pb_pct",
    "pe_basis", "eps_asof", "q_ni_yoy", "q_asof",
    "price_pct", "rev_cagr3", "eps_cagr3", "ni_last",
    "level", "level_dist_atr", "level_n", "zone_pos",
    "valuation", "entry", "fundamental", "score", "verdict",
]


def pct_rank(series, value):
    """Percentile of `value` inside `series` (0 = cheapest, 100 = dearest)."""
    s = pd.Series(series).dropna()
    if len(s) < 30 or value is None or not np.isfinite(value):
        return None
    return float((s < value).mean() * 100)


def _row(df, *names):
    """First matching row from a yfinance statement frame."""
    if df is None or df.empty:
        return None
    for n in names:
        for idx in df.index:
            if str(idx).strip().lower() == n.lower():
                return df.loc[idx]
    for n in names:
        for idx in df.index:
            if n.lower() in str(idx).lower():
                return df.loc[idx]
    return None


def latest_ttm(sym):
    """Trailing-twelve-month EPS and book value per share, with their as-of date.

    Fixes the defect that sank two recommendations: the CURRENT multiple was
    computed from annual earnings up to seven months old. yfinance carries only
    ~5 quarters, so the comparison history stays annual-stepped, but the point
    being valued is now current.
    """
    import yfinance as yf
    out = {"eps": None, "eps_asof": "", "bvps": None, "bvps_asof": "",
           "basis": "annual (stale)", "q_ni_yoy": None, "q_asof": ""}
    try:
        t = yf.Ticker(sym + ".NS")
        q = t.quarterly_income_stmt
    except Exception:
        return out

    eps_row = _row(q, "Basic EPS", "Diluted EPS")
    if eps_row is not None:
        vals = [float(v) for v in eps_row.values if v == v]
        if len(vals) >= 4:
            out["eps"] = sum(vals[:4])
            out["eps_asof"] = str(eps_row.index[0])[:10]
            out["basis"] = "TTM"

    niq = _row(q, "Net Income", "Net Income Common Stockholders")
    if out["eps"] is None and niq is not None:
        try:
            sh = _row(yf.Ticker(sym + ".NS").quarterly_balance_sheet,
                      "Ordinary Shares Number", "Share Issued")
        except Exception:
            sh = None
        if sh is not None:
            nv = [float(v) for v in niq.values if v == v]
            sv = [float(v) for v in sh.values if v == v]
            if len(nv) >= 4 and sv and sv[0]:
                out["eps"] = sum(nv[:4]) / sv[0]
                out["eps_asof"] = str(niq.index[0])[:10]
                out["basis"] = "TTM(ni/shares)"

    # latest quarter year-on-year: the annual figure can be stale and can carry
    # one-offs, so the most recent quarter gates the fundamental label
    if niq is not None:
        nv = [float(v) for v in niq.values if v == v]
        if len(nv) >= 5 and nv[4] != 0:
            out["q_ni_yoy"] = (nv[0] / nv[4] - 1) * 100
            out["q_asof"] = str(niq.index[0])[:10]

    try:
        bq = yf.Ticker(sym + ".NS").quarterly_balance_sheet
        eq = _row(bq, "Stockholders Equity", "Total Equity Gross Minority Interest")
        sh = _row(bq, "Ordinary Shares Number", "Share Issued")
        if eq is not None and sh is not None:
            ev = [float(v) for v in eq.values if v == v]
            sv = [float(v) for v in sh.values if v == v]
            if ev and sv and sv[0]:
                out["bvps"] = ev[0] / sv[0]
                out["bvps_asof"] = str(eq.index[0])[:10]
    except Exception:
        pass
    return out


def valuation_series(sym, years="5y"):
    """Reconstruct a daily P/E and P/B history from annual statements + prices."""
    import yfinance as yf
    t = yf.Ticker(sym + ".NS")
    px = t.history(period=years, auto_adjust=False)["Close"].dropna()
    if len(px) < 200:
        return None
    # yfinance returns a tz-aware index; statement dates are naive
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    try:
        inc = t.income_stmt
        bs = t.balance_sheet
    except Exception:
        return None

    eps = _row(inc, "Basic EPS", "Diluted EPS", "Basic Average Shares")
    if eps is None:
        ni = _row(inc, "Net Income", "Net Income Common Stockholders")
        sh = _row(bs, "Ordinary Shares Number", "Share Issued")
        eps = (ni / sh) if (ni is not None and sh is not None) else None
    equity = _row(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")
    shares = _row(bs, "Ordinary Shares Number", "Share Issued")
    bvps = (equity / shares) if (equity is not None and shares is not None) else None

    def stepped(row, lag_days=75):
        if row is None:
            return None
        vals = []
        for period_end, v in row.items():
            try:
                when = pd.Timestamp(period_end).tz_localize(None) \
                    if pd.Timestamp(period_end).tz is not None \
                    else pd.Timestamp(period_end)
                when = when + pd.Timedelta(days=lag_days)
                vals.append((when, float(v)))
            except (TypeError, ValueError):
                continue
        if not vals:
            return None
        vals.sort()
        out = []
        for day in px.index:
            cur = None
            for when, v in vals:
                if when <= day:
                    cur = v
                else:
                    break
            out.append(cur)
        return pd.Series(out, index=px.index)

    pe = pb = None
    e = stepped(eps)
    if e is not None:
        pe = (px / e.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    b = stepped(bvps)
    if b is not None:
        pb = (px / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return {"price": px, "pe": pe, "pb": pb}


def fundamentals(sym):
    """3-year revenue and EPS CAGR plus the latest year's net-income change."""
    import yfinance as yf
    try:
        inc = yf.Ticker(sym + ".NS").income_stmt
    except Exception:
        return {}
    rev = _row(inc, "Total Revenue", "Operating Revenue")
    eps = _row(inc, "Basic EPS", "Diluted EPS")
    ni = _row(inc, "Net Income", "Net Income Common Stockholders")

    def cagr(row, n=3):
        if row is None:
            return None
        v = [float(x) for x in row.values if x == x]
        if len(v) < n + 1:
            return None
        start, end = v[n], v[0]          # columns are newest-first
        if start <= 0 or end <= 0:
            return None
        return ((end / start) ** (1.0 / n) - 1) * 100

    last = None
    if ni is not None:
        v = [float(x) for x in ni.values if x == x]
        if len(v) >= 2 and v[1] != 0:
            last = (v[0] / v[1] - 1) * 100
    return {"rev_cagr3": cagr(rev), "eps_cagr3": cagr(eps), "ni_last": last}


def band(pct):
    if pct is None:
        return "n/a"
    if pct <= 25:
        return "CHEAP"
    if pct <= 60:
        return "FAIR"
    return "RICH"


def score_row(r):
    """Simple additive score, -3..+3. Parts are printed so it can be argued with."""
    s = 0.0
    pe, pb = r.get("pe_pct"), r.get("pb_pct")
    vals = [v for v in (pe, pb) if v is not None]
    if vals:
        avg = sum(vals) / len(vals)
        s += 1.0 if avg <= 30 else (0.0 if avg <= 60 else -1.0)
    d = r.get("level_dist_atr")
    if d is not None:
        s += 1.0 if d <= 0.5 else (0.5 if d <= 1.0 else 0.0)
    g = r.get("rev_cagr3")
    if g is not None:
        s += 1.0 if g >= 10 else (0.0 if g >= 0 else -1.0)
    n = r.get("ni_last")
    if n is not None:
        s += 0.5 if n >= 0 else -0.5
    # absolute sanity: a percentile against a bubble history is not cheapness
    pe_abs, pb_abs = r.get("pe"), r.get("pb")
    if isinstance(pe_abs, (int, float)) and pe_abs > 50:
        s -= 1.5
    if isinstance(pb_abs, (int, float)) and pb_abs > 10:
        s -= 1.0
    return round(s, 1)


def _absolute_ok(r):
    """Cheap relative to itself is only meaningful at a sane absolute multiple."""
    pe, pb = r.get("pe"), r.get("pb")
    if isinstance(pe, (int, float)) and pe > 30:
        return False
    if isinstance(pb, (int, float)) and pb > 5:
        return False
    return True


def verdict(r):
    pe, pb = r.get("pe"), r.get("pb")
    if (r["valuation"] == "CHEAP" and not _absolute_ok(r)
            and isinstance(pe, (int, float)) and pe > 50):
        return "DE-RATING"          # multiple collapsed from a bubble, not cheap
    if (r["valuation"] == "CHEAP" and _absolute_ok(r)
            and r["fundamental"] in ("IMPROVING", "STABLE")
            and r["entry"] in ("AT-LEVEL", "NEAR-LEVEL")):
        return "VALUE PICK"
    if r["valuation"] == "RICH" and r["fundamental"] == "DETERIORATING":
        return "AVOID"
    if r["valuation"] == "CHEAP" and r["fundamental"] == "DETERIORATING":
        return "VALUE TRAP?"
    return "WATCH"


def analyse(sym, min_levels=3):
    import yfinance as yf
    v = valuation_series(sym)
    if v is None:
        return None
    px = v["price"]
    price = float(px.iloc[-1])
    df = yf.Ticker(sym + ".NS").history(period="2y", auto_adjust=False).dropna(subset=["Close"])
    if len(df) < 260:
        return None
    r = hl.analyse_one(sym, df, 0.25) or {}
    if "error" in r:
        return None

    f = fundamentals(sym)
    ttm = latest_ttm(sym)
    pe_now = pb_now = None
    basis = "annual (stale)"
    if ttm["eps"] and ttm["eps"] > 0:
        pe_now = price / ttm["eps"]
        basis = ttm["basis"]
    elif v["pe"] is not None and v["pe"].dropna().size:
        pe_now = float(v["pe"].dropna().iloc[-1])
    if ttm["bvps"] and ttm["bvps"] > 0:
        pb_now = price / ttm["bvps"]
        basis = basis if basis != "annual (stale)" else "quarterly BV"
    elif v["pb"] is not None and v["pb"].dropna().size:
        pb_now = float(v["pb"].dropna().iloc[-1])

    sup_d = r.get("sup_dist_atr") or 99
    res_d = r.get("res_dist_atr") or 99
    best = min(sup_d, res_d)
    best_n = r.get("sup_levels") if sup_d <= res_d else r.get("res_levels")

    row = {
        "symbol": sym, "price": round(price, 2),
        "pe": round(pe_now, 1) if pe_now else "",
        "pe_pct": round(pct_rank(v["pe"], pe_now) or 0, 0) if pe_now else "",
        "pb": round(pb_now, 2) if pb_now else "",
        "pb_pct": round(pct_rank(v["pb"], pb_now) or 0, 0) if pb_now else "",
        "price_pct": round(pct_rank(px, price) or 0, 0),
        "rev_cagr3": round(f["rev_cagr3"], 1) if f.get("rev_cagr3") is not None else "",
        "eps_cagr3": round(f["eps_cagr3"], 1) if f.get("eps_cagr3") is not None else "",
        "ni_last": round(f["ni_last"], 1) if f.get("ni_last") is not None else "",
        "level": str(r.get("verdict", "")).split(" (")[0],
        "level_dist_atr": round(best, 2) if best < 99 else "",
        "level_n": best_n or 0,
        "zone_pos": r.get("zone_pos", ""),
        "pe_basis": basis,
        "eps_asof": ttm["eps_asof"],
        "q_ni_yoy": round(ttm["q_ni_yoy"], 1) if ttm["q_ni_yoy"] is not None else "",
        "q_asof": ttm["q_asof"],
    }
    pcts = [p for p in (row["pe_pct"], row["pb_pct"]) if p != ""]
    row["valuation"] = band(sum(pcts) / len(pcts)) if pcts else "n/a"
    if row["level_dist_atr"] == "":
        row["entry"] = "n/a"
    elif row["level_dist_atr"] <= 0.5 and row["level_n"] >= min_levels:
        row["entry"] = "AT-LEVEL"
    elif row["level_dist_atr"] <= 1.0:
        row["entry"] = "NEAR-LEVEL"
    else:
        row["entry"] = "MID-RANGE"

    g, n, q = row["rev_cagr3"], row["ni_last"], row["q_ni_yoy"]
    if g == "" or n == "":
        row["fundamental"] = "n/a"
    elif q != "" and q < 0:
        # the latest reported quarter is shrinking; the annual figure is history
        row["fundamental"] = "DETERIORATING"
    elif g >= 8 and n >= 0 and (q == "" or q >= 0):
        row["fundamental"] = "IMPROVING"
    elif g >= 0 and n >= -10:
        row["fundamental"] = "STABLE"
    else:
        row["fundamental"] = "DETERIORATING"

    row["score"] = score_row(row)
    row["verdict"] = verdict(row)
    return row


def selftest():
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            print(f"PASS  {name}")
            ok += 1
        else:
            print(f"FAIL  {name}")
            fail += 1

    s = list(range(1, 101))
    check("pct_rank: the minimum is the 0th percentile", pct_rank(s, 1) == 0.0)
    check("pct_rank: the maximum is the 99th percentile", pct_rank(s, 100) == 99.0)
    check("pct_rank: the median is ~50", 49 <= pct_rank(s, 51) <= 51)
    check("pct_rank: too few samples returns None", pct_rank([1, 2, 3], 2) is None)
    check("pct_rank: NaN value returns None", pct_rank(s, float("nan")) is None)

    check("band: bottom quartile is CHEAP", band(10) == "CHEAP")
    check("band: middle is FAIR", band(45) == "FAIR")
    check("band: top is RICH", band(90) == "RICH")
    check("band: None is n/a", band(None) == "n/a")

    check("verdict: cheap + improving + at level = VALUE PICK",
          verdict({"valuation": "CHEAP", "fundamental": "IMPROVING",
                   "entry": "AT-LEVEL"}) == "VALUE PICK")
    check("verdict: rich + deteriorating = AVOID",
          verdict({"valuation": "RICH", "fundamental": "DETERIORATING",
                   "entry": "MID-RANGE"}) == "AVOID")
    check("verdict: cheap percentile but 106x earnings is NOT a value pick",
          verdict({"valuation": "CHEAP", "fundamental": "IMPROVING",
                   "entry": "AT-LEVEL", "pe": 106, "pb": 30}) == "DE-RATING")
    check("verdict: 60x earnings is still not a value pick",
          verdict({"valuation": "CHEAP", "fundamental": "IMPROVING",
                   "entry": "AT-LEVEL", "pe": 60, "pb": 4}) != "VALUE PICK")
    check("verdict: a sane multiple still qualifies",
          verdict({"valuation": "CHEAP", "fundamental": "IMPROVING",
                   "entry": "AT-LEVEL", "pe": 19, "pb": 5}) == "VALUE PICK")
    check("verdict: cheap + deteriorating is flagged as a trap",
          verdict({"valuation": "CHEAP", "fundamental": "DETERIORATING",
                   "entry": "AT-LEVEL"}) == "VALUE TRAP?")

    row = {"pe_pct": 20, "pb_pct": 30, "level_dist_atr": 0.3,
           "rev_cagr3": 12, "ni_last": 5}
    check("score: cheap + at level + growing is positive", score_row(row) >= 2.5)
    row2 = {"pe_pct": 90, "pb_pct": 85, "level_dist_atr": 4,
            "rev_cagr3": -5, "ni_last": -20}
    check("score: rich + far + shrinking is negative", score_row(row2) <= -2)

    print(f"\n{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version",
                       version=f"{__SCRIPT_NAME} {__SCRIPT_VERSION} "
                               f"(build {__SCRIPT_BUILD}) — {__SCRIPT_NOTES}")
    ap.add_argument("--holdings", default=os.path.join(HERE, "holdings.csv"))
    ap.add_argument("--symbols", default="")
    ap.add_argument("--min-levels", type=int, default=3)
    ap.add_argument("--report", default=os.path.join(HERE, "value_rank.csv"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        if not os.path.exists(args.holdings):
            print(f"no holdings file at {args.holdings}")
            return 2
        syms = [h["symbol"] for h in hl.load_holdings(args.holdings)]

    rows = []
    for s in syms:
        try:
            r = analyse(s, args.min_levels)
        except Exception as e:
            print(f"  {s:12s} failed: {str(e)[:60]}")
            continue
        if not r:
            print(f"  {s:12s} skipped")
            continue
        rows.append(r)
        print(f"  {s:12s} {r['valuation']:6s} {r['entry']:11s} "
              f"{r['fundamental']:14s} score {r['score']:+.1f}  {r['verdict']}")
        sys.stdout.flush()

    if not rows:
        print("nothing analysed")
        return 1

    rows.sort(key=lambda r: -r["score"])
    print()
    print(f"{'sym':12s} {'price':>9s} {'P/E':>6s} {'PE%':>4s} {'P/B':>6s} {'PB%':>4s} "
          f"{'RevCAGR':>8s} {'NI':>7s} {'level':12s} {'d/ATR':>6s} {'zone':7s} "
          f"{'val':6s} {'fund':14s} {'score':>6s} {'verdict':14s}")
    print("-" * 150)
    for r in rows:
        print(f"{r['symbol']:12s} {r['price']:>9,.1f} {str(r['pe']):>6s} {str(r['pe_pct']):>4s} "
              f"{str(r['pb']):>6s} {str(r['pb_pct']):>4s} {str(r['rev_cagr3']):>8s} "
              f"{str(r['ni_last']):>7s} {r['level']:12s} {str(r['level_dist_atr']):>6s} "
              f"{r['zone_pos']:7s} {r['valuation']:6s} {r['fundamental']:14s} "
              f"{r['score']:>+6.1f} {r['verdict']:14s}")

    with open(args.report, "w", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print("PE%/PB% = percentile inside the stock's own 5-year history (0 = cheapest).")
    print("A percentile describes the past distribution; it is not a probability of return.")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
