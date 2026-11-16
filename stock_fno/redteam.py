#!/usr/bin/env python3
"""
redteam.py — the adversary. Run this BEFORE quoting a number or a conclusion.

Every check below exists because I actually failed it:

  1 AS-OF         Quoted a price off a 4-day-old bar and called it current.
                  -> Bank of Maharashtra, recommended as "a pullback at 79.6"
                     when the market was already at 86.9.
  2 INTEGRITY     An incomplete last bar poisoned an RSI.
                  -> L&T read RSI 90.4 instead of 85.9.
  3 SAMPLE        Concluded from a handful of episodes.
  4 OVERLAP       Reported z-scores from overlapping forward returns.
                  -> a z of -5.45 that became -2.1 when adjusted, and vanished
                     entirely over a proper window.
  5 CONTROL       Measured a rising sample and called it skill.
  6 ABSOLUTE      Called a stock "cheap" on a percentile while it traded at
                  106x earnings and 30x book.
  7 SELECTION     Backtested on 17 hand-picked winners.
  8 WINDOW        Claimed "2 years" on a 10-month test window.
  9 STATE/DIR     Answered "where is it" when the question was "where will it
                  go" — or the reverse.
 10 SPECIFICITY   Gave a number more precise than its basis ("80% confidence").

Usage
  python redteam.py --protocol
  python redteam.py --ticker TITAN
  python redteam.py --holdings                      # freshness for every holding
  python redteam.py --zscore 7.56 --n 799 --stride 3 --horizon 20
  python redteam.py --valuation 106 17 30 20        # pe, pe_pct, pb, pb_pct
  python redteam.py --sample 12
  python redteam.py --selftest
"""

import argparse
import datetime as dt
import os
import sys

PROTOCOL = """
THE TEN CHECKS  (any FAIL blocks the claim; say so rather than shipping it)

 1 AS-OF        Does every price/number carry its as-of date, and is the feed
                behind the market? A price without a date is not a price.
 2 INTEGRITY    Were NaN / partial bars dropped? Does the last bar look like a
                complete session (volume vs its own average)?
 3 SAMPLE       Is n large enough, and are the observations independent?
                Under ~30: an anecdote. Under ~100: a hint.
 4 OVERLAP      Do forward returns overlap? Then divide z by sqrt(h/stride)
                before calling anything significant.
 5 CONTROL      Is there a comparison group, or is the "signal" just the
                sample's own drift? Test on a universe you did not choose.
 6 ABSOLUTE     Does a relative claim hide an absurd absolute? "Cheap vs its
                own history" is meaningless at 106x earnings.
 7 SELECTION    Was the sample picked in a way that guarantees the answer?
                Winners chosen after the fact will always show mean reversion.
 8 WINDOW       Does the tested window match the claimed window? Warmups eat
                the first year of a "2-year" test.
 9 STATE/DIR    Am I answering "where is it" when asked "where will it go"?
                Levels describe position; they do not forecast direction.
10 SPECIFICITY  Is the number more precise than its basis? "80% confidence"
                needs a distribution behind it, or it is decoration.
"""


def _weekdays_between(a, b):
    n, d = 0, a
    while d < b:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def check_ticker(symbol, period="1y"):
    """Checks 1 and 2: is the data current, and is the last bar complete?"""
    import yfinance as yf
    t = yf.Ticker(symbol + ".NS")
    h = t.history(period=period, auto_adjust=False)
    if h is None or h.empty:
        return {"symbol": symbol, "error": "no data"}

    raw = h["Close"]
    clean = raw.dropna()
    last_date = clean.index[-1]
    if getattr(last_date, "tz", None) is not None:
        last_date = last_date.tz_localize(None)
    today = dt.date.today()
    behind = _weekdays_between(last_date.date(), today)

    nan_bars = int(raw.isna().sum())
    vol = h["Volume"].dropna()
    last_vol = float(vol.iloc[-1]) if len(vol) else 0.0
    avg_vol = float(vol.tail(21).head(20).mean()) if len(vol) > 21 else 0.0
    partial = bool(avg_vol and last_vol < 0.35 * avg_vol)

    price = float(clean.iloc[-1])
    return {
        "symbol": symbol,
        "as_of": str(last_date.date()),
        "price": price,
        "sessions_behind": behind,
        "nan_bars": nan_bars,
        "last_volume_ratio": round(last_vol / avg_vol, 2) if avg_vol else None,
        "partial_bar": partial,
    }


def report_ticker(r):
    if r.get("error"):
        print(f"  {r['symbol']:12s} ERROR {r['error']}")
        return False
    ok = True
    flags = []
    if r["sessions_behind"] >= 1:
        flags.append(f"AS-OF: {r['sessions_behind']} session(s) behind today")
        ok = False
    if r["nan_bars"]:
        flags.append(f"INTEGRITY: {r['nan_bars']} NaN bar(s) in the series")
        ok = False
    if r["partial_bar"]:
        flags.append(f"INTEGRITY: last bar volume {r['last_volume_ratio']}x average "
                     "-> looks like a partial session")
        ok = False
    tag = "OK  " if ok else "FAIL"
    print(f"  {tag} {r['symbol']:12s} {r['price']:>10,.2f}  as of {r['as_of']}"
          f"  (behind {r['sessions_behind']})")
    for f in flags:
        print(f"       -> {f}")
    return ok


def adjust_z(z, n, stride, horizon):
    """Check 4: overlapping windows shrink the effective sample."""
    import math
    factor = math.sqrt(horizon / stride) if stride > 0 else 1.0
    eff_n = n * stride / horizon
    return z / factor, eff_n


def check_valuation(pe, pe_pct, pb, pb_pct):
    """Check 6: relative cheapness vs absolute sanity."""
    flags = []
    if pe is not None and pe > 50:
        flags.append(f"ABSOLUTE: P/E {pe:.1f} is not cheap at any percentile")
    elif pe is not None and pe > 30:
        flags.append(f"ABSOLUTE: P/E {pe:.1f} is above a sane value band")
    if pb is not None and pb > 10:
        flags.append(f"ABSOLUTE: P/B {pb:.1f} is not cheap at any percentile")
    pcts = [p for p in (pe_pct, pb_pct) if p is not None]
    if pcts and sum(pcts) / len(pcts) <= 30 and flags:
        flags.append("CONTRADICTION: cheap on its own history, expensive in "
                     "absolute terms -> de-rating, not value")
    return flags


def check_fund_age(asof, today=None):
    """Check 11: how old is the fundamental data behind a valuation claim?

    redteam originally checked the as-of date of PRICES only. That is why it
    would not have caught the value_rank staleness: the multiple was computed
    from annual earnings seven months old while the price was current.
    """
    import datetime as _dt
    today = today or _dt.date.today()
    try:
        d = _dt.date.fromisoformat(str(asof)[:10])
    except (ValueError, TypeError):
        return "FUND-ASOF: unparseable date - treat every figure as unverified"
    months = (today - d).days / 30.44
    if months > 6:
        return (f"FUND-ASOF: {months:.1f} months old - two or more quarters are "
                "missing, so this is almost certainly an annual figure")
    if months > 4:
        return f"FUND-ASOF: {months:.1f} months old - a quarterly report is probably missing"
    return ""


def check_sample(n, independent=False):
    """Check 3: is there enough evidence to say anything?"""
    if n < 30:
        return f"SAMPLE: n={n} is an anecdote, not evidence"
    if n < 100:
        return f"SAMPLE: n={n} is a hint at best"
    if not independent:
        return f"SAMPLE: n={n} but observations may not be independent"
    return ""


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

    # weekdays_between: Mon -> next Mon is 5 weekdays
    a = dt.date(2026, 11, 2)      # Monday
    b = dt.date(2026, 11, 9)      # next Monday
    check("weekday count across a week is 5", _weekdays_between(a, b) == 5)
    check("same day is zero", _weekdays_between(a, a) == 0)

    z, n = adjust_z(7.56, 799, 3, 20)
    check("overlap adjustment shrinks the z", abs(z - 7.56 / (20 / 3) ** 0.5) < 1e-6)
    check("effective n is n*stride/horizon", abs(n - 799 * 3 / 20) < 1e-6)
    check("a stride equal to the horizon does not shrink",
          abs(adjust_z(2.0, 100, 20, 20)[0] - 2.0) < 1e-9)

    check("P/E 106 is flagged absolutely",
          any("ABSOLUTE" in f for f in check_valuation(106, 17, 30, 20)))
    check("P/E 19 with no contradiction is clean",
          check_valuation(19, 10, 5, 46) == [])
    check("cheap percentile + expensive absolute is called a de-rating",
          any("CONTRADICTION" in f for f in check_valuation(106, 17, 30, 20)))

    check("n=12 is an anecdote", "anecdote" in check_sample(12))
    check("n=60 is a hint", "hint" in check_sample(60))
    check("n=500 still warns on independence", "independent" in check_sample(500))
    check("n=500 independent passes", check_sample(500, True) == "")
    import datetime as _d
    check("annual earnings are flagged as stale",
          "annual" in check_fund_age("2026-03-31", _d.date(2026, 11, 16)))
    check("a fresh quarter passes",
          check_fund_age("2026-09-30", _d.date(2026, 11, 16)) == "")

    print(f"\n{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", action="store_true")
    ap.add_argument("--ticker", default="")
    ap.add_argument("--holdings", default="")
    ap.add_argument("--zscore", type=float)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--valuation", nargs=4, type=float,
                    metavar=("PE", "PE_PCT", "PB", "PB_PCT"))
    ap.add_argument("--sample", type=int)
    ap.add_argument("--fund-age", default="", help="YYYY-MM-DD of the earnings used")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.protocol:
        print(PROTOCOL)
        return 0

    if args.ticker:
        return 0 if report_ticker(check_ticker(args.ticker)) else 1

    if args.holdings:
        import holdings_levels as hl
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        rows = hl.load_holdings(args.holdings)
        bad = 0
        for h in rows:
            r = check_ticker(h["symbol"])
            if not report_ticker(r):
                bad += 1
        print(f"\n{bad} of {len(rows)} names failed the freshness/integrity check")
        return 0 if bad == 0 else 1

    if args.zscore is not None:
        if args.n <= 0:
            print("need --n")
            return 2
        z, eff = adjust_z(args.zscore, args.n, args.stride, args.horizon)
        print(f"reported z {args.zscore:+.2f} with n={args.n}, stride {args.stride}, "
              f"horizon {args.horizon}")
        print(f"effective independent n  {eff:,.0f}")
        print(f"adjusted z               {z:+.2f}")
        if abs(z) < 2.0:
            print("VERDICT: not significant after adjustment - do not report it as a finding")
        return 0

    if args.valuation:
        pe, pe_pct, pb, pb_pct = args.valuation
        flags = check_valuation(pe, pe_pct, pb, pb_pct)
        print(f"P/E {pe} ({pe_pct:.0f}th pct)  P/B {pb} ({pb_pct:.0f}th pct)")
        if flags:
            for f in flags:
                print(f"  -> {f}")
        else:
            print("  -> no absolute contradiction")
        return 0

    if args.fund_age:
        msg = check_fund_age(args.fund_age)
        print(msg or "fundamentals are current (under 4 months old)")
        return 0

    if args.sample is not None:
        msg = check_sample(args.sample)
        print(msg or "sample size adequate")
        return 0

    print("nothing to do - try --protocol")
    return 2


if __name__ == "__main__":
    sys.exit(main())
