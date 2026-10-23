#!/usr/bin/env python3
"""
rule_backtest.py — replay the collected guard logs against each candidate rule.

We have weeks of the guard's per-cycle MTM path for every structure. That is
enough to ask the question properly: on the days we actually traded, what would
each profit rule have done? This replays every episode against each rule and
reports the total, and the delta against what actually happened.

Rules
-----
  actual     what happened (last MTM in the episode)
  floor      exit at the first MTM <= MARK, once peak >= MARK      (book-profit)
  trail      exit at the first MTM <= LOCK x peak, once peak >= ARM
  eod        exit at the MTM at/after EOD
  combined   floor + eod, whichever comes first

Read-only. Uses the guard's OWN MTM, which we measured to understate fills by
~36%, so absolute numbers run light; the RELATIVE comparison is the point.

Usage:
    python futures_and_options/rule_backtest.py --mark 2000 --eod 15:20
    python futures_and_options/rule_backtest.py --selftest
"""
import argparse
import glob
import gzip
import os
import re
import sys

MTM = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+(.+?):\s+(\d+)\s+lot\(s\)\s+\|\s+"
                 r"MTM Rs\.(-?[\d,]+)")
START = re.compile(r"Started at\s+:\s+(\d{4}-\d\d-\d\d)")
FLAT = re.compile(r"No open F&O")


def _num(s):
    return float(s.replace(",", ""))


def _open(path):
    if path.endswith(".gz"):
        try:
            f = gzip.open(path, "rt", errors="replace")
            f.read(1)
            f.seek(0)
            return f
        except Exception:
            pass
    return open(path, errors="replace")


def _fdate(p):
    m = re.search(r"(\d{8})", os.path.basename(p))
    return m.group(1) if m else "99999999"


def parse_episodes(logs_dir):
    """One episode per continuous structure, with its full (time, MTM) path."""
    eps, cur, day = [], None, None
    for path in sorted(glob.glob(os.path.join(logs_dir, "fno_guard.log*")),
                       key=_fdate):
        for line in _open(path):
            m = START.search(line)
            if m:
                day = m.group(1)
                continue
            if FLAT.search(line):
                if cur:
                    eps.append(cur)
                    cur = None
                continue
            m = MTM.match(line)
            if not m:
                continue
            t, key, lots, mtm = (m.group(1), m.group(2), int(m.group(3)),
                                 _num(m.group(4)))
            norm = key.split()[0]
            if cur is None or cur["norm"] != norm:
                if cur:
                    eps.append(cur)
                cur = {"date": day, "key": key, "norm": norm, "lots": lots,
                       "path": [(t, mtm)]}
            else:
                cur["path"].append((t, mtm))
                cur["lots"] = max(cur["lots"], lots)
    if cur:
        eps.append(cur)
    return eps


def replay(ep, rule, arm, lock, mark, eod):
    lots = max(ep["lots"], 1)
    m = mark * lots
    a = arm * lots
    peak = -1e18
    for t, mtm in ep["path"]:
        peak = max(peak, mtm)
        if rule in ("floor", "combined") and peak >= m and mtm <= m:
            return mtm, "FLOOR"
        if rule == "trail" and peak >= a and mtm <= lock * peak:
            return mtm, "TRAIL"
        if rule in ("eod", "combined") and eod and t >= eod:
            return mtm, "EOD"
    return ep["path"][-1][1], "HELD"


def run(episodes, arm, lock, mark, eod):
    out = []
    base = sum(replay(e, "actual", arm, lock, mark, eod)[0] for e in episodes)
    out.append("=" * 68)
    out.append(f"RULE BACKTEST — replaying {len(episodes)} episodes "
               f"from the guard logs")
    out.append("=" * 68)
    out.append(f"  arm {arm:.0f}/lot | lock {lock*100:.0f}% | "
               f"mark {mark:.0f}/lot | eod {eod or 'off'}")
    out.append("")
    out.append("  rule       total P&L      delta vs actual   exits")
    for rule in ("actual", "floor", "trail", "eod", "combined"):
        res = [replay(e, rule, arm, lock, mark, eod) for e in episodes]
        tot = sum(r[0] for r in res)
        kinds = {}
        for r in res:
            kinds[r[1]] = kinds.get(r[1], 0) + 1
        ks = " ".join(f"{k}:{v}" for k, v in sorted(kinds.items()))
        out.append(f"  {rule:<9s}  Rs{tot:+,.0f}   Rs{tot - base:+,.0f}   {ks}")
    out.append("")
    out.append("  (MTM-based, and the guard's MTM runs ~36% light; read the")
    out.append("   deltas, not the absolutes. Small sample: treat as a smell,")
    out.append("   not a verdict.)")
    return "\n".join(out)


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  PASS  " if c else "  FAIL  ") + m)
        ok = ok and c

    ep = {"date": "2026-10-01", "key": "K", "norm": "K", "lots": 2,
          "path": [("10:00", 500), ("10:30", 5000), ("11:00", 4500),
                   ("12:00", 3800), ("14:00", 1000), ("15:20", -1000)]}
    chk(replay(ep, "actual", 2000, .5, 2000, "15:20")[0] == -1000,
        "actual = last MTM")
    chk(replay(ep, "floor", 2000, .5, 2000, "15:20")[0] == 3800,
        "floor exits at the mark")
    chk(replay(ep, "trail", 2000, .5, 2000, "15:20")[0] == 1000,
        "trail exits at 50% of peak")
    chk(replay(ep, "eod", 2000, .5, 2000, "15:20")[0] == -1000,
        "eod exits at the cutoff")
    chk(replay(ep, "combined", 2000, .5, 2000, "15:20")[0] == 3800,
        "combined takes the floor first")
    tbl = run([ep], 2000, .5, 2000, "15:20")   # exercise the table formatting
    chk("RULE BACKTEST" in tbl and "combined" in tbl, "run() renders the table")
    print()
    print("[SELFTEST] " + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Rule backtest over guard logs")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--arm", type=float, default=2000)
    ap.add_argument("--lock", type=float, default=0.5)
    ap.add_argument("--mark", type=float, default=2000)
    ap.add_argument("--eod", default="15:20")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    eps = parse_episodes(a.logs)
    if not eps:
        print("No episodes found under %s (looked for fno_guard.log*)" % a.logs)
        sys.exit(1)
    print(run(eps, a.arm, a.lock, a.mark, a.eod))


if __name__ == "__main__":
    main()
