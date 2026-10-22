#!/usr/bin/env python3
"""
profit_study.py — what does a winning day actually pay?
=======================================================
The number that sets a sensible book-profit is simple: on the days that
ENDED in profit, how high did the position actually get, per lot? That peak
is the "normal win". A --per-lot-profit set BELOW it fires on normal winners;
one set above it never fires at all (which is the current state — 4500/lot,
above every peak this book has ever produced).

This reads the guard logs, reconstructs each structure's MTM path, and prints:
  1. the win-day peak distribution, per lot
  2. a table simulating a book-profit lock at various per-lot levels against
     what actually happened

Read-only. No broker API.

Usage:
    python futures_and_options/profit_study.py
    python futures_and_options/profit_study.py --logs logs --min-peak 500
    python futures_and_options/profit_study.py --selftest
"""
import argparse
import glob
import gzip
import os
import re
import sys

MTM = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+(.+?):\s+(\d+)\s+lot\(s\)\s+\|\s+"
                 r"MTM Rs\.(-?[\d,]+)\s+\|\s+peak Rs\.(-?[\d,]+)")
START = re.compile(r"Started at\s+:\s+(\d{4}-\d\d-\d\d)")


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


def _file_date(path):
    m = re.search(r"(\d{8})", os.path.basename(path))
    return m.group(1) if m else "99999999"   # the live file sorts last


def parse_logs(logs_dir):
    """One episode per continuous structure. A structure ends when the book
    goes flat ('No open') or the key changes; a position held overnight stays
    ONE episode, which is what the previous version got wrong."""
    eps = []
    cur = None
    day = None
    for path in sorted(glob.glob(os.path.join(logs_dir, "fno_guard.log*")),
                       key=_file_date):
        for line in _open(path):
            m = START.search(line)
            if m:
                day = m.group(1)
                continue
            if "No open F&O positions" in line:
                if cur:
                    eps.append(cur)
                    cur = None
                continue
            m = MTM.match(line)
            if not m:
                continue
            _t, key, lots, mtm, peak = (m.group(1), m.group(2), int(m.group(3)),
                                        _num(m.group(4)), _num(m.group(5)))
            norm = key.split()[0]          # strip the expiry time, so a
            if cur is None or cur["norm"] != norm:   # carry-over merges
                if cur:
                    eps.append(cur)
                cur = dict(date=day, key=key, norm=norm, lots=lots,
                           peak=mtm, final=mtm, days=[day])
            else:
                cur["peak"] = max(cur["peak"], peak, mtm)
                cur["final"] = mtm
                cur["lots"] = max(cur["lots"], lots)
                if day and day not in cur["days"]:
                    cur["days"].append(day)
    if cur:
        eps.append(cur)
    return eps


def study(episodes, locks):
    out = []
    wins = [e for e in episodes if e["final"] > 0]
    per_lot = sorted(e["peak"] / e["lots"] for e in wins if e["lots"] > 0)

    out.append("=" * 64)
    out.append("PROFIT STUDY — what a winning day actually pays")
    out.append("=" * 64)
    out.append("Episodes: %d   wins: %d   losses: %d"
               % (len(episodes), len(wins), len(episodes) - len(wins)))
    if per_lot:
        n = len(per_lot)
        med = per_lot[n // 2] if n % 2 else (per_lot[n // 2 - 1] + per_lot[n // 2]) / 2
        out.append("")
        out.append("WIN-day peak, per lot (Rs):")
        for p in per_lot:
            out.append("   %8.0f" % p)
        out.append("   -> min %.0f | median %.0f | mean %.0f | max %.0f"
                   % (per_lot[0], med, sum(per_lot) / len(per_lot), per_lot[-1]))

    base = sum(e["final"] for e in episodes)
    out.append("")
    out.append("Baseline (what actually happened): Rs%+.0f" % base)
    out.append("")
    out.append("  lock/lot  fires   total P&L       delta")
    for lock in locks:
        fired = sum(1 for e in episodes if e["peak"] >= lock * e["lots"])
        tot = sum(lock * e["lots"] if e["peak"] >= lock * e["lots"] else e["final"]
                  for e in episodes)
        out.append("  %7d   %2d/%d   Rs%+12.0f   Rs%+.0f"
                   % (lock, fired, len(episodes), tot, tot - base))
    out.append("")
    out.append("Read: pick the lock BELOW the median win so it fires on normal")
    out.append("winners. Above the max peak it can never fire at all.")
    return "\n".join(out)


def selftest():
    import tempfile
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  PASS  " if c else "  FAIL  ") + m)
        ok = ok and c

    with tempfile.TemporaryDirectory() as tmp:
        g = os.path.join(tmp, "logs")
        os.makedirs(g)
        with open(os.path.join(g, "fno_guard.log"), "w") as f:
            f.write("  Started at  : 2026-10-12 10:15:00 IST\n")
            f.write("[10:15:00] NIFTY-OCT|2026-10-19 14:30:00: 2 lot(s) | "
                    "MTM Rs.100 | peak Rs.100 | loss-limit Rs.7,200 "
                    "(profit-limit Rs.9,000) -> ok\n")
            f.write("[11:00:00] NIFTY-OCT|2026-10-19 14:30:00: 2 lot(s) | "
                    "MTM Rs.8,642 | peak Rs.8,642 | loss-limit Rs.7,200 "
                    "(profit-limit Rs.9,000) -> ok\n")
            f.write("[13:00:00] NIFTY-OCT|2026-10-19 14:30:00: 2 lot(s) | "
                    "MTM Rs.8,213 | peak Rs.8,642 | loss-limit Rs.7,200 "
                    "(profit-limit Rs.9,000) -> ok\n")
            f.write("[14:00:00] No open F&O positions.\n")
            f.write("  Started at  : 2026-10-13 10:15:00 IST\n")
            f.write("[10:20:00] NIFTY-OCT|2026-10-19 14:30:00: 2 lot(s) | "
                    "MTM Rs.-5,000 | peak Rs.-100 | loss-limit Rs.7,200 "
                    "(profit-limit Rs.9,000) -> ok\n")
        eps = parse_logs(g)
        chk(len(eps) == 2, "two episodes parsed")
        w = [e for e in eps if e["date"] == "2026-10-12"][0]
        chk(w["peak"] == 8642 and w["final"] == 8213 and w["lots"] == 2,
            "peak/final/lots correct")
        rep = study(eps, [2000, 2500, 4500])
        chk("median 4321" in rep, "win-day median per lot = 4321")
        chk(re.search(r"4500\s+0/2", rep) is not None, "4500/lot fires on zero episodes")
    print()
    print("[SELFTEST] " + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Win-day profit study")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--locks", default="1000,1500,2000,2500,3000,3500,4000,4500")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    eps = parse_logs(a.logs)
    if not eps:
        print("No episodes found under %s (looked for fno_guard.log*)" % a.logs)
        sys.exit(1)
    locks = [int(x) for x in a.locks.split(",")]
    print(study(eps, locks))


if __name__ == "__main__":
    main()
