#!/usr/bin/env python3
"""
Weekly Digest — the week in one page
====================================
Aggregates the three tracks into a single weekly roll-up, meant for cron on
Saturday (a slot nothing else uses now that screening is monthly).

Read-only: it places no orders and touches no broker API.

Sections
--------
  1. Paper book  — trades CLOSED this week (win rate, PF, realized) and the
                   current open positions vs their SL/target.
  2. Signals     — the latest scan, plus per-day signal counts for the week.
  3. F&O guard   — this week's round-trips and their trail-audit verdicts,
                   plus the alerts that fired (WARN50/WARN75/flattenings).
  4. Health      — TOTP failures and token-cache hits seen in the logs.

Usage
-----
    python src/live/weekly_digest.py                 # last 7 days, live prices
    python src/live/weekly_digest.py --days 14       # a fortnight
    python src/live/weekly_digest.py --no-live       # skip price fetches
    python src/live/weekly_digest.py --selftest      # offline, synthetic data
    python src/live/weekly_digest.py --out logs/weekly_digest.log
"""

import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

DEFAULT_DATA = os.path.join(REPO_ROOT, "data")
DEFAULT_LOGS = os.path.join(REPO_ROOT, "logs")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _parse_day(s):
    if not s:
        return None
    s = str(s)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _open_maybe_gz(path):
    """Open a possibly-gzipped log. Tolerates a .gz name on plain text."""
    if path.endswith(".gz"):
        try:
            f = gzip.open(path, "rt", errors="replace")
            f.read(1)      # probe: raises BadGzipFile if it is not really gzip
            f.seek(0)
            return f
        except Exception:
            pass
    return open(path, errors="replace")


def _log_date(path, today):
    """Dated rotated logs carry YYYYMMDD in the name; the live file is 'today'."""
    m = re.search(r"(\d{8})", os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except Exception:
            pass
    return today


def _logs(logs_dir, pattern):
    return sorted(glob.glob(os.path.join(logs_dir, pattern)))


def _fetch_price(symbol):
    """Last close with retries; None on failure (never NaN). Mirrors performance_report."""
    try:
        import pandas as pd
        import yfinance as yf
    except Exception:
        return None
    ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
    for _ in range(3):
        try:
            df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].dropna()
            if len(closes) > 0:
                return float(closes.iloc[-1])
        except Exception:
            pass
        time.sleep(2)
    return None


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_paper(data_dir, start, end, live):
    out = []
    pp = _read_json(os.path.join(data_dir, "paper_positions.json"))
    if pp is None:
        return ["  (no paper_positions.json found)"]

    closed = pp.get("closed_trades") or []
    opens = [p for p in (pp.get("positions") or [])
             if str(p.get("status", "")).upper() == "OPEN"]

    wk = []
    for c in closed:
        d = _parse_day(c.get("exit_date"))
        if d and start <= d <= end:
            wk.append(c)

    out.append(f"  Closed this week: {len(wk)}")
    if wk:
        pnls = [float(c.get("pnl_rs", 0) or 0) for c in wk]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(wk) * 100
        out.append(f"    Wins {len(wins)} | Losses {len(losses)} | Win rate {wr:.0f}%")
        if wins:
            out.append(f"    Avg win  Rs.{sum(wins)/len(wins):,.0f}")
        if losses:
            out.append(f"    Avg loss Rs.{sum(losses)/len(losses):,.0f}")
        if wins and losses and sum(losses) != 0:
            out.append(f"    Profit factor {abs(sum(wins)/sum(losses)):.2f}")
        out.append(f"    Realized Rs.{sum(pnls):,.0f}")
        for c in sorted(wk, key=lambda x: x.get("exit_date", "")):
            out.append(f"    {c.get('symbol','?'):12s} {c.get('strategy','?'):9s} "
                       f"{c.get('exit_reason','?'):10s} "
                       f"{float(c.get('pnl_pct',0) or 0):6.1f}%  "
                       f"Rs.{float(c.get('pnl_rs',0) or 0):,.0f}")

    out.append(f"  Open now: {len(opens)}")
    total_unreal = 0.0
    for p in opens:
        sym = p.get("symbol", "?")
        entry = float(p.get("entry_price", 0) or 0)
        qty = float(p.get("quantity", 0) or 0)
        sl = float(p.get("stop_loss", 0) or 0)
        tgt = float(p.get("target", 0) or 0)
        if live:
            px = _fetch_price(sym)
            if px is not None:
                pnl = (px - entry) * qty
                total_unreal += pnl
                out.append(f"    {sym:12s} entry {entry:9.2f} SL {sl:9.2f} "
                           f"tgt {tgt:9.2f} now {px:9.2f}  Rs.{pnl:,.0f}")
            else:
                out.append(f"    {sym:12s} entry {entry:9.2f} SL {sl:9.2f} "
                           f"tgt {tgt:9.2f}  (price fetch failed)")
            time.sleep(0.3)
        else:
            out.append(f"    {sym:12s} entry {entry:9.2f} SL {sl:9.2f} tgt {tgt:9.2f}")
    if live and opens:
        out.append(f"    Unrealized Rs.{total_unreal:,.0f}")
    return out


def section_signals(data_dir, logs_dir, start, end, today):
    out = []
    sig = _read_json(os.path.join(data_dir, "signals_today.json"))
    if sig:
        out.append(f"  Latest scan {sig.get('date','?')}: "
                   f"{sig.get('signals_count',0)} signal(s) "
                   f"from {sig.get('basket_size','?')} names")
        for s in (sig.get("signals") or [])[:10]:
            out.append(f"    [{str(s.get('strategy','?')).upper()}] "
                       f"{s.get('symbol','?')} @ {s.get('entry_price','?')} "
                       f"(RSI {s.get('rsi','?')})")
    else:
        out.append("  (no signals_today.json found)")

    counts = {}
    for p in _logs(logs_dir, "scanner.log*"):
        if not re.search(r"\d{8}", os.path.basename(p)):
            continue   # live file, not yet rotated: its scan is already
                       # reported as "Latest scan" above; counting it here
                       # dated it "today" and double-counted the last session
        d = _log_date(p, today)
        if not (start <= d <= end):
            continue
        n = 0
        for line in _open_maybe_gz(p):
            if re.search(r"\[(PULLBACK|REVERSAL)\]", line, re.I):
                n += 1
        counts[d] = counts.get(d, 0) + n
    if counts:
        out.append("  Signal lines per day:")
        for d in sorted(counts):
            out.append(f"    {d.isoformat()}: {counts[d]}")
    return out


def section_guard(data_dir, logs_dir, start, end, today):
    out = []
    rows = _read_csv(os.path.join(data_dir, "trail_audit.csv"))
    wk = []
    for r in rows:
        d = _parse_day(r.get("date"))
        if d and start <= d <= end:
            wk.append(r)
    if wk:
        by_v = {}
        for r in wk:
            v = r.get("verdict_plain") or r.get("verdict", "?")
            by_v[v] = by_v.get(v, 0) + 1
        out.append(f"  Trail audit: {len(wk)} exit(s) this week")
        for v in sorted(by_v):
            out.append(f"    {v}: {by_v[v]}")
        for r in sorted(wk, key=lambda x: x.get("date", "")):
            out.append(f"    {r.get('date','?')} {(r.get('key','?') or '')[:26]:26s} "
                       f"exit Rs.{r.get('exit_mtm','?')} | "
                       f"hold {r.get('hold_worst','?')}..{r.get('hold_best','?')} | "
                       f"{r.get('verdict_plain') or r.get('verdict','?')} "
                       f"({r.get('confidence','?')})")
    else:
        out.append("  Trail audit: no exits this week")

    w50 = w75 = flat = 0
    for p in _logs(logs_dir, "fno_guard.log*"):
        d = _log_date(p, today)
        if not (start <= d <= end):
            continue
        for line in _open_maybe_gz(p):
            if "-> WARN50" in line:
                w50 += 1
            if "-> WARN75" in line:
                w75 += 1
            if "FLATTENING" in line:
                flat += 1
    out.append(f"  Guard alerts: WARN50 {w50}, WARN75 {w75}, flattenings {flat}")
    return out


def section_health(logs_dir, start, end, today):
    totp = cached = minted = 0
    totp_lines = []
    for p in _logs(logs_dir, "*.log*"):
        d = _log_date(p, today)
        if not (start <= d <= end):
            continue
        for line in _open_maybe_gz(p):
            low = line.lower()
            if "totp" in low and ("fail" in low or "error" in low):
                totp += 1
                if len(totp_lines) < 3:
                    totp_lines.append(line.strip()[:110])
            if "using cached dhan access token" in low:
                cached += 1
            if "access token generated" in low or "minting" in low:
                minted += 1
    out = [f"  TOTP failures: {totp}",
           f"  Token cache hits: {cached}",
           f"  Token mints: {minted}"]
    for l in totp_lines:          # show WHAT matched, so a false positive
        out.append("    matched: " + l)   # is visible rather than trusted
    return out


def send_telegram(text):
    """Best-effort delivery to Telegram in <=3900-char chunks. Never raises."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[digest] --telegram set but TELEGRAM_BOT_TOKEN / "
              "TELEGRAM_CHAT_ID are missing; skipping send")
        return False
    import urllib.parse
    import urllib.request
    ok = True
    for i in range(0, len(text), 3900):
        chunk = text[i:i + 3900]
        body = urllib.parse.urlencode(
            {"chat_id": chat, "text": chunk, "disable_web_page_preview": "true"}
        ).encode()
        url = "https://api.telegram.org/bot%s/sendMessage" % token
        try:
            with urllib.request.urlopen(url, data=body, timeout=20) as r:
                if r.status != 200:
                    ok = False
        except Exception as e:
            print("[digest] telegram send failed: %s" % e)
            ok = False
    return ok


def build_report(days, live, data_dir, logs_dir, today):
    start = today - timedelta(days=days - 1)
    end = today
    L = []
    L.append("=" * 70)
    L.append(f"WEEKLY DIGEST — {start.isoformat()} to {end.isoformat()} "
             f"({days} days)")
    L.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} IST")
    L.append("=" * 70)
    L.append("")
    L.append("1. PAPER BOOK")
    L.extend(section_paper(data_dir, start, end, live))
    L.append("")
    L.append("2. SIGNALS")
    L.extend(section_signals(data_dir, logs_dir, start, end, today))
    L.append("")
    L.append("3. F&O GUARD")
    L.extend(section_guard(data_dir, logs_dir, start, end, today))
    L.append("")
    L.append("4. HEALTH")
    L.extend(section_health(logs_dir, start, end, today))
    L.append("")
    L.append("=" * 70)
    return "\n".join(L)


# --------------------------------------------------------------------------
# selftest — fully offline, synthetic fixtures
# --------------------------------------------------------------------------
def selftest():
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "data")
        g = os.path.join(tmp, "logs")
        os.makedirs(d)
        os.makedirs(g)
        today = date(2026, 10, 9)

        with open(os.path.join(d, "paper_positions.json"), "w") as f:
            json.dump({
                "positions": [
                    {"strategy": "pullback", "symbol": "NYKAA",
                     "entry_price": 340.6, "stop_loss": 316.76,
                     "target": 391.69, "quantity": 73, "status": "OPEN"},
                    {"strategy": "reversal", "symbol": "OLDCO",
                     "entry_price": 100, "stop_loss": 90, "target": 120,
                     "quantity": 10, "status": "CLOSED"},
                ],
                "closed_trades": [
                    {"symbol": "GLENMARK", "strategy": "pullback",
                     "exit_date": "2026-10-08", "exit_price": 2240.0,
                     "exit_reason": "SL hit", "pnl_pct": -7.0, "pnl_rs": -1686},
                    {"symbol": "BAJAJ-AUTO", "strategy": "reversal",
                     "exit_date": "2026-10-01", "exit_price": 10435.75,
                     "exit_reason": "SL hit", "pnl_pct": -5.0, "pnl_rs": -1098},
                ],
            }, f)

        with open(os.path.join(d, "signals_today.json"), "w") as f:
            json.dump({"date": "2026-10-09", "basket_size": 30,
                       "signals_count": 1,
                       "signals": [{"strategy": "pullback", "symbol": "RADICO",
                                    "entry_price": 4544.8, "rsi": 54.7}]}, f)

        with open(os.path.join(d, "trail_audit.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "key", "exit_time", "exit_mtm", "peak",
                        "hold_worst", "hold_best", "hold_close", "verdict",
                        "confidence", "method", "delta_per_pt", "iv", "samples"])
            w.writerow(["2026-10-06", "NIFTY-OCT|2026-10-06 14:30:00", "12:00:02",
                        2623, 7966, -3266, 13063, 12763, "PREMATURE", "OK",
                        "bs", -92, 0.128, 20])
            w.writerow(["2026-09-28", "NIFTY-SEP|2026-09-29 14:30:00", "11:40:01",
                        2450, 5258, 729, 9998, 9998, "HELPED", "OK",
                        "bs", -108, 0.245, 17])

        # dated guard log inside window + one outside
        with open(os.path.join(g, "fno_guard.log-20261008.gz"), "wt") as f:
            f.write("[13:49:32] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | "
                    "MTM Rs.-6870 | peak Rs.1313 | loss-limit Rs.7,200 "
                    "(profit-limit Rs.9,000) -> WARN75\n")
        with open(os.path.join(g, "fno_guard.log"), "w") as f:
            f.write("[10:20:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | "
                    "MTM Rs.-98 | peak Rs.-98 | loss-limit Rs.7,200 "
                    "(profit-limit Rs.9,000) -> ok\n")
        with open(os.path.join(g, "scanner.log-20261009.gz"), "wt") as f:
            f.write("[PULLBACK] NYKAA @ 342.05 | SL=318.11 | RSI=55.6\n")
            f.write("[PULLBACK] RADICO @ 4544.8 | SL=4226.66 | RSI=54.7\n")

        rep = build_report(7, False, d, g, today)

    check("WEEKLY DIGEST" in rep, "header present")
    check("Closed this week: 1" in rep, "week filter: 1 closed (Oct 8), not the Oct 1 one")
    check("GLENMARK" in rep and "BAJAJ-AUTO" not in rep, "only in-window trade listed")
    check("Realized Rs.-1,686" in rep, "realized computed from in-window trade")
    check("Open now: 1" in rep, "only OPEN positions counted")
    check("PREMATURE" in rep and "HELPED" not in rep, "audit week filter (HELPED is 11 days back)")
    check("WARN75 1" in rep, "guard alert counted from dated .gz")
    check("2026-10-09: 2" in rep, "signal lines counted per day")
    check("TOTP failures: 0" in rep, "health section renders")

    print()
    print("[SELFTEST] " + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Weekly Digest")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--no-live", action="store_true",
                    help="skip live price fetches for open positions")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    ap.add_argument("--out", default=None,
                    help="also append the report to this file")
    ap.add_argument("--telegram", action="store_true",
                    help="also send the report to Telegram")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    rep = build_report(args.days, not args.no_live, args.data, args.logs,
                       date.today())
    print(rep)
    if args.out:
        with open(args.out, "a") as f:
            f.write("\n" + rep + "\n")
    if args.telegram:
        send_telegram(rep)


if __name__ == "__main__":
    main()
