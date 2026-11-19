#!/usr/bin/env python3
__SCRIPT_NAME = "auth_health"
__SCRIPT_VERSION = "v1"
__SCRIPT_BUILD = "2026-11-20"
__SCRIPT_NOTES = "mint and verify the Dhan token before the market, alert on failure"
"""
auth_health.py — mint the Dhan token before the trading day starts.

WHY THIS EXISTS
  On 20-Nov the token mint failed twice at 00:12 with "Invalid TOTP". The
  secret and the clock were both correct; nine minutes later the identical
  call succeeded. A transient broker-side rejection — but every job at 08:00
  mints on first use, so a repeat would have killed the scan, the guard and
  the manager silently, and nobody would have known until the morning
  shortlist failed to arrive.

  This runs BEFORE them, retries across TOTP windows, and Telegrams on
  failure. Silence means healthy; a message means act.

USAGE
  python auth_health.py            # mint or verify, alert on failure
  python auth_health.py --quiet    # no stdout on success (for cron)
  python auth_health.py --version
"""

import os
import sys
import time
import argparse

if "--version" in sys.argv:
    print(f"{__SCRIPT_NAME} {__SCRIPT_VERSION} (build {__SCRIPT_BUILD}) — {__SCRIPT_NOTES}")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# cross a TOTP window on retry: Dhan's TOTP is single-use per 30-second window
RETRY_GAP = 31
ATTEMPTS = 3


def telegram_send(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": message,
                                "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"[TELEGRAM] send failed: {str(e)[:80]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    last = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            from src.auth import get_access_token
            t = get_access_token()
            if t and len(t) > 50:
                if not args.quiet:
                    print(f"[OK] Dhan token valid ({len(t)} chars), "
                          f"attempt {attempt}/{ATTEMPTS}")
                return 0
            last = f"token returned but looks wrong (length {len(t) if t else 0})"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:160]}"

        print(f"[FAIL] attempt {attempt}/{ATTEMPTS}: {last}")
        if attempt < ATTEMPTS:
            print(f"       waiting {RETRY_GAP}s to cross into a fresh TOTP window")
            time.sleep(RETRY_GAP)

    # every attempt failed: this is the case that must never be silent
    msg = (f"<b>AUTH FAILED</b>\n"
           f"Dhan token could not be minted after {ATTEMPTS} attempts.\n"
           f"<code>{last}</code>\n\n"
           f"Every scheduled job will fail until this is fixed. "
           f"Check DHAN_PIN / DHAN_TOTP_SECRET in .env, and that the clock is right.")
    if not args.no_telegram:
        telegram_send(msg)
    print("[ALERT] Telegram sent" if not args.no_telegram else "[ALERT] not sent")
    return 1


if __name__ == "__main__":
    sys.exit(main())
