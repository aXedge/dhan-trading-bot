"""
Dhan authentication — daily token generation via PIN + TOTP.

TOKEN CACHE (added 2026-10-02)
------------------------------
Dhan's TOTP is single-use per 30-second window. The old in-memory cache was
per-process, so every cron job (guard */5, monitor */15, scanner, executor,
EOD, analyzer) minted its own token — and when two jobs landed on the same
minute, the first consumed the TOTP and the second got "Invalid TOTP".
Observed effect: ~70-90 failures/day and occasional fully blind cycles.

Now a token is shared across processes via a small file cache
(data/dhan_token.json, mode 600), guarded by an flock so concurrent
first-mints cannot race. In steady state exactly ONE process mints per day;
every other job reads the file. Mint attempts also retry across TOTP windows,
so a single window collision no longer kills a run.

The cached token is valid for the rest of the IST calendar day (Dhan tokens
expire ~midnight IST), so a stale file from yesterday is never reused.

NOTE for the guard cron: its wrapper retries get_access_token() itself. With
the file cache a mint happens at most once per day, so keep the guard's
`timeout 120` in mind if you also raise MINT_ATTEMPTS below.

Set DHAN_TOKEN_CACHE to override the cache path (used by the test suite).
"""

import fcntl
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pyotp
from dhanhq import DhanLogin
from utils import setup_logger

logger = setup_logger(__name__, "auth.log")

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_PATH = os.getenv("DHAN_TOKEN_CACHE", "data/dhan_token.json")
LOCK_PATH = CACHE_PATH + ".lock"
MINT_ATTEMPTS = 2          # retries cross TOTP windows
MINT_GAP_SECONDS = 20      # > 0 so attempt 2 lands in the next 30s window

# Cache the token in memory for the process lifetime
_access_token = None


def _read_cache():
    """Return today's cached token, or None if missing/stale/corrupt."""
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        generated = datetime.fromisoformat(data["generated_at"])
        token = data.get("token")
        if token and generated.astimezone(IST).date() == datetime.now(IST).date():
            return token
    except (FileNotFoundError, KeyError, ValueError, OSError,
            json.JSONDecodeError):
        return None
    return None


def _write_cache(token):
    """Atomically write the cache with owner-only permissions."""
    directory = os.path.dirname(CACHE_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"token": token,
                   "generated_at": datetime.now(IST).isoformat()}, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, CACHE_PATH)


def _mint():
    """Mint a fresh token, retrying across TOTP windows on failure."""
    client_id = os.getenv("DHAN_CLIENT_ID")
    pin = os.getenv("DHAN_PIN")
    totp_secret = os.getenv("DHAN_TOTP_SECRET")

    if not all([client_id, pin, totp_secret]):
        raise RuntimeError(
            "Missing Dhan credentials. Check .env file for: "
            "DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET"
        )

    last_err = None
    for attempt in range(1, MINT_ATTEMPTS + 1):
        try:
            logger.info("Generating Dhan access token via PIN + TOTP "
                        f"(attempt {attempt}/{MINT_ATTEMPTS})...")
            totp = pyotp.TOTP(totp_secret).now()
            login = DhanLogin(client_id)
            result = login.generate_token(pin, totp)
            token = result.get("accessToken")
            if not token:
                raise RuntimeError(f"No accessToken in response: {result}")
            logger.info("Access token generated successfully "
                        f"(expires: {result.get('expiryTime', 'unknown')})")
            return token
        except Exception as e:  # window collision, network, bad response
            last_err = e
            logger.error(f"Failed to generate access token: {e}")
            if attempt < MINT_ATTEMPTS:
                time.sleep(MINT_GAP_SECONDS)  # cross into a fresh TOTP window
    raise RuntimeError(f"Dhan authentication failed: {last_err}")


def get_access_token() -> str:
    """
    Return a valid Dhan access token.

    Order: in-process cache -> shared file cache -> mint (under flock).
    Only one process mints; everyone else reads the file.

    Returns:
        Access token string (JWT)

    Raises:
        RuntimeError if credentials are missing or auth fails
    """
    global _access_token

    if _access_token:
        return _access_token

    token = _read_cache()
    if token:
        logger.info("Using cached Dhan access token (shared file cache).")
        _access_token = token
        return token

    # No fresh token: mint, but serialize so concurrent jobs cannot race
    directory = os.path.dirname(LOCK_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            token = _read_cache()   # another process may have just minted
            if not token:
                token = _mint()
                _write_cache(token)
            else:
                logger.info("Another job minted first; using its token.")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    _access_token = token
    return token


def get_dhan_context():
    """
    Get an authenticated DhanContext for API calls.

    Returns:
        DhanContext instance ready for dhanhq() initialization
    """
    from dhanhq import DhanContext

    client_id = os.getenv("DHAN_CLIENT_ID")
    token = get_access_token()
    return DhanContext(client_id, token)


def get_dhan():
    """
    Get an authenticated dhanhq client instance.

    Returns:
        dhanhq instance ready for place_order(), option_chain(), etc.
    """
    from dhanhq import dhanhq

    ctx = get_dhan_context()
    return dhanhq(ctx)


def reset_token():
    """Force re-authentication on next get_access_token() call."""
    global _access_token
    _access_token = None
    try:
        os.remove(CACHE_PATH)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    token = get_access_token()
    print(f"Token generated: {token[:20]}... (expires in ~24h)")
