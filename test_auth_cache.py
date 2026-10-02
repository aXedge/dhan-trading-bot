"""
Offline tests for the shared token cache in auth.py.
Stubs dhanhq/pyotp/utils, so it needs no network and no credentials.

Run:  python test_auth_cache.py
"""
import json
import os
import sys
import tempfile
import threading
import types
from datetime import datetime, timedelta

TMP = tempfile.mkdtemp()
os.environ["DHAN_TOKEN_CACHE"] = os.path.join(TMP, "dhan_token.json")
os.environ.update({"DHAN_CLIENT_ID": "1000191656", "DHAN_PIN": "1234",
                   "DHAN_TOTP_SECRET": "JBSWY3DPEHPK3PXP"})

# ---- stub 'utils' -----------------------------------------------------------
_utils = types.ModuleType("utils")
_nop = lambda *a, **k: None
_utils.setup_logger = lambda name, logfile=None: types.SimpleNamespace(
    info=_nop, error=_nop, warning=_nop, debug=_nop)
sys.modules["utils"] = _utils

# ---- stub 'dhanhq' ----------------------------------------------------------
MINTS = {"count": 0, "fail_first": 0}


class FakeLogin:
    def __init__(self, client_id):
        self.client_id = client_id

    def generate_token(self, pin, totp):
        MINTS["count"] += 1
        if MINTS["fail_first"] > 0:
            MINTS["fail_first"] -= 1
            return {"message": "Invalid TOTP", "status": "error"}
        return {"accessToken": f"tok-{MINTS['count']}",
                "expiryTime": "2026-10-03T00:00:00.000"}


_dhanhq = types.ModuleType("dhanhq")
_dhanhq.DhanLogin = FakeLogin
_dhanhq.DhanContext = lambda cid, tok: {"client_id": cid, "token": tok}
_dhanhq.dhanhq = lambda ctx: ctx
sys.modules["dhanhq"] = _dhanhq

# ---- stub 'pyotp' -----------------------------------------------------------
_pyotp = types.ModuleType("pyotp")
_pyotp.TOTP = lambda secret: types.SimpleNamespace(now=lambda: "123456")
sys.modules["pyotp"] = _pyotp

import auth  # noqa: E402
auth.MINT_GAP_SECONDS = 0  # no sleeps in tests

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def new_process():
    """Simulate a fresh process: clear only the in-memory cache."""
    auth._access_token = None


# 1. cold start mints once and writes the cache
auth.reset_token()
new_process()
t1 = auth.get_access_token()
check("cold start mints exactly once", MINTS["count"] == 1 and t1 == "tok-1")
check("cache file written with 0600 perms",
      os.path.exists(auth.CACHE_PATH)
      and (os.stat(auth.CACHE_PATH).st_mode & 0o777) == 0o600)

# 2. a second process (e.g. monitor after guard) reuses the file token
new_process()
t2 = auth.get_access_token()
check("second process reuses token, no mint",
      t2 == t1 and MINTS["count"] == 1)

# 3. yesterday's token is never reused
with open(auth.CACHE_PATH) as f:
    data = json.load(f)
data["generated_at"] = (datetime.now(auth.IST) - timedelta(days=1)).isoformat()
with open(auth.CACHE_PATH, "w") as f:
    json.dump(data, f)
new_process()
t3 = auth.get_access_token()
check("stale (yesterday) token re-minted", MINTS["count"] == 2 and t3 != t1)

# 4. a TOTP window collision is retried, not fatal
MINTS["fail_first"] = 1
auth.reset_token()
new_process()
t4 = auth.get_access_token()
check("TOTP collision retried to success", t4 == "tok-4" and MINTS["count"] == 4)

# 5. corrupt cache file does not crash; it re-mints
auth.reset_token()
with open(auth.CACHE_PATH, "w") as f:
    f.write("{not json")
new_process()
before = MINTS["count"]
t5 = auth.get_access_token()
check("corrupt cache re-mints safely",
      t5 is not None and MINTS["count"] == before + 1)

# 6. five concurrent cold processes -> exactly one mint, one shared token
auth.reset_token()
MINTS["count"] = 0
results, errors = [], []


def worker():
    auth._access_token = None
    try:
        results.append(auth.get_access_token())
    except Exception as e:  # noqa: BLE001
        errors.append(e)


threads = [threading.Thread(target=worker) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("5 concurrent cold starts mint exactly once",
      MINTS["count"] == 1 and len(set(results)) == 1 and not errors)

# 7. reset_token removes the file so the next call re-mints
auth.reset_token()
check("reset_token deletes the cache file",
      not os.path.exists(auth.CACHE_PATH))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
