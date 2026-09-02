"""
Dhan authentication — daily token generation via PIN + TOTP.
"""

import os
import pyotp
from dhanhq import DhanLogin
from utils import setup_logger

logger = setup_logger(__name__, "auth.log")

# Cache the token in memory for the session
_access_token = None

def get_access_token() -> str:
    """
    Generate or return cached Dhan access token.

    Uses DhanLogin PIN/TOTP flow to generate a token valid until ~midnight IST.
    Caches the token in memory — subsequent calls reuse it.

    Returns:
        Access token string (JWT)

    Raises:
        RuntimeError if credentials are missing or auth fails
    """
    global _access_token

    if _access_token:
        return _access_token

    client_id = os.getenv("DHAN_CLIENT_ID")
    pin = os.getenv("DHAN_PIN")
    totp_secret = os.getenv("DHAN_TOTP_SECRET")

    if not all([client_id, pin, totp_secret]):
        raise RuntimeError(
            "Missing Dhan credentials. Check .env file for: "
            "DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET"
        )

    try:
        logger.info("Generating Dhan access token via PIN + TOTP...")

        totp = pyotp.TOTP(totp_secret).now()

        login = DhanLogin(client_id)
        result = login.generate_token(pin, totp)

        _access_token = result.get("accessToken")

        if not _access_token:
            raise RuntimeError(f"No accessToken in response: {result}")

        expiry = result.get("expiryTime", "unknown")
        logger.info(f"Access token generated successfully (expires: {expiry})")
        return _access_token

    except Exception as e:
        logger.error(f"Failed to generate access token: {e}")
        raise RuntimeError(f"Dhan authentication failed: {e}")


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


if __name__ == "__main__":
    token = get_access_token()
    print(f"Token generated: {token[:20]}... (expires in ~24h)")
