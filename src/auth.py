"""
Dhan authentication — daily token generation via OAuth + TOTP.
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

    Uses DhanLogin (OAuth flow with TOTP) to generate a token valid for 24 hours.
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
    api_key = os.getenv("DHAN_API_KEY")
    api_secret = os.getenv("DHAN_API_SECRET")
    totp_secret = os.getenv("DHAN_TOTP_SECRET")
    redirect_url = os.getenv("DHAN_REDIRECT_URL", "http://127.0.0.1:5000/dhan/callback")

    if not all([client_id, api_key, api_secret, totp_secret]):
        raise RuntimeError(
            "Missing Dhan credentials. Check .env file for: "
            "DHAN_CLIENT_ID, DHAN_API_KEY, DHAN_API_SECRET, DHAN_TOTP_SECRET"
        )

    try:
        logger.info("Generating Dhan access token via OAuth + TOTP...")

        def totp_generator():
            return pyotp.TOTP(totp_secret).now()

        login = DhanLogin(
            client_id=client_id,
            api_key=api_key,
            api_secret=api_secret,
            redirect_url=redirect_url,
            totp_generator=totp_generator,
        )
        _access_token = login.generate_access_token()

        logger.info("Access token generated successfully (valid 24 hours)")
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
