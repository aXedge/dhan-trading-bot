#!/usr/bin/env python3
"""
Download and cache the Dhan instrument master.
Maps NSE ticker symbols to Dhan security IDs.

Run once before first execution, then weekly to refresh.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth import get_dhan
from src.utils import setup_logger, save_json, DATA_DIR

logger = setup_logger(__name__, "instruments.log")


def download_instruments():
    """Download the full instrument master from Dhan and cache it locally."""
    logger.info("Downloading Dhan instrument master...")

    dhan = get_dhan()

    try:
        # Fetch compact security list
        instruments = dhan.fetch_security_list("compact")

        # Save full list
        save_json(instruments, "instrument_master.json")
        logger.info(f"Saved {len(instruments)} instruments to data/instrument_master.json")

        # Show NSE equity count
        nse_equities = [
            i for i in instruments
            if i.get("SEM_EXM_EXCH_ID") == "NSE"
        ]
        logger.info(f"  NSE instruments: {len(nse_equities)}")

        return instruments

    except Exception as e:
        logger.error(f"Failed to download instruments: {e}")
        raise


if __name__ == "__main__":
    download_instruments()
