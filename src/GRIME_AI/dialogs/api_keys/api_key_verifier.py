"""
api_key_verifier.py
-------------------
Lightweight, off-thread key verification for NEON and USGS.

Each verifier makes exactly one cheap API request and returns a
(success: bool, message: str) tuple.  Designed to be called from
QThread workers so the GUI never blocks.

NEON:  GET /api/v0/products/DP1.00001.001 with X-API-Token header.
       A 200 confirms the token is accepted; 401/403 means invalid.

USGS:  GET /ogcapi/v0/collections/daily/items?limit=1 with X-Api-Key header.
       Same logic.  Without a key the request still works (200) but at a
       lower rate limit, so we check the X-RateLimit-Limit header value
       to confirm the key is being honoured (token users get 1000/hr vs 100).
"""

from __future__ import annotations

import requests

_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# NEON
# ---------------------------------------------------------------------------

_NEON_TEST_URL = (
    "https://data.neonscience.org/api/v0/products/DP1.00001.001"
)


def verify_neon_token(token: str) -> tuple[bool, str]:
    """
    Returns (True, "Token accepted") or (False, "<reason>").
    """
    if not token or not token.strip():
        return False, "No token provided."
    headers = {"X-API-Token": token.strip()}
    try:
        r = requests.get(_NEON_TEST_URL, headers=headers, timeout=_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return False, "Network error — check your internet connection."
    except requests.exceptions.Timeout:
        return False, "Request timed out."
    except Exception as exc:
        return False, f"Unexpected error: {exc}"

    if r.status_code == 200:
        return True, "Token accepted  ✓"
    if r.status_code in (401, 403):
        return False, f"Token rejected by NEON (HTTP {r.status_code})."
    return False, f"Unexpected response: HTTP {r.status_code}."


# ---------------------------------------------------------------------------
# USGS
# ---------------------------------------------------------------------------

_USGS_TEST_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
    "?limit=1"
)


def verify_usgs_key(key: str) -> tuple[bool, str]:
    """
    Returns (True, "Key accepted (rate limit: N/hr)") or (False, "<reason>").
    An empty key is allowed (USGS is optional) — we verify only when one
    is supplied.
    """
    if not key or not key.strip():
        return False, "No key provided."
    headers = {"X-Api-Key": key.strip()}
    try:
        r = requests.get(_USGS_TEST_URL, headers=headers, timeout=_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return False, "Network error — check your internet connection."
    except requests.exceptions.Timeout:
        return False, "Request timed out."
    except Exception as exc:
        return False, f"Unexpected error: {exc}"

    if r.status_code in (401, 403):
        return False, f"Key rejected by USGS (HTTP {r.status_code})."
    if r.status_code == 200:
        limit = r.headers.get("X-RateLimit-Limit", "?")
        return True, f"Key accepted  ✓  (rate limit: {limit}/hr)"
    return False, f"Unexpected response: HTTP {r.status_code}."
