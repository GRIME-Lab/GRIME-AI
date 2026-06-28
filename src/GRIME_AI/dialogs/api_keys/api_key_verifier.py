"""
api_key_verifier.py
-------------------
Lightweight, off-thread key verification for NEON and USGS.

Returns (ok: bool, lines: list[str]) where lines are structured entries
for display in the APIKeyDialog results QTextEdit.

Line format conventions (parsed by _append_results for bold/colour):
  - Line 0         : timestamp header  (blue, bold)
  - "  Result: …"  : result line       (green/red, bold)
  - "  Note: …"    : note lines        (grey)
  - everything else: detail lines      (grey)
"""

from __future__ import annotations

import requests
from datetime import datetime, timezone

_TIMEOUT = 10  # seconds

_NEON_TEST_URL = "https://data.neonscience.org/api/v0/products/DP1.00001.001"
_USGS_TEST_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items?limit=1"

_ANON_NEON_LIMIT = 200
_ANON_USGS_LIMIT = 200


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _mask(key: str) -> str:
    if not key:
        return "(none)"
    return "*****"


def _rate_note(limit: int, anon_limit: int, credential: str) -> list[str]:
    """Build the Note lines comparing authenticated vs anonymous limits."""
    if limit > anon_limit:
        multiplier = round(limit / anon_limit)
        return [
            f"  Note:             Anonymous users receive {anon_limit} requests/window (rolling).",
            f"                    Your {credential} grants {multiplier}x the default rate limit.",
        ]
    return [
        f"  Note:             Anonymous users receive {anon_limit} requests/window (rolling).",
    ]


# ---------------------------------------------------------------------------
# NEON
# ---------------------------------------------------------------------------

def verify_neon_token(token: str, endpoint: str = "") -> tuple[bool, list[str]]:
    url = (endpoint.rstrip("/") + "/products/DP1.00001.001") if endpoint else _NEON_TEST_URL
    masked = _mask(token.strip() if token else "")

    lines = [
        f"[{_ts()}]  NEON API Token Verification",
        f"  URL:              {url}",
        f"  Token:            {masked}",
        "",
    ]

    if not token or not token.strip():
        lines += [
            "  HTTP:             (not sent)",
            "  API version:      Not available",
            "  Result:           FAIL \u2014 no token provided",
        ]
        return False, lines

    headers = {"X-API-Token": token.strip()}
    try:
        r = requests.get(url, headers=headers, timeout=_TIMEOUT)

        limit_raw     = r.headers.get("X-Ratelimit-Limit", "")
        remaining_raw = r.headers.get("X-Ratelimit-Remaining", "")
        reset_raw     = r.headers.get("X-Ratelimit-Reset", "")

        limit     = int(limit_raw)     if limit_raw.isdigit()     else None
        remaining = int(remaining_raw) if remaining_raw.isdigit() else None
        reset_sec = int(reset_raw)     if reset_raw.isdigit()     else None

        remaining_str = f"{remaining} of {limit}" if (limit is not None and remaining is not None) else "Not available"
        reset_str     = f"~{reset_sec} second{'s' if reset_sec != 1 else ''}" if reset_sec is not None else "Not available"
        limit_str     = f"{limit} requests" if limit is not None else "Not available"

        if r.status_code == 200:
            result_line = "  Result:           OK \u2014 token accepted \u2713"
            ok = True
        elif r.status_code in (401, 403):
            result_line = "  Result:           FAIL \u2014 token rejected by NEON"
            ok = False
        else:
            result_line = f"  Result:           UNEXPECTED \u2014 check NEON API status"
            ok = False

        http_note = ""
        if r.status_code == 200 and not token:
            http_note = "  (token not yet enforced \u2014 changes June 30 2026)"

        lines += [
            f"  HTTP:             {r.status_code} {r.reason}{http_note}",
            f"  API version:      Not available",
            result_line,
            "",
            f"  Rate limit:       {limit_str}",
            f"  Remaining:        {remaining_str}",
            f"  Rate reset:       {reset_str}",
        ]

        if ok and limit is not None:
            lines += [""] + _rate_note(limit, _ANON_NEON_LIMIT, "token")

        if not ok and r.status_code in (401, 403) and r.text:
            lines.append(f"  Detail:           {r.text[:120].strip()}")

        return ok, lines

    except requests.exceptions.ConnectionError:
        lines.append("  Result:           FAIL \u2014 network error (check internet connection)")
        return False, lines
    except requests.exceptions.Timeout:
        lines.append(f"  Result:           FAIL \u2014 request timed out after {_TIMEOUT}s")
        return False, lines
    except Exception as exc:
        lines.append(f"  Result:           FAIL \u2014 {exc}")
        return False, lines


# ---------------------------------------------------------------------------
# USGS
# ---------------------------------------------------------------------------

def verify_usgs_key(key: str, endpoint: str = "") -> tuple[bool, list[str]]:
    url = _USGS_TEST_URL
    masked = _mask(key.strip() if key else "")
    key_display = masked if key and key.strip() else "(none \u2014 anonymous request)"

    lines = [
        f"[{_ts()}]  USGS Water Data API Key Verification",
        f"  URL:              {url}",
        f"  Key:              {key_display}",
        "",
    ]

    headers = {"X-Api-Key": key.strip()} if key and key.strip() else {}
    try:
        r = requests.get(url, headers=headers, timeout=_TIMEOUT)

        limit_raw     = r.headers.get("X-Ratelimit-Limit", "")
        remaining_raw = r.headers.get("X-Ratelimit-Remaining", "")
        reset_raw     = r.headers.get("X-Ratelimit-Reset", "")
        api_version   = r.headers.get("Api-Version", "Not available")

        limit     = int(limit_raw)     if limit_raw.isdigit()     else None
        remaining = int(remaining_raw) if remaining_raw.isdigit() else None
        reset_sec = int(reset_raw)     if reset_raw.isdigit()     else None

        remaining_str = f"{remaining} of {limit}" if (limit is not None and remaining is not None) else "Not available"
        reset_str     = f"~{reset_sec} second{'s' if reset_sec != 1 else ''}" if reset_sec is not None else "Not available"
        limit_str     = f"{limit} requests" if limit is not None else "Not available"

        if r.status_code == 200:
            result_line = ("  Result:           OK \u2014 key accepted \u2713" if key
                           else "  Result:           OK \u2014 anonymous access (no key)")
            ok = True
        elif r.status_code in (401, 403):
            result_line = "  Result:           FAIL \u2014 key rejected by USGS"
            ok = False
        else:
            result_line = f"  Result:           UNEXPECTED \u2014 check USGS API status"
            ok = False

        lines += [
            f"  HTTP:             {r.status_code} {r.reason}",
            f"  API version:      {api_version}",
            result_line,
            "",
            f"  Rate limit:       {limit_str}",
            f"  Remaining:        {remaining_str}",
            f"  Rate reset:       {reset_str}",
        ]

        if ok and limit is not None:
            lines += [""] + _rate_note(limit, _ANON_USGS_LIMIT, "key")
            if not key:
                lines.append(
                    "  Note:             Register a free key at api.waterdata.usgs.gov/signup"
                )
                lines.append(
                    "                    USGS key is optional but recommended for regular use."
                )

        if not ok and r.text:
            lines.append(f"  Detail:           {r.text[:120].strip()}")

        return ok, lines

    except requests.exceptions.ConnectionError:
        lines.append("  Result:           FAIL \u2014 network error (check internet connection)")
        return False, lines
    except requests.exceptions.Timeout:
        lines.append(f"  Result:           FAIL \u2014 request timed out after {_TIMEOUT}s")
        return False, lines
    except Exception as exc:
        lines.append(f"  Result:           FAIL \u2014 {exc}")
        return False, lines
