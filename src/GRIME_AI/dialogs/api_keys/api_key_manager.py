"""
api_key_manager.py
------------------
Centralised read/write access to NEON and USGS API keys for GRIME AI.

Keys are stored in  <settings_folder>/api_keys.ini  under section [api_keys].
The file is created on first save and is excluded from version control by the
project .gitignore (api_keys.ini).

Usage
-----
# Read (GUI or CLI):
    from GRIME_AI.dialogs.api_keys.api_key_manager import APIKeyManager
    mgr  = APIKeyManager()
    neon = mgr.get_neon_token()   # str | None
    usgs = mgr.get_usgs_key()     # str | None

# Write (called from the dialog's Save button):
    mgr.save(neon_token="abc123", usgs_key="xyz789")

# CLI helper — call once at startup after argparse:
    mgr.apply_cli_overrides(args)   # args.neon_token / args.neon_token_file etc.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path


_SECTION         = "api_keys"
_KEY_NEON        = "neon_token"
_KEY_NEON_EP     = "neon_endpoint"
_KEY_USGS        = "usgs_api_key"
_KEY_USGS_EP     = "usgs_endpoint"
_FILENAME        = "api_keys.ini"
_DEFAULT_NEON_EP = "https://data.neonscience.org/api/v0/"
_DEFAULT_USGS_EP = "https://api.waterdata.usgs.gov/nims/v0"


def _settings_folder() -> Path:
    """Return the GRIME AI settings folder, creating it if necessary.

    Resolution order:
      1. GRIME_AI_Save_Utils().get_settings_folder()  — normal runtime path
      2. ~/Documents/GRIME-AI/Settings                — mirrors PROJECT_ROOT
         defined in the package __init__.py
    """
    try:
        from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
        folder = Path(GRIME_AI_Save_Utils().get_settings_folder())
    except Exception:
        folder = Path.home() / "Documents" / "GRIME-AI" / "Settings"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class APIKeyManager:
    """Thin wrapper around configparser for NEON / USGS API key persistence."""

    def __init__(self) -> None:
        self._path   = _settings_folder() / _FILENAME
        self._config = configparser.ConfigParser()
        if self._path.exists():
            self._config.read(self._path, encoding="utf-8")
        if not self._config.has_section(_SECTION):
            self._config.add_section(_SECTION)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_neon_token(self) -> str | None:
        """Return the stored NEON API token, or None if not set."""
        value = self._config.get(_SECTION, _KEY_NEON, fallback="").strip()
        return value if value else None

    def get_neon_endpoint(self) -> str:
        """Return the stored NEON API base URL, defaulting to data.neonscience.org."""
        value = self._config.get(_SECTION, _KEY_NEON_EP, fallback="").strip()
        return value if value else _DEFAULT_NEON_EP

    def get_usgs_key(self) -> str | None:
        """Return the stored USGS API key, or None if not set."""
        value = self._config.get(_SECTION, _KEY_USGS, fallback="").strip()
        return value if value else None

    def get_usgs_endpoint(self) -> str:
        """Return the stored NIMS API endpoint, defaulting to the new api.waterdata.usgs.gov URL."""
        value = self._config.get(_SECTION, _KEY_USGS_EP, fallback="").strip()
        return value if value else _DEFAULT_USGS_EP

    # ------------------------------------------------------------------
    # Setters / persistence
    # ------------------------------------------------------------------

    def save(
        self,
        neon_token:    str | None = None,
        neon_endpoint: str | None = None,
        usgs_key:      str | None = None,
        usgs_endpoint: str | None = None,
    ) -> None:
        """Persist one or more values.  Pass None to leave a value unchanged."""
        if neon_token is not None:
            self._config.set(_SECTION, _KEY_NEON, neon_token.strip())
        if neon_endpoint is not None:
            self._config.set(_SECTION, _KEY_NEON_EP, neon_endpoint.strip())
        if usgs_key is not None:
            self._config.set(_SECTION, _KEY_USGS, usgs_key.strip())
        if usgs_endpoint is not None:
            self._config.set(_SECTION, _KEY_USGS_EP, usgs_endpoint.strip())
        with open(self._path, "w", encoding="utf-8") as fh:
            self._config.write(fh)

    def clear(self, neon: bool = True, usgs: bool = True) -> None:
        """Erase stored keys (useful for testing or logout flows)."""
        if neon:
            self._config.set(_SECTION, _KEY_NEON, "")
        if usgs:
            self._config.set(_SECTION, _KEY_USGS, "")
        with open(self._path, "w", encoding="utf-8") as fh:
            self._config.write(fh)

    # ------------------------------------------------------------------
    # CLI helper
    # ------------------------------------------------------------------

    def apply_cli_overrides(self, args) -> None:
        """
        Merge CLI-supplied keys into the in-memory config (does NOT save to
        disk — command-line keys are session-only overrides).

        Expected argparse attributes (all optional):
            args.neon_token          str  — raw token string
            args.neon_token_file     str  — path to .txt file containing token
            args.usgs_api_key        str  — raw key string
            args.usgs_api_key_file   str  — path to .txt file containing key
        """
        neon_token = _resolve_cli_value(
            getattr(args, "neon_token", None),
            getattr(args, "neon_token_file", None),
        )
        usgs_key = _resolve_cli_value(
            getattr(args, "usgs_api_key", None),
            getattr(args, "usgs_api_key_file", None),
        )
        if neon_token:
            self._config.set(_SECTION, _KEY_NEON, neon_token)
        if usgs_key:
            self._config.set(_SECTION, _KEY_USGS, usgs_key)


def _resolve_cli_value(raw: str | None, file_path: str | None) -> str | None:
    """Return the first non-empty value from raw string or file contents."""
    if raw and raw.strip():
        return raw.strip()
    if file_path:
        path = Path(file_path)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        print(f"[APIKeyManager] Warning: key file not found: {file_path}")
    return None
