#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# http_client.py
#
# One place for outbound HTTP policy. Every PhenoCam / NEON / USGS request in
# GRIME must go through here so that a slow or dead server FAILS FAST instead
# of hanging a feature indefinitely.
#
# Root cause this fixes: `requests` and `urllib` have NO timeout by default.
# A server that accepts the connection and then goes silent blocks the call
# forever (the "read timeout=None" seen in the logs). Background threads keep
# the GUI alive, but the feature never completes.
#
# Policy:
#   - connect timeout: fail if the TCP/TLS handshake takes too long
#   - read timeout:    fail if the server stops sending mid-response
#   - bounded retries with exponential backoff on transient failures
#   - never retry forever; surface a clear error so the UI can degrade
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# (connect, read) seconds. Connect is short: a reachable host answers in
# well under 5 s. Read is longer to allow large image bodies to stream.
DEFAULT_TIMEOUT = (5, 20)
# For single large image downloads that legitimately take a while.
IMAGE_TIMEOUT = (5, 60)

_RETRY = Retry(
    total=3,                     # at most 3 retries (4 attempts)
    connect=2,
    read=2,
    backoff_factor=0.5,          # 0.5s, 1s, 2s between retries
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "HEAD"]),
    raise_on_status=False,
)


class TimeoutSession(requests.Session):
    """requests.Session that ALWAYS applies a timeout, so no call site can
    accidentally issue an unbounded request."""

    def __init__(self, timeout=DEFAULT_TIMEOUT):
        super().__init__()
        self._default_timeout = timeout
        adapter = HTTPAdapter(max_retries=_RETRY, pool_maxsize=8)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


# Module-level shared session (connection reuse across the app).
_session = TimeoutSession()


def get(url, timeout=None, **kwargs):
    """GET with a guaranteed timeout and bounded retries. Raises
    requests.RequestException on failure (caller decides how to degrade)."""
    if timeout is not None:
        kwargs["timeout"] = timeout
    return _session.get(url, **kwargs)


def get_bytes(url, timeout=IMAGE_TIMEOUT):
    """Download a binary body (image). Raises on failure."""
    r = get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def get_text(url, timeout=DEFAULT_TIMEOUT):
    """Fetch a page body as text. Raises on failure."""
    r = get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def is_reachable(url, timeout=(3, 5)):
    """Cheap liveness probe (HEAD). Returns True/False, never raises. Use this
    at startup so features can be marked unavailable instead of blocking."""
    try:
        r = _session.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code < 500
    except requests.RequestException:
        return False
