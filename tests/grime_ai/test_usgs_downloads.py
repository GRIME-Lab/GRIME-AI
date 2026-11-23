"""USGS image downloads: cancellation, skipping existing files, failures, and use of the cached image list.
No network: urllib.request.urlretrieve is replaced with a fake that writes a small file."""
import os
from datetime import date, time

import pytest

from GRIME_AI.usgs import usgs_service


class _NoKeys:
    """Stands in for APIKeyManager so the tests never read or write user settings."""
    def get_usgs_key(self):
        return ""

    def get_usgs_endpoint(self):
        return usgs_service.ENDPOINT_NEW


class _FakeHIVIS:
    """Stands in for USGS_HIVIS: returns a cached image list for one set of parameters."""
    def __init__(self, names, params):
        self.names, self.params, self.calls = names, params, 0

    def get_cached_filenames(self, *params):
        self.calls += 1
        return self.names if params == self.params else None


SITE = "NE_Test_Site"
NAMES = [f"{SITE}___2026-05-01T{h:02d}-00-00Z.jpg" for h in range(8, 18)]


@pytest.fixture
def downloads(monkeypatch):
    """A USGSService with no network access. Returns (service, list of URLs requested, gap-report folders)."""
    monkeypatch.setattr(usgs_service, "APIKeyManager", _NoKeys)
    requested, reports = [], []

    def fake_urlretrieve(url, dst):
        requested.append(url)
        if "FAIL" in url:
            raise OSError("simulated download failure")
        with open(dst, "wb") as f:
            f.write(b"jpg")

    monkeypatch.setattr(usgs_service.urllib.request, "urlretrieve", fake_urlretrieve)
    svc = usgs_service.USGSService()
    monkeypatch.setattr(svc, "_write_gap_report", lambda site, folder: reports.append(folder))
    monkeypatch.setattr(svc, "_collect_image_names",
                        lambda *a, **k: pytest.fail("image list was fetched instead of using the cache"))
    return svc, requested, reports


def test_downloads_every_image_without_cancel(downloads, tmp_path):
    svc, requested, reports = downloads
    downloaded, missing = svc.download_images_from_list(SITE, NAMES, str(tmp_path))
    assert (downloaded, missing) == (len(NAMES), 0)
    assert sorted(os.listdir(tmp_path)) == sorted(NAMES)
    assert reports == [str(tmp_path)]


def test_cancel_mid_download_stops_at_next_image(downloads, tmp_path):
    svc, requested, _ = downloads
    stop_after = 4
    downloaded, missing = svc.download_images_from_list(
        SITE, NAMES, str(tmp_path), cancel_check=lambda: len(requested) >= stop_after)
    assert downloaded == stop_after
    assert len(os.listdir(tmp_path)) == stop_after


def test_cancel_before_first_image_downloads_nothing(downloads, tmp_path):
    svc, requested, _ = downloads
    assert svc.download_images_from_list(SITE, NAMES, str(tmp_path), cancel_check=lambda: True) == (0, 0)
    assert requested == []


def test_gap_report_written_after_cancel(downloads, tmp_path):
    svc, _, reports = downloads
    svc.download_images_from_list(SITE, NAMES, str(tmp_path), cancel_check=lambda: True)
    assert reports == [str(tmp_path)]


def test_existing_images_are_skipped_and_not_counted(downloads, tmp_path):
    svc, requested, _ = downloads
    (tmp_path / NAMES[0]).write_bytes(b"already here")
    (tmp_path / NAMES[1]).write_bytes(b"")                 # empty file: downloaded again
    downloaded, _ = svc.download_images_from_list(SITE, NAMES, str(tmp_path))
    assert downloaded == len(NAMES) - 1
    assert not any(url.endswith(NAMES[0]) for url in requested)
    assert (tmp_path / NAMES[0]).read_bytes() == b"already here"


def test_failed_image_counted_as_missing(downloads, tmp_path):
    svc, _, _ = downloads
    names = NAMES[:3] + [f"{SITE}___FAIL.jpg"]
    assert svc.download_images_from_list(SITE, names, str(tmp_path)) == (3, 1)


def test_progress_reported_for_each_image(downloads, tmp_path):
    svc, _, _ = downloads
    calls = []
    svc.download_images_from_list(SITE, NAMES, str(tmp_path), progress=lambda i, n, label: calls.append((i, n)))
    assert calls == [(i, len(NAMES)) for i in range(len(NAMES))]


def test_download_images_uses_cached_list_and_honors_cancel(downloads, tmp_path):
    svc, requested, _ = downloads
    params = (SITE, date(2026, 5, 1), date(2026, 5, 1), time(8, 0), time(17, 59))
    svc.hivis = _FakeHIVIS(NAMES, params)
    downloaded, _ = svc.download_images(*params, str(tmp_path), cancel_check=lambda: len(requested) >= 2)
    assert svc.hivis.calls == 1
    assert downloaded == 2
