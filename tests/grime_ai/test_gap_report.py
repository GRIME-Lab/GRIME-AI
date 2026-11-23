"""Download gap report: timestamp parsing, capture cadence, and missing-image detection."""
from datetime import datetime

import pytest

from GRIME_AI.reporting import gap_report as gr


@pytest.mark.parametrize("name, expected, is_utc", [
    ("NE_Site___2026-01-01T17-00-31Z.jpg", datetime(2026, 1, 1, 17, 0, 31), True),
    ("x_2026-01-01T17:00:31Z.jpg", datetime(2026, 1, 1, 17, 0, 31), True),
    ("harvard_2026_01_01_170031.jpg", datetime(2026, 1, 1, 17, 0, 31), False),
    ("cam_20260101_170031.jpg", datetime(2026, 1, 1, 17, 0, 31), False),
])
def test_parse_timestamp_formats(name, expected, is_utc):
    assert gr.parse_timestamp(name) == (expected, is_utc)


def test_parse_timestamp_rejects_unrecognized_name():
    assert gr.parse_timestamp("IMG_0001.jpg") is None


@pytest.mark.parametrize("tz, iana", [("CST", "America/Chicago"), ("cdt", "America/Chicago"),
                                      ("America/Denver", "America/Denver"), ("", None)])
def test_resolve_usgs_tz(tz, iana):
    assert gr.resolve_usgs_tz(tz) == iana


def listing(tmp_path, names):
    path = tmp_path / "listing.txt"
    path.write_text("\n".join(names))
    return path


def test_analyze_finds_missing_slots(tmp_path):
    """Hourly captures 08:00-16:00 over three days, with 12:00 missing on day 2."""
    names = [f"site_2026_05_{d:02d}_{h:02d}0000.jpg" for d in (1, 2, 3) for h in range(8, 17)
             if not (d == 2 and h == 12)]
    a = gr.analyze(listing(tmp_path, names))
    assert a["cadence"] == 60
    assert a["window"] == (8 * 60, 16 * 60)
    assert a["n_expected"] == 27
    assert a["n_missing_slots"] == 1
    assert a["missing_days"] == []
    assert len(a["gap_events"]) == 1


def test_analyze_reports_missing_day(tmp_path):
    names = [f"site_2026_05_{d:02d}_{h:02d}0000.jpg" for d in (1, 3) for h in range(8, 17)]
    a = gr.analyze(listing(tmp_path, names))
    assert [d.day for d in a["missing_days"]] == [2]
    assert a["n_missing_slots"] == 9


def test_analyze_absorbs_clock_jitter(tmp_path):
    """Captures a few seconds off schedule still land in their slots."""
    names = [f"site_2026_05_01_{h:02d}{m:02d}{s:02d}.jpg"
             for h, m, s in [(8, 0, 4), (8, 59, 57), (10, 0, 11), (10, 59, 50), (12, 0, 2)]]
    a = gr.analyze(listing(tmp_path, names))
    assert a["n_missing_slots"] == 0


def test_analyze_rejects_folder_without_timestamps(tmp_path):
    with pytest.raises(ValueError):
        gr.analyze(listing(tmp_path, ["IMG_0001.jpg", "IMG_0002.jpg"]))
