# geomaps/ISUSM.py
"""
ISU Soil Moisture Network (Iowa State University), served by the Iowa Environmental Mesonet (IEM).

Station table for map pins, plus minute, hourly and daily observations from IEM's documented ISU Soil Moisture
download service (/cgi-bin/request/isusm.py; parameters at isusm.py?help).
Self-contained: does not depend on the dataset manager.
"""
import io
import datetime
import urllib.parse
import urllib.request

from ..app_identity import APP_NAME

IEM_URL            = "https://mesonet.agron.iastate.edu"
ISUSM_STATIONS_URL = f"{IEM_URL}/geojson/network/ISUSM.geojson"
ISUSM_DATA_URL     = f"{IEM_URL}/cgi-bin/request/isusm.py"
ISUSM_STATION_PAGE = f"{IEM_URL}/sites/site.php?station={{station_id}}&network=ISUSM"
ISUSM_TIMEZONE     = "America/Chicago"     # local time with daylight saving (CST/CDT)
ISUSM_MISSING      = -99                   # IEM missing-value marker, in data and flag columns

# Data type -> (label, request parameters, True if daily)
DATA_TYPES = {
    "minute": ("Minute", {"mode": "hourly", "timeres": "minute"}, False),
    "hourly": ("Hourly", {"mode": "hourly"},                      False),
    "daily":  ("Daily",  {"mode": "daily"},                       True),
}

# From IEM's ISU Soil Moisture download page.
UNITS_TEXT = ("as provided by IEM: air and soil temperature degF, relative humidity %, solar radiation J/m2, "
              "precipitation inch, wind mph (10 ft), direction degrees, ET (alfalfa) inch, soil VWC %; "
              "soil depths in the names are inches (soil04t = 4 in)")
FLAGS_TEXT = ("columns ending in _f are IEM quality flags: E instrument flagged until repaired, R estimate from "
              "surrounding stations, e low-confidence estimate, M missing; B appears in the data but is not "
              "defined in IEM's documentation; empty = no flag")

# The ear-of-corn map pin in black and gold, drawn for this module.
IOWA_CORN_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 40' width='24' height='40'><defs>"
    "<clipPath id='cob'><ellipse cx='12' cy='18.5' rx='5.8' ry='15.5'/></clipPath></defs>"
    "<ellipse cx='12' cy='18.5' rx='5.8' ry='15.5' fill='#b88a00' stroke='#000000' stroke-width='1.3'/>"
    "<g clip-path='url(#cob)'><rect x='6.10' y='3.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='9.00' y='3.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='11.90' y='3.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='14.80' y='3.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='17.70' y='3.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='7.55' y='6.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='10.45' y='6.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='13.35' y='6.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='16.25' y='6.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='6.10' y='9.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='9.00' y='9.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='11.90' y='9.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='14.80' y='9.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='17.70' y='9.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='7.55' y='12.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='10.45' y='12.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='13.35' y='12.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='16.25' y='12.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='6.10' y='14.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='9.00' y='14.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='11.90' y='14.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='14.80' y='14.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='17.70' y='14.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='7.55' y='17.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='10.45' y='17.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='13.35' y='17.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='16.25' y='17.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='6.10' y='20.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='9.00' y='20.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='11.90' y='20.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='14.80' y='20.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='17.70' y='20.40' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='7.55' y='23.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='10.45' y='23.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='13.35' y='23.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='16.25' y='23.20' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='6.10' y='26.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='9.00' y='26.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='11.90' y='26.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='14.80' y='26.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='17.70' y='26.00' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='7.55' y='28.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='10.45' y='28.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='13.35' y='28.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='16.25' y='28.80' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='6.10' y='31.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='9.00' y='31.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='11.90' y='31.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='14.80' y='31.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/>"
    "<rect x='17.70' y='31.60' width='2.5' height='2.45' rx='1.0' fill='#ffcd00'/></g>"
    "<path d='M12 39.2 C4.4 34 2.4 23 4.6 13.5 C7.2 21.5 9.6 29 12.6 33 Z' fill='#141414' stroke='#000000' stroke-width='1' stroke-linejoin='round'/>"
    "<path d='M12 39.2 C19.6 34 21.6 23 19.4 13.5 C16.8 21.5 14.4 29 11.4 33 Z' fill='#141414' stroke='#000000' stroke-width='1' stroke-linejoin='round'/>"
    "<path d='M6 19.5 C7.4 26 9.4 31 12 35.6 M18 19.5 C16.6 26 14.6 31 12 35.6' fill='none' stroke='#b88a00' stroke-width='0.7'/>"
    "</svg>"
)


class ISUSoilMoisture:
    def __init__(self, log_fn=None, timeout_sec=120, chunk_days_minute=7, chunk_days_hourly=366,
                 chunk_days_daily=3660, default_days=1, include_qc_flags=True):
        """
        chunk_days_*:     longest span per request for each data type; longer ranges are split.
        default_days:     span used when no dates are given ("most recent").
        include_qc_flags: ask IEM for the per-variable QC flag columns.
        """
        self.log_fn = log_fn
        self.timeout_sec = timeout_sec
        self.chunk_days = {"minute": chunk_days_minute, "hourly": chunk_days_hourly, "daily": chunk_days_daily}
        self.default_days = default_days
        self.include_qc_flags = include_qc_flags

    # ---- stations ----
    def get_dataframe(self):
        """
        Stations with numeric lat/lon. Columns: station_id, station, county, lat, lon, elevation_m,
        archive_begin, archive_end, online (True for active stations), timezone, station_url.
        """
        import json
        import pandas as pd
        payload = json.loads(self._fetch_text(ISUSM_STATIONS_URL))
        rows = []
        for f in payload.get("features", []):
            p = f.get("properties", {}) or {}
            coords = (f.get("geometry") or {}).get("coordinates") or [None, None]
            sid = p.get("sid") or f.get("id")
            rows.append({
                "station_id":    sid,
                "station":       p.get("sname", ""),
                "county":        p.get("county", ""),
                "lat":           coords[1],
                "lon":           coords[0],
                "elevation_m":   p.get("elevation"),
                "archive_begin": p.get("archive_begin") or "",
                "archive_end":   p.get("archive_end") or "",
                "online":        bool(p.get("online")),
                "timezone":      p.get("tzname") or ISUSM_TIMEZONE,
                "station_url":   ISUSM_STATION_PAGE.format(station_id=sid),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[pd.to_numeric(df["lat"], errors="coerce").notna() & pd.to_numeric(df["lon"], errors="coerce").notna()]
        return df.reset_index(drop=True)

    # ---- observations ----
    def get_observations(self, data_type, station_id, start_date=None, end_date=None):
        """
        Observations as a DataFrame; start_date/end_date are datetime.date (Iowa local), inclusive. Without dates,
        the last `default_days` days are returned.

        Sub-daily: timestamp_local (Iowa local time, CST/CDT) and timestamp_utc. Daily: date.
        Then every column IEM returns, with its own name (tmpf, soil12vwc, ...), each QC flag column (_f) next to
        its variable. IEM's -99 becomes an empty cell in data and flag columns.
        """
        import pandas as pd
        self._check_type(data_type)
        _label, params, daily = DATA_TYPES[data_type]
        if start_date is None or end_date is None:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=self.default_days)

        parts, d = [], start_date
        while d <= end_date:
            last = min(end_date, d + datetime.timedelta(days=self.chunk_days[data_type] - 1))
            query = dict(params, station=station_id, format="comma", tz=ISUSM_TIMEZONE,
                         sts=self._utc_z(d), ets=self._utc_z(last + datetime.timedelta(days=1)))
            if self.include_qc_flags:
                query["qcflags"] = 1
            text = self._fetch_text(f"{ISUSM_DATA_URL}?{urllib.parse.urlencode(query)}")
            if not text.lstrip().startswith("station"):
                raise ValueError(f"Unexpected response from IEM: {text[:300]!r}")
            parts.append(pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False))
            d = last + datetime.timedelta(days=1)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return self._finish(df, daily)

    @staticmethod
    def _utc_z(day):
        """Local midnight of `day` as a UTC timestamp string (IEM requires time-zone-aware times)."""
        import pandas as pd
        local = pd.Timestamp(datetime.datetime.combine(day, datetime.time())).tz_localize(ISUSM_TIMEZONE)
        return local.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")

    @staticmethod
    def _finish(df, daily):
        import pandas as pd
        if df.empty or "valid" not in df.columns:
            return pd.DataFrame(columns=["date"] if daily else ["timestamp_local", "timestamp_utc"])
        df = df.drop(columns=[c for c in ("station",) if c in df.columns])
        missing = str(ISUSM_MISSING)
        for c in df.columns:
            if c == "valid":
                continue
            col = df[c].str.strip()
            col = col.where(~col.isin([missing, f"{missing}.0", f"{missing}.0000"]), "")
            if c.endswith("_f"):
                df[c] = col                                              # flags stay text
            else:
                df[c] = pd.to_numeric(col.replace("", None), errors="coerce")
        stamp = pd.to_datetime(df.pop("valid"), errors="coerce")
        if daily:
            df.insert(0, "date", stamp.dt.date)
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
        else:
            try:
                local = stamp.dt.tz_localize(ISUSM_TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                local = stamp.dt.tz_localize(ISUSM_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
            df.insert(0, "timestamp_utc", local.dt.tz_convert("UTC"))
            df.insert(0, "timestamp_local", local)
            df = df.drop_duplicates(subset=["timestamp_local"]).sort_values("timestamp_local")
        return df.reset_index(drop=True)

    @staticmethod
    def _check_type(data_type):
        if data_type not in DATA_TYPES:
            raise ValueError(f"data_type={data_type!r} is not valid. Use one of: {', '.join(DATA_TYPES)}")

    # ---- fetch ----
    def _fetch_text(self, url):
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=self.timeout_sec).read().decode("utf-8", errors="replace")
        except Exception as e1:
            if self.log_fn:
                self.log_fn(f"urllib fetch failed ({e1.__class__.__name__}); trying requests\u2026")
        import requests
        resp = requests.get(url, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.text
