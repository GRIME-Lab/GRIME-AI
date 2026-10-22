# geomaps/AZMET.py
"""
Arizona Meteorological Network (AZMet).

Station table for map pins, plus observations from the public AZMet API (api.azmet.arizona.edu):
15-minute, hourly, daily, leaf wetness 15-minute and leaf wetness daily data.
Self-contained: does not depend on the dataset manager.

The station table comes from the station file AZMet keeps in its official R package repository
(uace-azmet/azmetr), because the API has no station-list endpoint.
"""
import json
import datetime
import urllib.request

from ..app_identity import APP_NAME

AZMET_API_URL      = "https://api.azmet.arizona.edu/v1/observations"
AZMET_STATIONS_URL = "https://raw.githubusercontent.com/uace-azmet/azmetr/main/data-raw/azmet-station-info.csv"
AZMET_STATION_PAGE = "https://azmet.arizona.edu/about/station-metadata/{station_id}"
AZMET_TIMEZONE     = "America/Phoenix"      # Arizona: UTC-7 all year, no daylight saving time
AZMET_TEST_STATION = "az99"
EARLIEST_DATA_DATE = "2021-01-01"           # start of the data the API serves

# Data type -> (label, time field in each record, True if values are daily)
DATA_TYPES = {
    "15min":   ("15-minute",               "datetime",      False),
    "hourly":  ("Hourly",                  "date_datetime", False),
    "daily":   ("Daily",                   "datetime",      True),
    "lw15min": ("Leaf wetness 15-minute",  "datetime",      False),
    "lwdaily": ("Leaf wetness daily",      "datetime",      True),
}

# Values AZMet uses for "no measurement" (e.g. a sensor the station does not have).
MISSING_VALUES = {-999.0, -9999.0, -99999.0}

# The saguaro map pin, as an SVG drawn for this module.
SAGUARO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 40' width='24' height='40'>"
    "<g fill='none' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M12 4.5 V37' stroke='#1b5e20' stroke-width='7.6'/>"
    "<path d='M11 26 H6.6 Q4.6 26 4.6 24 V15' stroke='#1b5e20' stroke-width='5.6'/>"
    "<path d='M13 21 H17.4 Q19.4 21 19.4 19 V10' stroke='#1b5e20' stroke-width='5.6'/>"
    "<path d='M12 4.5 V37' stroke='#43a047' stroke-width='5.6'/>"
    "<path d='M11 26 H6.6 Q4.6 26 4.6 24 V15' stroke='#43a047' stroke-width='3.6'/>"
    "<path d='M13 21 H17.4 Q19.4 21 19.4 19 V10' stroke='#43a047' stroke-width='3.6'/>"
    "<path d='M10.2 6 V36.5 M12 3.6 V37 M13.8 6 V36.5' stroke='#1b5e20' stroke-width='0.7'/>"
    "<path d='M4.6 15.5 V23.5 M19.4 10.5 V18.5' stroke='#1b5e20' stroke-width='0.7'/>"
    "</g></svg>"
)


class AZMet:
    def __init__(self, log_fn=None, timeout_sec=60, chunk_hours=720, chunk_days=365):
        """
        chunk_hours: longest span per request for 15-minute and hourly data (requests are split into chunks).
        chunk_days:  longest span per request for daily data.
        """
        self.log_fn = log_fn
        self.timeout_sec = timeout_sec
        self.chunk_hours = chunk_hours
        self.chunk_days = chunk_days

    # ---- stations ----
    def get_dataframe(self):
        """Return a DataFrame of AZMet stations (test station excluded) with numeric lat/lon."""
        import io
        import pandas as pd
        text = self._fetch_text(AZMET_STATIONS_URL)
        df = pd.read_csv(io.StringIO(text.lstrip("\ufeff")))
        df = df[df["id"] != AZMET_TEST_STATION].rename(columns={
            "name": "station", "symbol": "symbol", "id": "station_id", "latitude": "lat", "longitude": "lon"})
        df = df[pd.to_numeric(df["lat"], errors="coerce").notna() & pd.to_numeric(df["lon"], errors="coerce").notna()]
        df["station_url"] = df["station_id"].apply(self.station_page_url)
        return df.reset_index(drop=True)

    @staticmethod
    def station_page_url(station_id):
        """AZMet's page for one station: location, sensors and heights, previous locations, link to past data."""
        return AZMET_STATION_PAGE.format(station_id=station_id)

    # ---- observations ----
    def get_observations_raw(self, data_type, station_id, start="*", interval="*"):
        """One API call, untouched JSON. start/interval follow the API: YYYY-MM-DDTHH:MM and PT<h>H / P<d>DT0H."""
        self._check_type(data_type)
        return json.loads(self._fetch_text(f"{AZMET_API_URL}/{data_type}/{station_id}/{start}/{interval}"))

    def get_observations(self, data_type, station_id, start_date=None, end_date=None):
        """
        Observations as a DataFrame. start_date/end_date are datetime.date (Arizona local), inclusive.
        Without dates the API default is used (the most recent period it serves for that data type).

        Columns: timestamp_local, timestamp_utc (sub-daily types) or date (daily types), then every field
        AZMet returns, with its own name and both unit systems as provided (e.g. temp_airC and temp_airF).
        Numbers are converted from text; AZMet's no-measurement values (-999, -9999, -99999) become empty.
        """
        self._check_type(data_type)
        records, errors = [], []
        for start, interval in self._chunks(data_type, start_date, end_date):
            payload = self.get_observations_raw(data_type, station_id, start, interval)
            records.extend(payload.get("data") or [])
            errors.extend(payload.get("errors") or [])
        df = self._to_dataframe(data_type, records)
        df.attrs["api_errors"] = errors
        return df

    def _chunks(self, data_type, start_date, end_date):
        if start_date is None or end_date is None:
            return [("*", "*")]
        daily = DATA_TYPES[data_type][2]
        out = []
        if daily:
            d = start_date
            while d <= end_date:
                last = min(end_date, d + datetime.timedelta(days=self.chunk_days - 1))
                out.append((f"{d.isoformat()}T00:00", f"P{(last - d).days}DT0H"))
                d = last + datetime.timedelta(days=1)
        else:
            t = datetime.datetime.combine(start_date, datetime.time(0, 0))
            stop = datetime.datetime.combine(end_date + datetime.timedelta(days=1), datetime.time(0, 0))
            while t < stop:
                hours = min(self.chunk_hours, int((stop - t).total_seconds() // 3600))
                out.append((t.strftime("%Y-%m-%dT%H:%M"), f"PT{hours}H"))
                t += datetime.timedelta(hours=hours)
        return out

    def _to_dataframe(self, data_type, records):
        import pandas as pd
        label, time_field, daily = DATA_TYPES[data_type]
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=["date"] if daily else ["timestamp_local", "timestamp_utc"])

        # Text -> numbers where every value in a column is numeric; other text columns stay text.
        # Codes with leading zeros (date_hour "0100") also stay text.
        for c in df.columns:
            if c == time_field or df[c].astype(str).str.match(r"^0\d").any():
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() == df[c].replace("", None).notna().sum():
                df[c] = converted.where(~converted.isin(MISSING_VALUES))

        stamp = df.pop(time_field) if time_field in df.columns else pd.Series([None] * len(df))
        if daily:
            df.insert(0, "date", pd.to_datetime(stamp, errors="coerce").dt.date)
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
        else:
            local = pd.to_datetime(stamp, errors="coerce").dt.tz_localize(AZMET_TIMEZONE)
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
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)", "Accept": "application/json, text/csv"}
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
