# geomaps/KSMESONET.py
"""
Kansas Mesonet (Kansas State University).

Station table for map pins, plus 5-minute, hourly and daily observations from the Kansas Mesonet REST service
(mesonet.k-state.edu/rest), the same service K-State's own ksmesopy package uses.
Self-contained: does not depend on the dataset manager.
"""
import io
import datetime
import urllib.parse
import urllib.request

from ..app_identity import APP_NAME

KS_API_URL        = "http://mesonet.k-state.edu/rest"
KS_TIMEZONE       = "Etc/GMT+6"         # Kansas Mesonet times are Central Standard Time (UTC-6) all year
KS_TIMEZONE_LABEL = "Central Standard Time (UTC-6), no daylight saving time"
KS_MISSING        = "M"

# Data type -> (label, interval in seconds as used by the station availability table, True if daily)
DATA_TYPES = {
    "5min": ("5-minute", 300,   False),
    "hour": ("Hourly",   3600,  False),
    "day":  ("Daily",    86400, True),
}

UNITS_TEXT = ("metric, as provided: temperature degC, pressure kPa, precipitation mm, solar radiation W/m2, "
              "wind m/s, direction degrees, VPD kPa, soil VWC m3/m3, soil EC dS/m "
              "(units per K-State's ksmesopy documentation; the service does not return units)")

# The sunflower map pin (Kansas state flower), drawn for this module. Centered on the station.
SUNFLOWER_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30' width='30' height='30'>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(0.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(22.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(45.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(67.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(90.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(112.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(135.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(157.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(180.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(202.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(225.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(247.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(270.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(292.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(315.0 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<ellipse cx='15' cy='6.50' rx='2.0' ry='4.1' transform='rotate(337.5 15 15)' fill='#ffc20e' stroke='#a35f00' stroke-width='0.6'/>"
    "<circle cx='15' cy='15' r='5.4' fill='#6b4219' stroke='#2b1a0b' stroke-width='0.8'/>"
    "<circle cx='15.67' cy='15.00' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='14.14' cy='15.79' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='15.13' cy='13.50' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='16.08' cy='16.41' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='13.02' cy='14.65' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='16.88' cy='13.80' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='14.37' cy='17.34' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='13.80' cy='12.69' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='17.60' cy='15.95' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='12.29' cy='16.12' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='16.30' cy='12.21' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='15.97' cy='18.07' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='12.09' cy='13.32' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='18.41' cy='14.24' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='12.93' cy='17.96' r='0.45' fill='#2b1a0b'/>"
    "<circle cx='14.51' cy='11.29' r='0.45' fill='#2b1a0b'/></svg>"
)


class KSMesonet:
    def __init__(self, log_fn=None, timeout_sec=60, max_records=3000, default_days=1):
        """
        max_records:  most rows per request; longer ranges are split (K-State's ksmesopy uses 3,000).
        default_days: span used when no dates are given ("most recent").
        """
        self.log_fn = log_fn
        self.timeout_sec = timeout_sec
        self.max_records = max_records
        self.default_days = default_days

    # ---- stations ----
    def get_dataframe(self):
        """
        Stations with numeric lat/lon. Columns: station_id (the station name, which the service uses as its ID),
        station, abbreviation, county, lat, lon, elevation_m, network, operator_name, fire_weather_number,
        and data_start_/data_end_ for each data type (5min, hour, day) from the availability table.
        """
        import pandas as pd
        names = pd.read_csv(io.StringIO(self._fetch_text(f"{KS_API_URL}/stationnames/")))
        df = names.rename(columns={
            "NAME": "station", "COUNTY": "county", "LATITUDE": "lat", "LONGITUDE": "lon",
            "ELEVATION": "elevation_m", "NETWORK": "network", "ABBR": "abbreviation",
            "OPER_NAME": "operator_name", "FW13_NUM": "fire_weather_number"})
        df.insert(0, "station_id", df["station"])
        for c in ("county", "network", "abbreviation", "operator_name"):
            df[c] = df[c].fillna("").astype(str)
        try:
            active = pd.read_csv(io.StringIO(self._fetch_text(f"{KS_API_URL}/stationactive/")))
            for t, (_label, seconds, _daily) in DATA_TYPES.items():
                part = active[active["OBS_INTERVAL"] == seconds][["STATION", "START", "END"]]
                part = part.rename(columns={"STATION": "station", "START": f"data_start_{t}",
                                            "END": f"data_end_{t}"})
                df = df.merge(part, on="station", how="left")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Kansas station availability not loaded: {e}")
        df = df[pd.to_numeric(df["lat"], errors="coerce").notna() & pd.to_numeric(df["lon"], errors="coerce").notna()]
        return df.reset_index(drop=True)

    # ---- observations ----
    def get_observations(self, data_type, station_id, start_date=None, end_date=None):
        """
        Observations as a DataFrame; start_date/end_date are datetime.date, inclusive. Without dates, the last
        `default_days` days are returned.

        Sub-daily: timestamp_local (Central Standard Time) and timestamp_utc.
        Daily: date (the day the values describe) and timestamp_api (the service stamps each day at 00:00 of the
        FOLLOWING day; per K-State's ksmesopy).
        Then every variable the service returns, with its own name (TEMP2MAVG, PRECIP, VWC5CM, ...).
        Missing values ("M") become empty.
        """
        import pandas as pd
        self._check_type(data_type)
        _label, seconds, daily = DATA_TYPES[data_type]
        if start_date is None or end_date is None:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=self.default_days)

        if daily:   # day D is stamped at 00:00 of D+1
            first = datetime.datetime.combine(start_date + datetime.timedelta(days=1), datetime.time())
            last = datetime.datetime.combine(end_date + datetime.timedelta(days=1), datetime.time())
        else:
            first = datetime.datetime.combine(start_date, datetime.time())
            last = datetime.datetime.combine(end_date + datetime.timedelta(days=1), datetime.time()) \
                - datetime.timedelta(seconds=seconds)

        step = datetime.timedelta(seconds=seconds)
        parts, cur = [], first
        while cur <= last:
            chunk_end = min(cur + step * (self.max_records - 1), last)
            query = urllib.parse.urlencode({"stn": station_id, "int": data_type,
                                            "t_start": cur.strftime("%Y%m%d%H%M%S"),
                                            "t_end": chunk_end.strftime("%Y%m%d%H%M%S")})
            text = self._fetch_text(f"{KS_API_URL}/stationdata/?{query}")
            if not text.lstrip().startswith("TIMESTAMP"):
                raise ValueError(f"Unexpected response from the Kansas Mesonet: {text[:200]!r}")
            parts.append(pd.read_csv(io.StringIO(text), na_values=KS_MISSING, keep_default_na=False))
            cur = chunk_end + step
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return self._finish(df, daily)

    @staticmethod
    def _finish(df, daily):
        import pandas as pd
        if df.empty or "TIMESTAMP" not in df.columns:
            return pd.DataFrame(columns=["date", "timestamp_api"] if daily else ["timestamp_local", "timestamp_utc"])
        df = df.drop(columns=[c for c in ("STATION",) if c in df.columns])
        stamp = pd.to_datetime(df.pop("TIMESTAMP"), errors="coerce")
        if daily:
            df.insert(0, "timestamp_api", stamp)
            df.insert(0, "date", (stamp - pd.Timedelta(days=1)).dt.date)
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
        else:
            local = stamp.dt.tz_localize(KS_TIMEZONE)
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
