# geomaps/NEMESONET.py
"""
Nebraska Mesonet station table.

Reads the station list the Nebraska Mesonet website uses (nemesonet.unl.edu/api/frontend/stations) and returns a
pandas DataFrame of stations for map pins. Self-contained: does not depend on the dataset manager.

The endpoint is internal to the website; its responses say it "is not intended for public use and may change
without notice".
"""
import re
import html
import json
import urllib.parse
import urllib.request

from ..app_identity import APP_NAME

NE_MESONET_BASE_URL     = "https://nemesonet.unl.edu"
NE_MESONET_STATIONS_URL = f"{NE_MESONET_BASE_URL}/api/frontend/stations"

METERS_TO_FEET = 3.28084

# The ear-of-corn map pin (Cornhuskers: red husks, yellow corn), as an SVG drawn for this module.
CORN_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 40' width='24' height='40'><defs>"
    "<clipPath id='cob'><ellipse cx='12' cy='18.5' rx='5.8' ry='15.5'/></clipPath></defs>"
    "<ellipse cx='12' cy='18.5' rx='5.8' ry='15.5' fill='#b88a00' stroke='#4a0008' stroke-width='1.3'/>"
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
    "<path d='M12 39.2 C4.4 34 2.4 23 4.6 13.5 C7.2 21.5 9.6 29 12.6 33 Z' fill='#c8102e' stroke='#4a0008' stroke-width='1' stroke-linejoin='round'/>"
    "<path d='M12 39.2 C19.6 34 21.6 23 19.4 13.5 C16.8 21.5 14.4 29 11.4 33 Z' fill='#c8102e' stroke='#4a0008' stroke-width='1' stroke-linejoin='round'/>"
    "<path d='M6 19.5 C7.4 26 9.4 31 12 35.6 M18 19.5 C16.6 26 14.6 31 12 35.6' fill='none' stroke='#8a0016' stroke-width='0.7'/>"
    "</svg>"
)

# Valid parameter values, taken from the website's JavaScript (enums Kt, Zt, ge, ys, ni).
RESOLUTIONS       = ("minute", "5-minute", "10-minute", "hourly", "daily")
PERIODS           = ("past-hour", "today", "yesterday", "last-3-days", "last-7-days", "last-30-days",
                     "this-month", "last-month", "last-year", "year-to-date", "season-to-date", "custom")
AGGREGATES        = ("avg", "min", "max")
SOIL_MEASUREMENTS = ("soil-moisture", "bare-soil-temperature", "vegetated-soil-temperature")
WIND_SOURCES      = ("wind-3m", "gust-3m", "wind-10m", "gust-10m")

# Periods the server accepts for each resolution, found by testing station-history on 2026-10-05.
# These are observed rules, not documented ones, and may change. Resolutions not listed here ("minute" was not
# tested) are not checked. Set NEMesonet(enforce_period_rules=False) to send requests unchecked if the server changes.
# Assumed to apply to soil and wind history too, since the website runs the same period check for every chart.
PERIODS_BY_RESOLUTION = {
    "5-minute":  ("past-hour", "today", "yesterday", "last-3-days", "last-7-days"),
    "10-minute": ("past-hour", "today", "yesterday", "last-3-days", "last-7-days", "last-30-days"),
    "hourly":    ("today", "yesterday", "last-3-days", "last-7-days"),
    "daily":     ("yesterday", "last-3-days", "last-7-days", "last-30-days", "this-month", "last-month",
                  "last-year", "year-to-date", "season-to-date", "custom"),
}
# Earliest date with data in the current system (daily custom ranges returned nothing before 2024).
EARLIEST_DATA_DATE = "2024-01-01"


class NEMesonet:
    def __init__(self, log_fn=None, timeout_sec=30, enforce_period_rules=True):
        self.log_fn = log_fn
        self.timeout_sec = timeout_sec
        self.enforce_period_rules = enforce_period_rules

    # ---- public ----
    def get_dataframe(self):
        """Return a DataFrame of NE Mesonet stations with numeric lat/lon."""
        import pandas as pd
        payload = self._fetch_json(NE_MESONET_STATIONS_URL)
        rows = [self._flatten(s) for s in payload.get("data", [])]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["lat"].apply(lambda v: isinstance(v, (int, float)))]
            df = df[df["lon"].apply(lambda v: isinstance(v, (int, float)))]
            df = df.reset_index(drop=True)
        return df

    @staticmethod
    def station_page_url(station_id):
        return f"{NE_MESONET_BASE_URL}/station?station={station_id}"

    # ---- station history: weather, precipitation, radiation, ET, degree days ----
    def get_station_history_raw(self, station_id, resolution, period, aggregate="avg", start=None, end=None):
        """Raw JSON from /stations/{id}/station-history."""
        self._check("resolution", resolution, RESOLUTIONS)
        self._check("period", period, PERIODS)
        self._check_period_for_resolution(resolution, period)
        self._check("aggregate", aggregate, AGGREGATES)
        self._check_custom(period, start, end)
        return self._fetch_json(f"{NE_MESONET_BASE_URL}/api/frontend/stations/{station_id}/station-history",
                                {"resolution": resolution, "period": period, "aggregate": aggregate,
                                 "start": start, "end": end})

    def get_station_history(self, station_id, resolution, period, aggregate="avg", start=None, end=None):
        """
        Station history as a DataFrame: one row per timestamp, one column per variable.
          timestamp_utc, timestamp_local   (local = the station's own time zone)
          <variable>                       for value series, e.g. temperature, precipitationTotal
          <variable>_low, <variable>_high  for range series, e.g. solarRadiationRange_low
        Variables with no data are left out. Units and request details are in df.attrs.
        """
        payload = self.get_station_history_raw(station_id, resolution, period, aggregate, start, end)
        return self._series_to_dataframe(payload)

    # ---- soil and wind ----
    def get_soil_history(self, station_id, resolution, period, measurement="soil-moisture", aggregate="avg",
                         start=None, end=None):
        """
        Soil history as a DataFrame: timestamp_utc, timestamp_local, one column per sensor depth, and precipitation.
        Depth columns are named <measurement>_<depth>in, e.g. soil_moisture_2in, bare_soil_temperature_20in.
        Units, depths and request details are in df.attrs.
        """
        payload = self.get_soil_history_raw(station_id, resolution, period, measurement, aggregate, start, end)
        return self._soil_to_dataframe(payload)

    def get_wind_history(self, station_id, resolution, period, wind_source="wind-3m", aggregate="avg",
                         start=None, end=None):
        """
        Wind time series as a DataFrame: timestamp_utc, timestamp_local, wind, windRange_low, windRange_high, gust.
        Units and request details are in df.attrs.
        """
        payload = self.get_wind_history_raw(station_id, resolution, period, wind_source, aggregate, start, end)
        data = payload.get("data", {}) or {}
        df = self._series_dict_to_dataframe(data.get("lineSeries", {}) or {}, data.get("timezone"))
        self._attach_attrs(df, payload, ("windSource", "lineSampleCount"))
        return df

    def get_wind_rose(self, station_id, resolution, period, wind_source="wind-3m", aggregate="avg",
                      start=None, end=None):
        """
        Wind rose as a DataFrame: one row per compass direction (N, NNE, ...), one column per speed bin,
        values = percent of samples. Bin edges are in df.attrs["speed_bins"].
        """
        import pandas as pd
        payload = self.get_wind_history_raw(station_id, resolution, period, wind_source, aggregate, start, end)
        data = payload.get("data", {}) or {}
        unit = (payload.get("meta") or {}).get("speedUnit", "")
        bins = data.get("series", []) or []
        df = pd.DataFrame({f"{b.get('name')} {unit}".strip(): b.get("data") for b in bins},
                          index=data.get("directions", []))
        df.index.name = "direction"
        self._attach_attrs(df, payload, ("windSource",))
        df.attrs["speed_bins"] = [{"label": b.get("name"), "from": b.get("from"), "to": b.get("to")} for b in bins]
        return df

    def get_soil_history_raw(self, station_id, resolution, period, measurement="soil-moisture", aggregate="avg",
                             start=None, end=None):
        """Raw JSON from /stations/{id}/soil-moisture-chart."""
        self._check("resolution", resolution, RESOLUTIONS)
        self._check("period", period, PERIODS)
        self._check_period_for_resolution(resolution, period)
        self._check("measurement", measurement, SOIL_MEASUREMENTS)
        self._check("aggregate", aggregate, AGGREGATES)
        self._check_custom(period, start, end)
        return self._fetch_json(f"{NE_MESONET_BASE_URL}/api/frontend/stations/{station_id}/soil-moisture-chart",
                                {"resolution": resolution, "period": period, "measurement": measurement,
                                 "aggregate": aggregate, "start": start, "end": end})

    def get_wind_history_raw(self, station_id, resolution, period, wind_source="wind-3m", aggregate="avg",
                             start=None, end=None):
        """Raw JSON from /stations/{id}/wind-rose."""
        self._check("resolution", resolution, RESOLUTIONS)
        self._check("period", period, PERIODS)
        self._check_period_for_resolution(resolution, period)
        self._check("wind_source", wind_source, WIND_SOURCES)
        self._check("aggregate", aggregate, AGGREGATES)
        self._check_custom(period, start, end)
        return self._fetch_json(f"{NE_MESONET_BASE_URL}/api/frontend/stations/{station_id}/wind-rose",
                                {"resolution": resolution, "period": period, "windSource": wind_source,
                                 "aggregate": aggregate, "start": start, "end": end})

    # ---- camera images ----
    def get_station_images(self, page_station_id):
        """
        Latest camera images for every station, read from the all-station-images attribute embedded in a station
        page (any station's page carries the full table). Returns {station_id: [image, ...]}, each image:
          {"direction": "North", "direction_code": "n", "url": "...webp", "last_modified_utc": "2026-10-05T04:30:05Z"}
        Stations without cameras map to an empty list. Images are "latest only"; there is no archive.
        """
        page = self._fetch_text(self.station_page_url(page_station_id))
        m = re.search(r""":?all-station-images=(['"])(.*?)\1""", page, re.S)
        if not m:
            raise ValueError("Camera image table (all-station-images) not found on the station page.")
        raw = json.loads(html.unescape(m.group(2)))
        out = {}
        for sid, items in raw.items():
            out[int(sid)] = [{
                "direction":         (it.get("direction") or {}).get("label", ""),
                "direction_code":    (it.get("direction") or {}).get("value", ""),
                "url":               it.get("url", ""),
                "last_modified_utc": it.get("lastModified", ""),
            } for it in (items or [])]
        return out

    def download_bytes(self, url):
        """Raw bytes of a file, e.g. a camera image."""
        import requests
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)"}
        resp = requests.get(url, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.content

    # ---- parameter checks ----
    @staticmethod
    def _check(name, value, allowed):
        if value not in allowed:
            raise ValueError(f"{name}={value!r} is not valid. Use one of: {', '.join(allowed)}")

    @staticmethod
    def _check_custom(period, start, end):
        if period == "custom" and (start is None or end is None):
            raise ValueError("period='custom' requires both start and end (YYYY-MM-DD).")

    def _check_period_for_resolution(self, resolution, period):
        if not self.enforce_period_rules:
            return
        allowed = PERIODS_BY_RESOLUTION.get(resolution)
        if allowed is not None and period not in allowed:
            raise ValueError(f"period={period!r} is not accepted with resolution={resolution!r}. "
                             f"Allowed: {', '.join(allowed)}. (Rules observed 2026-10-05; "
                             f"use enforce_period_rules=False if the server has changed.)")

    # ---- fetch ----
    def _fetch_text(self, url):
        import requests
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)"}
        resp = requests.get(url, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.text

    def _fetch_json(self, url, params=None):
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)", "Accept": "application/json"}
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers=headers)
            text = urllib.request.urlopen(req, timeout=self.timeout_sec).read().decode("utf-8", errors="replace")
            return json.loads(text)
        except Exception as e1:
            if self.log_fn:
                self.log_fn(f"urllib fetch failed ({e1.__class__.__name__}); trying requests\u2026")
        import requests
        resp = requests.get(url, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.json()

    # ---- parse ----
    def _series_to_dataframe(self, payload):
        data = payload.get("data", {}) or {}
        df = self._series_dict_to_dataframe(data.get("series", {}) or {}, data.get("timezone"))
        self._attach_attrs(df, payload)
        return df

    @staticmethod
    def _series_dict_to_dataframe(series, timezone):
        """
        {name: [[ms, value], ...]} value series and {name: [{"x": ms, "low": v, "high": v}, ...]} range series
        -> one DataFrame with timestamp_utc, timestamp_local and one column per series (range: _low and _high).
        Empty series are left out.
        """
        import pandas as pd
        columns = {}
        for name, points in series.items():
            if not points:
                continue
            if isinstance(points[0], dict):
                idx = [p.get("x") for p in points]
                columns[f"{name}_low"] = pd.Series([p.get("low") for p in points], index=idx)
                columns[f"{name}_high"] = pd.Series([p.get("high") for p in points], index=idx)
            else:
                columns[name] = pd.Series([p[1] for p in points], index=[p[0] for p in points])
        return NEMesonet._add_timestamps(pd.DataFrame(columns).sort_index(), timezone)

    @staticmethod
    def _add_timestamps(df, timezone):
        """Index of Unix milliseconds -> timestamp_utc and timestamp_local columns, plain integer index."""
        import pandas as pd
        timestamp_utc = pd.to_datetime(df.index, unit="ms", utc=True)
        df.insert(0, "timestamp_utc", timestamp_utc)
        if timezone:
            df.insert(1, "timestamp_local", timestamp_utc.tz_convert(timezone))
        return df.reset_index(drop=True)

    @staticmethod
    def _attach_attrs(df, payload, extra_request_keys=()):
        data = payload.get("data", {}) or {}
        meta = payload.get("meta") or {}
        df.attrs["units"] = {k: v for k, v in meta.items() if k.endswith("Unit")}
        keys = ("stationId", "timezone", "resolution", "period", "aggregate", "start", "end", "sampleCount")
        df.attrs["request"] = {k: data.get(k) for k in keys + tuple(extra_request_keys)}

    def _soil_to_dataframe(self, payload):
        """points = [[ms, depth_index, value], ...]; depth_index points into data["depths"] (inches)."""
        import pandas as pd
        data = payload.get("data", {}) or {}
        depths = data.get("depths", []) or []
        prefix = str(data.get("measurement") or "soil").replace("-", "_")

        columns = {}
        for ms, depth_index, value in data.get("points", []) or []:
            depth = depths[depth_index] if 0 <= depth_index < len(depths) else depth_index
            columns.setdefault(f"{prefix}_{depth}in", {})[ms] = value
        precip = data.get("precipitation", []) or []
        if precip:
            columns["precipitation"] = {p[0]: p[1] for p in precip}

        df = self._add_timestamps(pd.DataFrame(columns).sort_index(), data.get("timezone"))
        self._attach_attrs(df, payload, ("measurement", "readingCount"))
        df.attrs["depths_in"] = depths
        return df

    def _flatten(self, s):
        sensors = s.get("sensors") or {}
        soil = sensors.get("hasSoilSensors") or []
        soil_depths_in = [str(x.get("depth")) for x in soil if x.get("enabled")]

        elevation_m = s.get("elevation")
        elevation_ft = elevation_m * METERS_TO_FEET if isinstance(elevation_m, (int, float)) else ""

        return {
            "station_id":            s.get("id"),
            "station":               s.get("name") or "",
            "station_url":           self.station_page_url(s.get("id")) if s.get("id") is not None else "",
            "nwsli":                 s.get("nwsli") or "",
            "county":                s.get("county") or "",
            "nrd":                   s.get("nrd") or "",
            "huc6":                  s.get("huc6") or "",
            "huc8":                  s.get("huc8") or "",
            "rfc_region":            s.get("rfc_region") or "",
            "lat":                   s.get("latitude"),
            "lon":                   s.get("longitude"),
            "elevation_m":           elevation_m if elevation_m is not None else "",
            "elevation_ft":          elevation_ft,
            "timezone":              s.get("timezone") or "",
            "sponsored_by":          s.get("sponsoredBy") or "",
            "has_10m_temp_humidity": bool(sensors.get("hasTenMeterTempHumidity")),
            "has_10m_wind":          bool(sensors.get("hasTenMeterWind")),
            "soil_sensor_depths_in": ", ".join(soil_depths_in),
        }
