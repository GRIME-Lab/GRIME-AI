# geomaps/NEMESONET.py
"""
Nebraska Mesonet station table.

Reads the station list the Nebraska Mesonet website uses (nemesonet.unl.edu/api/frontend/stations) and returns a
pandas DataFrame of stations for map pins. Self-contained: does not depend on the dataset manager.

The endpoint is internal to the website; its responses say it "is not intended for public use and may change
without notice".
"""
import json

from ..app_identity import APP_NAME

NE_MESONET_BASE_URL     = "https://nemesonet.unl.edu"
NE_MESONET_STATIONS_URL = f"{NE_MESONET_BASE_URL}/api/frontend/stations"

METERS_TO_FEET = 3.28084


class NEMesonet:
    def __init__(self, log_fn=None, timeout_sec=30):
        self.log_fn = log_fn
        self.timeout_sec = timeout_sec

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
        return f"{NE_MESONET_BASE_URL}/?station={station_id}"

    # ---- fetch ----
    def _fetch_json(self, url):
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)", "Accept": "application/json"}
        try:
            import urllib.request
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
