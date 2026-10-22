# geomaps/SDMESONET.py
"""
SD Mesonet station table scraper.

Scrapes the live SD State Mesonet station page and returns a pandas DataFrame
of stations (station, nwsli, county, lat, lon, elv_ft, status) for map pins.
Self-contained: does not depend on the dataset manager.

Also reads, for the SD Mesonet plugin, the 48-hour 5-minute history behind each station's History page
(/history/h.asp?num=<code>) and the camera images listed on each station's dashboard. These are internal,
unpublished website endpoints that may change without notice. SD Mesonet terms: credit required, no
redistribution, no commercial use, data provisional.
"""
import re
import json

from ..app_identity import APP_NAME

SD_MESONET_BASE_URL  = "https://climate.sdstate.edu"
MESONET_STATIONS_URL = f"{SD_MESONET_BASE_URL}/information/stations/"
SD_HISTORY_URL       = f"{SD_MESONET_BASE_URL}/history/h.asp"
SD_DASHBOARD_URL     = f"{SD_MESONET_BASE_URL}/weather/"

# The station table gives standard-time UTC offsets; both zones observe daylight saving time.
TIMEZONE_BY_UTC_OFFSET = {"-6": "America/Chicago", "-7": "America/Denver"}

# History series key -> column name. Units are appended from each series' own "unit" field.
HISTORY_COLUMN_NAMES = {
    "temp":        "air_temperature",
    "appTemp":     "apparent_temperature",          # wind chill or heat index; empty when neither applies
    "dewpoint":    "dew_point",
    "rh":          "relative_humidity",
    "windspeed":   "wind_speed",
    "windgust":    "wind_gust",
    "rainfall":    "precipitation",
    "accrainfall": "precipitation_accumulated",
    "radiation":   "solar_radiation",
    "rso":         "clear_sky_radiation",
    "skyPercent":  "sky_percent_of_clear",
    "stnpressure": "station_pressure",
    "sd":          "snow_depth",
    "inversion":   "inversion",
    "xinversion":  "inversion_x",                   # second inversion series; the site does not explain it
}
UNIT_SUFFIX = {"f": "F", "%": "pct", "mph": "mph", "inches": "in", "w/m2": "W_m2", "in hg": "inHg"}


class SDMesonet:
    def __init__(self, log_fn=None):
        self.log_fn = log_fn

    # ---- public ----
    def get_dataframe(self):
        """Return a DataFrame of SD Mesonet stations with numeric lat/lon."""
        import pandas as pd
        html = self._fetch_html(MESONET_STATIONS_URL)
        rows = self._parse_tables(html)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["lat"].apply(lambda v: isinstance(v, (int, float)))]
            df = df[df["lon"].apply(lambda v: isinstance(v, (int, float)))]
            df = df.reset_index(drop=True)
        return df

    # ---- station codes and time zones ----
    @staticmethod
    def station_code(station_url):
        """Numeric code used by the dashboard, history and camera pages (…/weather/?num=778 -> 778)."""
        m = re.search(r"[?&]num=(\d+)", station_url or "")
        return int(m.group(1)) if m else None

    @staticmethod
    def timezone_for(utc_offset):
        return TIMEZONE_BY_UTC_OFFSET.get(str(utc_offset).strip())

    # ---- 48-hour history ----
    def get_history_raw(self, code):
        """Raw JSON behind the station History page: the last 48 hours at 5-minute steps."""
        return json.loads(self._fetch_html(f"{SD_HISTORY_URL}?num={code}"))

    def get_history(self, code, timezone):
        """
        48-hour history as a DataFrame: timestamp_utc, timestamp_local, one column per series named
        <variable>_<unit>, e.g. air_temperature_F, precipitation_in. Empty series values become empty cells.
        The site's timestamps are the station's local clock time written as if UTC, so they are localized
        to `timezone` (America/Chicago or America/Denver) before conversion to UTC.
        Units and the station generation are in df.attrs.
        """
        payload = self.get_history_raw(code)
        return self._history_to_dataframe(payload, timezone)

    @staticmethod
    def _column_name(key, unit):
        base = HISTORY_COLUMN_NAMES.get(key, re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower())
        suffix = UNIT_SUFFIX.get(str(unit or "").strip().lower(), re.sub(r"[^A-Za-z0-9]+", "_", str(unit or "")))
        return f"{base}_{suffix}".rstrip("_")

    def _history_to_dataframe(self, payload, timezone):
        import pandas as pd
        sets = (payload.get("datasets") or [{}])[0] or {}
        columns, units = {}, {}
        for key, series in sets.items():
            points = series.get("data") or []
            if not points:
                continue
            name = self._column_name(key, series.get("unit"))
            units[name] = series.get("unit", "")
            columns[name] = pd.Series([p[1] for p in points], index=[p[0] for p in points])
        df = pd.DataFrame(columns).sort_index()
        local_naive = pd.to_datetime(df.index, unit="ms")
        try:
            local = local_naive.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            local = local_naive.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
        df.insert(0, "timestamp_local", local)
        df.insert(0, "timestamp_utc", local.tz_convert("UTC"))
        df = df.reset_index(drop=True)
        df.attrs["units"] = units
        df.attrs["station_generation"] = payload.get("gen")
        return df

    # ---- camera images ----
    def get_station_images(self, code):
        """
        Camera views listed on the station dashboard. Returns a list (empty = no cameras):
          {"view": 1, "direction": "Southwest", "still_url": ".../mostrecent1.jpg", "timelapse_url": ".../current1.gif"}
        Stills update every 5 minutes, time-lapse loops every 30 minutes; latest only, no archive.
        """
        page = self._fetch_html(f"{SD_DASHBOARD_URL}?num={code}")
        stills = {int(n): f"{SD_MESONET_BASE_URL}{path}" for path, n in
                  re.findall(r"(/pictures/[^/'\"?]+/mostrecent(\d+)\.jpg)", page)}
        loops = {}
        for path, n, label in re.findall(r"href=['\"](/pictures/[^/'\"?]+/current(\d+)\.gif)[^'\"]*['\"]>\s*([^<]*?)\s*Time Lapse",
                                         page, re.I):
            loops[int(n)] = (f"{SD_MESONET_BASE_URL}{path}", label.strip())
        views = []
        for n in sorted(set(stills) | set(loops)):
            gif, label = loops.get(n, ("", ""))
            views.append({"view": n, "direction": label or f"View {n}",
                          "still_url": stills.get(n, ""), "timelapse_url": gif})
        return views

    def download_file(self, url, timeout_sec=30):
        """(bytes, Last-Modified header or "") for an image or time-lapse file."""
        import requests
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)"}
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
        resp.raise_for_status()
        return resp.content, resp.headers.get("Last-Modified", "")

    # ---- fetch ----
    def _fetch_html(self, url):
        headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME} geomaps)"}
        try:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
        except Exception as e1:
            if self.log_fn:
                self.log_fn(f"urllib fetch failed ({e1.__class__.__name__}); trying requests\u2026")
        import requests
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    # ---- parse ----
    def _parse_tables(self, html):
        active_marker = html.find("Active Stations")
        inactive_marker = html.find("Inactive Stations")
        if active_marker == -1 or inactive_marker == -1 or inactive_marker <= active_marker:
            active_html, inactive_html = html, ""
        else:
            active_html = html[active_marker:inactive_marker]
            inactive_html = html[inactive_marker:]
        rows = []
        rows.extend(self._parse_one_table(active_html, "Active"))
        rows.extend(self._parse_one_table(inactive_html, "Inactive"))
        return rows

    def _parse_one_table(self, section_html, status):
        out = []
        table_match = re.search(r"<table.*?</table>", section_html, re.S | re.I)
        if not table_match:
            return out
        table_html = table_match.group(0)
        tr_blocks = re.findall(r"<tr.*?</tr>", table_html, re.S | re.I)
        if not tr_blocks:
            return out

        header_cells = re.findall(r"<t[hd].*?>(.*?)</t[hd]>", tr_blocks[0], re.S | re.I)
        header_cells = [re.sub(r"<.*?>", "", c).strip().lower() for c in header_cells]

        def col_index(*names):
            for n in names:
                for i, h in enumerate(header_cells):
                    if n in h:
                        return i
            return None

        idx = {
            "station": col_index("station"),
            "nwsli":   col_index("nwsli"),
            "detail":  col_index("detail"),
            "county":  col_index("county"),
            "start":   col_index("start"),
            "end":     col_index("end"),
            "lat":     col_index("lat"),
            "lon":     col_index("lon"),
            "elv":     col_index("elv"),
            "tz":      col_index("time zone", "timezone"),
        }

        for tr in tr_blocks[1:]:
            cells = re.findall(r"<t[hd].*?>(.*?)</t[hd]>", tr, re.S | re.I)
            if not cells:
                continue
            clean = []
            for c in cells:
                text = re.sub(r"<.*?>", "", c)
                text = (text.replace("&amp;", "&").replace("&nbsp;", " ")
                             .replace("&#39;", "'").strip())
                clean.append(text)

            def get(k):
                i = idx[k]
                return clean[i] if i is not None and i < len(clean) else ""

            def num(k):
                v = get(k)
                try:
                    return float(v) if v not in ("", None) else ""
                except ValueError:
                    return ""

            # Pull the station-column hyperlink from the raw cell HTML.
            station_url = ""
            si = idx["station"]
            if si is not None and si < len(cells):
                m = re.search(r'href=["\']([^"\']+)["\']', cells[si], re.I)
                if m:
                    station_url = m.group(1).replace("&amp;", "&")

            row = {
                "station": get("station"),
                "station_url": station_url,
                "nwsli":   get("nwsli"),
                "detail":  get("detail"),
                "county":  get("county"),
                "start":   get("start"),
                "end":     get("end"),
                "lat":     num("lat"),
                "lon":     num("lon"),
                "elv_ft":  num("elv"),
                "utc_offset": get("tz"),
                "status":  status,
                "active":  (status == "Active"),
            }
            if row["station"] or row["nwsli"]:
                out.append(row)
        return out
