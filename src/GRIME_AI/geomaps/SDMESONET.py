# geomaps/SDMESONET.py
"""
SD Mesonet station table scraper.

Scrapes the live SD State Mesonet station page and returns a pandas DataFrame
of stations (station, nwsli, county, lat, lon, elv_ft, status) for map pins.
Self-contained: does not depend on the dataset manager.
"""
import re

MESONET_STATIONS_URL = "https://climate.sdstate.edu/information/stations/"


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

    # ---- fetch ----
    def _fetch_html(self, url):
        headers = {"User-Agent": "Mozilla/5.0 (GRIME-AI geomaps)"}
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
