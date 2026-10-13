#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
AmeriFlux_API: a lightweight read-only client for the AmeriFlux web API, kept in
its own class so the PhenoCam co-location plugin can ask "what sensors does this
AmeriFlux site measure?" without carrying AmeriFlux's specifics.

Endpoints (derived from the amerifluxr R package, R/zzz.R amf_server()):
  Base API:   https://amfcdn.lbl.gov/api/v1
    sites         GET  /site_display/AmeriFlux
    data years    GET  /data_availability/AmeriFlux/<product>/<policy>
    FP variables  GET  /fp_var?limits=True
    data policy   GET  /site_availability/AmeriFlux/BIF/CCBY4.0
  Per-site variable (sensor) availability is a CSV published with amerifluxr:
    https://raw.githubusercontent.com/chuhousen/amerifluxr/master/data-summary/
        AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_LATEST.csv   (its first line is a title; real header is line 2)

The metadata and availability calls above are public (no key). Only downloading
the actual BASE/FLUXNET data needs an AmeriFlux account, which this client does
not do: it reports which sensors exist and over what years, not the data itself.
"""

import io
import re
from pathlib import Path

import pandas as pd

from appcore.http_client import TimeoutSession


class AmeriFlux_API:
    """Read-only AmeriFlux metadata / variable-availability client (DataFrames)."""

    BASE_API = "https://amfcdn.lbl.gov/api/v1"
    SITES_URL = f"{BASE_API}/site_display/AmeriFlux"
    DATA_YEARS_URL = f"{BASE_API}/data_availability/AmeriFlux"   # + /<product>/<policy>
    FP_VARIABLES_URL = f"{BASE_API}/fp_var?limits=True"
    # Per-site BASE variable availability, maintained alongside the amerifluxr package.
    VARIABLE_AVAILABILITY_URL = ("https://raw.githubusercontent.com/chuhousen/amerifluxr/"
                                 "master/data-summary/AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_LATEST.csv")
    DATA_DOWNLOAD_URL = f"{BASE_API}/data_download"
    SITE_CCBY4_URL = f"{BASE_API}/site_availability/AmeriFlux/BIF/CCBY4.0"

    # intended_use categories AmeriFlux accepts (the verbose string is what gets sent).
    INTENDED_USE = {
        "synthesis":      "Research - Multi-site synthesis",
        "remote_sensing": "Research - Remote sensing",
        "model":          "Research - Land model/Earth system model",
        "other_research": "Research - Other",
        "education":      "Education (Teacher or Student)",
        "other":          "Other",
    }

    def __init__(self):
        # Shared session with a timeout + bounded retries, like Phenocam_API, so a
        # silent server cannot block a call forever.
        self.session = TimeoutSession()

    # ------------------------------------------------------------------------------------------------------------------
    def _get_json(self, url):
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------------------------------------------------------
    def get_sites(self) -> pd.DataFrame:
        """Every AmeriFlux site with general info. Lat/Lon/Elev are flattened and renamed to
        LOCATION_LAT / LOCATION_LONG / LOCATION_ELEV to match amerifluxr's output."""
        df = pd.json_normalize(self._get_json(self.SITES_URL))
        renames = {
            "GRP_LOCATION.LOCATION_LAT": "LOCATION_LAT",
            "GRP_LOCATION.LOCATION_LONG": "LOCATION_LONG",
            "GRP_LOCATION.LOCATION_ELEV": "LOCATION_ELEV",
            "GRP_CLIM_AVG.MAT": "MAT",
            "GRP_CLIM_AVG.MAP": "MAP",
            "GRP_CLIM_AVG.CLIMATE_KOEPPEN": "CLIMATE_KOEPPEN",
        }
        df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
        if "SITE_ID" in df.columns:
            df = df.sort_values("SITE_ID").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------------------------------------------------------
    def get_data_coverage(self, data_product: str = "BASE-BADM", data_policy: str = "CCBY4.0") -> pd.DataFrame:
        """Per-site published data years: SITE_ID, URL, publish_years (list of YYYY)."""
        url = f"{self.DATA_YEARS_URL}/{data_product}/{data_policy}"
        df = pd.json_normalize(self._get_json(url))
        if "SITE_ID" in df.columns:
            df = df.sort_values("SITE_ID").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------------------------------------------------------
    def get_ccby4_site_ids(self) -> set:
        """SITE_IDs whose data is published under the CC-BY-4.0 policy. Every other AmeriFlux site
        is under the older LEGACY policy."""
        df = pd.json_normalize(self._get_json(self.SITE_CCBY4_URL))
        if df.empty:
            return set()
        col = "SITE_ID" if "SITE_ID" in df.columns else df.columns[0]
        return set(df[col].astype(str))

    def get_fp_variables(self) -> pd.DataFrame:
        """AmeriFlux FP (Flux-Processing) standard variable definitions:
        Name, Description, Units, Min, Max, Type."""
        return pd.json_normalize(self._get_json(self.FP_VARIABLES_URL))

    # ------------------------------------------------------------------------------------------------------------------
    def get_variable_availability(self) -> pd.DataFrame:
        """Per-site BASE variable availability, in long form: one row per SITE_ID and VARIABLE,
        with columns SITE_ID, VARIABLE, BASENAME, GAP_FILLED and a Y<year> column per year holding
        the fraction of that year with data (0 = not measured). The CSV's first line is a title,
        so the real header is the second line."""
        resp = self.session.get(self.VARIABLE_AVAILABILITY_URL)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text), skiprows=1)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _has_value(v) -> bool:
        """Treat an availability cell as "the site measures this variable" when it is present
        and not zero. Blanks, NaN, 0 and "0" mean not measured."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        s = str(v).strip()
        return s not in ("", "0", "0.0", "nan", "NaN", "NA", "False")

    # ------------------------------------------------------------------------------------------------------------------
    def site_variables(self, site_id: str, availability: pd.DataFrame = None,
                       variables: pd.DataFrame = None) -> list:
        """The variables (sensors) a site measures, as a list of dicts with name, description,
        units and the first/last year each has data. The availability file is long form (one row
        per site and variable, a column per year), so this gathers the site's rows and groups them
        by base measurement name. Pass preloaded `availability` (get_variable_availability) and
        `variables` (get_fp_variables) to avoid refetching them per site."""
        if availability is None:
            availability = self.get_variable_availability()
        if "SITE_ID" not in availability.columns:
            return []
        rows = availability[availability["SITE_ID"].astype(str) == str(site_id)]
        if rows.empty:
            return []

        year_cols = [c for c in availability.columns if re.fullmatch(r"Y\d{4}", str(c))]

        defs = {}
        if variables is None:
            variables = self.get_fp_variables()
        if "Name" in variables.columns:
            for _, r in variables.iterrows():
                defs[str(r["Name"])] = (str(r.get("Description", "") or ""), str(r.get("Units", "") or ""))

        grouped = {}   # base measurement name -> set of years with data
        for _, r in rows.iterrows():
            name = str(r.get("BASENAME") or r.get("VARIABLE") or "").strip()
            if not name:
                continue
            years = grouped.setdefault(name, set())
            for c in year_cols:
                if self._has_value(r[c]):
                    years.add(int(str(c)[1:]))

        out = []
        for name in sorted(grouped):
            years = sorted(grouped[name])
            desc, units = defs.get(name, ("", ""))
            out.append({"name": name, "description": desc, "units": units,
                        "first_year": years[0] if years else None,
                        "last_year": years[-1] if years else None})
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Actual data download (CO2 / methane / water-vapor / energy fluxes). Needs an AmeriFlux account.
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _first_method(mgr, names):
        """Return the first non-empty result from the first matching getter on the key manager."""
        for n in names:
            fn = getattr(mgr, n, None)
            if callable(fn):
                try:
                    v = fn()
                    if v:
                        return str(v)
                except Exception:
                    pass
        return ""

    def _load_credentials(self):
        """AmeriFlux downloads need a registered user id + email. Load them from the app's key
        storage (APIKeyManager), the same place the USGS and NEON keys live, rather than asking
        the caller to pass them in."""
        mgr = None
        for path in ("appcore.dialogs.api_keys.api_key_manager", "appcore.dialogs.api_keys"):
            try:
                module = __import__(path, fromlist=["APIKeyManager"])
                mgr = module.APIKeyManager()
                break
            except Exception:
                continue
        if mgr is None:
            return "", ""
        user_id = self._first_method(mgr, ["get_ameriflux_user_id", "get_ameriflux_userid",
                                           "get_ameriflux_user", "get_ameriflux_username"])
        email = self._first_method(mgr, ["get_ameriflux_email", "get_ameriflux_user_email"])
        return user_id, email

    def download_base(self, site_ids, save_dir=None, intended_use="other_research",
                      intended_use_text="GRIME AI flux co-location analysis",
                      data_product="BASE-BADM", data_policy="auto", extract=True) -> list:
        """Request AmeriFlux BASE data (CO2 flux FC, methane FCH4, water-vapor / latent-heat LE,
        sensible heat H, net radiation NETRAD, and so on) for one or more sites.

        The AmeriFlux user id and email come from the app's key storage, not from arguments.
        With save_dir, each returned file is downloaded there (zips are extracted when extract is
        True) and the saved paths are returned; without save_dir, the download URLs are returned.

        Downloading actual data requires a free AmeriFlux account and acceptance of the data
        policy; the metadata and variable-availability methods above do not.
        """
        if isinstance(site_ids, str):
            site_ids = [site_ids]
        site_ids = [s for s in site_ids if s]
        if not site_ids:
            raise ValueError("download_base needs at least one AmeriFlux site id.")

        use_text = self.INTENDED_USE.get(intended_use)
        if use_text is None:
            raise ValueError(f"intended_use must be one of {sorted(self.INTENDED_USE)}.")

        user_id, user_email = self._load_credentials()
        if not user_id or not user_email:
            raise RuntimeError("AmeriFlux credentials are not set. Add your AmeriFlux user id and email "
                               "in the API keys dialog (the same place as the USGS and NEON keys).")

        # Each site publishes under CC-BY-4.0 or the older LEGACY policy. "auto" looks up which and
        # requests each group under its own policy, so the caller does not have to know or guess.
        if data_policy == "auto":
            ccby4 = self.get_ccby4_site_ids()
            groups = {"CCBY4.0": [s for s in site_ids if s in ccby4],
                      "LEGACY":  [s for s in site_ids if s not in ccby4]}
        else:
            groups = {data_policy: site_ids}

        urls = []
        for policy, group_ids in groups.items():
            if not group_ids:
                continue
            payload = {
                "user_id": user_id,
                "user_email": user_email,
                "data_product": data_product,
                "data_policy": policy,
                "site_ids": group_ids,
                "intended_use": use_text,
                "description": f"{intended_use_text} [GRIME AI download]",
                "is_test": "",
            }
            resp = self.session.post(self.DATA_DOWNLOAD_URL, json=payload,
                                     headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            urls += [d.get("url") for d in (resp.json().get("data_urls") or []) if d.get("url")]
        if not urls:
            raise RuntimeError(f"AmeriFlux returned no data for {', '.join(site_ids)}.")
        if save_dir is None:
            return urls

        saved = []
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            name = url.rstrip("/").split("/")[-1] or "ameriflux_download"
            dest = out_dir / name
            with self.session.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            saved.append(str(dest))
            if extract and name.lower().endswith(".zip"):
                import zipfile
                try:
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(out_dir)
                except zipfile.BadZipFile:
                    pass
        return saved
