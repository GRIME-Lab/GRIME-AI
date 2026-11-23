#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# Input / Output
import csv
import multiprocessing
# OS / system
import os
import shutil
import tempfile
import time
# Error management
import traceback
from io import StringIO

# Third-party
import requests
from PyQt5.QtGui import QPixmap
from bs4 import BeautifulSoup
from neonutilities import zips_by_product, stack_by_table

# Timeouts and bounded retries for every request come from the app-wide HTTP policy
from appcore.http_client import get as http_get, TimeoutSession, IMAGE_TIMEOUT
from appcore.JSON_Editor import JsonEditor
# GRIME AI MODULES
from appcore.QProgressWheel import QProgressWheel
from appcore.App_Utils import App_Utils
from appcore.nitrateData import nitrateData
from appcore.siteData import siteData
from appcore.dialogs.api_keys import APIKeyManager
from ..app_identity import DOWNLOADS_DIR

SERVER = 'https://data.neonscience.org/api/v0/'  # default; overridden per-instance by APIKeyManager

# https://www.neonscience.org/sites/default/files/NEON_Field_Site_Metadata_20240423.csv

# ======================================================================================================================
# ======================================================================================================================
# =====    =====     =====     =====     =====       HELPER FUNCTIONS      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================

def parse_NEON_field_site_metadata(filename_with_path):
    """
    Module: parse_NEON_field_site_metadata
    Author: John Edward Stranzl, Jr.
    Created: 2021-04-xx
    License: Apache License 2.0

        This file is licensed under the Apache License, Version 2.0.
        You may not use this file except in compliance with the License.
        You may obtain a copy of the License at:
            http://www.apache.org/licenses/LICENSE-2.0

    Description:
        Parses the NEON field site metadata CSV (downloaded by NEON_API.Download_Field_Site_Metadata)
        and extracts the following fields for each site:
          - site_id
          - site_name
          - phenocams
          - latitude
          - longitude
        It automatically handles headers that may be prefixed with 'field_'.

    Dependencies:
        - csv (stdlib)
        - is_valid_csv (custom validator for CSV integrity)
        - find_field_index (helper to locate columns with or without 'field_' prefix)
        - siteData (data class for site attributes)

    Function:
        parse_NEON_field_site_metadata(filename_with_path) -> List[siteData]
            Parse the CSV at the given path and return a list of siteData objects.
            Raises ValueError if the CSV fails validation. Returns an empty list
            on any parsing error.

    Usage Example:
        from appcore.neon.NEON_API import parse_NEON_field_site_metadata

        try:
            sites = parse_NEON_field_site_metadata("/data/NEON_Field_Site_Metadata.csv")
        except ValueError as e:
            print(f"Validation error: {e}")
    """
    # Validate before parsing
    if not is_valid_csv(filename_with_path):
        raise ValueError(f"Invalid CSV content detected in '{filename_with_path}'.")

    rows = []
    siteList = []

    try:
        # Read all rows into memory (you could stream this if files get huge)
        with open(filename_with_path, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            # Extract header
            fields = next(csvreader)
            # Buffer the rest of the rows
            for row in csvreader:
                rows.append(row)

        # Locate the needed columns, with or without 'field_' prefix
        site_id_idx = find_field_index(fields, "site_id")
        site_name_idx = find_field_index(fields, "site_name")
        phenocam_idx = find_field_index(fields, "phenocams")
        lat_idx = find_field_index(fields, "latitude")
        lon_idx = find_field_index(fields, "longitude")

        # Build your siteData objects
        for row in rows:
            siteList.append(
                siteData(
                    row[site_id_idx],
                    row[site_name_idx],
                    row[phenocam_idx],
                    row[lat_idx],
                    row[lon_idx]
                )
            )

    except Exception:
        # Swallow any parsing errors and return empty list
        siteList = []

    return siteList


def is_valid_csv(source, sniff_lines=5, max_rows=10, encoding="utf-8-sig"):
    """
    Function: is_valid_csv
    Author: John Edward Stranzl, Jr.
    Created: 2025-09-04
    License: Apache License 2.0

        This file is licensed under the Apache License, Version 2.0.
        You may not use this file except in compliance with the License.
        You may obtain a copy of the License at:
            http://www.apache.org/licenses/LICENSE-2.0

    Description:
        Determine whether the provided source (a filesystem path or raw CSV text)
        constitutes a well-formed CSV. Checks include:
          - Quick HTML rejection
          - Sniffing a sample of lines for dialect
          - Consistent column counts across up to max_rows
          - At least a header plus one data row

    Dependencies:
        - os (stdlib)
        - csv (stdlib)
        - io.StringIO (stdlib)

    Parameters:
        source       Path to a CSV file or a raw CSV string.
        sniff_lines  Number of initial lines to sample for dialect sniffing (default: 5).
        max_rows     Maximum number of rows to read for validation (default: 10).
        encoding     Encoding used when reading a file source (default: "utf-8-sig").

    Returns:
        bool         True if source passes all CSV validity checks; False otherwise.
    """

    # Load the file’s contents if 'source' is a valid path
    if isinstance(source, str) and os.path.exists(source):
        with open(source, 'r', encoding=encoding) as f:
            text = f.read()
    else:
        text = source

    # Strip BOM if present
    if text.startswith('\ufeff'):
        text = text.lstrip('\ufeff')

    # Quick HTML rejection
    head = text[:2048].lower()
    for sig in ['<!doctype html', '<html', '<head', '<body', '<script']:
        if sig in head:
            return False

    # Build a sample of complete lines for sniffing
    lines = text.splitlines(keepends=True)
    sample = ''.join(lines[:sniff_lines])
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        # Fallback to basic CSV with commas
        dialect = csv.excel
        dialect.delimiter = ','
        dialect.quotechar = '"'
        dialect.doublequote = True

    # Read up to max_rows and collect them
    reader = csv.reader(StringIO(text), dialect)
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        rows.append(row)

    # Need at least header + one data row
    if len(rows) < 2:
        return False

    expected_cols = len(rows[0])
    if expected_cols < 1:
        return False

    # Ensure every sampled row matches the header’s column count
    return all(len(r) == expected_cols for r in rows)


def find_field_index(fields, base_name):
    """
    Return the index of either 'field_<base_name>' or '<base_name>' in the header list.
    Raises ValueError if neither is present.
    """
    for prefix in ("field_", ""):
        key = prefix + base_name
        if key in fields:
            return fields.index(key)
    raise ValueError(f"Could not find column '{base_name}' (with or without 'field_' prefix) in CSV header.")


class  NEON_API:
    def __init__(self, parent=None):
        self.instance = 1
        self.className = "NEON API"
        self.dest = ""
        self._token: str = ""
        self._server: str = SERVER
        try:
            _mgr = APIKeyManager()
            self._token  = _mgr.get_neon_token() or ""
            self._server = _mgr.get_neon_endpoint() or SERVER
        except Exception as _e:
            print(f"[NEON_API] Could not load API settings: {_e}")
        # Same normalization as set_server, so URLs never come out as ".../v0sites/..."
        self._server = self._server.rstrip("/") + "/"
        # Reason for the most recent failed request, for callers to show the user
        self.last_error: str = ""

    # ------------------------------------------------------------------------
    def set_token(self, token) -> None:
        """Update the NEON API token at runtime (called from MainWindow after dialog save)."""
        self._token = token or ""

    # ------------------------------------------------------------------------
    def set_server(self, server) -> None:
        """Update the NEON API base URL at runtime (called from MainWindow after dialog save)."""
        self._server = (server or SERVER).rstrip("/") + "/"

    # ------------------------------------------------------------------------
    def _auth_headers(self) -> dict:
        """Return the X-API-Token header dict if a token is set, else empty dict."""
        if self._token:
            return {"X-API-Token": self._token}
        return {}

    # ------------------------------------------------------------------------
    def _get_json(self, url) -> dict:
        """
        GET a NEON API URL and return its JSON, or {} on failure with the reason in
        self.last_error. Timeouts and retries come from appcore.http_client.
        """
        self.last_error = ""
        try:
            r = http_get(url, headers=self._auth_headers())
        except requests.exceptions.RequestException as e:
            self.last_error = f"Could not reach NEON ({e})"
            print(f"[NEON_API] {self.last_error}: {url}")
            return {}

        if r.status_code in (401, 403):
            self.last_error = (f"NEON rejected the request (HTTP {r.status_code}). "
                               "Check the NEON API token in the API Key Manager.")
        elif r.status_code != 200:
            self.last_error = f"NEON returned HTTP {r.status_code} {r.reason}"
        else:
            try:
                return r.json()
            except ValueError:
                self.last_error = "NEON returned a response that is not JSON"

        print(f"[NEON_API] {self.last_error}: {url}")
        return {}


    # ======================================================================================================================
    # The purpose of this function is to query information above a specific product. The information
    # contains the product description, sites for which the product is available among other information.
    # ======================================================================================================================
    def QueryProductInfo(self, productCode):
        """Product JSON, or {} on failure with the reason in self.last_error."""
        return self._get_json(self._server + 'products/' + productCode)


    # ======================================================================================================================
    # Site JSON from the NEON endpoint stored in settings, or {} on failure with the reason in self.last_error.
    # ======================================================================================================================
    def FetchSiteInfoFromNEON(self, siteCode):
        return self._get_json(self._server + 'sites/' + siteCode)

    # ======================================================================================================================
    # Every site's JSON in one request (GET <endpoint>sites), or {} on failure with the reason in self.last_error.
    # ======================================================================================================================
    def FetchAllSitesFromNEON(self):
        return self._get_json(self._server + 'sites')


    # ======================================================================================================================
    # THIS FUNCTION FETCHES THE FIELD SITE TABLE FROM THE NEON SITE AND PARSES ITS INFORMATION.
    # ======================================================================================================================
    def FetchFieldSiteTableURL(self, my_url):
        """
        Return the href of the field site metadata link on the NEON page, or None if the
        page has no such link. Certificate checking stays on (no unverified SSL context).
        """
        r = http_get(my_url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html5lib')

        # FIND ALL CSV LINKS ON THE WEB-PAGE. CURRENTLY THERE IS ONLY ONE. HOWEVER, THERE COULD BE MULTIPLES IN THE FUTURE
        links = [link['href'] for link in soup.find_all("a", href=lambda href: href and "Metadata" in href)]

        if not links:
            print(f"[NEON_API] No field site metadata link found on {my_url}")
            return None

        if len(links) > 1:
            # Same choice as before (the last link); flag it so a page change is noticed
            print(f"[NEON_API] {len(links)} metadata links found on {my_url}; using the last: {links}")

        return links[-1]


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def Download_Field_Site_Metadata(self, csv_links):
        link = 'https://www.neonscience.org' + csv_links + '.csv'
        file_name = link.split('/')[-1]

        # Build target directory
        configFilePath = os.path.join(str(DOWNLOADS_DIR), 'NEON', 'Metadata')
        os.makedirs(configFilePath, exist_ok=True)
        filename_with_path = os.path.join(configFilePath, file_name)

        # One retry layer: the app-wide policy in appcore.http_client (bounded retries,
        # backoff, timeouts). IMAGE_TIMEOUT allows a larger file to stream.
        session = TimeoutSession(timeout=IMAGE_TIMEOUT)
        temp_name = None
        try:
            with session.get(link, stream=True) as r:
                r.raise_for_status()
                # write atomically to avoid partial files
                with tempfile.NamedTemporaryFile('wb', delete=False, dir=configFilePath) as tmp:
                    temp_name = tmp.name
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
            os.replace(temp_name, filename_with_path)
            return filename_with_path  # success
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download {link}: {e}")
        finally:
            # A failed attempt must not leave its temp file in the Metadata folder
            if temp_name and os.path.exists(temp_name):
                os.remove(temp_name)
            session.close()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def FetchData(self, SiteCode, strProduct, strStartDate, strEndDate, downloadsFilePath):

        nError = 0

        print("=== FetchData called ===")
        print("Initial args:")
        print("  SiteCode:", SiteCode)
        print("  strProduct:", strProduct)
        print("  strStartDate:", strStartDate)
        print("  strEndDate:", strEndDate)
        print("  downloadsFilePath:", downloadsFilePath)

        if not os.path.exists(downloadsFilePath):
            print("Creating downloadsFilePath:", downloadsFilePath)
            os.makedirs(downloadsFilePath)

        foldername = strProduct.split('.')[1].zfill(5) + ' -' + strProduct.split('.')[2].split(':')[1]
        self.dest = os.path.join(downloadsFilePath, foldername)
        print("Computed foldername:", foldername)
        print("Destination folder (self.dest):", self.dest)

        if not os.path.exists(self.dest):
            print("Creating destination folder:", self.dest)
            os.makedirs(self.dest)

        # JES OVERRIDE DOWNLOAD PATH DUE TO WINDOWS FILE PATH LENGTH LIMITATION.
        scratchPath = os.path.normpath(JsonEditor().getValue("Scratchpad_folder"))
        print("Scratchpad_folder override:", scratchPath)

        strProduct = 'DP1.' + strProduct.split('.')[1] + '.001'
        print("Normalized strProduct:", strProduct)

        progressBar = QProgressWheel()
        progressBar.setRange(0, 100)
        progressBar.show()
        progressBar.setValue(20)

        try:
            # Download zip files for the product into Scratchpad
            _zbp_kwargs = dict(
                dpid=strProduct,
                site=SiteCode,
                savepath=scratchPath,
                startdate=strStartDate,
                enddate=strEndDate,
                package="basic",
                include_provisional=True,
                check_size=False
            )
            if self._token:
                _zbp_kwargs["token"] = self._token
            result = zips_by_product(**_zbp_kwargs)
            print("zips_by_product returned:", result)

            progressBar.setValue(40)

            # PATH WHERE zipsByProduct PLACED THE DOWNLOADED ZIP FILES (Scratchpad)
            myFolderPath = os.path.join(scratchPath, 'filesToStack' + strProduct.split('.')[1].zfill(5))
            print("myFolderPath:", myFolderPath, "exists?", os.path.exists(myFolderPath))

            # TEMPORARY STACKING PATH IN SCRATCHPAD
            mySavePath = os.path.join(scratchPath, foldername)
            print("mySavePath:", mySavePath)

            if os.path.exists(mySavePath):
                print("Removing existing mySavePath:", mySavePath)
                shutil.rmtree(mySavePath)

            nMyCores = multiprocessing.cpu_count()
            print("CPU cores available:", nMyCores)

            progressBar.setValue(60)

            # Stack the monthly files into one CSV per table
            print("Calling stack_by_table...")
            stack_by_table(
                filepath=myFolderPath,
                savepath=mySavePath
            )
            print("stack_by_table completed")

            # REMOVE THE ZIP FILE DOWNLOAD FOLDER
            if os.path.exists(myFolderPath):
                print("Cleaning up myFolderPath:", myFolderPath)
                shutil.rmtree(myFolderPath)

            # MOVE STACKED FILES FROM SCRATCHPAD TO FINAL DESTINATION (Downloads/NEON/data)
            src = os.path.join(scratchPath, foldername, 'stackedFiles')
            print("src:", src)
            if os.path.exists(src) and os.path.exists(self.dest):
                for filename in os.listdir(src):
                    shutil.move(os.path.join(src, filename), self.dest)
                shutil.rmtree(os.path.join(scratchPath, foldername))

            progressBar.setValue(100)

            # Final check
            if os.path.exists(self.dest):
                print("Final destination contents:", os.listdir(self.dest))
            else:
                print("Final destination does not exist:", self.dest)

        except Exception as e:
            print("Exception occurred in FetchData:", e)
            traceback.print_exc()
            nError = -1

        progressBar.close()
        del progressBar

        print("=== FetchData completed with nError =", nError, "===")
        return nError

    # ======================================================================================================================
    # THIS FUNCTION DOWNLOADS THE LATEST IMAGE FOR THE SITE SELECTED BY THE END-USER AND DISPLAYS IT IN THE GUI SO THE
    # END-USER CAN CAN SEE WHAT THE PARTICULAR SITE LOOKS LIKE.
    # ======================================================================================================================
    def DownloadLatestImage(self, siteCode, domainCode, strProductID=20002):
        nStatus = -1
        nWebImageCount = 0
        latestImage = []

        latestImageURL = f'https://phenocam.nau.edu/data/latest/NEON.{domainCode}.{siteCode}.DP1.{strProductID}.jpg'

        start_time = time.time()

        # Retries on server errors and timeouts come from appcore.http_client.
        # A 404 means the site has no latest image, so it is not retried.
        try:
            r = http_get(latestImageURL, timeout=IMAGE_TIMEOUT)
            nStatus = r.status_code
            if r.status_code == 200:
                nWebImageCount = 1
                latestImage = QPixmap()
                latestImage.loadFromData(r.content)
            elif r.status_code == 404:
                print(f"404: PhenoCam - no latest image at {latestImageURL}")
            else:
                print(f"PhenoCam latest image: unexpected status {r.status_code} for {latestImageURL}")
        except requests.exceptions.RequestException as e:
            print(f"PhenoCam latest image: connection error: {e}")

        end_time = time.time()
        print("Download Latest Image Elapsed Time:", end_time - start_time)

        return nStatus, latestImage, nWebImageCount

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getAvailableMonths(self, SITE, PRODUCTCODE):
        """
        Available months ("YYYY-MM") for a product at a site, or [] when none or on failure.
        On failure the reason is in self.last_error. Retries come from appcore.http_client.
        """
        availableMonths = []

        site_json = self.FetchSiteInfoFromNEON(SITE)

        # Loop through the 'dataProducts' list items (each one is a dictionary) at the site
        for product in site_json.get('data', {}).get('dataProducts', []):
            # if a list item's 'dataProductCode' dict element equals the product code string
            if (product.get('dataProductCode') == PRODUCTCODE):
                availableMonths = product.get('availableMonths', [])

        return availableMonths


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getProductAbstract(self, SITE, PRODUCTCODE):
        productAbstract = []

        product_json = self.QueryProductInfo(PRODUCTCODE)

        try:
            productAbstract = product_json['data']['productAbstract']
        except Exception:
            pass

        return productAbstract


    # ======================================================================================================================
    # THIS FUNCTION WILL FETCH AND READ THE NEON FIELD SITE TABLE THAT CONTAINS INFORMATION ABOUT ALL THE
    # NEON SITES INCLUDING ALL THE IMAGES THAT ARE AVAILABLE FOR A SITE ON THE PHENOCAM WEBSITE.
    # ======================================================================================================================
    def readFieldSiteTable(self):
        siteList = []
        url = 'https://www.neonscience.org/field-sites/explore-field-sites'

        nErrorCode = App_Utils().check_url_validity(url)

        # IF AT LEAST ONE FIELD SITE TABLE IS FOUND ON THE NEON SITE...
        if nErrorCode == 0:
            csv_links = self.FetchFieldSiteTableURL(url)

            if csv_links is None:
                # The page loaded but has no field site metadata link
                return -1, []

            # download all CSV files
            filename_with_path = self.Download_Field_Site_Metadata(csv_links)

            try:
                siteList = parse_NEON_field_site_metadata(filename_with_path)
            except Exception as e:
                nErrorCode = -2
        # ELSE IF NO FIELD SITE TABLES ARE FOUND, RETURN AN EMPTY LIST
        elif nErrorCode == -1:
            siteList = []

        return nErrorCode, siteList


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def parseNitrateCSV(self):
        # FULLY QUALIFIED PATH OF THE CSV DOWNLOADED ONTO THE LOCAL COMPUTER
        filename = os.path.join(self.dest, 'NSW_15_minute.csv')

        nitrateList = []
        fields = []
        rows = []

        with open(filename, 'r') as data:
            dict_Reader = csv.DictReader(data)
            ordered_dict_from_csv = list(dict_Reader)[0]
            dict_from_csv = dict(ordered_dict_from_csv)
            keys = dict_from_csv.keys()

        # READ CSV FILE
        with open(filename, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)

            # EXTRACT FIELD NAMES FROM THE FIRST ROW OF THE CSV
            fields = next(csvreader)

            # READ ONE ROW AT A TIME AND APPEND INTO A LIST
            for row in csvreader:
                rows.append(row)

        for row in rows:
            # VALIDATE DATA: SOME OF THE FIELDS IN EACH DATA RECORD ARE NULL SO WE HAVE TO HANDLE THESE CASES. 'NOT SURE
            # WHY RECORDS WITH MISSING DATA ARE NOT EXCLUDED FROM THE DATA FILES PROVIDED BY NEON
            # Columns 6-13 are converted with float() below, so all of them need a default
            for i in range(6, 14):
                if len(row[i]) == 0:
                    row[i] = '0.0'

            nitrateList.append(nitrateData(row[0],
                                           row[1],
                                           row[2],
                                           row[3],
                                           row[4],
                                           row[5],
                                           float(row[6]),
                                           float(row[7]),
                                           float(row[8]),
                                           float(row[9]),
                                           float(row[10]),
                                           float(row[11]),
                                           float(row[12]),
                                           float(row[13]),
                                           row[14],
                                           row[15]))

        return nitrateList

    def scrape_phenocam_table(self, url="https://phenocam.nau.edu/webcam/network/table/"):
        """
        Scrape the PhenoCam site table and return a dictionary keyed by camera name.
        Each entry contains latitude, longitude, elevation, site description, and hyperlink.
        """
        response = http_get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html5lib")
        table = soup.find("table")
        if not table:
            raise ValueError("No table found on the page")

        data = {}
        rows = table.find_all("tr")[1:]  # skip header

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            # Camera column has the hyperlink
            camera_cell = cols[0]
            camera_link = camera_cell.find("a")
            camera_name = camera_link.get_text(strip=True) if camera_link else camera_cell.get_text(strip=True)
            camera_href = camera_link["href"] if camera_link and camera_link.has_attr("href") else None

            lat = cols[1].get_text(strip=True)
            lon = cols[2].get_text(strip=True)
            elev = cols[3].get_text(strip=True)
            desc = cols[4].get_text(strip=True)

            try:
                lat = float(lat)
                lon = float(lon)
                elev = float(elev)
            except ValueError:
                continue

            data[camera_name] = {
                "lat": lat,
                "lon": lon,
                "elev_m": elev,
                "site_description": desc,
                "link": camera_href
            }

        return data
