#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# tnm_download.py
#
# Download NAIP aerial imagery and 3DEP elevation data for a reach, from The
# National Map. No account, no API key: TNMAccess is open.
#
# The area can be given as a bounding box, or as an NWIS gage number, in which
# case the gage's coordinates come from the NWIS site service and the box is
# built around it. That second form matches how the camera sites are already
# identified elsewhere, so the same gage number that pulls stage also pulls the
# imagery for the reach.
#
# Dataset names are looked up from TNM rather than hard-coded, because they are
# renamed from time to time and a stale string returns nothing with no error.
#
# Examples
#   List what covers a reach, without downloading:
#       python tnm_download.py --gage 06768000 --buffer 3 --list
#
#   NAIP imagery for that reach, every year available:
#       python tnm_download.py --gage 06768000 --buffer 3 --naip --out D:\NAIP
#
#   NAIP for one year, plus the 1 m DEM:
#       python tnm_download.py --bbox -99.60 40.68 -99.50 40.74 --naip --dem --year 2022
#
#   Lidar point clouds (large files):
#       python tnm_download.py --gage 06768000 --buffer 2 --lidar

import os
import re
import sys
import json
import math
import time
import argparse

import requests

TNM_BASE = "https://tnmaccess.nationalmap.gov/api/v1"
NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USER_AGENT = "GRIME-AI tnm_download"

# What to match in the TNM dataset list for each kind of data. The lists are
# matched case-insensitively against the dataset titles TNM reports.
DATASET_PATTERNS = {
    "naip":  ["naip"],
    "dem":   ["digital elevation model (dem) 1 meter", "1 meter dem"],
    "lidar": ["lidar point cloud"],
}
PRODUCT_FORMATS = {
    "naip":  "JPEG2000,GeoTIFF",
    "dem":   "GeoTIFF",
    "lidar": "LAS,LAZ",
}

REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1 << 20          # 1 MiB
MAX_ITEMS_PER_QUERY = 1000    # TNM's own ceiling for one call


# ======================================================================================================================
# Area of interest
# ======================================================================================================================
def gage_location(gage_number: str) -> tuple:
    """(longitude, latitude, name) for an NWIS gage."""
    params = {"format": "rdb", "sites": gage_number, "siteOutput": "expanded"}
    response = requests.get(NWIS_SITE_URL, params=params,
                            headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    rows = [line for line in response.text.splitlines()
            if line and not line.startswith("#")]
    if len(rows) < 3:
        raise ValueError(f"No site record for gage {gage_number}.")
    header = rows[0].split("\t")
    values = rows[2].split("\t")            # row 1 is the field-width line
    record = dict(zip(header, values))
    return (float(record["dec_long_va"]), float(record["dec_lat_va"]),
            record.get("station_nm", gage_number))


def box_around(longitude: float, latitude: float, buffer_km: float) -> list:
    """Bounding box [minx, miny, maxx, maxy] of the given half-width, in degrees."""
    lat_degrees = buffer_km / 111.32
    lon_degrees = buffer_km / (111.32 * max(math.cos(math.radians(latitude)), 1e-6))
    return [longitude - lon_degrees, latitude - lat_degrees,
            longitude + lon_degrees, latitude + lat_degrees]


# ======================================================================================================================
# The National Map
# ======================================================================================================================
def tnm_datasets() -> list:
    """Every dataset TNM publishes, as returned by the API."""
    response = requests.get(f"{TNM_BASE}/datasets", headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else data.get("items", [])


def _flatten(datasets) -> list:
    """Dataset titles, including the sub-datasets TNM nests under some entries."""
    titles = []
    for entry in datasets:
        title = entry.get("sbDatasetTag") or entry.get("title") or ""
        if title:
            titles.append(title)
        titles += _flatten(entry.get("tags", []) or [])
    return titles


def dataset_names(kind: str, datasets=None) -> list:
    """The TNM dataset names matching one kind, so the strings are never guessed."""
    titles = _flatten(datasets if datasets is not None else tnm_datasets())
    patterns = DATASET_PATTERNS[kind]
    matched = [t for t in titles
               if any(p in t.lower() for p in patterns)]
    return sorted(set(matched))


def search(bbox, dataset, prod_formats=None, max_items=MAX_ITEMS_PER_QUERY) -> list:
    """Products of one dataset covering the box. Paged until TNM runs out."""
    items, offset = [], 0
    while True:
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "datasets": dataset,
            "outputFormat": "JSON",
            "max": min(max_items, MAX_ITEMS_PER_QUERY),
            "offset": offset,
        }
        if prod_formats:
            params["prodFormats"] = prod_formats
        response = requests.get(f"{TNM_BASE}/products", params=params,
                                headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("items", [])
        items += batch
        offset += len(batch)
        if not batch or offset >= payload.get("total", 0) or len(items) >= max_items:
            return items


def item_year(item) -> str:
    """The item's year, from its publication date or its title."""
    for key in ("publicationDate", "lastUpdated", "dateCreated"):
        value = str(item.get(key) or "")
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return match.group(0)
    match = re.search(r"(19|20)\d{2}", str(item.get("title") or ""))
    return match.group(0) if match else "unknown"


# ======================================================================================================================
# Download
# ======================================================================================================================
def download(url, destination) -> bool:
    """
    Fetch one file, resuming a partial download and skipping a complete one.
    Returns True when the file is present and complete afterwards.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".part"
    existing = os.path.getsize(temporary) if os.path.exists(temporary) else 0

    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT) as response:
        if response.status_code == 416:            # already complete
            os.replace(temporary, destination)
            return True
        response.raise_for_status()
        resumed = response.status_code == 206
        total = int(response.headers.get("Content-Length", 0)) + (existing if resumed else 0)

        mode = "ab" if resumed and existing else "wb"
        written = existing if resumed and existing else 0
        with open(temporary, mode) as handle:
            for chunk in response.iter_content(CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total:
                    percent = written * 100.0 / total
                    print(f"\r    {os.path.basename(destination)}  "
                          f"{written / 1e6:,.1f} / {total / 1e6:,.1f} MB  ({percent:4.1f}%)",
                          end="", flush=True)
        print()

    os.replace(temporary, destination)
    return True


def fetch_all(items, out_dir, kind, dry_run=False) -> dict:
    """Download every item into out_dir/<kind>/<year>/, and write a manifest."""
    summary = {"downloaded": 0, "skipped": 0, "failed": 0, "files": []}
    for index, item in enumerate(items, start=1):
        url = item.get("downloadURL") or item.get("urls", {}).get("TIFF")
        if not url:
            summary["failed"] += 1
            continue
        name = os.path.basename(url.split("?")[0])
        destination = os.path.join(out_dir, kind, item_year(item), name)

        print(f"[{index}/{len(items)}] {name}  ({(item.get('sizeInBytes') or 0) / 1e6:,.1f} MB)")
        if dry_run:
            continue
        if os.path.exists(destination):
            print("    already present")
            summary["skipped"] += 1
            summary["files"].append(destination)
            continue
        try:
            download(url, destination)
            summary["downloaded"] += 1
            summary["files"].append(destination)
        except Exception as err:
            print(f"    failed: {type(err).__name__}: {err}")
            summary["failed"] += 1

    if not dry_run and items:
        manifest = os.path.join(out_dir, kind, f"{kind}_manifest.json")
        os.makedirs(os.path.dirname(manifest), exist_ok=True)
        with open(manifest, "w") as handle:
            json.dump({"retrieved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "items": items, "summary": summary}, handle, indent=2)
    return summary


# ======================================================================================================================
# Command line
# ======================================================================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download NAIP imagery and 3DEP elevation data from The National Map.")
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                      help="Area of interest in decimal degrees.")
    area.add_argument("--gage", help="NWIS gage number; the box is built around the gage.")
    parser.add_argument("--buffer", type=float, default=2.0,
                        help="Half-width in km around the gage (default 2).")
    parser.add_argument("--naip", action="store_true", help="NAIP aerial imagery.")
    parser.add_argument("--dem", action="store_true", help="3DEP 1 m digital elevation model.")
    parser.add_argument("--lidar", action="store_true", help="3DEP lidar point clouds (large).")
    parser.add_argument("--year", help="Keep only products from this year.")
    parser.add_argument("--out", default="tnm_data", help="Output folder (default tnm_data).")
    parser.add_argument("--list", action="store_true",
                        help="List what is available without downloading.")
    parser.add_argument("--datasets", action="store_true",
                        help="Print the TNM dataset names this script matches, then exit.")
    args = parser.parse_args(argv)

    try:
        catalog = tnm_datasets()
    except Exception as err:
        print(f"Could not reach The National Map: {type(err).__name__}: {err}")
        return 1

    if args.datasets:
        for kind in DATASET_PATTERNS:
            print(f"{kind}:")
            for name in dataset_names(kind, catalog) or ["  (nothing matched)"]:
                print(f"  {name}")
        return 0

    if args.gage:
        longitude, latitude, name = gage_location(args.gage)
        bbox = box_around(longitude, latitude, args.buffer)
        print(f"Gage {args.gage}: {name}")
        print(f"  {latitude:.5f}, {longitude:.5f}  buffer {args.buffer} km")
    else:
        bbox = list(args.bbox)
    print("Bounding box: " + ", ".join(f"{v:.5f}" for v in bbox))

    kinds = [kind for kind, wanted in
             (("naip", args.naip), ("dem", args.dem), ("lidar", args.lidar)) if wanted]
    if not kinds:
        kinds = ["naip"]
        print("No data type given; using NAIP.")

    exit_code = 0
    for kind in kinds:
        names = dataset_names(kind, catalog)
        if not names:
            print(f"\n{kind}: no matching dataset in The National Map "
                  f"(patterns: {DATASET_PATTERNS[kind]}). Run --datasets to see the list.")
            exit_code = 1
            continue

        items = []
        for name in names:
            try:
                items += search(bbox, name, PRODUCT_FORMATS[kind])
            except Exception as err:
                print(f"\n{kind}: search failed for '{name}': {type(err).__name__}: {err}")
                exit_code = 1

        if args.year:
            items = [item for item in items if item_year(item) == str(args.year)]

        seen, unique = set(), []
        for item in items:
            url = item.get("downloadURL")
            if url and url not in seen:
                seen.add(url)
                unique.append(item)

        size = sum(item.get("sizeInBytes") or 0 for item in unique)
        print(f"\n{kind}: {len(unique)} product(s), {size / 1e9:,.2f} GB")
        years = sorted({item_year(item) for item in unique})
        if years:
            print(f"  years: {', '.join(years)}")

        summary = fetch_all(unique, args.out, kind, dry_run=args.list)
        if not args.list:
            print(f"  downloaded {summary['downloaded']}, skipped {summary['skipped']}, "
                  f"failed {summary['failed']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
