#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# image_timestamp_sidecar.py
#
# Plugin: Image Timestamp Sidecar.
#
# Scans a folder of images, works out which network they came from, and writes
# a sidecar CSV giving each image's canonical USGS-style UTC filename
# (SITE___YYYY-MM-DDTHH-MM-SSZ), so folders whose filenames do not follow that
# convention can still be read by the rest of the application.
#
# Per-source resolution:
#   USGS     filename is already UTC. If it cannot be parsed, the watermark is
#            local with no stated offset, so the row is left unresolved.
#   NEON     the watermark states UTC directly, and the filename and EXIF carry
#            no usable offset, so OCR is required.
#   PhenoCam the watermark carries local time and an explicit UTC offset, so OCR
#            is required; the filename alone is local-only.
#   Mesonet  local time and the NWSLI station code come from the filename, and
#            the offset from SD_Mesonet_Stations.xlsx (South Dakota Mesonet stays
#            on standard time year round). There is no watermark to fall back on.
#
# Output: <folder>/sidecars/image_timestamp_sidecar.csv, written when the scan
# finishes, with Original Filename, Canonical Filename, Status and UTC Offset.
#
# Ported from grime_ai_dataset_manager.py.

import os
import re
import sys
import csv
import json
import math
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import cv2
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QPushButton, QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView, QPlainTextEdit, QFileDialog, QMessageBox,
                             QSizePolicy, QApplication)

# The only host dependency, and an optional one: it locates the application's
# user folder. Run standalone (python image_timestamp_sidecar.py) and the
# folder beside this file is used instead.
try:
    from appcore.app_identity import PLUGINS_DIR
    USER_ROOT = Path(str(PLUGINS_DIR)).parent
except Exception:
    USER_ROOT = Path(__file__).resolve().parent

PLUGIN = {
    "title":       "Image Timestamp Sidecar",
    "class":       "ImageSidecarGeneratorTab",
    "description": "Write a sidecar CSV mapping image filenames to canonical USGS UTC timestamps",
    "surface":     "tools",
    "api_version": 2,
}

# OCR reads the watermark strip across the top of the frame. NEON writes two
# lines there and the lower one carries the time, so the strip must be deep
# enough to hold both with room to spare; at 0.06 the time line is clipped at
# its baseline and reads intermittently.
CROP_LEFT = 0.0
CROP_TOP = 0.0
CROP_RIGHT = 0.75
CROP_BOTTOM = 0.12

OCR_TESSERACT = "Tesseract"
OCR_EASYOCR = "EasyOCR"
# Models live beside the plugins folder, under the application's user root, so
# this works for any build without naming the application here. Sets left by
# other tools are used where they are, rather than downloading a second copy.
EASYOCR_MODEL_DIR = str(USER_ROOT / "EasyOCR")
# Deployments that pre-install the models (conda, Docker, an air-gapped machine)
# set these: EASYOCR_MODEL_PATH names the folder holding craft_mlt_25k.pth and
# english_g2.pth, and EASYOCR_ALLOW_DOWNLOAD=0 forbids fetching them at runtime.
EASYOCR_ENV_DIR = os.environ.get("EASYOCR_MODEL_PATH", "")
EASYOCR_ALLOW_DOWNLOAD = os.environ.get("EASYOCR_ALLOW_DOWNLOAD", "1").strip().lower() \
    not in ("0", "false", "no", "off")
EASYOCR_SEARCH_DIRS = [
    EASYOCR_ENV_DIR,
    EASYOCR_MODEL_DIR,
    str(Path.home() / "Documents" / "GRIME-AI" / "EasyOCR"),
    str(Path.home() / "Documents" / "OpsiLumAI" / "EasyOCR"),
    str(Path.home() / ".EasyOCR" / "model"),
]
SETTINGS_FILE = str(USER_ROOT / "Settings" / "image_timestamp_sidecar.json")

RE_CANONICAL_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z$")
RE_WATERMARK_TS = re.compile(r"(\d{4})[\/\-](\d{2})[\/\-](\d{2})\s+(\d{2})[:\.](\d{2})[:\.](\d{2})")
# Legacy USGS local-time filename: 2019-06-01_12-00-00-0000-05-00
RE_OLD_USGS_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})-\d+-(\d{2})-(\d{2})$")

_easyocr_reader = None


def _load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(settings: dict):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"[image_timestamp_sidecar] Could not save settings: {e}")


def _exif_datetime(image_path):
    """Capture time from EXIF DateTime/DateTimeOriginal, or None."""
    try:
        from PIL.ExifTags import TAGS as _TAGS
        exif = Image.open(image_path)._getexif()
        if exif:
            for tag_id, value in exif.items():
                if _TAGS.get(tag_id, tag_id) in ("DateTime", "DateTimeOriginal"):
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def _local_to_utc(local_dt, iana_tz):
    tz = ZoneInfo(iana_tz)
    return local_dt.replace(tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)


def easyocr_model_dir():
    """The first folder holding EasyOCR model files, or None if there is none."""
    for folder in EASYOCR_SEARCH_DIRS:
        try:
            if os.path.isdir(folder) and any(f.endswith(".pth") for f in os.listdir(folder)):
                return folder
        except Exception:
            continue
    return None


# Set by the GUI when the user agrees to fetch the models for this run.
_download_approved = False


def approve_model_download(approved=True):
    global _download_approved
    _download_approved = bool(approved)


def _get_easyocr_reader(allow_download=None):
    """
    The shared EasyOCR reader, using whichever model folder already has the
    models. With none found, the models are fetched once into the application's
    own folder (about 100 MB, needs a network connection).
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        if allow_download is None:
            allow_download = EASYOCR_ALLOW_DOWNLOAD and _download_approved
        folder = easyocr_model_dir()
        download = folder is None and allow_download
        if folder is None:
            if not allow_download:
                raise FileNotFoundError(
                    "EasyOCR models were not found and were not downloaded. "
                    "Put craft_mlt_25k.pth and english_g2.pth in "
                    + (EASYOCR_ENV_DIR or EASYOCR_MODEL_DIR)
                    + ", or set EASYOCR_MODEL_PATH to the folder holding them.")
            folder = EASYOCR_MODEL_DIR
            os.makedirs(folder, exist_ok=True)
        _easyocr_reader=easyocr.Reader(["en"],gpu=False,verbose=False,
            model_storage_directory=folder,download_enabled=download)
    return _easyocr_reader


# Set by _ocr_watermark when it fails, so the status log can say why.
_last_ocr_error = ""


def last_ocr_error() -> str:
    return _last_ocr_error

def _ocr_watermark(image_path,save_crop_dir=None,engine=OCR_TESSERACT):
    global _last_ocr_error
    def _run_engine(eng,cv_img,gray,thresh):
        if eng==OCR_EASYOCR:
            reader=_get_easyocr_reader(); results=reader.readtext(cv_img,detail=0,paragraph=True)
            text="\n".join(results)
        else:
            import pytesseract
            processed=Image.fromarray(thresh); text=pytesseract.image_to_string(processed,config="--psm 6")
        lines=[l.strip() for l in text.splitlines() if l.strip()]
        ts_line=None; loc_parts=[]
        for line in lines:
            if RE_WATERMARK_TS.search(line): ts_line=line
            else: loc_parts.append(line)
        return " ".join(loc_parts).strip() or None,ts_line
    try:
        img=Image.open(image_path); w,h=img.size
        box=(int(CROP_LEFT*w),int(CROP_TOP*h),int(CROP_RIGHT*w),int(CROP_BOTTOM*h))
        crop=img.crop(box)
        cv_img=cv2.cvtColor(np.array(crop),cv2.COLOR_RGB2BGR)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(gray.shape[1]*3,gray.shape[0]*3),interpolation=cv2.INTER_LANCZOS4)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=31,C=15)
        if save_crop_dir:
            os.makedirs(save_crop_dir,exist_ok=True)
            stem=os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(save_crop_dir,f"{stem}_crop.png"),thresh if engine==OCR_TESSERACT else cv_img)
        try:
            location,ts_line=_run_engine(engine,cv_img,gray,thresh)
        except Exception as engine_err:
            # EasyOCR missing or its models unavailable: try Tesseract rather
            # than giving up, and keep the reason for the status column.
            _last_ocr_error=f"{type(engine_err).__name__}: {engine_err}"
            if engine!=OCR_TESSERACT:
                location,ts_line=_run_engine(OCR_TESSERACT,cv_img,gray,thresh)
            else:
                raise
        if ts_line is None and engine==OCR_EASYOCR:
            # Second opinion only. If Tesseract is not installed, keep what
            # EasyOCR read rather than losing it to an import error.
            try:
                location2,ts_line2=_run_engine(OCR_TESSERACT,cv_img,gray,thresh)
                if ts_line2: location,ts_line=location2,ts_line2
            except Exception as fallback_err:
                _last_ocr_error=f"no timestamp line found; second engine unavailable ({fallback_err})"
        return location,ts_line
    except Exception as err:
        _last_ocr_error = f"{os.path.basename(image_path)}: {type(err).__name__}: {err}"
        return None,None

def _watermark_line_images(image_path):
    """
    The watermark's text lines, cropped one per line. NEON draws its text on a
    solid blue plate, which is found by color, so each line is read without the
    sky behind it and without the other line's characters running into it.
    Returns [] when no plate is found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    top = img[0:int(0.20 * h), 0:int(CROP_RIGHT * w)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int)
    plate = ((hue > 95) & (hue < 115) & (sat > 120)).astype(np.uint8)

    rows = np.where(plate.sum(axis=1) > 40)[0]
    if rows.size == 0:
        return []
    runs = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)

    crops = []
    for run in runs:
        if len(run) < 10:
            continue
        y0, y1 = max(int(run[0]) - 4, 0), min(int(run[-1]) + 5, top.shape[0])
        cols = np.where(plate[y0:y1].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = max(int(cols[0]) - 4, 0), min(int(cols[-1]) + 5, top.shape[1])
        crops.append(top[y0:y1, x0:x1])
    return crops


def ocr_watermark_lines(image_path, engine=OCR_EASYOCR):
    """Text read from the watermark line by line, joined with spaces, or ""."""
    global _last_ocr_error
    texts = []
    for line in _watermark_line_images(image_path):
        gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.bitwise_not(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        try:
            if engine == OCR_EASYOCR:
                reader = _get_easyocr_reader()
                texts += reader.readtext(line, detail=0, paragraph=True)
            else:
                import pytesseract
                texts.append(pytesseract.image_to_string(Image.fromarray(thresh),
                                                         config="--psm 7"))
        except Exception as err:
            _last_ocr_error = f"{os.path.basename(image_path)}: {type(err).__name__}: {err}"
            break
    return " ".join(t.strip() for t in texts if t and t.strip())


def _ocr_worker(args):
    image_path,save_crop_dir,engine=args; loc,ts_line=_ocr_watermark(image_path,save_crop_dir,engine)
    return image_path,loc,ts_line

def _convert_old_usgs_ts(ts_part):
    m=RE_OLD_USGS_TS.match(ts_part)
    if not m: return None
    date_str,hh,mm,ss,off_h,off_m=m.groups()
    local_dt=datetime.strptime(f"{date_str} {hh}:{mm}:{ss}","%Y-%m-%d %H:%M:%S")
    offset=timedelta(hours=int(off_h),minutes=int(off_m)); utc_dt=local_dt+offset
    return utc_dt.strftime("%Y-%m-%dT%H-%M-%SZ")

def _parse_watermark_ts(ts_line):
    """
    Parse a USGS watermark timestamp tolerating common OCR errors:
    - Merged digits: '12200:20' -> 12:00:20 (OCR drops separator between HH and MM)
    - Letter substitution: '09.00e11' -> 09:00:11 (OCR reads '1' as 'e')
    """
    # 1. Strict full match
    m=RE_WATERMARK_TS.search(ts_line)
    if m:
        try:
            return datetime(int(m.group(1)),int(m.group(2)),int(m.group(3)),
                            int(m.group(4)),int(m.group(5)),int(m.group(6)))
        except ValueError: pass
    dm=re.search(r'(\d{4})[\/\-](\d{2})[\/\-](\d{2})',ts_line)
    if not dm: return None
    after=ts_line[dm.end():]
    # 2. HH:MM:SS or HH.MM.SS
    m2=re.search(r'(\d{2})[:\.](\d{2})[:\.](\d{2})',after)
    if m2:
        hh,mm,ss=int(m2.group(1)),int(m2.group(2)),int(m2.group(3))
        if hh<=23 and mm<=59 and ss<=59:
            try: return datetime(int(dm.group(1)),int(dm.group(2)),int(dm.group(3)),hh,mm,ss)
            except ValueError: pass
    # 3. HH.MM with noise chars then SS: '09.00e11'
    m3=re.search(r'(\d{2})[:\.](\d{2})\D*(\d{2})',after)
    if m3:
        hh,mm,ss=int(m3.group(1)),int(m3.group(2)),int(m3.group(3))
        if hh<=23 and mm<=59 and ss<=59:
            try: return datetime(int(dm.group(1)),int(dm.group(2)),int(dm.group(3)),hh,mm,ss)
            except ValueError: pass
    # 4. Merged HHMM+sep+SS: '12200:20' -> first 2 digits=HH, last 2 before sep=MM
    m4=re.search(r'(\d{4,})[:\.](\d{2})',after)
    if m4:
        merged=m4.group(1); ss=int(m4.group(2))
        hh,mm=int(merged[0:2]),int(merged[-2:])
        if hh<=23 and mm<=59 and ss<=59:
            try: return datetime(int(dm.group(1)),int(dm.group(2)),int(dm.group(3)),hh,mm,ss)
            except ValueError: pass
    return None

def _utc_to_filename_ts(utc_dt):
    return utc_dt.strftime("%Y-%m-%dT%H-%M-%SZ")

def _parse_filename_utc(stem):
    m=RE_CANONICAL_TS.search(stem)
    if not m: return None
    return datetime(int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)),int(m.group(5)),int(m.group(6)))

def _folder_row(label_text,browse_slot,default=""):
    group=QGroupBox(label_text); layout=QHBoxLayout(group)
    lbl=QLabel(default or "No folder selected")
    lbl.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred); lbl.setWordWrap(True)
    btn=QPushButton("Browse\u2026"); btn.setFixedWidth(90); btn.clicked.connect(browse_slot)
    layout.addWidget(lbl); layout.addWidget(btn)
    return group,lbl

# --- Tab 0: SD Mesonet Station Scraper ---
MESONET_STATIONS_URL = "https://climate.sdstate.edu/information/stations/"


def _fetch_html_with_fallback(url: str, log_fn=None) -> str:
    """
    Fetch page HTML, trying urllib first and falling back to requests.

    urllib.request does NOT automatically pick up OS/browser proxy
    settings on all platforms; requests (via its underlying use of
    environment variables HTTP_PROXY/HTTPS_PROXY) often succeeds where
    urllib fails on machines behind a proxy or certain VPN configurations.
    If a browser on the same machine can load the page but this still
    fails, the machine is very likely behind a proxy that needs to be
    configured via the HTTPS_PROXY environment variable.

    log_fn, if given, is called with progress strings (e.g. a pyqtSignal.emit).
    """
    headers = {"User-Agent": "Mozilla/5.0 (GRIME-AI sidecar tool)"}

    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
    except Exception as e1:
        if log_fn:
            log_fn(f"urllib fetch failed ({e1.__class__.__name__}: {e1}); trying requests\u2026")

    try:
        import requests
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e2:
        raise RuntimeError(
            "Both urllib and requests failed to reach the page.\n\n"
            f"requests error: {e2.__class__.__name__}: {e2}\n\n"
            "Since the page loads fine in a regular browser on this machine, "
            "this strongly suggests the network is behind a proxy or VPN that "
            "only browser traffic is routed through. Try setting the "
            "HTTPS_PROXY environment variable to your proxy address before "
            "running GRIME AI, e.g.:\n"
            "  set HTTPS_PROXY=http://your-proxy-host:port"
        )


def _parse_mesonet_tables(html: str) -> list:
    """
    Parse the two HTML tables on the SD Mesonet station info page
    (Active / Inactive) into row dicts.
    """
    active_marker   = html.find("Active Stations")
    inactive_marker = html.find("Inactive Stations")
    if active_marker == -1 or inactive_marker == -1 or inactive_marker <= active_marker:
        active_html, inactive_html = html, ""
    else:
        active_html   = html[active_marker:inactive_marker]
        inactive_html = html[inactive_marker:]

    rows = []
    rows.extend(_parse_mesonet_one_table(active_html, "Active"))
    rows.extend(_parse_mesonet_one_table(inactive_html, "Inactive"))
    return rows


def _parse_mesonet_one_table(section_html: str, status: str) -> list:
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

    def _col_index(*names):
        for n in names:
            for i, h in enumerate(header_cells):
                if n in h:
                    return i
        return None

    idx_station = _col_index("station")
    idx_nwsli   = _col_index("nwsli")
    idx_detail  = _col_index("detail")
    idx_county  = _col_index("county")
    idx_start   = _col_index("start")
    idx_end     = _col_index("end")
    idx_lat     = _col_index("lat")
    idx_lon     = _col_index("lon")
    idx_elv     = _col_index("elv")
    idx_tz      = _col_index("time zone", "timezone")

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

        def _get(idx):
            return clean[idx] if idx is not None and idx < len(clean) else ""

        def _num(idx):
            v = _get(idx)
            try:
                return float(v) if v not in ("", None) else ""
            except ValueError:
                return ""

        row = {
            "station": _get(idx_station),
            "nwsli":   _get(idx_nwsli),
            "detail":  _get(idx_detail),
            "county":  _get(idx_county),
            "start":   _get(idx_start),
            "end":     _get(idx_end),
            "lat":     _num(idx_lat),
            "lon":     _num(idx_lon),
            "elv_ft":  _num(idx_elv),
            "utc_offset": _get(idx_tz),
            "status":  status,
        }
        if row["station"] or row["nwsli"]:
            out.append(row)

    return out


def _save_mesonet_xlsx(rows: list, out_path: str) -> None:
    """Save scraped Mesonet station rows to a single-sheet xlsx (Active first, then Inactive)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Mesonet Stations"

    headers = ["Station", "NWSLI", "Detail", "County", "Start", "End",
               "Lat", "Lon", "Elv(Feet)", "UTC Offset", "Status"]
    ws.append(headers)

    ordered = ([r for r in rows if r["status"] == "Active"] +
               [r for r in rows if r["status"] == "Inactive"])

    for row in ordered:
        ws.append([
            row["station"], row["nwsli"], row["detail"], row["county"],
            row["start"], row["end"], row["lat"], row["lon"],
            row["elv_ft"], row["utc_offset"], row["status"],
        ])

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    inactive_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"

    status_col = headers.index("Status")
    for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
        if row_cells[status_col].value == "Inactive":
            for cell in row_cells:
                cell.fill = inactive_fill

    for col_cells in ws.columns:
        max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        col_letter = get_column_letter(col_cells[0].column)
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 8), 28)

    wb.save(out_path)


def _scrape_and_save_mesonet_stations(sidecars_dir: str, log_fn=None) -> dict:
    """
    Scrape the live SD Mesonet station table, save it as
    sidecars/SD_Mesonet_Stations.xlsx, and return the NWSLI -> UTC offset
    dict. Raises on failure (caller decides how to surface the error).
    """
    if log_fn:
        log_fn(f"No Mesonet station table found -- fetching {MESONET_STATIONS_URL} \u2026")
    html = _fetch_html_with_fallback(MESONET_STATIONS_URL, log_fn=log_fn)
    rows = _parse_mesonet_tables(html)
    if not rows:
        raise RuntimeError("No station rows found on the page (page layout may have changed).")

    out_path = os.path.join(sidecars_dir, "SD_Mesonet_Stations.xlsx")
    _save_mesonet_xlsx(rows, out_path)
    if log_fn:
        n_active   = sum(1 for r in rows if r["status"] == "Active")
        n_inactive = sum(1 for r in rows if r["status"] == "Inactive")
        log_fn(f"Saved {out_path} ({n_active} active, {n_inactive} inactive stations).")

    return {r["nwsli"].upper(): int(r["utc_offset"])
            for r in rows if r["nwsli"] and str(r["utc_offset"]).strip() not in ("", "None")}


class MesonetScrapeWorker(QThread):
    """
    Fetches the SD Mesonet station table from
    https://climate.sdstate.edu/information/stations/ and parses the
    Active and Inactive station tables into row dicts.

    NOTE: SD Mesonet image timestamps use fixed standard time year-round
    (CST = UTC-6, MST = UTC-7) -- there is no DST adjustment to apply.
    The "UTC Offset" column scraped here can be used directly.
    """
    log    = pyqtSignal(str)
    done   = pyqtSignal(list)   # list of row dicts (active first, then inactive)
    failed = pyqtSignal(str)

    URL = MESONET_STATIONS_URL

    def run(self):
        try:
            self.log.emit(f"Fetching {self.URL} \u2026")
            html = _fetch_html_with_fallback(self.URL, log_fn=self.log.emit)
            self.log.emit("Page fetched. Parsing tables\u2026")

            rows = _parse_mesonet_tables(html)
            if not rows:
                self.failed.emit("No station rows found on the page (page layout may have changed).")
                return

            n_active   = sum(1 for r in rows if r["status"] == "Active")
            n_inactive = sum(1 for r in rows if r["status"] == "Inactive")
            self.log.emit(f"Parsed {n_active} active and {n_inactive} inactive station(s).")
            self.done.emit(rows)
        except Exception as e:
            self.failed.emit(f"Scrape failed: {e}\n{traceback.format_exc()}")


# --- Tab 0: Image Sidecar Generator ---

# NEON watermark: "Sun Jun 28 2026  21:55:00 - UTC"  (explicit UTC, no offset math needed)

# OCR often returns "15: 10/00" or "15:40\u00b000" for "15:10:00", and clips the
# trailing "UTC" to "UT", so the separators and the tail are both loose here.
RE_NEON_WATERMARK = re.compile(
    r"(\d{1,2})\s*[:;.,/'\u00b0]\s*(\d{2})\s*[:;.,/'\u00b0]\s*(\d{2})(?=.{0,8}?UT)",
    re.IGNORECASE | re.DOTALL)

# PhenoCam watermark: "Mon Jan 15 2024 10:21:13 MST - UTC-7" (local time + explicit offset)
RE_PHENOCAM_WATERMARK = re.compile(
    r"(\d{1,2}:\d{2}:\d{2})\s+\w{2,5}\s*-\s*UTC([+-]\d{1,2})", re.IGNORECASE)

# Full date pulled from a watermark line, used alongside the time regexes above
RE_WATERMARK_DATE_TEXT = re.compile(
    r"(\w{3})\s+(\w{3})\s+(\d{1,2})\s+(\d{4})")  # "Sun Jun 28 2026" / "Mon Jan 15 2024"

_MONTH_ABBR = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

# PhenoCam filename: <name>_yyyy_mm_dd_hhmmss (local time, no offset)
RE_PHENOCAM_FILENAME = re.compile(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})")

# MESONET filename: <NWSLI>_<cam#>_yyyy_mm_dd_hh_mm_ss (local time, 24h, no offset)
RE_MESONET_FILENAME = re.compile(
    r"^([A-Z0-9]{4,7})_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


def _load_mesonet_offsets(sidecars_dir: str) -> dict:
    """
    Load NWSLI -> UTC offset (int hours, fixed standard time, no DST) from
    SD_Mesonet_Stations.xlsx if present in the given sidecars folder.
    Returns {} if the file is missing.
    """
    path = os.path.join(sidecars_dir, "SD_Mesonet_Stations.xlsx")
    if not os.path.isfile(path):
        return {}
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb["Mesonet Stations"] if "Mesonet Stations" in wb.sheetnames else wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return {}
        header = [str(h).strip().lower() if h else "" for h in rows[0]]
        idx_nwsli = header.index("nwsli") if "nwsli" in header else None
        idx_offset = header.index("utc offset") if "utc offset" in header else None
        if idx_nwsli is None or idx_offset is None:
            return {}
        out = {}
        for row in rows[1:]:
            if row is None or idx_nwsli >= len(row):
                continue
            code = row[idx_nwsli]
            off  = row[idx_offset] if idx_offset < len(row) else None
            if code and off not in (None, ""):
                try:
                    out[str(code).strip().upper()] = int(off)
                except (TypeError, ValueError):
                    pass
        return out
    except Exception:
        return {}


def _detect_source(filenames: list) -> str:
    """
    Auto-detect the imagery source from a sample of filenames in a folder.
    Returns one of: "USGS", "NEON", "PHENOCAM", "MESONET", "UNKNOWN".
    """
    sample = filenames[:25] if len(filenames) > 25 else filenames
    votes = {"USGS": 0, "NEON": 0, "PHENOCAM": 0, "MESONET": 0}
    for fname in sample:
        stem = os.path.splitext(fname)[0]
        if RE_CANONICAL_TS.search(stem):
            votes["USGS"] += 1
        elif stem.upper().startswith("NEON."):
            votes["NEON"] += 1
        elif RE_MESONET_FILENAME.match(stem):
            votes["MESONET"] += 1
        elif RE_PHENOCAM_FILENAME.search(stem):
            votes["PHENOCAM"] += 1
    if not any(votes.values()):
        return "UNKNOWN"
    return max(votes, key=votes.get)


def _build_canonical_filename(prefix: str, utc_dt) -> str:
    """
    Build the full canonical USGS-style filename: SITE___YYYY-MM-DDTHH-MM-SSZ
    The prefix is whatever identifies the site/camera in the original
    filename (with any trailing separator stripped), joined to the UTC
    timestamp with a triple underscore, matching the existing USGS convention
    used elsewhere in this codebase.
    """
    prefix = prefix.rstrip("_")
    return f"{prefix}___{_utc_to_filename_ts(utc_dt)}"


def _resolve_usgs(image_path: str, fname: str) -> tuple:
    """Returns (proposed_filename_or_None, status_str, utc_offset_or_None)."""
    stem = os.path.splitext(fname)[0]
    dt = _parse_filename_utc(stem)
    if dt is not None:
        # Already canonical -- prefix is everything before the final "___".
        prefix = stem.rsplit("___", 1)[0] if "___" in stem else stem
        return _build_canonical_filename(prefix, dt), "RESOLVED-FILENAME", 0

    # Old USGS local-time-with-offset format embedded in filename
    parts = stem.split("___")
    if len(parts) == 2:
        m = RE_OLD_USGS_TS.match(parts[1])
        converted = _convert_old_usgs_ts(parts[1])
        if converted:
            off_h = int(m.group(5)) if m else None
            # converted is "YYYY-MM-DDTHH-MM-SSZ"; re-attach the original prefix.
            return f"{parts[0]}___{converted}", "RESOLVED-FILENAME", off_h

    # Fall back to OCR -- USGS watermark is LOCAL TIME with no offset stated,
    # so without a separately-known site UTC offset we cannot safely produce
    # a canonical UTC filename from OCR alone.
    _, ts_line = _ocr_watermark(image_path, engine=OCR_TESSERACT)
    if ts_line:
        local_dt = _parse_watermark_ts(ts_line)
        if local_dt:
            return None, "UNRESOLVED-NEEDS-OFFSET", None
    return None, "UNRESOLVED", None


def _neon_utc_datetime(date_m, stem, hh, mm, ss):
    """
    The UTC date for a NEON image. The watermark date is used when the OCR read
    it. Otherwise the date comes from the filename's local date plus the offset
    implied by the local and UTC clock times, since NEON sites are west of UTC
    and the watermark time is always the later of the two.
    """
    if date_m is not None:
        mon = _MONTH_ABBR.get(date_m.group(2)[:3].title())
        if mon is not None:
            try:
                return datetime(int(date_m.group(4)), mon, int(date_m.group(3)), hh, mm, ss)
            except ValueError:
                return None

    local_m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})$", stem)
    if not local_m:
        return None
    try:
        local_dt = datetime(*(int(g) for g in local_m.groups()))
    except ValueError:
        return None
    local_seconds = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
    utc_seconds = hh * 3600 + mm * 60 + ss
    offset = (utc_seconds - local_seconds) % 86400        # UTC is ahead at these sites
    return local_dt + timedelta(seconds=offset)


def _ocr_failure_status() -> str:
    """Status for an image the OCR could not read, naming the cause when known."""
    err = last_ocr_error()
    if not err or "no timestamp line" in err:
        return "UNRESOLVED-OCR-NO-TEXT"
    if "no module named" in err.lower():
        return "UNRESOLVED-OCR-NOT-INSTALLED"
    if "easyocr" in err.lower() or "model" in err.lower() or "download" in err.lower():
        return "UNRESOLVED-OCR-UNAVAILABLE"
    if "urlerror" in err.lower() or "connection" in err.lower() or "timed out" in err.lower():
        return "UNRESOLVED-OCR-NO-MODELS"
    return "UNRESOLVED-OCR-FAILED"


def _resolve_neon(image_path: str, fname: str) -> tuple:
    """
    NEON: filename is local time with no offset info anywhere -- OCR of the
    watermark is the ONLY path, since the watermark states UTC directly
    (e.g. "Sun Jun 28 2026  21:55:00 - UTC"). Offset is always 0 since the
    watermark IS UTC, no conversion math involved.

    Returns (proposed_filename_or_None, status_str, utc_offset_or_None).
    """
    stem = os.path.splitext(fname)[0]
    loc, ts_line = _ocr_watermark(image_path, engine=OCR_EASYOCR)
    combined = f"{loc or ''} {ts_line or ''}".strip()

    time_m = RE_NEON_WATERMARK.search(combined) if combined else None
    date_m = RE_WATERMARK_DATE_TEXT.search(combined) if combined else None
    if not time_m:
        # Second attempt: read the watermark one line at a time, which keeps the
        # sky out of the crop and the two lines from running together.
        combined = ocr_watermark_lines(image_path, engine=OCR_EASYOCR)
        if not combined:
            return None, _ocr_failure_status(), None
        time_m = RE_NEON_WATERMARK.search(combined)
        date_m = RE_WATERMARK_DATE_TEXT.search(combined)
    if not time_m:
        return None, "UNRESOLVED-OCR-UNREADABLE", None

    try:
        hh, mm, ss = (int(x) for x in time_m.groups())
        utc_dt = _neon_utc_datetime(date_m, stem, hh, mm, ss)
        if utc_dt is None:
            return None, "UNRESOLVED-OCR-UNREADABLE", None
        # Prefix is the full original stem (e.g. "NEON.D03.BARC.DP1.20002_..."
        # up to where the timestamp would have started). NEON filenames have
        # no embedded UTC timestamp to strip, so use the part before the
        # local _yyyy_mm_dd_hhmmss block if present, else the whole stem.
        date_block = re.search(r"_\d{4}_\d{2}_\d{2}_\d{6}$", stem)
        prefix = stem[:date_block.start()] if date_block else stem
        return _build_canonical_filename(prefix, utc_dt), "RESOLVED-OCR", 0
    except (ValueError, TypeError):
        return None, "UNRESOLVED-OCR-UNREADABLE", None


def _resolve_phenocam(image_path: str, fname: str) -> tuple:
    """
    PhenoCam: filename and EXIF both give LOCAL time with no offset, so
    neither alone is sufficient. OCR of the watermark IS self-contained --
    it states local time AND the UTC offset explicitly
    (e.g. "Mon Jan 15 2024 10:21:13 MST - UTC-7"), so OCR is the resolving path.
    Filename/EXIF are recorded for status purposes but do not resolve alone.

    Returns (proposed_filename_or_None, status_str, utc_offset_or_None).
    """
    stem = os.path.splitext(fname)[0]

    # OCR first -- this is the only self-contained path for PhenoCam.
    loc, ts_line = _ocr_watermark(image_path, engine=OCR_EASYOCR)
    combined = f"{loc or ''} {ts_line or ''}".strip()
    if combined:
        time_off_m = RE_PHENOCAM_WATERMARK.search(combined)
        date_m = RE_WATERMARK_DATE_TEXT.search(combined)
        if time_off_m and date_m:
            try:
                mon = _MONTH_ABBR.get(date_m.group(2)[:3].title())
                day = int(date_m.group(3))
                year = int(date_m.group(4))
                hh, mm, ss = (int(x) for x in time_off_m.group(1).split(":"))
                offset_h = int(time_off_m.group(2))
                if mon is not None:
                    local_dt = datetime(year, mon, day, hh, mm, ss)
                    utc_dt = local_dt - timedelta(hours=offset_h)
                    pm = RE_PHENOCAM_FILENAME.search(stem)
                    prefix = stem[:pm.start()] if pm else stem
                    return _build_canonical_filename(prefix, utc_dt), "RESOLVED-OCR", offset_h
            except (ValueError, TypeError):
                pass

    # Filename / EXIF give local time only, with no offset source --
    # cannot be safely converted to canonical UTC.
    if RE_PHENOCAM_FILENAME.search(stem):
        return None, "UNRESOLVED-NEEDS-OFFSET", None

    exif_dt = _exif_datetime(image_path)
    if exif_dt:
        return None, "EXIF-LOCAL-UNVERIFIED", None

    return None, "UNRESOLVED", None


def _resolve_mesonet(fname: str, offsets: dict) -> tuple:
    """
    MESONET: filename alone gives local time (yyyy_mm_dd_hh_mm_ss, 24h) plus
    the NWSLI station code, which is looked up against the scraped station
    table for a fixed UTC offset (SD Mesonet uses standard time year-round,
    no DST adjustment). No watermark exists, so there is no OCR fallback --
    a missing/unknown NWSLI code is UNRESOLVED.

    Example: MDMS2_2_2025_10_13_00_02_45.jpg -> MDMS2_2___2025-10-13T06-02-45Z
    (prefix "MDMS2_2" = NWSLI code + camera index, preserved from the original
    filename ahead of the date block.)

    Returns (proposed_filename_or_None, status_str, utc_offset_or_None).
    """
    stem = os.path.splitext(fname)[0]
    m = RE_MESONET_FILENAME.match(stem)
    if not m:
        return None, "UNRESOLVED", None

    nwsli = m.group(1).upper()
    year, mon, day, hh, mm, ss = (int(g) for g in m.groups()[1:])

    offset_h = offsets.get(nwsli)
    if offset_h is None:
        return None, "UNRESOLVED-UNKNOWN-STATION", None

    try:
        local_dt = datetime(year, mon, day, hh, mm, ss)
        utc_dt = local_dt - timedelta(hours=offset_h)
        # Prefix is everything in the original stem up to the date block
        # (preserves the NWSLI code AND any camera index, e.g. "MDMS2_2").
        prefix = stem[:m.start(2) - 1]  # -1 to drop the trailing underscore before the year
        return _build_canonical_filename(prefix, utc_dt), "RESOLVED-FILENAME", offset_h
    except ValueError:
        return None, "UNRESOLVED", None

class SidecarGenWorker(QThread):
    progress = pyqtSignal(int, int)   # done, total
    status   = pyqtSignal(str)
    row      = pyqtSignal(str, str, str, str)  # original, proposed, status, utc_offset
    finished = pyqtSignal(str, int, int, int)  # source, resolved, unresolved, total

    def __init__(self, folder: str):
        super().__init__()
        self._folder = folder
        self._cancelled = False

    def cancel(self):
        """Stop after the image being processed; the rows already produced are kept."""
        self._cancelled = True

    def run(self):
        try:
            exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
            filenames = sorted(
                f for f in os.listdir(self._folder)
                if f.lower().endswith(exts) and os.path.isfile(os.path.join(self._folder, f))
            )
            if not filenames:
                self.status.emit("No image files found in this folder.")
                self.finished.emit("UNKNOWN", 0, 0, 0)
                return

            source = _detect_source(filenames)
            self.status.emit(f"Detected source: {source}  ({len(filenames)} image(s) found)")

            if source in ("NEON", "PHENOCAM"):
                try:
                    import easyocr        # noqa: F401  (presence check only)
                except Exception as err:
                    self.status.emit(f"Primary OCR engine unavailable ({err}); "
                                     "the secondary engine will be tried instead.")
                found = easyocr_model_dir()
                if found:
                    self.status.emit(f"OCR models: {found}")
                elif EASYOCR_ALLOW_DOWNLOAD:
                    self.status.emit("OCR models not found; they will be downloaded into "
                                     f"{EASYOCR_MODEL_DIR} on first use.")
                else:
                    self.status.emit("OCR models not found and downloading is disabled "
                                     "for this installation.")

            offsets = {}
            if source == "MESONET":
                sidecars_dir = os.path.join(self._folder, "sidecars")
                offsets = _load_mesonet_offsets(sidecars_dir)
                if not offsets:
                    try:
                        offsets = _scrape_and_save_mesonet_stations(
                            sidecars_dir, log_fn=self.status.emit)
                    except Exception as e:
                        self.status.emit(
                            f"WARNING: Could not auto-fetch SD Mesonet station table "
                            f"({e}) -- all MESONET files will be UNRESOLVED.")

            resolved = unresolved = 0
            ocr_reported = False
            total = len(filenames)
            for i, fname in enumerate(filenames, start=1):
                if self._cancelled:
                    self.status.emit(f"Cancelled after {i - 1} of {total} image(s).")
                    break
                image_path = os.path.join(self._folder, fname)

                if source == "USGS":
                    proposed, status, offset_h = _resolve_usgs(image_path, fname)
                elif source == "NEON":
                    proposed, status, offset_h = _resolve_neon(image_path, fname)
                elif source == "PHENOCAM":
                    proposed, status, offset_h = _resolve_phenocam(image_path, fname)
                elif source == "MESONET":
                    proposed, status, offset_h = _resolve_mesonet(fname, offsets)
                else:
                    proposed, status, offset_h = None, "UNRESOLVED-UNKNOWN-SOURCE", None

                if status.startswith("RESOLVED"):
                    resolved += 1
                else:
                    unresolved += 1
                    if status.startswith("UNRESOLVED-OCR") and last_ocr_error() and not ocr_reported:
                        ocr_reported = True
                        self.status.emit(f"OCR error: {last_ocr_error()}")
                        if status.endswith("NOT-INSTALLED"):
                            self.status.emit(
                                "No Optical Character Recognition (OCR) engine is available to "
                                "this Python environment, and NEON and PhenoCam timestamps come "
                                "only from the watermark. Install one of: pip install easyocr, "
                                "or pip install pytesseract with the Tesseract program itself on "
                                "PATH. The message above names the Python package that is "
                                "missing, which is not the same as the Tesseract program.")
                        elif status.endswith("UNAVAILABLE") or status.endswith("NO-MODELS"):
                            self.status.emit(
                                "EasyOCR models were not found in any of: "
                                + "; ".join(EASYOCR_SEARCH_DIRS)
                                + f". They are downloaded into {EASYOCR_MODEL_DIR} on first use, "
                                  "which needs a network connection.")

                offset_str = f"UTC{offset_h:+d}" if offset_h is not None else ""
                self.row.emit(fname, proposed or "", status, offset_str)
                self.progress.emit(i, total)

            self.finished.emit(source, resolved, unresolved, resolved + unresolved)
        except Exception as e:
            self.status.emit(f"ERROR: {e}\n{traceback.format_exc()}")
            self.finished.emit("UNKNOWN", 0, 0, 0)


class ImageSidecarGeneratorTab(QWidget):
    """
    Tab 0 - Image Sidecar Generator

    Scans a folder of images, auto-detects the imagery source (USGS, NEON,
    PhenoCam, or MESONET), and resolves each filename to the canonical USGS
    UTC filename format (SITE___YYYY-MM-DDTHH-MM-SSZ), writing a 3-column
    sidecar CSV (Original Filename, Canonical Filename, Status, UTC Offset) into a
    "sidecars" subfolder alongside the images.

    Per-source resolution order:
      USGS     : filename (already UTC) -> OCR watermark (local, needs an
                 offset we don't have -> UNRESOLVED-NEEDS-OFFSET)
      NEON     : OCR watermark only (mandatory -- watermark states UTC
                 directly, filename/EXIF have no usable offset info)
      PhenoCam : OCR watermark (self-contained -- states local time AND
                 explicit UTC offset); filename/EXIF alone are local-time-only
                 and cannot be resolved without OCR
      MESONET  : filename only (local time + NWSLI station code) looked up
                 against SD_Mesonet_Stations.xlsx (sidecars/) for a fixed
                 UTC offset (SD Mesonet uses standard time year-round, no
                 DST). No watermark exists, so no OCR fallback is possible.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._worker = None
        self._mesonet_worker = None
        self._rows = []  # (original, proposed, status, utc_offset)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        info = QLabel(
            "Auto-detects the imagery source (USGS, NEON, PhenoCam, or MESONET) "
            "from filenames in the selected folder, resolves each image's "
            "timestamp to the canonical USGS UTC filename format using "
            "filename parsing, EXIF, and/or OCR as appropriate for that "
            "source, and writes a 3-column sidecar CSV (Original Filename, "
            "Canonical Filename, Status) into a \"sidecars\" subfolder.\n\n"
            "Mesonet folders need SD_Mesonet_Stations.xlsx in the sidecars subfolder; "
            "use Download Station Metadata below, or let the scan fetch it."
        )
        info.setStyleSheet("color: #444; font-size: 9pt; background: #f5f5f5; "
                           "padding: 6px; border-radius: 3px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        saved = self._settings.get("sidecar_gen_folder", "")
        grp, self._folder_lbl = _folder_row(
            "Folder containing images to resolve", self._browse, saved)
        layout.addWidget(grp)

        btn_row = QHBoxLayout()
        self._scan_btn = QPushButton("Generate Sidecar")
        self._scan_btn.setFixedHeight(32)
        self._scan_btn.clicked.connect(self._run)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setToolTip("Stop the scan. The images already processed are kept "
                                    "and written to the sidecar CSV.")
        self._cancel_btn.clicked.connect(self._cancel_run)

        self._mesonet_btn = QPushButton("Download Station Metadata")
        self._mesonet_btn.setFixedHeight(32)
        self._mesonet_btn.setToolTip(
            "Scrapes the live SD Mesonet station table from climate.sdstate.edu "
            "and saves it as sidecars/SD_Mesonet_Stations.xlsx in the selected folder. "
            "Run this before generating sidecars for MESONET images, or if the "
            "station list has changed.")
        self._mesonet_btn.clicked.connect(self._refresh_mesonet)

        btn_row.addWidget(self._scan_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._mesonet_btn)
        layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet("font-size: 9pt; color: #333;")
        layout.addWidget(self._summary_lbl)

        tbl_group = QGroupBox("Results")
        tbl_layout = QVBoxLayout(tbl_group)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Original Filename", "Canonical Filename (USGS UTC)", "Status", "UTC Offset"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        tbl_layout.addWidget(self._table)
        layout.addWidget(tbl_group)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier", 8))
        self._log.setMaximumHeight(100)
        status_layout.addWidget(self._log)
        layout.addWidget(status_group)

    def _browse(self):
        start  = self._settings.get("sidecar_gen_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            self._folder_lbl.setText(folder)
            self._settings["sidecar_gen_folder"] = folder
            _save_settings(self._settings)

    def _run(self):
        folder = self._settings.get("sidecar_gen_folder", "")
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please select a valid folder.")
            return

        if not self._confirm_ocr_models(folder):
            return

        self._rows = []
        self._table.setRowCount(0)
        self._summary_lbl.setText("")
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._scan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._mesonet_btn.setEnabled(False)
        self._log.clear()
        self._log.appendPlainText("Scanning folder\u2026")

        self._worker = SidecarGenWorker(folder)
        self._worker.status.connect(self._log.appendPlainText)
        self._worker.progress.connect(self._on_progress)
        self._worker.row.connect(self._on_row)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _confirm_ocr_models(self, folder) -> bool:
        """
        Ask before anything is downloaded. NEON and PhenoCam folders need the OCR
        models; when none are installed the user decides whether to fetch them.
        Returns False only if the user cancels the scan itself.
        """
        approve_model_download(False)
        try:
            filenames = [f for f in os.listdir(folder)
                         if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
        except OSError:
            return True
        if _detect_source(filenames) not in ("NEON", "PHENOCAM"):
            return True            # these sources read the filename, no OCR needed
        if easyocr_model_dir() is not None:
            return True            # models already installed

        if not EASYOCR_ALLOW_DOWNLOAD:
            QMessageBox.warning(
                self, "OCR Models Missing",
                "The Optical Character Recognition (OCR) model files were not found, and "
                "downloading is disabled for this installation.\n\n"
                "They are expected in:\n"
                f"{EASYOCR_ENV_DIR or EASYOCR_MODEL_DIR}\n\n"
                "The scan will continue, but these images cannot be read.")
            return True

        answer = QMessageBox.question(
            self, "OCR Models Missing",
            "These images carry their timestamp only in the watermark, and reading it needs "
            "Optical Character Recognition (OCR). The OCR model files are not installed.\n\n"
            "Download them now? This requires an internet connection, transfers about "
            "100 MB, and happens once.\n\n"
            f"They will be saved in:\n{EASYOCR_MODEL_DIR}\n\n"
            "Choose No to run without OCR: these images will be reported as unresolved.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)

        if answer == QMessageBox.Cancel:
            return False
        approve_model_download(answer == QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._log.appendPlainText(
                f"Downloading OCR models into {EASYOCR_MODEL_DIR} on first use\u2026")
        return True

    def _cancel_run(self):
        if self._worker is not None and self._worker.isRunning():
            self._cancel_btn.setEnabled(False)
            self._log.appendPlainText("Cancelling\u2026")
            self._worker.cancel()

    def _on_progress(self, done, total):
        if total > 0:
            self._progress.setValue(int(done * 100 / total))

    def _on_row(self, original, proposed, status, utc_offset):
        self._rows.append((original, proposed, status, utc_offset))
        r = self._table.rowCount()
        self._table.insertRow(r)
        for c, val in enumerate([original, proposed, status, utc_offset]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            if status.startswith("RESOLVED"):
                item.setForeground(Qt.darkGreen)
            elif status.startswith("EXIF-LOCAL-UNVERIFIED"):
                item.setForeground(QColor(180, 130, 0))
            else:
                item.setForeground(Qt.red)
            self._table.setItem(r, c, item)

    def _on_finished(self, source, resolved, unresolved, total):
        self._progress.setVisible(False)
        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._mesonet_btn.setEnabled(True)
        if total == 0:
            self._summary_lbl.setText("No images processed.")
            return
        self._summary_lbl.setText(
            f"Source: {source}  |  {total} image(s)  |  "
            f"{resolved} resolved  |  {unresolved} unresolved")
        self._log.appendPlainText(
            f"Done. {resolved} resolved, {unresolved} unresolved out of {total}.")
        if self._rows:
            self._write_sidecar_csv()

    def _refresh_mesonet(self):
        """Manually trigger a fresh scrape of the SD Mesonet station table."""
        folder = self._settings.get("sidecar_gen_folder", "")
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder",
                "Please select a valid image folder first — the station table "
                "will be saved into its sidecars/ subfolder.")
            return

        self._mesonet_btn.setEnabled(False)
        self._log.appendPlainText("Fetching SD Mesonet station table\u2026")

        self._mesonet_worker = MesonetScrapeWorker()
        self._mesonet_worker.log.connect(self._log.appendPlainText)
        self._mesonet_worker.done.connect(self._on_mesonet_done)
        self._mesonet_worker.failed.connect(self._on_mesonet_failed)
        self._mesonet_worker.start()

    def _on_mesonet_done(self, rows):
        folder = self._settings.get("sidecar_gen_folder", "")
        sidecars_dir = os.path.join(folder, "sidecars")
        out_path = os.path.join(sidecars_dir, "SD_Mesonet_Stations.xlsx")
        try:
            _save_mesonet_xlsx(rows, out_path)
            n_active   = sum(1 for r in rows if r["status"] == "Active")
            n_inactive = sum(1 for r in rows if r["status"] == "Inactive")
            self._log.appendPlainText(
                f"Saved {out_path} ({n_active} active, {n_inactive} inactive stations).")
        except Exception as e:
            self._log.appendPlainText(f"ERROR saving station table: {e}")
        self._mesonet_btn.setEnabled(True)

    def _on_mesonet_failed(self, msg):
        self._log.appendPlainText(f"ERROR: {msg}")
        self._mesonet_btn.setEnabled(True)

    @staticmethod
    def _hyperlink(folder, filename):
        """Excel HYPERLINK formula opening the image, shown as its file name.
        Plain CSV readers see the formula text, so the name is kept inside it."""
        path = os.path.join(folder, filename).replace('"', '""')
        return f'=HYPERLINK("{path}", "{filename}")'

    def _write_sidecar_csv(self):
        """Auto-write the sidecar CSV into the sidecars/ subfolder on scan completion."""
        folder = self._settings.get("sidecar_gen_folder", "")
        if not folder or not os.path.isdir(folder) or not self._rows:
            return

        sidecars_dir = os.path.join(folder, "sidecars")
        os.makedirs(sidecars_dir, exist_ok=True)
        out_path = os.path.join(sidecars_dir, "image_timestamp_sidecar.csv")

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Original Filename", "Canonical Filename (USGS UTC)", "Status", "UTC Offset"])
                for original, proposed, status, offset in self._rows:
                    writer.writerow([self._hyperlink(folder, original), proposed, status, offset])
            self._log.appendPlainText(f"Sidecar saved: {out_path}")
        except Exception as e:
            self._log.appendPlainText(f"ERROR saving sidecar CSV: {e}")


# --- Tab 0b: SD Mesonet Station Scraper ---


# ======================================================================================================================
# Standalone use: python image_timestamp_sidecar.py [folder]
# ======================================================================================================================
def main(argv=None):
    """Open the tool in its own window, outside the host application."""
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    window = ImageSidecarGeneratorTab()
    window.setWindowTitle(PLUGIN["title"])
    if len(argv) > 1 and os.path.isdir(argv[1]):
        window._folder_lbl.setText(argv[1])
        window._settings["sidecar_gen_folder"] = argv[1]
    window.resize(1000, 700)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
