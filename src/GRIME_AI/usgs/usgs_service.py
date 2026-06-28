import os
import json
import urllib.request
import requests
import pandas as pd

from typing import Dict, Optional

from .usgs_types import CameraInfo, LatestImage
from GRIME_AI.dialogs.api_keys import APIKeyManager

# Legacy AWS endpoint - kept as fallback until USGS decommissions it (after July 2026)
ENDPOINT_LEGACY  = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com"
# New NIMS API endpoint (api.waterdata.usgs.gov)
ENDPOINT_NEW     = "https://api.waterdata.usgs.gov/nims/v0"
# S3 image base - overlayDir from /cameras confirms this per-camera
IMAGE_ENDPOINT   = "https://usgs-nims-images.s3.amazonaws.com/overlay"


# ================================================================================
# ================================================================================
#                               HELPER FUNCTIONS
# ================================================================================
# ================================================================================
@staticmethod
def _to_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


from datetime import datetime, date, time, timedelta
from typing import List, Tuple, Union
from urllib.parse import quote


def _to_date(d):
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.fromisoformat(str(d)).date()


def _to_time(t):
    if isinstance(t, time):
        return t
    if isinstance(t, datetime):
        return t.time()
    parts = str(t).split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]), 0)
    if len(parts) == 3:
        return time(int(parts[0]), int(parts[1]), int(parts[2]))
    return datetime.fromisoformat(str(t)).time()


def build_hivis_urls_and_count(
        endpoint: str,
        cam_id: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        start_time: Union[str, time, datetime],
        end_time: Union[str, time, datetime],
) -> Tuple[int, List[str] | int]:
    def format_timestamp(this_day, start_time_for_day, end_time_for_day):

        if start_time_for_day == time(0, 0, 0) and end_time_for_day == time(0, 0, 0):
            after_dt = datetime.combine(this_day, time(0, 0, 0))
            before_dt = datetime.combine(this_day, time(23, 59, 59))
        else:
            after_dt = datetime.combine(this_day, start_time_for_day)
            before_dt = datetime.combine(this_day, end_time_for_day)

        formatted_after_val = after_dt.strftime("%Y-%m-%d:%H:%M:%S")
        formatted_before_val = before_dt.strftime("%Y-%m-%d:%H:%M:%S")

        return formatted_after_val, formatted_before_val

    # INITIALIZE VARIABLES
    start_d = _to_date(start_date)
    end_d = _to_date(end_date)
    t_start = _to_time(start_time)
    t_end = _to_time(end_time)

    endpoint = endpoint.rstrip("?")
    cam_q = quote(str(cam_id), safe="")

    urls: List[str] = []

    # ------------------------------------------------------------------------
    # IF THE START DATE IS AFTER THE END DATE, RETURN INVALID!
    # ------------------------------------------------------------------------
    if start_d == end_d and t_start > t_end:
        return -2, []

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    day = start_d
    while day <= end_d:
        after_val, before_val = format_timestamp(day, t_start, t_end)

        url = f"{endpoint}?camId={cam_q}&after={after_val}&before={before_val}"
        urls.append(url)

        day += timedelta(days=1)

    return len(urls), urls


# ================================================================================
# ================================================================================
#                               class USGSService
# ================================================================================
# ================================================================================
class USGSService:
    """
    Pure service: fetches camera metadata, lists, images, and USGS discharge data.
    No Qt. No GUI side-effects. Raises exceptions on hard failures.
    """

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def __init__(self, hivis_instance=None):
        self._camera_dict: Dict[str, dict] = {}
        self._site_count: int = 0
        self._nwis_id: Optional[str] = None
        self._cam_id: Optional[str] = None
        self._cam_name: Optional[str] = None
        self._api_key: str = ""

        # Load endpoint and API key from settings
        try:
            _mgr = APIKeyManager()
            self._api_key   = _mgr.get_usgs_key() or ""
            self._endpoint  = _mgr.get_usgs_endpoint() or ENDPOINT_NEW
        except Exception as _e:
            print(f"[USGSService] Could not load settings: {_e}")
            self._endpoint  = ENDPOINT_NEW

        # ============================================================================
        # REFERENCE TO USGS_HIVIS FOR CACHE ACCESS
        # This allows us to use cached filenames from get_image_count
        # ============================================================================
        self.hivis = hivis_instance

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def set_api_key(self, key: str) -> None:
        """Store the USGS Water Data API key. Pass empty string to clear."""
        self._api_key = key or ""

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def set_endpoint(self, endpoint: str) -> None:
        """Update the NIMS API base endpoint at runtime."""
        self._endpoint = endpoint.rstrip("/") if endpoint else ENDPOINT_NEW

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def _api_headers(self) -> dict:
        """Return X-Api-Key header if a key is set, else empty dict.
        USGS downloads work without a key; the key only raises rate limits.
        """
        if self._api_key:
            return {"X-Api-Key": self._api_key}
        return {}

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def initialize(self) -> None:
        """Fetch the camera list.  Tries the new NIMS endpoint first,
        falls back to the legacy AWS Lambda endpoint if it fails.
        The active endpoint is whatever was loaded from APIKeyManager
        (defaults to ENDPOINT_NEW).
        """
        cam_dict: Dict[str, dict] = {}

        endpoints_to_try = [
            (self._endpoint, "new"),
            (ENDPOINT_LEGACY, "legacy"),
        ]
        # Avoid trying the same URL twice if the user has set the endpoint to legacy
        if self._endpoint == ENDPOINT_LEGACY:
            endpoints_to_try = [(ENDPOINT_LEGACY, "legacy")]

        for endpoint, label in endpoints_to_try:
            try:
                if "api.waterdata.usgs.gov" in endpoint:
                    # New NIMS API - returns a list directly, no locus/hideCam filter
                    uri = f"{endpoint}/cameras"
                    resp = requests.get(uri, headers=self._api_headers(), timeout=15)
                    resp.raise_for_status()
                    camera_data = resp.json()
                    for element in camera_data:
                        cam_id = element.get("camId")
                        if isinstance(cam_id, str):
                            cam_dict[cam_id] = element
                else:
                    # Legacy AWS Lambda endpoint
                    uri = f"{endpoint}/prod/cameras?enabled=true"
                    data = urllib.request.urlopen(uri, timeout=15).read()
                    camera_data = json.loads(data.decode("utf-8"))
                    for element in camera_data:
                        if element.get("locus") == "aws" and not element.get("hideCam", True):
                            cam_id = element.get("camId")
                            if isinstance(cam_id, str):
                                cam_dict[cam_id] = element

                print(f"[USGSService] Loaded {len(cam_dict)} cameras from {label} endpoint")
                self._camera_dict = cam_dict
                break  # success - no need to try fallback

            except Exception as e:
                print(f"[USGSService] {label} endpoint failed: {e}")
                cam_dict = {}

        self._camera_dict = cam_dict
        self._site_count = len(self._camera_dict)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def camera_dictionary(self) -> Dict[str, dict]:
        return self._camera_dict

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def camera_list(self) -> List[str]:
        return sorted(self._camera_dict.keys())

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def camera_info(self, camera_id: str) -> CameraInfo:
        cam = self._camera_dict.get(camera_id)
        if cam is None:
            return CameraInfo(camera_id, None, None, None, None, None, None)
        self._nwis_id = cam.get("nwisId")
        self._cam_id = cam.get("camId")
        self._cam_name = cam.get("camName")
        return CameraInfo(
            cam_id=cam.get("camId", camera_id),
            nwis_id=cam.get("nwisId"),
            cam_name=cam.get("camName"),
            lat=_to_float(cam.get("lat")),
            lng=_to_float(cam.get("lng")),
            tz=cam.get("tz"),
            description=cam.get("camDesc")
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def latest_image(self, site_name: str) -> LatestImage:
        url = f"{IMAGE_ENDPOINT}/{site_name}/{site_name}_newest.jpg"
        r = requests.get(url, stream=True)
        if r.status_code == 404:
            return LatestImage(error_code=404, content=None)
        r.raise_for_status()
        content = urllib.request.urlopen(url).read()
        return LatestImage(error_code=0, content=content)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def image_count(self, site_name: str, start_date: date, end_date: date,
                    start_time: time, end_time: time,
                    progress: Optional[callable] = None) -> int:
        names = self._collect_image_names(site_name, start_date, end_date, start_time, end_time, progress)
        return len(names)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def download_images(self, site_name: str, start_date: date, end_date: date,
                        start_time: time, end_time: time, save_folder: str,
                        progress: Optional[callable] = None, cancel_check=None) -> Tuple[int, int]:
        """
        Download images. Now optimized to use cached filenames from get_image_count.
        Falls back to old method if cache unavailable.
        """
        os.makedirs(save_folder, exist_ok=True)

        # ============================================================================
        # TRY TO USE CACHED FILENAMES FIRST (AVOIDS DUPLICATE API CALL)
        # ============================================================================
        names = None
        if self.hivis is not None:
            try:
                names = self.hivis.get_cached_filenames(site_name, start_date, end_date, start_time, end_time)
                if names:
                    print(f"✓ Using {len(names)} cached filenames for download")
            except Exception as e:
                print(f"Could not access cache: {e}")
                names = None

        # ============================================================================
        # CACHE MISS - FETCH FILENAMES THE OLD WAY
        # ============================================================================
        if names is None:
            print("✗ Cache miss - fetching image list...")
            names = self._collect_image_names(site_name, start_date, end_date, start_time, end_time, progress)

        # ============================================================================
        # DOWNLOAD IMAGES USING THE LIST
        # ============================================================================
        downloaded, missing = 0, 0
        total = len(names)
        for idx, image in enumerate(names):
            if progress:
                progress(idx, total, image)
            if cancel_check and cancel_check():
                print("USGS image download cancelled by user.")
                break
            if not image or image == "[]":
                continue
            try:
                file_url = f"{IMAGE_ENDPOINT}/{site_name}/{image}"
                dst = os.path.join(save_folder, image)

                # Normalize: if `image` is absolute, strip it down
                dst = os.path.join(save_folder, os.path.basename(image))

                # Only increment if we actually download
                if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
                    urllib.request.urlretrieve(file_url, dst)
                    downloaded += 1
                else:
                    # Optional: track skipped files separately
                    # skipped += 1
                    pass
            except Exception as e:
                print(f"Download failed for {dst}: {e}")
                missing += 1
        return downloaded, missing

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def download_images_from_list(self, site_name: str, names: List[str],
                                  save_folder: str,
                                  progress: Optional[callable] = None,
                                  cancel_check=None) -> Tuple[int, int]:
        """
        Download a pre-filtered list of image filenames.
        Used by USGS_HIVIS.download_images() to avoid re-querying the API
        when the timezone-aware filtered list is already cached.
        """
        os.makedirs(save_folder, exist_ok=True)
        downloaded, missing = 0, 0
        total = len(names)
        for idx, image in enumerate(names):
            if progress:
                progress(idx, total, image)
            if cancel_check and cancel_check():
                print("USGS image download cancelled by user.")
                break
            if not image or image == "[]":
                continue
            try:
                file_url = f"{IMAGE_ENDPOINT}/{site_name}/{image}"
                dst = os.path.join(save_folder, os.path.basename(image))
                if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
                    urllib.request.urlretrieve(file_url, dst)
                    downloaded += 1
            except Exception as e:
                print(f"Download failed for {image}: {e}")
                missing += 1
        return downloaded, missing

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def get_available_parameters(self, nwis_id: str) -> List[Dict]:
        """
        Query NWIS waterservices for all available time series parameters
        at a given site.  Returns a list of dicts with keys:
            'code'        - parameter code e.g. '00060'
            'description' - human-readable name e.g. 'Discharge, cfs'
            'ts_id'       - time series ID
        Returns [] if nwis_id is None/empty or the request fails.
        """
        if not nwis_id:
            print(f"[USGS] get_available_parameters: no nwis_id, returning []")
            return []
        url = (
            f"https://waterservices.usgs.gov/nwis/iv/"
            f"?format=json&sites={nwis_id}&siteStatus=all"
        )
        print(f"[USGS] Fetching parameters for nwis_id={nwis_id}")
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            params = []
            for ts in data.get('value', {}).get('timeSeries', []):
                var = ts.get('variable', {})
                code = var.get('variableCode', [{}])[0].get('value', '')
                desc = var.get('variableDescription', '')
                ts_id = ts.get('name', '')
                if code:
                    params.append({
                        'code': code,
                        'description': desc,
                        'ts_id': ts_id,
                    })
            print(f"[USGS] Found {len(params)} parameters for nwis_id={nwis_id}")
            return params
        except Exception as e:
            print(f"[USGS] get_available_parameters error for nwis_id={nwis_id}: {e}")
            return []

    def midday_image(self, site_name: str, tz_str: Optional[str]) -> LatestImage:
        """
        Return the image whose timestamp is closest to 12:00 local time
        for the most recent date that has any images.
        Uses _collect_image_names with a 10:00-14:00 local time window.
        Returns LatestImage(404, None) if no images are found.
        """
        # Resolve UTC offset
        TZ_OFFSETS = {
            'EST': -5, 'EDT': -4, 'CST': -6, 'CDT': -5,
            'MST': -7, 'MDT': -6, 'PST': -8, 'PDT': -7,
            'AKST': -9, 'AKDT': -8, 'HST': -10, 'UTC': 0,
        }
        PYTZ_OFFSETS = {
            'US/EASTERN': -5, 'AMERICA/NEW_YORK': -5,
            'US/CENTRAL': -6, 'AMERICA/CHICAGO': -6,
            'US/MOUNTAIN': -7, 'AMERICA/DENVER': -7,
            'US/PACIFIC': -8, 'AMERICA/LOS_ANGELES': -8,
            'US/ALASKA': -9, 'AMERICA/ANCHORAGE': -9,
            'US/HAWAII': -10, 'PACIFIC/HONOLULU': -10,
            'UTC': 0,
        }
        tz_key = (tz_str or '').upper().strip().replace(' ', '_')
        if tz_key in TZ_OFFSETS:
            offset_h = TZ_OFFSETS[tz_key]
        elif tz_key in PYTZ_OFFSETS:
            offset_h = PYTZ_OFFSETS[tz_key]
        else:
            try:
                import pytz
                tz_obj = pytz.timezone(tz_str)
                now_aware = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(tz_obj)
                offset_h = now_aware.utcoffset().total_seconds() / 3600
            except Exception:
                offset_h = 0
        utc_delta = timedelta(hours=offset_h)

        # Today in local time
        today_local = datetime.utcnow() + utc_delta
        today_date  = today_local.date()

        # Convert 10:00-14:00 local to UTC for _collect_image_names
        win_start_utc = datetime.combine(today_date, time(10, 0, 0)) - utc_delta
        win_end_utc   = datetime.combine(today_date, time(14, 0, 0)) - utc_delta

        # _collect_image_names treats times as UTC
        names = self._collect_image_names(
            site_name,
            win_start_utc.date(), win_end_utc.date(),
            win_start_utc.time(), win_end_utc.time(),
            progress=None
        )

        if not names:
            print(f"[USGS midday] No images in window for {site_name}, falling back to latest")
            return self.latest_image(site_name), False

        print(f"[USGS midday] Got {len(names)} filenames. First: {names[0]}")

        # Find the image closest to noon local time.
        # Filename format: SITE_NAME___YYYY-MM-DDTHH-MM-SSZ.jpg  (timestamps are UTC)
        noon_local = datetime.combine(today_date, time(12, 0, 0))
        best_name  = None
        best_delta = None

        for name in names:
            base = os.path.basename(name)
            if '___' not in base:
                continue
            try:
                ts_str = base.split('___')[1]
                ts_str = ts_str.replace('.jpg', '').replace('.JPG', '')
                dt_utc   = datetime.strptime(ts_str, '%Y-%m-%dT%H-%M-%SZ')
                dt_local = dt_utc + utc_delta
                delta    = abs((dt_local - noon_local).total_seconds())
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_name  = name
            except (ValueError, IndexError):
                continue

        if not best_name:
            print(f"[USGS midday] Could not parse timestamp from any filename, falling back to latest")
            return self.latest_image(site_name), False

        print(f"[USGS midday] Best match: {best_name} (delta={best_delta}s from noon)")

        url = f"{IMAGE_ENDPOINT}/{site_name}/{os.path.basename(best_name)}"
        try:
            content = urllib.request.urlopen(url, timeout=15).read()
            return LatestImage(error_code=0, content=content), True
        except Exception:
            return self.latest_image(site_name), False

    def fetch_stage_and_discharge(self, nwis_id: str, site_name: str,
                                  start_date: date, end_date: date,
                                  start_time: time, end_time: time,
                                  save_folder: str) -> Tuple[str, str]:
        os.makedirs(save_folder, exist_ok=True)
        base = "https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites="
        url = f"{base}{nwis_id}&startDT={start_date.strftime('%Y-%m-%d')}&endDT={end_date.strftime('%Y-%m-%d')}&siteStatus=all"
        timestamp = f"{start_date.strftime('%Y-%m-%d')}T{start_time.strftime('%H%M')} - {end_date.strftime('%Y-%m-%d')}T{end_time.strftime('%H%M')}"
        txt_path = os.path.join(save_folder, f"{site_name} - {nwis_id} - {timestamp}.txt")
        csv_path = os.path.join(save_folder, f"{site_name} - {nwis_id} - {timestamp}.csv")
        with urllib.request.urlopen(url) as resp:
            _ = resp.read()
        urllib.request.urlretrieve(url, txt_path)
        self._reformat_file(txt_path, csv_path)
        return txt_path, csv_path

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _collect_image_names(self, site_name: str, start_date: date, end_date: date,
                             start_time: time, end_time: time,
                             progress: Optional[callable]) -> List[str]:
        """
        Collects image names using per-day API calls.
        Now includes retry logic with DNS error handling.
        """
        names: List[str] = []
        days = (end_date - start_date).days + 1
        for i in range(days):
            if progress:
                progress(i, days, None)
            after, before = self._build_image_datetime_filter(i, start_date, start_time, end_time)
            text = self._fetch_list_of_images(site_name, after, before)
            if text and text != "[]":
                cleaned = text.replace("[", "").replace("]", "").replace('"', "")
                if cleaned:
                    parts = [p for p in cleaned.split(",") if p]
                    names.extend(parts)
        return names

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _build_image_datetime_filter(self, index: int, start_date: date,
                                     start_time: time, end_time: time) -> Tuple[str, str]:

        '''
        https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2025-02-07:22:59:00&before=2025-02-07:23:01:00
        https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=CA_SALINAS_R_NR_BRADLEY_CA&after=2025-12-01:11:59:30&before=2025-12-01:13:00:30
        :param index:
        :param start_date:
        :param start_time:
        :param end_time:
        :return:
        '''
        start_day = start_date + timedelta(days=index)
        if start_time.hour == 0 and start_time.minute == 0 and end_time.hour == 0 and end_time.minute == 0:
            day_start = datetime.combine(start_day, time(0, 0, 0))
            day_end = datetime.combine(start_day, time(23, 59, 59))
        else:
            day_start = datetime.combine(start_day, start_time)
            day_end = datetime.combine(start_day, end_time)
        after_dt = day_start - timedelta(seconds=30)
        before_dt = day_end + timedelta(seconds=30)
        after = f"&after={after_dt.strftime('%Y-%m-%d:%H:%M:%S')}"
        before = f"&before={before_dt.strftime('%Y-%m-%d:%H:%M:%S')}"
        return after, before

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _fetch_list_of_images(self, site_name: str, after: str, before: str) -> str:
        """
        Fetch list of images with retry logic and DNS error handling.
        """
        import time as time_module

        if "api.waterdata.usgs.gov" in self._endpoint:
            # New NIMS API - ISO 8601 timestamps, no /prod/ prefix
            url = f"{self._endpoint}/listFiles?camId={site_name}{after}{before}"
        else:
            # Legacy AWS Lambda endpoint
            url = f"{self._endpoint}/prod/listFiles?camId={site_name}{after}{before}"

        # ============================================================================
        # RETRY LOGIC FOR NETWORK ISSUES
        # ============================================================================
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=self._api_headers(), timeout=30)
                resp.raise_for_status()
                return resp.text

            except requests.exceptions.ConnectionError as e:
                # Check if it's a DNS error specifically
                if "getaddrinfo failed" in str(e) or "Failed to resolve" in str(e):
                    print(f"DNS resolution failed (attempt {attempt + 1}/{max_retries}): {url}")

                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time_module.sleep(retry_delay)
                        continue
                    else:
                        print(f"All retries exhausted. DNS resolution failed for: {url}")
                        print("Please check:")
                        print("  1. Internet connection is active")
                        print("  2. Firewall/VPN is not blocking AWS domains")
                        print("  3. DNS server is working")
                        raise  # Re-raise to let caller handle
                else:
                    # Other connection error (not DNS)
                    print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time_module.sleep(retry_delay)
                        continue
                    else:
                        raise

            except requests.exceptions.Timeout as e:
                print(f"Timeout (attempt {attempt + 1}/{max_retries}): {url}")
                if attempt < max_retries - 1:
                    time_module.sleep(retry_delay)
                    continue
                else:
                    raise

            except requests.exceptions.HTTPError as e:
                # Don't retry HTTP errors (404, 500, etc.)
                print(f"HTTP error: {e}")
                raise

            except Exception as e:
                # Unexpected error
                print(f"Unexpected error fetching {url}: {e}")
                if attempt < max_retries - 1:
                    time_module.sleep(retry_delay)
                    continue
                else:
                    raise

        # Should never reach here, but just in case
        return "[]"

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _reformat_file(self, input_txt: str, output_csv: str) -> None:
        df = pd.read_csv(input_txt, delimiter="\t", comment="#")
        df = df[~df["agency_cd"].astype(str).str.contains("5s")]
        df.to_csv(output_csv, index=False)

    def build_hivis_command(self, endpoint, cam_id, start_date, end_date, start_time, end_time):
        """
        Construct a GET command URL for the USGS HIVIS listFiles endpoint.

        Parameters
        - endpoint (str): base endpoint, e.g. "https://.../prod/listFiles"
        - cam_id (str): camera identifier, e.g. "CA_SALINAS_R_NR_BRADLEY_CA"
        - start_date (str|date|datetime): "YYYY-MM-DD" or date object
        - end_date (str|date|datetime): "YYYY-MM-DD" or date object
        - start_time (str|time|datetime): "HH:MM" or "HH:MM:SS" or time object
        - end_time (str|time|datetime): "HH:MM" or "HH:MM:SS" or time object

        Returns
        - str: full URL like "https://.../listFiles?camId=...&after=YYYY-MM-DD:HH:MM:SS&before=YYYY-MM-DD:HH:MM:SS"
        """
        date_a = _to_date_str(start_date)
        date_b = _to_date_str(end_date)
        time_a = _to_time_str(start_time)
        time_b = _to_time_str(end_time)

        after = f"{date_a}:{time_a}"
        before = f"{date_b}:{time_b}"

        # Quote values; keep colons in the timestamp readable by using safe=':'
        cam_q = quote(str(cam_id), safe='')
        after_q = quote(after, safe=':')
        before_q = quote(before, safe=':')

        # Ensure endpoint has no trailing '?'
        endpoint = endpoint.rstrip('?')

        return f"{endpoint}?camId={cam_q}&after={after_q}&before={before_q}"