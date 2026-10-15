import os
import json
import urllib.request
import requests
import pandas as pd

from typing import Dict, Optional

from .usgs_types import CameraInfo, LatestImage
from appcore.reporting.gap_report import generate_gap_report
from appcore.dialogs.api_keys import APIKeyManager

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

# Local noon, the center of the midday window for the midday preview image
MIDDAY           = time(12, 0, 0)
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
        self._all_camera_dict: Dict[str, dict] = {}   # unfiltered (includes hidden cameras)
        self._show_hidden: bool = False
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
                    # New NIMS API - returns a list directly; apply hideCam filter (no locus field)
                    uri = f"{endpoint}/cameras"
                    resp = requests.get(uri, headers=self._api_headers(), timeout=15)
                    resp.raise_for_status()
                    camera_data = resp.json()
                    for element in camera_data:
                        # Keep ALL cameras (including hidden); visibility is
                        # applied later by _apply_hidden_filter().
                        cam_id = element.get("camId")
                        if isinstance(cam_id, str):
                            cam_dict[cam_id] = element
                else:
                    # Legacy AWS Lambda endpoint
                    uri = f"{endpoint}/prod/cameras?enabled=true"
                    data = urllib.request.urlopen(uri, timeout=15).read()
                    camera_data = json.loads(data.decode("utf-8"))
                    for element in camera_data:
                        if element.get("locus") == "aws":
                            cam_id = element.get("camId")
                            if isinstance(cam_id, str):
                                cam_dict[cam_id] = element

                print(f"[USGSService] Loaded {len(cam_dict)} cameras from {label} endpoint")
                break  # success - no need to try fallback

            except Exception as e:
                print(f"[USGSService] {label} endpoint failed: {e}")
                cam_dict = {}

        self._all_camera_dict = cam_dict
        self._apply_hidden_filter()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def _apply_hidden_filter(self) -> None:
        """Rebuild the visible camera dict from the unfiltered one.
        A camera is considered hidden when hideCam is missing or True.
        """
        if self._show_hidden:
            self._camera_dict = dict(self._all_camera_dict)
        else:
            self._camera_dict = {
                cam_id: element
                for cam_id, element in self._all_camera_dict.items()
                if not element.get("hideCam", True)
            }
        self._site_count = len(self._camera_dict)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def set_show_hidden(self, show: bool) -> None:
        """Include (True) or exclude (False) hidden cameras in the camera list."""
        self._show_hidden = bool(show)
        self._apply_hidden_filter()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def show_hidden(self) -> bool:
        return self._show_hidden

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def is_hidden(self, cam_id: str) -> bool:
        """True if the camera is flagged hidden (hideCam missing or True)."""
        element = self._all_camera_dict.get(cam_id, {})
        return element.get("hideCam", True)

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
    def image_date_range(self, cam_id: str, list_keys: int, timeout_s: float):
        """
        (first image UTC, newest image UTC) for a camera, each a timezone-aware datetime
        or None when unknown. Never raises: a failed image-folder request returns first=None.

        First: S3 lists a folder's files in name order, and image names carry their UTC
        capture time (CAMID___YYYY-MM-DDTHH-MM-SSZ.jpg), so the first image name in the
        camera's overlay folder is its oldest image. list_keys names are requested so a
        non-image file at the start of the folder does not hide the first image.
        Newest: newestImageDT from the camera list.
        """
        from datetime import timezone
        from urllib.parse import urlparse
        import xml.etree.ElementTree as ET

        cam = self._all_camera_dict.get(cam_id, {})

        newest = None
        text = cam.get("newestImageDT")
        if not cam:
            print(f"[USGSService] Newest image for {cam_id}: camera not in the loaded camera list")
        elif not text:
            print(f"[USGSService] Newest image for {cam_id}: no newestImageDT in its camera record "
                  f"(camera list from {self._endpoint})")
        else:
            # e.g. 2026-10-15T00:15:03.000Z; the fraction is dropped so any length parses
            stamp = str(text).strip().replace("Z", "").split(".")[0].split("+")[0]
            try:
                newest = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                print(f"[USGSService] Newest image for {cam_id}: could not read newestImageDT {text!r}")

        folder = cam.get("overlayDir") or f"{IMAGE_ENDPOINT}/{cam_id}/"
        u = urlparse(folder)
        bucket = f"{u.scheme}://{u.netloc}/"
        prefix = u.path.lstrip("/")
        if not prefix.endswith("/"):
            prefix += "/"

        first = None
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"   # S3 ListObjectsV2 XML namespace
        try:
            resp = requests.get(bucket, params={"list-type": "2", "prefix": prefix, "max-keys": int(list_keys)},
                                timeout=float(timeout_s))
            resp.raise_for_status()
            nodes = ET.fromstring(resp.text).findall(f"{ns}Contents")
        except Exception as e:
            # The newest date is still known from the camera list; only the first is lost.
            print(f"[USGSService] First image for {cam_id} unavailable: {e}")
            nodes = []
        for node in nodes:
            name = os.path.basename(node.findtext(f"{ns}Key") or "")
            if "___" not in name:
                continue
            stamp = os.path.splitext(name.split("___", 1)[1])[0]
            try:
                first = datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        return first, newest

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
    def _write_gap_report(self, site_name: str, save_folder: str) -> None:
        """
        Emit image_download_report.html / image_gaps.csv into the download
        folder, summarizing temporal discontinuities in the imagery just
        downloaded.  Timezone comes from the camera dictionary so the report
        reads in site-local, DST-aware time.  Never raises: a report failure
        must not fail a download.
        """
        try:
            tz = None
            try:
                tz = self.camera_info(site_name).tz
            except Exception:
                pass
            html_path, _ = generate_gap_report(save_folder, tz=tz)
            print(f"[USGS] Gap report written: {html_path}")
        except Exception as e:
            print(f"[USGS] Gap report skipped: {e}")

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
        self._write_gap_report(site_name, save_folder)
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
        self._write_gap_report(site_name, save_folder)
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

    def midday_image(self, site_name: str, tz_str: Optional[str], window_hours: float,
                     search_days: int, list_keys: int, timeout_s: float):
        """
        (LatestImage, is_midday): the image closest to local noon on the most recent day
        whose local noon has passed, searching back up to search_days days.

        Images within window_hours of local noon count as midday. The camera's image folder
        on S3 is listed one UTC hour at a time (image names carry their UTC capture time,
        CAMID___YYYY-MM-DDTHH-MM-SSZ.jpg), starting with the hour around noon and working
        outward, so a day usually needs one small request. Local time comes from the
        camera's time zone, so daylight saving time is handled.

        When no midday image is found, returns the newest image with is_midday=False.
        """
        from datetime import timezone
        from urllib.parse import urlparse
        import xml.etree.ElementTree as ET

        tz = None
        if tz_str:
            try:
                import pytz
                tz = pytz.timezone(tz_str)
            except Exception:
                try:
                    from zoneinfo import ZoneInfo
                    tz = ZoneInfo(tz_str)
                except Exception:
                    tz = None
        if tz is None:
            print(f"[USGS midday] Unknown time zone {tz_str!r} for {site_name}; using UTC")
            tz = timezone.utc

        def localize(naive):
            return tz.localize(naive) if hasattr(tz, "localize") else naive.replace(tzinfo=tz)

        cam = self._all_camera_dict.get(site_name, {})
        folder = cam.get("overlayDir") or f"{IMAGE_ENDPOINT}/{site_name}/"
        u = urlparse(folder)
        bucket = f"{u.scheme}://{u.netloc}/"
        prefix = u.path.lstrip("/")
        if not prefix.endswith("/"):
            prefix += "/"
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"   # S3 ListObjectsV2 XML namespace

        def images_in_hour(hour_utc):
            """(UTC time, file name) for every image whose name falls in this UTC hour."""
            stamp = hour_utc.strftime("%Y-%m-%dT%H")
            resp = requests.get(bucket, params={"list-type": "2", "max-keys": int(list_keys),
                                                "prefix": f"{prefix}{site_name}___{stamp}"},
                                timeout=float(timeout_s))
            resp.raise_for_status()
            found = []
            for node in ET.fromstring(resp.text).findall(f"{ns}Contents"):
                name = os.path.basename(node.findtext(f"{ns}Key") or "")
                try:
                    t = datetime.strptime(os.path.splitext(name.split("___", 1)[1])[0], "%Y-%m-%dT%H-%M-%SZ")
                    found.append((t.replace(tzinfo=timezone.utc), name))
                except (IndexError, ValueError):
                    continue
            return found

        now_local = datetime.now(timezone.utc).astimezone(tz)
        day = now_local.date()
        if now_local < localize(datetime.combine(day, MIDDAY)):
            day -= timedelta(days=1)            # today's noon hasn't happened yet
        half = timedelta(hours=float(window_hours))

        for _ in range(int(search_days)):
            noon_utc = localize(datetime.combine(day, MIDDAY)).astimezone(timezone.utc)
            lo, hi = noon_utc - half, noon_utc + half
            one_hour = timedelta(hours=1)

            def closest_possible(hour):
                """Smallest distance from noon that any image in this UTC hour can have."""
                if hour <= noon_utc < hour + one_hour:
                    return timedelta(0)
                return hour - noon_utc if hour > noon_utc else noon_utc - (hour + one_hour)

            # UTC hours covering the window, nearest to noon first
            hours, h = [], lo.replace(minute=0, second=0, microsecond=0)
            while h <= hi:
                hours.append(h)
                h += one_hour
            hours.sort(key=closest_possible)

            best = None
            try:
                for i, hour in enumerate(hours):
                    for t, name in images_in_hour(hour):
                        if lo <= t <= hi and (best is None or abs(t - noon_utc) < abs(best[0] - noon_utc)):
                            best = (t, name)
                    # Stop once no unchecked hour can hold an image closer to noon
                    if best is not None and (i + 1 == len(hours)
                                             or abs(best[0] - noon_utc) <= closest_possible(hours[i + 1])):
                        break
            except Exception as e:
                print(f"[USGS midday] Could not list images for {site_name}: {e}")
                break

            if best is not None:
                t, name = best
                print(f"[USGS midday] {site_name}: {name} ({t.astimezone(tz):%Y-%m-%d %H:%M %Z})")
                try:
                    content = requests.get(f"{IMAGE_ENDPOINT}/{site_name}/{name}", timeout=float(timeout_s)).content
                    return LatestImage(error_code=0, content=content), True
                except Exception as e:
                    print(f"[USGS midday] Could not download {name}: {e}")
                    break
            day -= timedelta(days=1)

        print(f"[USGS midday] No midday image for {site_name} in the last {search_days} days; showing the newest image")
        return self.latest_image(site_name), False

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
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