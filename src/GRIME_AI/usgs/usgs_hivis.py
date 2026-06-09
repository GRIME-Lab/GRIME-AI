# usgs/usgs_hivis.py
import datetime
from PyQt5.QtWidgets import QMessageBox
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from .usgs_client import USGSClient

class USGS_HIVIS:

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __init__(self):
        self._client = USGSClient()
        self._last_cam_id = None
        self._last_nwis_id = None
        self._last_cam_name = None

        # ============================================================================
        # CACHE FOR FILTERED IMAGE NAMES
        # Stores the result of the last get_image_count call
        # ============================================================================
        self._cached_filenames = []  # Store filtered list
        self._cache_params = None  # Validate cache is current

        try:
            self._client.initialize()
        except Exception:
            msg = GRIME_AI_QMessageBox('USGS NIMS Error',
                                       'Unable to access USGS NIMS Database!')
            msg.displayMsgBox()


    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def init_camera_dictionary(self):
        # Legacy method existed; provide access via client. Return dict.
        try:
            return self._client._svc.camera_dictionary()
        except Exception:
            return {}

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camera_dictionary(self):
        return self._client._svc.camera_dictionary()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camera_list(self):
        return self._client.get_sites()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camera_info(self, camera_id):
        lines = self._client.get_camera_info_lines(camera_id)
        # Track legacy attributes
        self._last_cam_id = camera_id
        self._last_nwis_id = None
        self._last_cam_name = None
        for line in lines:
            if line.startswith("nwisId:"):
                self._last_nwis_id = line.split(":", 1)[1].strip()
            elif line.startswith("camName:"):
                self._last_cam_name = line.split(":", 1)[1].strip()
        return lines if lines else ["No information available for this site."]


    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_nwisID(self):
        return self._last_nwis_id

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camName(self):
        return self._last_cam_name

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camId(self):
        return self._last_cam_id

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_latest_image(self, siteName):
        code, pixmap = self._client.get_latest_pixmap(siteName)
        if code == 404:
            return 404, []
        return 0, pixmap

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _clear_cache(self):
        """Clear the cached filename list"""
        self._cached_filenames = []
        self._cache_params = None
        print("Cache cleared")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _validate_cache(self, siteName, startDate, endDate, startTime, endTime):
        """
        Check if cached data matches the requested parameters.
        Returns True if cache is valid, False otherwise.
        """
        if not self._cache_params:
            return False

        return (self._cache_params['siteName'] == siteName and
                   self._cache_params['startDate'] == startDate and
                   self._cache_params['endDate'] == endDate and
                   self._cache_params['startTime'] == startTime and
                   self._cache_params['endTime'] == endTime)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_cached_filenames(self, siteName, startDate, endDate, startTime, endTime):
        """
        Get cached filenames if parameters match.
        Returns cached list or None if cache invalid.
        """
        if self._validate_cache(siteName, startDate, endDate, startTime, endTime):
            print(f"✓ Using cached list of {len(self._cached_filenames)} filenames")
            return self._cached_filenames
        else:
            print("✗ Cache invalid - parameters don't match")
            return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_image_count(self, siteName, nwisID, startDate, endDate, startTime, endTime):
        """
        Fetch all images for the date range in one API call, then filter by time.
        CACHES the filtered list for later use by download_images.
        Much faster than building per-day URLs.

        Args:
            siteName: Camera ID (e.g., "NE_Platte_River_near_Grand_Island")
            nwisID: NWIS site ID (unused in this implementation)
            startDate: datetime.date object
            endDate: datetime.date object
            startTime: datetime.time object
            endTime: datetime.time object

        Returns:
            int: Number of images within the specified date and time range
        """
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        from urllib.parse import urlencode
        from datetime import datetime
        import json
        import time
        import socket
        import pytz

        # ============================================================================
        # GET SITE TIMEZONE FROM CAMERA INFO
        # ============================================================================
        camera_info = self.get_camera_info(siteName)
        site_tz_str = None
        for line in camera_info:
            if line.startswith("tz:"):
                site_tz_str = line.split(":", 1)[1].strip()
                break
        
        if not site_tz_str:
            print(f"Warning: No timezone info for {siteName}, assuming UTC")
            site_tz = pytz.UTC
        else:
            try:
                site_tz = pytz.timezone(site_tz_str)
                print(f"Site timezone: {site_tz_str}")
            except Exception as e:
                print(f"Warning: Invalid timezone '{site_tz_str}', using UTC: {e}")
                site_tz = pytz.UTC

        # ============================================================================
        # RETRY LOGIC FOR NETWORK ISSUES
        # ============================================================================
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # ============================================================================
                # BUILD SINGLE API REQUEST FOR ENTIRE DATE RANGE
                # THIS IS MUCH FASTER THAN MULTIPLE PER-DAY REQUESTS
                # ============================================================================
                base_url = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles"

                # Convert the user's local start/end datetimes to UTC so the API
                # window correctly spans the full local date+time range, including
                # cases where the local window crosses midnight UTC.
                from datetime import datetime as _dt
                local_start_aware = site_tz.localize(_dt.combine(startDate, startTime))
                local_end_aware   = site_tz.localize(_dt.combine(endDate,   endTime))
                utc_start = local_start_aware.astimezone(pytz.UTC)
                utc_end   = local_end_aware.astimezone(pytz.UTC)

                params = {
                    "camId": siteName,
                    "after":  utc_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "before": utc_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 50000,  # High limit to get all images
                }

                url = f"{base_url}?{urlencode(params)}"
                print(f"REQUEST URL: {url}")
                print(f"Attempt {attempt + 1}/{max_retries}")

                # Make API request
                req = Request(url, headers={"User-Agent": "python-urllib/3"})

                with urlopen(req, timeout=30) as resp:
                    filenames = json.load(resp)

                # Success - break out of retry loop
                break

            except socket.gaierror as e:
                # DNS resolution failed
                print(f"DNS resolution failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # All retries exhausted
                    msg = GRIME_AI_QMessageBox(
                        'Network Error',
                        'Cannot connect to USGS HIVIS API. Please check:\n\n'
                        '1. Internet connection is active\n'
                        '2. Firewall/VPN is not blocking AWS domains\n'
                        '3. DNS server is working\n\n'
                        'Try: ping jj5utwupk5.execute-api.us-east-1.amazonaws.com',
                        QMessageBox.Close
                    )
                    msg.displayMsgBox()
                    self._clear_cache()
                    return 0

            except URLError as e:
                # Network error (timeout, connection refused, etc.)
                print(f"Network error (attempt {attempt + 1}/{max_retries}): {e.reason}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    msg = GRIME_AI_QMessageBox(
                        'Network Error',
                        f'Could not connect to USGS HIVIS API after {max_retries} attempts.\n\n'
                        f'Error: {e.reason}',
                        QMessageBox.Close
                    )
                    msg.displayMsgBox()
                    self._clear_cache()
                    return 0

            except HTTPError as e:
                # HTTP error from server (don't retry these)
                print(f"HTTP error: {e.code} {e.reason}")
                msg = GRIME_AI_QMessageBox(
                    'API Error',
                    f'Server returned error {e.code}: {e.reason}',
                    QMessageBox.Close
                )
                msg.displayMsgBox()
                self._clear_cache()
                return 0

            except json.JSONDecodeError as e:
                # Invalid JSON response (don't retry)
                print(f"JSON decode error: {e}")
                msg = GRIME_AI_QMessageBox(
                    'API Error',
                    'Received invalid response from USGS HIVIS API',
                    QMessageBox.Close
                )
                msg.displayMsgBox()
                self._clear_cache()
                return 0

        # ============================================================================
        # PROCESS RESULTS (outside retry loop)
        # ============================================================================
        try:
            # Validate response
            if not isinstance(filenames, list):
                print(f"Unexpected response type: {type(filenames)}")
                raise ValueError("API did not return a list of filenames")

            print(f"API returned {len(filenames)} total filenames")

            if len(filenames) == 0:
                # No images in date range
                msg = GRIME_AI_QMessageBox(
                    'Images unavailable',
                    'No images available for the site or for the time/date range specified.',
                    QMessageBox.Close
                )
                msg.displayMsgBox()
                self._clear_cache()
                return 0

            # ============================================================================
            # FILTER BY TIME RANGE AND CACHE RESULTS
            # PARSE TIMESTAMPS FROM FILENAMES AND CHECK IF WITHIN startTime-endTime
            # ============================================================================
            filtered_filenames = []
            parse_errors = 0

            for fname in filenames:
                try:
                    # Extract timestamp from filename
                    # Format: SITE_NAME___YYYY-MM-DDTHH-MM-SSZ.jpg
                    # Example: NE_Platte_River_near_Grand_Island___2025-07-31T23-45-02Z.jpg

                    if "___" not in fname:
                        parse_errors += 1
                        continue

                    # Get timestamp part (between ___ and .jpg)
                    ts_str = fname.split("___")[1]

                    # Remove file extension
                    if ts_str.endswith(".jpg"):
                        ts_str = ts_str[:-4]
                    elif ts_str.endswith(".JPG"):
                        ts_str = ts_str[:-4]

                    # ========================================================================
                    # Parse UTC timestamp and convert to site's local timezone
                    # Filename format: YYYY-MM-DDTHH-MM-SSZ (Z = UTC)
                    # Example: 2026-01-09T15-45-02Z = 15:45:02 UTC = 9:45:02 AM CST
                    # ========================================================================
                    ts_utc = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%SZ")
                    ts_utc = pytz.UTC.localize(ts_utc)  # Make timezone-aware

                    # Convert to site's local timezone
                    ts_local = ts_utc.astimezone(site_tz)
                    
                    # Compare in local time (what user entered in UI)
                    if startTime <= ts_local.time() <= endTime:
                        filtered_filenames.append(fname)

                except (ValueError, IndexError, AttributeError) as e:
                    # Failed to parse this filename
                    parse_errors += 1
                    continue

            # Log parsing issues if any
            if parse_errors > 0:
                print(f"Warning: Could not parse {parse_errors} of {len(filenames)} filenames")

            filtered_count = len(filtered_filenames)

            print(f"Filtered to {filtered_count} images within time range "
                  f"{startTime.strftime('%H:%M')}-{endTime.strftime('%H:%M')}")

            # ============================================================================
            # CACHE THE FILTERED LIST FOR DOWNLOAD_IMAGES TO USE
            # ============================================================================
            self._cached_filenames = filtered_filenames
            self._cache_params = {
                'siteName': siteName,
                'startDate': startDate,
                'endDate': endDate,
                'startTime': startTime,
                'endTime': endTime
            }
            print(f"✓ Cached {len(self._cached_filenames)} filtered filenames")

            # Log when the time filter eliminates all results, but don't pop a
            # blocking dialog — the user may still be editing the time fields,
            # and the "0" shown in the image count column is sufficient feedback.
            if filtered_count == 0:
                print(f"[get_image_count] 0 images match local time range "
                      f"{startTime.strftime('%H:%M')}–{endTime.strftime('%H:%M')} "
                      f"(site tz: {site_tz_str}); {len(filenames)} images exist in date range.")

            return filtered_count

        except Exception as e:
            # Catch-all for unexpected errors in filtering logic
            print(f"Unexpected error processing results: {e}")
            msg = GRIME_AI_QMessageBox(
                'Error',
                f'Failed to process image list: {str(e)}',
                QMessageBox.Close
            )
            msg.displayMsgBox()
            self._clear_cache()
            return 0

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def download_images(self, siteName, startDate, endDate, startTime, endTime, saveFolder, progress=None):
        try:
            # Use the timezone-aware filtered list cached by get_image_count.
            # That list already has UTC→local conversion applied, so each filename
            # is confirmed to fall within the user's requested local time window.
            names = self.get_cached_filenames(siteName, startDate, endDate, startTime, endTime)

            if names is None:
                # Cache is stale or get_image_count was never called for this range.
                # Re-run get_image_count to populate the timezone-aware cache before downloading.
                print("[download_images] Cache miss — re-running get_image_count for timezone-aware filtering")
                self.get_image_count(siteName, self._last_nwis_id, startDate, endDate, startTime, endTime)
                names = self._cached_filenames

            downloaded, missing = self._client.download_images_from_list(
                siteName, names, saveFolder, progress
            )
            return downloaded, missing
        except Exception:
            msg = GRIME_AI_QMessageBox('Images unavailable',
                                       'One or more images reported as available by NIMS are not available.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return 0, 0

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def fetchStageAndDischarge(self, nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder):
        try:
            return self._client.fetch_stage_and_discharge(
                nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder
            )
        except Exception:
            msg = GRIME_AI_QMessageBox('USGS - Retrieval Error',
                                       'Unable to retrieve data from the USGS site.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return None, None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def buildImageDateTimeFilter(self, index, startDate, endDate, startTime, endTime):
        startDay = startDate + datetime.timedelta(days=index)
        if startTime.hour == 0 and startTime.minute == 0 and endTime.hour == 0 and endTime.minute == 0:
            day_start = datetime.datetime.combine(startDay, datetime.time(0, 0, 0))
            day_end = datetime.datetime.combine(startDay, datetime.time(23, 59, 59))
        else:
            day_start = datetime.datetime.combine(startDay, startTime)
            day_end = datetime.datetime.combine(startDay, endTime)
        after_dt = day_start - datetime.timedelta(seconds=30)
        before_dt = day_end + datetime.timedelta(seconds=30)
        return "&after=" + after_dt.strftime("%Y-%m-%d:%H:%M:%S"), "&before=" + before_dt.strftime("%Y-%m-%d:%H:%M:%S")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def fetchListOfImages(self, siteName, after, before):
        # Delegate to client; return legacy string payload (already formatted)
        return self._client.fetch_list_of_images(siteName, after, before)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reformat_file(self, input_path, output_path):
        return self._client.reformat_file(input_path, output_path)
