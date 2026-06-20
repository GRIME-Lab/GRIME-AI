from typing import Callable, Optional, Tuple, List
from PyQt5.QtGui import QPixmap

from .usgs_service import USGSService
from .usgs_types import CameraInfo

# Progress callback signature: (index, total, label)
ProgressFn = Callable[[int, int, Optional[str]], None]


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===            class USGSClient            ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class USGSClient:
    """
    Thin façade for GUI code.
    Wraps USGSService and converts raw results into Qt‑friendly objects
    (e.g. QPixmap) or legacy string lists for listboxes.
    """

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __init__(self, service: Optional[USGSService] = None):
        self._svc = service or USGSService()
        self._initialized = False

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def initialize(self) -> None:
        """Initialize the underlying service (fetch camera dictionary)."""
        self._svc.initialize()
        self._initialized = True

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_sites(self) -> List[str]:
        """Return a sorted list of site/camera IDs."""
        return self._svc.camera_list()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camera_info_lines(self, cam_id: str) -> List[str]:
        """
        Return formatted info lines for a site.
        Keeps the old 'list of strings' format so MainWindow can
        populate listboxes without change.
        """
        info: CameraInfo = self._svc.camera_info(cam_id)
        lines = []
        if info.cam_name:
            lines.append(f"camName: {info.cam_name}")
        if info.cam_id:
            lines.append(f"camId: {info.cam_id}")
        if info.nwis_id:
            lines.append(f"nwisId: {info.nwis_id}")
        if info.lat is not None:
            lines.append(f"lat: {info.lat}")
        if info.lng is not None:
            lines.append(f"lng: {info.lng}")
        if info.tz:
            lines.append(f"tz: {info.tz}")
        if info.description:
            lines.append(f"camDesc: {info.description}")
        return lines or ["No information available for this site."]

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_latest_pixmap(self, site_name: str) -> Tuple[int, Optional[QPixmap]]:
        """
        Return (error_code, QPixmap) for the latest image.
        error_code = 0 if ok, 404 if not found.
        """
        latest = self._svc.latest_image(site_name)
        if latest.error_code == 404 or latest.content is None:
            return 404, None
        pix = QPixmap()
        pix.loadFromData(latest.content)
        return 0, pix

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def image_count(
        self,
        site_name,
        start_date,
        end_date,
        start_time,
        end_time,
        progress: Optional[ProgressFn] = None
    ) -> int:
        """Return number of images available in the given date/time range."""
        return self._svc.image_count(site_name, start_date, end_date, start_time, end_time, progress)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def download_images(
        self,
        site_name,
        start_date,
        end_date,
        start_time,
        end_time,
        folder: str,
        progress: Optional[ProgressFn] = None,
        cancel_check=None
    ) -> Tuple[int, int]:
        """
        Download images for a site into the given folder.
        Returns (downloaded_count, missing_count).
        """
        return self._svc.download_images(site_name, start_date, end_date, start_time, end_time, folder, progress=progress, cancel_check=cancel_check)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def download_images_from_list(
        self,
        site_name: str,
        names: List[str],
        folder: str,
        progress: Optional[ProgressFn] = None
    ) -> Tuple[int, int]:
        """
        Download a pre-filtered list of image filenames.
        Called by USGS_HIVIS.download_images() after timezone-aware filtering
        has already been applied by get_image_count().
        """
        return self._svc.download_images_from_list(site_name, names, folder, progress)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_available_parameters(self, cam_id: str) -> List[dict]:
        """Return available NWIS time series parameters for the site."""
        info: CameraInfo = self._svc.camera_info(cam_id)
        return self._svc.get_available_parameters(info.nwis_id)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_midday_pixmap(self, site_name: str) -> Tuple[int, Optional[QPixmap], bool]:
        """
        Return (error_code, QPixmap, is_midday).
        is_midday=True if a midday window image was found,
        False if fell back to latest image.
        """
        info = self._svc.camera_info(site_name)
        latest, is_midday = self._svc.midday_image(site_name, info.tz)
        if latest.error_code == 404 or latest.content is None:
            return 404, None, False
        pix = QPixmap()
        pix.loadFromData(latest.content)
        return 0, pix, is_midday

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def fetch_stage_and_discharge(
        self,
        nwis_id,
        site_name,
        start_date,
        end_date,
        start_time,
        end_time,
        folder: str
    ) -> Tuple[str, str]:
        """
        Fetch stage/discharge data for a site.
        Returns (txt_path, csv_path).
        """
        return self._svc.fetch_stage_and_discharge(nwis_id, site_name, start_date, end_date, start_time, end_time, folder)
