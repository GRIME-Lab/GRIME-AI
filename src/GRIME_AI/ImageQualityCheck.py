#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import numpy as np

from appcore.utils.fft_utils import (fft_ring_power, fft_blur_score, blur_scores_at,
                                     choose_blur_cutoff, flag_blurry, calibrated_blur_threshold,
                                     DEFAULT_BLUR_MAD_K)


class ImageQualityAnalyzer:
    """
    Centralised image quality analysis for triage and related workflows.

    All blur, brightness, contrast, and exposure metrics are computed here so that
    ImageTriage and any other callers do not contain duplicated algorithm code.

    FFT blur is judged per folder (one camera): analyze() records each image's
    FFT ring power, then finalize_fft_blur() chooses the frequency cutoff and the
    threshold from all the images together and sets the FFT blur flags
    (see utils/fft_utils.py). An image is FFT-blurry when its score falls well
    below the folder's typical score.

    If fft_calibration (from TriageCalibrator) is given and was made with the same
    focus region and resize, its cutoff is used and the folder threshold is kept
    between its Good and Blurry bounds (fft_utils.calibrated_blur_threshold).
    Otherwise the cutoff and threshold come from the folder alone.

    Parameters
    ----------
    focus_roi : list or None
        Normalised [x, y, w, h] (values 0.0–1.0) defining the subregion used for
        blur scoring. Brightness is always computed over the full frame.
        If None, the full frame is used for all metrics.
    """

    def __init__(
        self,
        # ── blur ──────────────────────────────────────────────────────────────
        use_fft_blur=True,
        use_laplacian_blur=True,
        fft_blur_mad_k=DEFAULT_BLUR_MAD_K,
        laplacian_threshold=150.0,
        blur_logic="AND",
        # ── brightness ────────────────────────────────────────────────────────
        use_brightness=True,
        brightness_min=40.0,
        brightness_max=215.0,
        # ── contrast ──────────────────────────────────────────────────────────
        use_contrast=True,
        contrast_threshold=30.0,
        # ── exposure clipping ─────────────────────────────────────────────────
        use_exposure_clipping=True,
        clip_percent=0.25,
        dark_clip=10,
        bright_clip=245,
        # ── preprocessing ─────────────────────────────────────────────────────
        resize_percent=50.0,
        focus_roi=None,             # normalised [x, y, w, h] or None
        fft_calibration=None,       # dict from TriageCalibrator, or None
        # ── color imbalance ───────────────────────────────────────────────
        use_color_imbalance=False,
        color_imbalance_threshold=0.5,  # flag if max channel fraction exceeds this
    ):
        self.use_fft_blur          = use_fft_blur
        self.use_laplacian_blur    = use_laplacian_blur
        self.fft_blur_mad_k        = fft_blur_mad_k
        self.laplacian_threshold   = laplacian_threshold
        self.blur_logic            = blur_logic.upper()

        self.use_brightness        = use_brightness
        self.brightness_min        = brightness_min
        self.brightness_max        = brightness_max

        self.use_contrast          = use_contrast
        self.contrast_threshold    = contrast_threshold

        self.use_exposure_clipping = use_exposure_clipping
        self.clip_percent          = clip_percent
        self.dark_clip             = dark_clip
        self.bright_clip           = bright_clip

        self.resize_percent             = resize_percent
        self.focus_roi                  = focus_roi
        self.fft_calibration            = fft_calibration
        self.fft_blur_mode              = None   # how the last finalize_fft_blur() decided
        self.use_color_imbalance        = use_color_imbalance
        self.color_imbalance_threshold  = color_imbalance_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, image) -> dict:
        """
        Analyse a single BGR/RGB numpy image.

        Blur metrics operate on the focus ROI (if defined); brightness, contrast,
        and exposure metrics always operate on the full frame.

        Returns a dict with raw metrics, boolean flags, and top-level summary:
            is_bad  (bool) – True if any enabled check fails
            reason  (str)  – comma-separated failing checks, or 'Nominal'

        The FFT blur flag is not known yet: result["fft_ring_power"] holds this
        image's FFT ring power and result["is_blurry_fft"] is None until
        finalize_fft_blur() is called with the whole folder's results.
        """
        gray_full  = self._preprocess(image, apply_roi=False)
        gray_blur  = self._preprocess(image, apply_roi=True)
        result     = {}

        # ── blur (focus ROI if defined) ────────────────────────────────────
        if self.use_fft_blur:
            result["fft_ring_power"] = fft_ring_power(gray_blur)
            result["blur_fft"]       = None    # set by finalize_fft_blur()
            result["is_blurry_fft"]  = None

        if self.use_laplacian_blur:
            lap_var                       = self._compute_laplacian_var(gray_blur)
            result["blur_laplacian"]      = lap_var
            result["is_blurry_laplacian"] = lap_var < self.laplacian_threshold

        # ── brightness (always full frame) ────────────────────────────────
        if self.use_brightness:
            brightness              = self._compute_brightness(gray_full)
            result["brightness"]    = brightness
            too_dark                = brightness < self.brightness_min
            too_light               = brightness > self.brightness_max
            result["is_too_dark"]   = too_dark
            result["is_too_light"]  = too_light

        # ── contrast (always full frame) ──────────────────────────────────
        if self.use_contrast:
            contrast                  = self._compute_contrast(gray_full)
            result["contrast"]        = contrast
            low_contrast              = contrast < self.contrast_threshold
            result["is_low_contrast"] = low_contrast

        # ── exposure clipping (always full frame) ─────────────────────────
        if self.use_exposure_clipping:
            clip_fraction           = self._compute_clip_fraction(gray_full)
            result["clip_fraction"] = clip_fraction
            clipped                 = clip_fraction > self.clip_percent
            result["is_clipped"]    = clipped

        # ── color imbalance (full frame, RGB) ────────────────────────────
        if self.use_color_imbalance:
            imbalance = self._compute_color_imbalance(image)
            result["color_imbalance"]    = imbalance
            is_imbalanced                = imbalance > self.color_imbalance_threshold
            result["is_color_imbalanced"] = is_imbalanced

        self._summarize(result)
        return result

    def finalize_fft_blur(self, results):
        """
        Set the FFT blur flags for one folder's results (a list of analyze() dicts),
        in place. Chooses the cutoff and threshold from all the images together,
        then recomputes is_blurry, is_bad and reason for every result.

        Without a usable calibration and with fewer images than
        fft_utils.MIN_IMAGES_FOR_AUTO_CUTOFF the FFT check is skipped (is_blurry_fft
        stays None) and blur uses the Laplacian alone. self.fft_blur_mode records
        which method decided.
        Returns (cutoff, threshold); threshold is None when nothing could be flagged.
        """
        scored = [r for r in results if r.get("fft_ring_power") is not None]
        if not scored:
            return None, None

        ring_powers = np.array([r["fft_ring_power"] for r in scored])
        calibration, mismatch = self._usable_fft_calibration(ring_powers.shape[1])
        threshold = None
        flags = None

        if calibration is not None:
            cutoff_ring = calibration["cutoff_ring"]
            cutoff      = calibration["cutoff"]
            scores      = blur_scores_at(ring_powers, cutoff_ring)
            threshold, self.fft_blur_mode = calibrated_blur_threshold(
                scores, calibration, k=self.fft_blur_mad_k)
            flags = scores < threshold
        else:
            cutoff, cutoff_ring, _ = choose_blur_cutoff(ring_powers)
            self.fft_blur_mode = "automatic (from this folder)"
            if mismatch:
                self.fft_blur_mode += f"; calibration not used: {mismatch}"
            if cutoff_ring is not None:
                scores = blur_scores_at(ring_powers, cutoff_ring)
                flags, threshold = flag_blurry(scores, k=self.fft_blur_mad_k)
            else:
                self.fft_blur_mode += "; skipped, too few images"

        if flags is not None:
            for r, score, flag in zip(scored, scores, flags):
                r["blur_fft"]      = float(score)
                r["is_blurry_fft"] = bool(flag)

        for r in results:
            self._summarize(r)
        return cutoff, threshold

    def _usable_fft_calibration(self, num_bins):
        """
        Return (calibration, reason). calibration is None when there is none or it
        was made under different conditions; reason then says why (or is "" when
        there is no calibration at all).
        """
        cal = self.fft_calibration
        if not cal:
            return None, ""
        if cal.get("num_bins") != num_bins:
            return None, "different FFT ring count"
        if abs(float(cal.get("resize_percent", -1)) - float(self.resize_percent)) > 1e-6:
            return None, "different image resize"
        if not self._same_roi(cal.get("focus_roi"), self.focus_roi):
            return None, "different focus region"
        return cal, ""

    @staticmethod
    def _same_roi(a, b):
        if a is None or b is None:
            return a is None and b is None
        return len(a) == len(b) and all(abs(float(p) - float(q)) < 1e-4 for p, q in zip(a, b))

    def _summarize(self, result):
        """Combine the blur flags and build is_bad / reason from all flags in result."""
        fft = result.get("is_blurry_fft")        # None: not decided (yet)
        lap = result.get("is_blurry_laplacian")  # None: check disabled
        checks = [f for f in (fft, lap) if f is not None]
        if checks:
            if self.blur_logic == "AND" and len(checks) == 2:
                result["is_blurry"] = fft and lap
            else:
                result["is_blurry"] = any(checks)

        reasons = []
        for key, label in (("is_blurry", "Blurry"), ("is_too_dark", "Too Dark"),
                           ("is_too_light", "Too Light"), ("is_low_contrast", "Low Contrast"),
                           ("is_clipped", "Exposure Clipped"), ("is_color_imbalanced", "Color Imbalance")):
            if result.get(key):
                reasons.append(label)
        result["is_bad"] = bool(reasons)
        result["reason"] = ", ".join(reasons) if reasons else "Nominal"

    def is_bad_image(self, image) -> tuple:
        r = self.analyze(image)
        return r["is_bad"], r["reason"]

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, image, apply_roi: bool = True) -> np.ndarray:
        """
        Convert to grayscale, optionally resize, then optionally crop to focus_roi.

        Parameters
        ----------
        apply_roi : bool
            If True and self.focus_roi is set, crop to the ROI after resize.
            Pass False to get the full-frame gray (used for brightness etc.).
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.resize_percent != 100.0:
            scale = self.resize_percent / 100.0
            gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if apply_roi and self.focus_roi is not None:
            gray = self._apply_focus_roi(gray)

        return gray

    def _apply_focus_roi(self, gray: np.ndarray) -> np.ndarray:
        """Crop gray image to the normalised focus_roi. Returns full image if crop is invalid."""
        h, w = gray.shape
        x  = int(self.focus_roi[0] * w)
        y  = int(self.focus_roi[1] * h)
        rw = int(self.focus_roi[2] * w)
        rh = int(self.focus_roi[3] * h)

        # Clamp
        x  = max(0, min(w - 1, x))
        y  = max(0, min(h - 1, y))
        rw = max(1, min(w - x, rw))
        rh = max(1, min(h - y, rh))

        cropped = gray[y:y + rh, x:x + rw]
        if cropped.size == 0:
            return gray
        return cropped

    def _compute_laplacian_var(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_brightness(self, gray: np.ndarray) -> float:
        smoothed = cv2.GaussianBlur(gray, (0, 0), 1)
        return float(cv2.mean(smoothed)[0])

    def _compute_contrast(self, gray: np.ndarray) -> float:
        return float(int(gray.max()) - int(gray.min()))

    def _compute_color_imbalance(self, image: np.ndarray) -> float:
        """
        Max channel fraction: R_mean / (R+G+B), G_mean / (R+G+B), B_mean / (R+G+B).
        Returns the highest fraction. Perfectly balanced = 0.333. Flag if > threshold.
        Operates on the full frame regardless of focus_roi (color is a global property).
        """
        if image.ndim == 2:
            return 0.333   # grayscale — no color info, treat as balanced
        # Support both BGR (cv2) and RGB inputs
        b = float(np.mean(image[:, :, 0]))
        g = float(np.mean(image[:, :, 1]))
        r = float(np.mean(image[:, :, 2]))
        total = r + g + b
        if total < 1e-6:
            return 0.333
        return max(r / total, g / total, b / total)

    def _compute_clip_fraction(self, gray: np.ndarray) -> float:
        clipped = np.sum((gray <= self.dark_clip) | (gray >= self.bright_clip))
        return float(clipped) / gray.size
