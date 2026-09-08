#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRIME_AI_Sandbar_Analyzer.py
# Engine for the Sandbar Analyzer tab. Detects INTERNAL edges within the
# segmented ROI (mask eroded so the segmentation boundary itself does not
# register as an edge) and reports per-frame edge statistics.
#
# Companion to GRIME_AI_ROI_Analyzer. Shares its folder-pairing convention
# (<base>.jpg  <->  <base>_mask.png, '_overlay' ignored) and its mask
# handling (grayscale read, binarized > 0).
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import numpy as np
import cv2


class _SandbarEdgeSettings:
    """Adapter exposing the same getter surface as constants.edgeMethodsClass,
    so GRIME_AI_ProcessImage.canny_edges() drives off the Sandbar tab's spin
    boxes through the identical interface the Edge Detection dialog uses.
    """
    def __init__(self, low, high, canny_kernel=3):
        self._low = int(low)
        self._high = int(high)
        self.canny_kernel = canny_kernel

    def getCannyThresholdLow(self):
        return self._low

    def getCannyThresholdHigh(self):
        return self._high


class GRIME_AI_Sandbar_Analyzer:
    """
    Per-frame internal-edge analysis inside a single segmented region.

    Typical use (headless):
        a = GRIME_AI_Sandbar_Analyzer(orig_path, mask_path)
        a.run(operator='canny', canny_low=25, canny_high=150, erode_px=3)
        edge_rgb = a.edge_overlay_rgb   # for display
        stats    = a.stats              # dict of per-frame edge metrics

    The mask is treated as a single ROI (everything > 0), matching the
    'it is already segmented' contract. No color / sub-unit work here; this
    is the edge-first v1.
    """

    OPERATORS = ('canny', 'sobel', 'laplacian', 'prewitt')

    def __init__(self, image_filename, mask_filename):
        self.image_filename = image_filename
        self.mask_filename = mask_filename

        # loaded data
        self.image = None          # BGR original
        self.mask = None           # raw grayscale mask
        self.mask_bin = None       # 0/255 ROI
        self.mask_eroded = None    # ROI shrunk by erode_px (edge-search region)

        # results
        self.edges = None          # uint8 0/255 internal edge map (inside eroded ROI)
        self.grad_mag = None       # float gradient magnitude (inside ROI), for stats
        self.edge_overlay_rgb = None
        self.poc_composite_rgb = None   # POC-style: red boundary + cyan clusters + yellow edges
        self.cluster_result = None      # last ClusterResult (spatial labels)
        self.stats = {}

    # ------------------------------------------------------------------
    # folder pairing (mirrors GRIME_AI_ROI_Analyzer.generate_file_pairs)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_file_pairs(folder):
        files = [f for f in os.listdir(folder) if '_overlay' not in f.lower()]
        originals, masks = {}, {}
        for f in files:
            base, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext == '.jpg':
                if not base.lower().endswith('_mask'):
                    originals[base] = f
            elif ext == '.png':
                if base.lower().endswith('_mask'):
                    masks[base[:-5]] = f
        pairs = []
        for base, orig_file in originals.items():
            if base in masks:
                pairs.append((os.path.join(folder, orig_file),
                              os.path.join(folder, masks[base])))
        return pairs

    # ------------------------------------------------------------------
    def load_data(self):
        self.image = cv2.imread(self.image_filename)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_filename}")
        self.mask = cv2.imread(self.mask_filename, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise FileNotFoundError(f"Could not load mask: {self.mask_filename}")
        if self.image.shape[:2] != self.mask.shape:
            self.mask = cv2.resize(self.mask, (self.image.shape[1], self.image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        _, self.mask_bin = cv2.threshold(self.mask, 1, 255, cv2.THRESH_BINARY)

    # ------------------------------------------------------------------
    def _erode_mask(self, erode_px):
        """Shrink the ROI so the segmentation boundary is excluded from edges."""
        if erode_px <= 0:
            self.mask_eroded = self.mask_bin.copy()
            return
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        self.mask_eroded = cv2.erode(self.mask_bin, k, iterations=1)

    # ------------------------------------------------------------------
    def _compute_edges(self, operator, gray, canny_low, canny_high,
                       sobel_ksize, blur_ksize):
        """Return (edge_binary_uint8, grad_magnitude_float) over the full frame.

        Edge operators are NOT reimplemented here — they are delegated to the
        app's single edge engine, GRIME_AI_ProcessImage, so a Canny/Sobel/
        Laplacian in the Sandbar tab is identical to one anywhere else. This
        method only builds the edgeMethodSettings the shared cores expect and
        adapts their output to (binary, magnitude).
        """
        from GRIME_AI.GRIME_AI_ProcessImage import GRIME_AI_ProcessImage

        if blur_ksize and blur_ksize >= 3:
            gray = cv2.GaussianBlur(gray, (blur_ksize | 1, blur_ksize | 1), 0)

        pi = GRIME_AI_ProcessImage()
        op = operator.lower()

        if op == 'canny':
            ems = _SandbarEdgeSettings(low=canny_low, high=canny_high,
                                       canny_kernel=3)
            edges = pi.canny_edges(gray, ems)
            _, _, mag = pi.sobel_components(gray, 3)     # magnitude for stats only
        elif op == 'sobel':
            _, _, mag = pi.sobel_components(gray, sobel_ksize)
            edges = self._mag_to_binary(mag)
        elif op == 'laplacian':
            resp = pi.laplacian_response(gray, use_log=True)
            mag = np.abs(resp.astype(np.float64))
            edges = self._mag_to_binary(mag)
        elif op == 'prewitt':
            kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
            ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
            gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
            mag = cv2.magnitude(gx, gy)
            edges = self._mag_to_binary(mag)
        else:
            raise ValueError(f"Unknown operator: {operator!r}. Use one of {self.OPERATORS}.")
        return edges, mag

    @staticmethod
    def _fit_to(arr, h, w):
        """Center-pad or crop a 2-D array to exactly (h, w)."""
        if arr.shape[0] == h and arr.shape[1] == w:
            return arr
        out = np.zeros((h, w), dtype=arr.dtype)
        sh, sw = min(h, arr.shape[0]), min(w, arr.shape[1])
        oy, ox = (h - sh) // 2, (w - sw) // 2
        ay, ax = (arr.shape[0] - sh) // 2, (arr.shape[1] - sw) // 2
        out[oy:oy+sh, ox:ox+sw] = arr[ay:ay+sh, ax:ax+sw]
        return out

    @staticmethod
    def _mag_to_binary(mag):
        """Otsu-threshold a gradient magnitude image to a 0/255 edge map."""
        m = mag.copy()
        m -= m.min()
        peak = m.max()
        if peak > 0:
            m = (m / peak) * 255.0
        m = m.astype(np.uint8)
        _, binary = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # ------------------------------------------------------------------
    def run(self, operator='canny', canny_low=25, canny_high=150,
            sobel_ksize=3, blur_ksize=5, erode_px=3):
        """Full pipeline: load (if needed) -> erode ROI -> edges -> mask -> stats."""
        if self.image is None or self.mask_bin is None:
            self.load_data()
        self._erode_mask(erode_px)

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges_full, mag_full = self._compute_edges(
            operator, gray, canny_low, canny_high, sobel_ksize, blur_ksize)

        # Some shared cores (e.g. Laplacian-of-Gaussian) trim a border, returning a
        # slightly smaller array. Pad/crop back to the frame so ROI masking broadcasts.
        h, w = gray.shape[:2]
        edges_full = self._fit_to(edges_full, h, w)
        mag_full = self._fit_to(mag_full, h, w)

        roi = self.mask_eroded > 0
        self.edges = np.where(roi, edges_full, 0).astype(np.uint8)
        self.grad_mag = np.where(roi, mag_full, 0.0)

        self._build_overlay()
        self._compute_stats(roi)
        return self.stats

    # ------------------------------------------------------------------
    def _build_overlay(self):
        """Original (RGB) dimmed outside ROI, with internal edges drawn yellow."""
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).copy()
        outside = self.mask_bin == 0
        rgb[outside] = (rgb[outside] * 0.35).astype(np.uint8)
        rgb[self.edges > 0] = (255, 255, 0)
        self.edge_overlay_rgb = rgb

    # ------------------------------------------------------------------
    def build_poc_composite(self, method='kmeans', clusters=4,
                            outline_mode='all', bandwidth=None, quantile=0.20,
                            boundary_color=(255, 0, 0),   # red   (RGB)
                            cluster_color=(0, 200, 255),  # cyan  (RGB)
                            edge_color=(255, 255, 0)):    # yellow(RGB)
        """
        Recreate the POC's three-layer look on the confirmed sandbar region:
          red    = sandbar mask boundary (from segmentation, not computed)
          cyan   = color sub-unit outlines (clustering WITHIN the sand)
          yellow = internal Canny edges (self.edges, already computed by run())
        outline_mode: 'all' outlines every tonal cluster; 'dominant' outlines
        only the largest cluster (POC-style single sub-class).
        Requires run() to have been called first (needs self.edges).
        """
        from GRIME_AI.GRIME_AI_Clustering import cluster_masked_pixels

        if self.edges is None:
            raise RuntimeError("Call run() before build_poc_composite().")

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).copy()
        # dim outside the ROI so the bar reads like the POC's isolated view
        outside = self.mask_bin == 0
        rgb[outside] = (rgb[outside] * 0.30).astype(np.uint8)

        # --- cyan: cluster the sand, outline sub-units ---
        cr = cluster_masked_pixels(self.image, self.mask_bin, method=method,
                                   clusters=clusters, bandwidth=bandwidth,
                                   quantile=quantile)
        self.cluster_result = cr
        if cr.centers_hsv.shape[0] > 0:
            if outline_mode == 'dominant':
                ids = [cr.dominant_index()]
            else:
                ids = list(range(cr.centers_hsv.shape[0]))
            for cid in ids:
                sub = (cr.labels_image == cid).astype(np.uint8) * 255
                # clean pinholes so outlines aren't shattered
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k, iterations=1)
                cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnts, -1, cluster_color, 1)

        # --- yellow: internal edges on top ---
        rgb[self.edges > 0] = edge_color

        # --- red: mask boundary last, so it frames everything ---
        bnd_cnts, _ = cv2.findContours(self.mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, bnd_cnts, -1, boundary_color, 2)

        self.poc_composite_rgb = rgb
        return self.poc_composite_rgb

    # ------------------------------------------------------------------
    def _compute_stats(self, roi):
        roi_area = int(np.count_nonzero(roi))
        edge_px = int(np.count_nonzero(self.edges))
        density = (edge_px / roi_area) if roi_area else 0.0
        grad_vals = self.grad_mag[roi]
        mean_grad = float(np.mean(grad_vals)) if grad_vals.size else 0.0
        self.stats = {
            'roi_pixels': roi_area,
            'edge_pixels': edge_px,
            'edge_density': density,          # edge px / ROI px
            'mean_gradient': mean_grad,       # over ROI
        }
