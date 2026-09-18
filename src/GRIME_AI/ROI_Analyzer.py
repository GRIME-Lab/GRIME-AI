#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
from io import BytesIO

import cv2
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from PyQt5.QtGui import QImage, QPixmap

# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     class ROI_Analyzer     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class ROI_Analyzer:
    def __init__(self, image_filename, mask_filename, clusters=3, clustering_method='kmeans',
                 bandwidth=None, quantile=0.2):
        self.image_filename = image_filename
        self.mask_filename = mask_filename
        self.clusters = clusters
        self.clustering_method = clustering_method  # 'kmeans', 'gmm', or 'meanshift'
        self.bandwidth = bandwidth    # None = auto-estimate using quantile
        self.quantile = quantile if quantile is not None else 0.2

        self.image = None
        self.mask = None
        self.mask_bin = None
        self.composite = None

        self.roi_intensity = None
        self.roi_entropy = None
        self.roi_texture = None
        self.mean_gli = None
        self.mean_gcc = None
        self.ROI_total_pixels = None
        self.ROI_total_area = None

        self.dominant_hsv_list = []
        self.dominant_rgb_list = []
        self.percentages_list = []


    def generate_file_pairs(self, folder):
        """
        Scan the folder for paired files. Each pair consists of an original image (JPG)
        and its corresponding mask (PNG) with the same base filename, where the mask filename
        ends with '_mask' before the extension. Files with '_overlay' in their name are ignored.

        Parameters:
            folder (str): The folder in which to look for files.

        Returns:
            pairs (list of tuples): A list of tuples (original_file_path, mask_file_path).
        """
        # Get a list of all files in the folder.
        files = os.listdir(folder)
        # Filter out files that contain '_overlay' (case-insensitive).
        files = [f for f in files if '_overlay' not in f.lower()]

        originals = {}  # key: base filename (without extension), value: filename of original image
        masks = {}  # key: base filename corresponding to the original, value: filename of mask image

        for f in files:
            base, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext == ".jpg":
                # Ensure this file does NOT already end with '_mask'.
                if not base.lower().endswith("_mask"):
                    originals[base] = f
            elif ext == ".png":
                # Consider only files that end with '_mask'.
                if base.lower().endswith("_mask"):
                    # Remove the '_mask' suffix to obtain the corresponding original's base name.
                    base_original = base[:-5]  # removes exactly "_mask"
                    masks[base_original] = f

        # Construct the list of paired file paths.
        pairs = []
        for base, orig_file in originals.items():
            if base in masks:
                orig_full = os.path.join(folder, orig_file)
                mask_full = os.path.join(folder, masks[base])
                pairs.append((orig_full, mask_full))

        return pairs


    def load_data(self):
        if not os.path.exists(self.image_filename):
            print(f"Error: Image file not found: {self.image_filename}")
            sys.exit(1)
        if not os.path.exists(self.mask_filename):
            print(f"Error: Mask file not found: {self.mask_filename}")
            sys.exit(1)
        self.image = cv2.imread(self.image_filename)
        if self.image is None:
            print(f"Error: Could not load image from: {self.image_filename}")
            sys.exit(1)
        self.mask = cv2.imread(self.mask_filename, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            print(f"Error: Could not load mask from: {self.mask_filename}")
            sys.exit(1)

        # Resize mask if necessary.
        if self.image.shape[:2] != self.mask.shape:
            self.mask = cv2.resize(self.mask, (self.image.shape[1], self.image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Create binary mask.
        ret, self.mask_bin = cv2.threshold(self.mask, 1, 255, cv2.THRESH_BINARY)


    def compute_composite(self):
        # Convert image to BGRA.
        image_bgra = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        # Set alpha channel using the mask: 255 inside ROI; 0 outside.
        image_bgra[:, :, 3] = self.mask_bin

        # Create a white background.
        white_bg = np.ones_like(self.image, dtype=np.uint8) * 255
        white_bg = cv2.cvtColor(white_bg, cv2.COLOR_BGR2BGRA)

        # Composite the image over the white background using the alpha mask.
        alpha = self.mask_bin.astype(np.float32) / 255.0
        alpha_4 = np.dstack([alpha, alpha, alpha, alpha])
        self.composite = (image_bgra.astype(np.float32) * alpha_4 +
                          white_bg.astype(np.float32) * (1 - alpha_4)).astype(np.uint8)


    @staticmethod
    def calculate_shannon_entropy(gray_roi):
        hist, _ = np.histogram(gray_roi, bins=256, range=(0, 256))
        hist_sum = np.sum(hist)
        if hist_sum == 0:
            return 0.0
        probs = hist.astype(np.float32) / hist_sum
        entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        return entropy


    def analyze_roi(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        roi_pixels = gray_image[self.mask_bin > 0]
        self.roi_intensity = np.mean(roi_pixels)
        self.roi_entropy = self.calculate_shannon_entropy(roi_pixels)

        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        roi_grad = grad_magnitude[self.mask_bin > 0]
        self.roi_texture = np.mean(roi_grad)


    def compute_mask_area(self):
        """
        Calculate the total number of pixels in the masked region and
        store both the pixel count and area in class variables.

        Also stores:
            self.image_width, self.image_height  - dimensions of the image
            self.image_total_pixels              - total number of pixels in the image
            self.ROI_percentage                  - ROI pixel count as % of total image pixels
        """
        if self.mask_bin is None:
            raise ValueError("Mask binary (self.mask_bin) is not initialized. Run load_data() first.")

        # Get image dimensions from the mask
        self.image_height, self.image_width = self.mask_bin.shape[:2]
        self.image_total_pixels = self.image_width * self.image_height

        # Count nonzero pixels in the binary mask
        self.ROI_total_pixels = int(np.count_nonzero(self.mask_bin))

        # Area in pixels² (same as pixel count unless scaling is applied)
        self.ROI_total_area = float(self.ROI_total_pixels)

        # Percentage of ROI pixels relative to total image pixels
        self.ROI_percentage = (self.ROI_total_pixels / self.image_total_pixels) * 100.0


    def compute_greenness(self):
        roi_pixels_color = self.image[self.mask_bin > 0].astype(np.float32)
        B = roi_pixels_color[:, 0]
        G = roi_pixels_color[:, 1]
        R = roi_pixels_color[:, 2]
        eps = 1e-6
        gli_values = (2 * G - R - B) / (2 * G + R + B + eps)
        self.mean_gli = np.mean(gli_values)
        gcc_values = G / (R + G + B + eps)
        self.mean_gcc = np.mean(gcc_values)


    def extract_dominant_colors(self):
        """Dispatch to the selected clustering algorithm."""
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        if len(self.mask.shape) != 2:
            mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = self.mask.copy()
        _, mask_bin_local = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        masked_pixels = hsv_image[mask_bin_local > 0].astype(np.float32)

        if masked_pixels.size == 0:
            print("Error: No pixels found under the provided mask.")
            sys.exit(1)

        self.dominant_hsv_list = []
        self.dominant_rgb_list = []
        self.percentages_list = []

        method = self.clustering_method.lower()
        if method == 'gmm':
            self._extract_gmm(masked_pixels)
        elif method == 'meanshift':
            self._extract_meanshift(masked_pixels)
        else:
            self._extract_kmeans(masked_pixels)

    def _extract_kmeans(self, masked_pixels):
        clusters = self.clusters if masked_pixels.shape[0] >= self.clusters else 1
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        kmeans.fit(masked_pixels)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        counts = np.bincount(labels)
        percentages = counts / float(np.sum(counts)) * 100
        sorted_indices = np.argsort(percentages)[::-1]
        for idx in sorted_indices:
            self._append_cluster(cluster_centers[idx], percentages[idx])

    def _extract_gmm(self, masked_pixels):
        from sklearn.mixture import GaussianMixture
        n_components = self.clusters if masked_pixels.shape[0] >= self.clusters else 1
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
        gmm.fit(masked_pixels)
        labels = gmm.predict(masked_pixels)
        counts = np.bincount(labels, minlength=n_components)
        percentages = counts / float(np.sum(counts)) * 100
        sorted_indices = np.argsort(percentages)[::-1]
        for idx in sorted_indices:
            self._append_cluster(gmm.means_[idx], percentages[idx])

    def _extract_meanshift(self, masked_pixels):
        from sklearn.cluster import MeanShift, estimate_bandwidth
        # Subsample for speed on large masks (mean-shift is O(n²))
        max_samples = 5000
        if masked_pixels.shape[0] > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(masked_pixels.shape[0], max_samples, replace=False)
            sample = masked_pixels[idx]
        else:
            sample = masked_pixels

        if self.bandwidth is not None:
            bw = self.bandwidth
        else:
            bw = estimate_bandwidth(sample, quantile=self.quantile,
                                    n_samples=min(500, sample.shape[0]))
            if bw <= 0:
                bw = 10.0  # safe fallback

        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        ms.fit(sample)

        # Assign ALL masked pixels to nearest cluster center for accurate percentages
        centers = ms.cluster_centers_
        diffs = masked_pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]
        labels_all = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)

        counts = np.bincount(labels_all, minlength=len(centers))
        percentages = counts / float(np.sum(counts)) * 100
        sorted_indices = np.argsort(percentages)[::-1]
        for idx in sorted_indices:
            self._append_cluster(centers[idx], percentages[idx])

    def _append_cluster(self, center_hsv, percentage):
        """Convert an HSV cluster center to RGB and append to result lists."""
        center_uint8 = np.clip(center_hsv, 0, 255).astype(np.uint8)
        center_hsv_arr = np.array([[center_uint8]])
        center_rgb = cv2.cvtColor(center_hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        self.dominant_hsv_list.append(tuple(int(x) for x in center_uint8))
        self.dominant_rgb_list.append(tuple(int(x) for x in center_rgb))
        self.percentages_list.append(float(percentage))


    def get_results_pixmap(self, capture_date=None, capture_time=None):
        # 1) Prepare images
        orig_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        composite_rgba = cv2.cvtColor(self.composite, cv2.COLOR_BGRA2RGBA)

        # 2) Try to load an overlay image if it exists
        base, _ = os.path.splitext(self.image_filename)
        overlay_rgb = None
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = f"{base}_overlay{ext}"
            if os.path.exists(candidate):
                ov = cv2.imread(candidate)
                overlay_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
                break

        # 3) Layout: render at a fixed large DPI so images fill the label.
        #    Use a wide figure (10" per col at 100 dpi = 1000px per col) and
        #    derive height from the image aspect ratio so images aren't letterboxed.
        img_h, img_w = orig_rgb.shape[:2]
        aspect = img_w / max(img_h, 1)
        ncols_img = 3 if overlay_rgb is not None else 2

        dpi = 100
        fig_w = 10.0 * ncols_img   # wide enough to fill a typical label
        img_row_h = fig_w / (ncols_img * aspect)
        swatch_row_h = 0.35

        # Reserve space at top for date/time suptitle if present
        has_timestamp = capture_date or capture_time
        top_margin = 0.94 if has_timestamp else 0.97
        fig_h = img_row_h + swatch_row_h + (0.25 if has_timestamp else 0.0)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(
            2, ncols_img,
            height_ratios=[img_row_h, swatch_row_h],
            hspace=0.06,
            wspace=0.04,
        )

        # --- Date/time suptitle ---
        if has_timestamp:
            parts = []
            if capture_date and capture_date != 'n/a':
                parts.append(capture_date)
            if capture_time and capture_time != 'n/a':
                parts.append(capture_time)
            fig.suptitle("  ".join(parts), fontsize=9, y=0.99)

        # --- Row 0: images ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(orig_rgb)
        ax0.set_title("Original", fontsize=8, pad=2)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(composite_rgba)
        ax1.set_title("Extracted Features", fontsize=8, pad=2)
        ax1.axis('off')

        if overlay_rgb is not None:
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(overlay_rgb)
            ax2.set_title("Overlay", fontsize=8, pad=2)
            ax2.axis('off')

        # --- Row 1: single proportional-width swatch strip spanning all columns ---
        ax_sw = fig.add_subplot(gs[1, :])
        ax_sw.axis('off')

        swatch_count = len(self.dominant_rgb_list)
        if swatch_count > 0:
            strip_w = 1000
            strip = np.zeros((1, strip_w, 3), dtype=np.uint8)
            x = 0
            segments = []
            for i, (rgb, pct) in enumerate(zip(self.dominant_rgb_list, self.percentages_list)):
                w = int(round(strip_w * pct / 100.0))
                if i == swatch_count - 1:
                    w = strip_w - x  # absorb rounding remainder
                w = max(w, 1)
                strip[0, x:x + w] = rgb
                segments.append((x / strip_w, (x + w) / strip_w, rgb, pct))
                x += w

            ax_sw.imshow(strip, aspect='auto', extent=[0, 1, 0, 1])

            for (x0, x1, rgb, pct) in segments:
                cx = (x0 + x1) / 2.0
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                txt_color = 'white' if lum < 128 else 'black'
                ax_sw.text(
                    cx, 0.5, f"{rgb}\n{pct:.1f}%",
                    ha='center', va='center',
                    fontsize=5, color=txt_color,
                    transform=ax_sw.transAxes,
                    clip_on=True,
                )

        fig.subplots_adjust(left=0.01, right=0.99, top=top_margin, bottom=0.01)

        # 4) Render to QPixmap and return
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        plt.close(fig)
        buf.seek(0)

        qimg = QImage()
        qimg.loadFromData(buf.getvalue(), 'PNG')
        return QPixmap.fromImage(qimg)


    def run_analysis(self):
        self.load_data()
        self.compute_composite()
        self.analyze_roi()
        self.compute_greenness()
        self.compute_mask_area()
        self.extract_dominant_colors()
        # You can choose to use the traditional display_results() method here,
        # but for integration with the UI we'll use get_results_pixmap().
