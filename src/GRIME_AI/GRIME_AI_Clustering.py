#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRIME_AI_Clustering.py
# Shared color-clustering used by BOTH the ROI Analyzer and the Sandbar
# Analyzer, so a K-means/GMM/Mean-Shift is identical across the app.
#
# This lifts the exact algorithm parameters already in GRIME_AI_ROI_Analyzer
# (HSV space, masked pixels only, random_state=42, GMM max_iter=200,
# mean-shift subsampled at 5000 with full reassignment) into one place.
# The difference from the ROI path is only what the caller keeps:
#   - ROI Analyzer collapses the result to dominant colors + percentages
#   - Sandbar Analyzer keeps the per-pixel label image to draw sub-unit outlines
# Both call cluster_masked_pixels(); neither reimplements the algorithm.
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import numpy as np
import cv2

from sklearn.cluster import KMeans


class ClusterResult:
    """
    Outcome of clustering the masked pixels of one image.

    labels_image : int32 HxW, cluster id per pixel; -1 outside the mask.
    centers_hsv  : (k,3) float cluster centers in HSV (algorithm-native space).
    centers_rgb  : (k,3) uint8 same centers converted to RGB (display).
    percentages  : (k,) float, share of masked pixels per center.
    order        : indices sorting clusters by descending percentage.
    """
    __slots__ = ('labels_image', 'centers_hsv', 'centers_rgb',
                 'percentages', 'order')

    def __init__(self, labels_image, centers_hsv, centers_rgb, percentages, order):
        self.labels_image = labels_image
        self.centers_hsv = centers_hsv
        self.centers_rgb = centers_rgb
        self.percentages = percentages
        self.order = order

    def dominant_index(self):
        """Cluster id with the largest pixel share."""
        return int(self.order[0]) if len(self.order) else -1


def _centers_to_rgb(centers_hsv):
    out = []
    for c in centers_hsv:
        u8 = np.clip(c, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(np.array([[u8]]), cv2.COLOR_HSV2RGB)[0][0]
        out.append(tuple(int(x) for x in rgb))
    return np.array(out, dtype=np.uint8)


def cluster_masked_pixels(image_bgr, mask_bin, method='kmeans', clusters=3,
                          bandwidth=None, quantile=0.20, max_meanshift=5000,
                          random_state=42):
    """
    Cluster the HSV pixels under mask_bin (>0) and return a ClusterResult with a
    full-frame per-pixel label image. Algorithm parameters match
    GRIME_AI_ROI_Analyzer exactly.

    image_bgr : HxWx3 BGR (as cv2.imread returns).
    mask_bin  : HxW, nonzero = inside ROI.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    inside = mask_bin > 0
    coords = np.flatnonzero(inside)                 # flat indices of ROI pixels
    masked_pixels = hsv.reshape(-1, 3)[coords].astype(np.float32)

    h, w = mask_bin.shape[:2]
    labels_image = np.full(h * w, -1, dtype=np.int32)

    if masked_pixels.shape[0] == 0:
        return ClusterResult(labels_image.reshape(h, w),
                             np.empty((0, 3)), np.empty((0, 3), np.uint8),
                             np.empty((0,)), np.empty((0,), np.int32))

    m = method.lower()
    if m == 'gmm':
        labels, centers = _gmm(masked_pixels, clusters, random_state)
    elif m == 'meanshift':
        labels, centers = _meanshift(masked_pixels, bandwidth, quantile,
                                     max_meanshift, random_state)
    else:
        labels, centers = _kmeans(masked_pixels, clusters, random_state)

    counts = np.bincount(labels, minlength=centers.shape[0]).astype(float)
    percentages = counts / counts.sum() * 100.0
    order = np.argsort(percentages)[::-1]

    labels_image[coords] = labels.astype(np.int32)
    return ClusterResult(labels_image.reshape(h, w), centers,
                         _centers_to_rgb(centers), percentages, order)


def _kmeans(pixels, clusters, random_state):
    k = clusters if pixels.shape[0] >= clusters else 1
    km = KMeans(n_clusters=k, random_state=random_state)
    km.fit(pixels)
    return km.labels_, km.cluster_centers_


def _gmm(pixels, clusters, random_state):
    from sklearn.mixture import GaussianMixture
    n = clusters if pixels.shape[0] >= clusters else 1
    gmm = GaussianMixture(n_components=n, random_state=random_state, max_iter=200)
    gmm.fit(pixels)
    return gmm.predict(pixels), gmm.means_


def _meanshift(pixels, bandwidth, quantile, max_samples, random_state):
    from sklearn.cluster import MeanShift, estimate_bandwidth
    if pixels.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        sample = pixels[rng.choice(pixels.shape[0], max_samples, replace=False)]
    else:
        sample = pixels

    if bandwidth is not None:
        bw = bandwidth
    else:
        bw = estimate_bandwidth(sample, quantile=quantile,
                                n_samples=min(500, sample.shape[0]))
        if bw <= 0:
            bw = 10.0

    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(sample)
    centers = ms.cluster_centers_
    diffs = pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]
    labels = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)
    return labels, centers
