import math
import traceback
import numpy as np
import cv2
import pycocotools.mask as maskutils


class MaskVisualizer:
    """
    Mask rendering and centroid utilities for COCO-style annotations.

    Key improvements:
      - canonical polygon handling
      - collapsed decoded masks (H,W) boolean
      - logging on exceptions
      - safe contour shaping for cv2.moments
      - optional caching of decoded masks on the annotation dict
    """

    def __init__(self, category_color_fn=None, marker_color=(255, 255, 255), enable_cache=True):
        self.category_color_fn = category_color_fn or (lambda cid: tuple(np.random.RandomState(int(cid)).randint(0, 255, 3).tolist()))
        self.marker_color = marker_color
        self.enable_cache = enable_cache

    def category_color(self, cid):
        return self.category_color_fn(cid)

    # -------------------------
    # Helper: canonicalize polygons
    # -------------------------
    def _polygons_from_seg(self, seg):
        """
        Return a list-of-polygons where each polygon is a flat list [x0,y0,x1,y1,...].
        """
        if not isinstance(seg, list) or len(seg) == 0:
            return []
        if isinstance(seg[0], (list, tuple)):
            # already list-of-polygons
            return seg[:]
        # single flattened polygon
        return [seg]

    # -------------------------
    # Helper: decode and collapse RLE to single-channel boolean mask
    # -------------------------
    def _decode_mask(self, ann, h, w):
        """
        Decode ann['segmentation'] (RLE or polygon) to a boolean mask (H,W).
        Caches the decoded mask on the annotation dict under key '_decoded_mask_{h}x{w}' if enabled.
        Returns None on failure.
        """
        seg = ann.get("segmentation", None)
        cache_key = f"_decoded_mask_{h}x{w}"
        if self.enable_cache and cache_key in ann:
            m_comb = ann[cache_key]
            # ensure shape matches
            if isinstance(m_comb, np.ndarray) and m_comb.shape[:2] == (h, w):
                return m_comb
            # else fall through to re-decode

        # Polygon case: rasterize
        if isinstance(seg, list) and len(seg) > 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            polys = self._polygons_from_seg(seg)
            for poly in polys:
                try:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                except Exception:
                    # malformed polygon; skip
                    continue
                if pts.size == 0:
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [pts], 1)
            m_comb = (mask > 0).astype(np.uint8)
            if self.enable_cache:
                ann[cache_key] = m_comb
            return m_comb

        # RLE case
        if isinstance(seg, dict):
            try:
                counts = seg.get("counts", None)
                size = seg.get("size", [h, w])
                if isinstance(counts, list):
                    # uncompressed RLE -> compressed (maskutils.decode needs bytes counts)
                    seg_for_decode = maskutils.frPyObjects({"counts": counts, "size": size}, size[0], size[1])
                else:
                    seg_for_decode = seg if "size" in seg else {"counts": counts, "size": [h, w]}
                m = maskutils.decode(seg_for_decode)
                if m is None:
                    return None
                if m.ndim == 3:
                    m_comb = np.any(m, axis=2).astype(np.uint8)
                else:
                    m_comb = (m > 0).astype(np.uint8)
                if self.enable_cache:
                    ann[cache_key] = m_comb
                return m_comb
            except Exception as e:
                print(f"[MaskVisualizer] _decode_mask: decode error: {e}")
                traceback.print_exc()
                return None

        return None

    # -------------------------
    # compute_mask_centroid
    # -------------------------
    def compute_mask_centroid(self, ann, h, w):
        """
        Compute centroid (cx, cy) in pixel coordinates for an annotation.
        Returns (x, y) where x is column (0..w-1) and y is row (0..h-1).
        Strategy:
          - For polygons: try contour moments first; if degenerate, rasterize and use pixel mean.
          - For RLE: decode to mask, try mask moments and fall back to pixel mean.
        """
        seg = ann.get("segmentation", None)
        try:
            # Polygon branch: try contour moments first
            if isinstance(seg, list) and len(seg) > 0:
                polys = self._polygons_from_seg(seg)
                try:
                    contours = []
                    for poly in polys:
                        try:
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        except Exception:
                            continue
                        if pts.shape[0] >= 3:
                            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                            # reshape to Nx1x2 for OpenCV safety
                            cnt = pts.reshape(-1, 1, 2).astype(np.int32)
                            contours.append(cnt)
                    if contours:
                        cx_sum = 0.0
                        cy_sum = 0.0
                        area_sum = 0.0
                        for cnt in contours:
                            M = cv2.moments(cnt)
                            area = M.get("m00", 0.0)
                            if area and not math.isclose(area, 0.0):
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]
                            else:
                                # fallback to contour mean
                                pts_flat = cnt.reshape(-1, 2)
                                cx = float(np.mean(pts_flat[:, 0]))
                                cy = float(np.mean(pts_flat[:, 1]))
                                area = abs(cv2.contourArea(pts_flat)) if pts_flat.shape[0] >= 3 else 1.0
                            cx_sum += cx * area
                            cy_sum += cy * area
                            area_sum += area
                        if area_sum > 0:
                            return (cx_sum / area_sum, cy_sum / area_sum)
                except Exception as e:
                    print(f"[MaskVisualizer] compute_mask_centroid polygon moments error: {e}")
                    traceback.print_exc()

                # Raster fallback (robust)
                mask = self._decode_mask(ann, h, w)
                if mask is None:
                    return (None, None)
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    return (None, None)
                return (float(xs.mean()), float(ys.mean()))

            # RLE branch: decode and use mask moments or pixel mean
            elif isinstance(seg, dict):
                mask = self._decode_mask(ann, h, w)
                if mask is None:
                    return (None, None)
                try:
                    M = cv2.moments(mask.astype(np.uint8))
                    if M.get("m00", 0) != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        return (float(cx), float(cy))
                except Exception as e:
                    print(f"[MaskVisualizer] compute_mask_centroid RLE moments error: {e}")
                    traceback.print_exc()
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    return (None, None)
                return (float(xs.mean()), float(ys.mean()))

        except Exception as e:
            print(f"[MaskVisualizer] compute_mask_centroid unexpected error: {e}")
            traceback.print_exc()
            return (None, None)

        return (None, None)

    # -------------------------
    # draw_centroid
    # -------------------------
    def draw_centroid(self, img, cx, cy, color=None, label=None):
        if cx is None or cy is None:
            return img
        color = color or self.marker_color
        ix = int(round(cx)); iy = int(round(cy))
        h, w = img.shape[:2]
        ix = max(0, min(w - 1, ix)); iy = max(0, min(h - 1, iy))

        try:
            cv2.drawMarker(img, (ix, iy), color, markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=2, line_type=cv2.LINE_AA)
        except Exception:
            l = 8
            cv2.line(img, (ix - l, iy), (ix + l, iy), color, 2, lineType=cv2.LINE_AA)
            cv2.line(img, (ix, iy - l), (ix, iy + l), color, 2, lineType=cv2.LINE_AA)

        text = label if label is not None else f"({ix}, {iy})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        tx = ix + 10; ty = iy - 10
        if tx + tw > w - 1:
            tx = ix - 10 - tw
        if ty - th < 0:
            ty = iy + 10 + th
        tx = max(0, min(w - tw - 1, tx)); ty = max(th, min(h - 1, ty))
        cv2.rectangle(img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        return img

    # -------------------------
    # overlay_all_masks
    # -------------------------
    def overlay_all_masks(self, img, anns):
        overlay = img.copy()
        h, w = img.shape[:2]
        for ann in anns:
            seg = ann.get("segmentation", [])
            color = self.category_color(ann.get("category_id", 0))
            # polygons
            if isinstance(seg, list) and len(seg) > 0:
                try:
                    polys = self._polygons_from_seg(seg)
                    for poly in polys:
                        try:
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        except Exception:
                            continue
                        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                        if pts.shape[0] >= 3:
                            cv2.fillPoly(overlay, [pts], color)
                    cx, cy = self.compute_mask_centroid(ann, h, w)
                    if cx is not None:
                        self.draw_centroid(overlay, cx, cy)
                except Exception as e:
                    print(f"[MaskVisualizer] overlay_all_masks polygon error: {e}")
                    traceback.print_exc()
                    continue

            # RLE
            elif isinstance(seg, dict):
                try:
                    mask = self._decode_mask(ann, h, w)
                    if mask is None:
                        continue
                    overlay[mask > 0] = color
                    cx, cy = self.compute_mask_centroid(ann, h, w)
                    if cx is not None:
                        self.draw_centroid(overlay, cx, cy)
                except Exception as e:
                    print(f"[MaskVisualizer] overlay_all_masks RLE error: {e}")
                    traceback.print_exc()
                    continue

        self.draw_corner_markers(overlay, margin=8, color=(0, 255, 0), marker_size=12, font_scale=0.5)
        return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # -------------------------
    # overlay_single_mask
    # -------------------------
    def overlay_single_mask(self, img, ann):
        overlay = img.copy()
        h, w = img.shape[:2]
        seg = ann.get("segmentation", [])
        color = self.category_color(ann.get("category_id", 0))
        if isinstance(seg, list) and len(seg) > 0:
            try:
                polys = self._polygons_from_seg(seg)
                for poly in polys:
                    try:
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    except Exception:
                        continue
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                    if pts.shape[0] >= 3:
                        cv2.fillPoly(overlay, [pts], color)
                cx, cy = self.compute_mask_centroid(ann, h, w)
                if cx is not None:
                    self.draw_centroid(overlay, cx, cy)
            except Exception as e:
                print(f"[MaskVisualizer] overlay_single_mask polygon error: {e}")
                traceback.print_exc()
        elif isinstance(seg, dict):
            try:
                mask = self._decode_mask(ann, h, w)
                if mask is not None:
                    overlay[mask > 0] = color
                    cx, cy = self.compute_mask_centroid(ann, h, w)
                    if cx is not None:
                        self.draw_centroid(overlay, cx, cy)
            except Exception as e:
                print(f"[MaskVisualizer] overlay_single_mask RLE error: {e}")
                traceback.print_exc()

        self.draw_corner_markers(overlay, margin=8, color=(0, 255, 0), marker_size=12, font_scale=0.5)
        return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # -------------------------
    # draw_mask_border
    # -------------------------
    def draw_mask_border(self, img, ann, color=(0, 255, 255), thickness=4, draw_inplace=True):
        if not draw_inplace:
            img = img.copy()
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and len(seg) > 0:
            try:
                polys = self._polygons_from_seg(seg)
                for poly in polys:
                    try:
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    except Exception:
                        continue
                    pts[:, 0] = np.clip(pts[:, 0], 0, img.shape[1] - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, img.shape[0] - 1)
                    if pts.shape[0] >= 2:
                        cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)
            except Exception as e:
                print(f"[MaskVisualizer] draw_mask_border polygon error: {e}")
                traceback.print_exc()
        elif isinstance(seg, dict):
            try:
                h, w = img.shape[:2]
                mask = self._decode_mask(ann, h, w)
                if mask is None:
                    return img
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(img, contours, -1, color, thickness, lineType=cv2.LINE_AA)
            except Exception as e:
                print(f"[MaskVisualizer] draw_mask_border RLE error: {e}")
                traceback.print_exc()
        return img

    # -------------------------
    # draw_corner_markers
    # -------------------------
    def draw_corner_markers(self, img, margin=8, color=(0, 255, 0), marker_size=12, font_scale=0.5, thickness=1, draw_labels=True):
        h, w = img.shape[:2]
        corners = {
            "TL": (max(margin, 0), max(margin, 0)),
            "TR": (max(w - 1 - margin, 0), max(margin, 0)),
            "BL": (max(margin, 0), max(h - 1 - margin, 0)),
            "BR": (max(w - 1 - margin, 0), max(h - 1 - margin, 0))
        }
        for name, (cx, cy) in corners.items():
            try:
                cv2.drawMarker(img, (int(cx), int(cy)), color, markerType=cv2.MARKER_CROSS,
                               markerSize=marker_size, thickness=thickness, line_type=cv2.LINE_AA)
            except Exception:
                l = marker_size // 2
                cv2.line(img, (cx - l, cy), (cx + l, cy), color, thickness, lineType=cv2.LINE_AA)
                cv2.line(img, (cx, cy - l), (cx, cy + l), color, thickness, lineType=cv2.LINE_AA)
            if draw_labels:
                if name == "TL":
                    coord_text = "(0, 0)"
                elif name == "TR":
                    coord_text = f"({w-1}, 0)"
                elif name == "BL":
                    coord_text = f"(0, {h-1})"
                else:
                    coord_text = f"({w-1}, {h-1})"
                text = f"{name} {coord_text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = font_scale
                th_text = thickness
                (tw, tht), _ = cv2.getTextSize(text, font, fs, th_text)
                if name == "TL":
                    tx = cx + 6; ty = cy + tht + 6
                elif name == "TR":
                    tx = cx - 6 - tw; ty = cy + tht + 6
                elif name == "BL":
                    tx = cx + 6; ty = cy - 6
                else:
                    tx = cx - 6 - tw; ty = cy - 6
                tx = int(max(0, min(w - tw - 1, tx)))
                ty = int(max(tht, min(h - 1, ty)))
                cv2.rectangle(img, (tx - 2, ty - tht - 2), (tx + tw + 2, ty + 2), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, text, (tx, ty), font, fs, (0, 0, 0), th_text, lineType=cv2.LINE_AA)
        return img
