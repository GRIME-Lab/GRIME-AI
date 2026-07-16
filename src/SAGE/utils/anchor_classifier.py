# SAGE/utils/anchor_classifier.py
"""
Two-class anchor-based region classifier.

Given a set of class-agnostic candidate masks (e.g. from SAM2's automatic
mask generator) plus a few positive positive anchors and optional negative anchors, assign every candidate to target or exclude by
Mahalanobis distance in a standardized per-region feature space, then return
the union of the target-assigned candidates..

Generic: positive anchors define the target class, negative anchors define
exclude. No domain assumptions. Uses only numpy + OpenCV.
"""
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Per-region feature extraction
# ---------------------------------------------------------------------------
def _image_feature_planes(image_np):
    """
    Precompute pixel-wise planes once per image so per-mask reduction is cheap.
    Returns (hsv float32 HxWx3, grad float32 HxW).
    """
    if image_np.ndim == 2:
        rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image_np[..., :3]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    return hsv, grad


def _region_feature(hsv, grad, region_mask):
    """
    8-D feature for one region: [H,S,V mean], [H,S,V std], [grad mean, grad std].
    Returns None if the region is empty.
    """
    m = region_mask.astype(bool)
    if not m.any():
        return None

    h = hsv[..., 0][m]
    s = hsv[..., 1][m]
    v = hsv[..., 2][m]
    g = grad[m]

    return np.array([
        h.mean(), s.mean(), v.mean(),
        h.std(),  s.std(),  v.std(),
        g.mean(), g.std(),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Robust Mahalanobis with shrinkage (works with very few samples per class)
# ---------------------------------------------------------------------------
def _class_model(samples, dim, shrink=0.25, eps=1e-3):
    """
    Build (mean, inv_cov) for a class from standardized samples.
    Shrinks the sample covariance toward a scaled identity so that a class
    with 1-4 samples in an 8-D space still yields a usable inverse.
    """
    samples = np.atleast_2d(samples)
    mean = samples.mean(axis=0)

    if samples.shape[0] >= 2:
        cov = np.cov(samples.T)
        if cov.ndim == 0:               # dim == 1 edge case
            cov = np.array([[float(cov)]])
    else:
        cov = np.zeros((dim, dim))

    # Shrink toward diagonal average variance (Ledoit-Wolf style, simplified).
    trace_mean = np.trace(cov) / dim if dim else 1.0
    target = np.eye(dim) * max(trace_mean, eps)
    cov = (1.0 - shrink) * cov + shrink * target
    cov += np.eye(dim) * eps            # final guard against singularity

    return mean, np.linalg.inv(cov)


def _mahalanobis(x, mean, inv_cov):
    d = x - mean
    return float(np.sqrt(max(d @ inv_cov @ d, 0.0)))


# ---------------------------------------------------------------------------
# Per-pixel exclude veto (target+exclude sharing one candidate region)
# ---------------------------------------------------------------------------
def _pixel_samples(hsv, grad, points, region_mask, radius=4):
    """Collect per-pixel [H,S,V,grad] feature rows from windows around points
    that fall inside region_mask."""
    H, W = region_mask.shape
    rows = []
    for x, y in points:
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < W and 0 <= py < H) or not region_mask[py, px]:
            continue
        y0, y1 = max(0, py - radius), min(H, py + radius + 1)
        x0, x1 = max(0, px - radius), min(W, px + radius + 1)
        win = region_mask[y0:y1, x0:x1]
        if not win.any():
            continue
        h = hsv[y0:y1, x0:x1, 0][win]
        s = hsv[y0:y1, x0:x1, 1][win]
        v = hsv[y0:y1, x0:x1, 2][win]
        g = grad[y0:y1, x0:x1][win]
        rows.append(np.stack([h, s, v, g], axis=1))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 4))


def _carve_excludes(region_mask, hsv, grad, fg_points, bg_points):
    """
    Within a target-assigned region that also contains exclude (negative) anchors,
    keep only pixels closer to the target distribution than the exclude one.
    Returns a boolean sub-mask.
    """
    target_px = _pixel_samples(hsv, grad, fg_points, region_mask)
    exclude_px   = _pixel_samples(hsv, grad, bg_points, region_mask)

    if exclude_px.shape[0] < 3:
        return region_mask                      # nothing reliable to subtract

    # If no positive anchor sits in this region, derive a target prototype from
    # region pixels far from the exclude samples (median is robust to leftover exclude).
    if target_px.shape[0] < 3:
        feats = np.stack([
            hsv[..., 0][region_mask], hsv[..., 1][region_mask],
            hsv[..., 2][region_mask], grad[region_mask]
        ], axis=1)
        target_px = feats

    dim = 4
    w_mean, w_inv = _class_model(_std(target_px), dim)
    b_mean, b_inv = _class_model(_std(exclude_px, ref=target_px), dim)

    ys, xs = np.nonzero(region_mask)
    P = np.stack([hsv[..., 0][ys, xs], hsv[..., 1][ys, xs],
                  hsv[..., 2][ys, xs], grad[ys, xs]], axis=1)
    Pz = _std(P, ref=target_px)

    keep = np.array([
        _mahalanobis(p, w_mean, w_inv) <= _mahalanobis(p, b_mean, b_inv)
        for p in Pz
    ])
    out = np.zeros_like(region_mask)
    out[ys[keep], xs[keep]] = True
    return out


_STD_MU = {}
def _std(X, ref=None):
    """Z-score X. If ref given, standardize X using ref's stats (shared space)."""
    base = ref if ref is not None else X
    mu = base.mean(axis=0)
    sig = base.std(axis=0) + 1e-8
    return (X - mu) / sig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def classify_by_anchors(image_np, candidate_masks, fg_points, bg_points,
                   min_region_px=25):
    """
    Parameters
    ----------
    image_np       : HxWx3 RGB uint8
    candidate_masks: list of HxW bool/uint8 masks (class-agnostic)
    fg_points      : list of (x, y) target anchors
    bg_points      : list of (x, y) exclude anchors (may be empty)
    min_region_px  : ignore candidate regions smaller than this

    Returns
    -------
    target_union : HxW bool mask (union of target-assigned candidates), or None
    info        : dict with diagnostics (per-candidate assignment, counts)
    """
    if not candidate_masks or not fg_points:
        return None, {"reason": "need candidates and >=1 target anchor"}

    H, W = image_np.shape[:2]
    hsv, grad = _image_feature_planes(image_np)

    # --- build feature matrix for all candidates -------------------------
    regions = []          # (index, mask_bool, feature_vec)
    for i, cm in enumerate(candidate_masks):
        mb = cm.astype(bool)
        if mb.sum() < min_region_px:
            continue
        f = _region_feature(hsv, grad, mb)
        if f is not None:
            regions.append((i, mb, f))

    if not regions:
        return None, {"reason": "no usable candidate regions"}

    F = np.stack([r[2] for r in regions])          # (N, 8)
    dim = F.shape[1]

    # --- standardize feature space --------------------------------------
    mu = F.mean(axis=0)
    sigma = F.std(axis=0) + 1e-8
    Z = (F - mu) / sigma                           # (N, 8)

    # --- map anchors to the candidate region they fall inside -----------
    def region_at(px, py):
        px, py = int(round(px)), int(round(py))
        if not (0 <= px < W and 0 <= py < H):
            return None
        for ridx, (_, mb, _) in enumerate(regions):
            if mb[py, px]:
                return ridx
        return None

    target_idx, exclude_idx = set(), set()
    for x, y in fg_points:
        r = region_at(x, y)
        if r is not None:
            target_idx.add(r)
    for x, y in bg_points:
        r = region_at(x, y)
        if r is not None:
            exclude_idx.add(r)

    if not target_idx:
        return None, {"reason": "no target anchor landed on a candidate region"}

    target_samples = Z[sorted(target_idx)]
    w_mean, w_inv = _class_model(target_samples, dim)

    # Absolute keep-radius derived from the spread of the target anchors.
    # A region must be *this close* to the target distribution to qualify,
    # regardless of whether it happens to beat the exclude class.
    target_cutoff = _single_class_cutoff(target_samples, w_mean, w_inv)

    # 
    have_exclude = len(exclude_idx) > 0
    if have_exclude:
        exclude_samples = Z[sorted(exclude_idx)]
        b_mean, b_inv = _class_model(exclude_samples, dim)

    # --- assign every candidate -----------------------------------------
    assignment = []          # list of (candidate_index, "target"/"exclude", d_target, d_exclude)
    target_union = np.zeros((H, W), dtype=bool)

    # Precompute which regions contain exclude anchors (for sub-region veto).
    def _exclude_points_in(mb):
        pts = []
        for x, y in bg_points:
            px, py = int(round(x)), int(round(y))
            if 0 <= px < W and 0 <= py < H and mb[py, px]:
                pts.append((x, y))
        return pts

    for ridx, (orig_i, mb, _) in enumerate(regions):
        z = Z[ridx]
        dw = _mahalanobis(z, w_mean, w_inv)

        if ridx in target_idx:
            label = "target"                         # hard positive constraint
            db = float("nan")
        elif ridx in exclude_idx:
            label = "exclude"                            # hard negative constraint
            db = 0.0
        else:
            if have_exclude:
                db = _mahalanobis(z, b_mean, b_inv)
                # Nearest class AND inside the absolute target radius.
                label = "target" if (dw <= db and dw <= target_cutoff) else "exclude"
            else:
                db = float("nan")
                label = "target" if dw <= target_cutoff else "exclude"

        if label == "target":
            # Sub-region veto: if this target region also holds exclude anchors,
            # carve out exclude pixels per-pixel instead of taking the whole blob.
            local_excludes = _exclude_points_in(mb)
            keep = _carve_excludes(mb, hsv, grad, fg_points, local_excludes) if local_excludes else mb
            target_union |= keep
        assignment.append((orig_i, label, dw, db))

    info = {
        "n_candidates": len(candidate_masks),
        "n_regions": len(regions),
        "n_target_anchors_matched": len(target_idx),
        "n_exclude_anchors_matched": len(exclude_idx),
        "used_exclude_class": have_exclude,
        "assignment": assignment,
    }
    if not target_union.any():
        info["reason"] = "no region matched the target anchors"
        return None, info
    return target_union, info


def _single_class_cutoff(samples, mean, inv_cov, pad=1.8):
    """
    When no exclude anchors are given, derive a keep-radius from the spread of the
    target anchors themselves: max in-sample distance times a padding factor,
    with a sane floor so a single anchor still admits similar regions.
    """
    if samples.shape[0] >= 2:
        dmax = max(_mahalanobis(s, mean, inv_cov) for s in samples)
        return max(dmax * pad, 3.0)
    return 3.0
