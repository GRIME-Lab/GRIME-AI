from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from grime2py.core import detect_octagon_points


@dataclass
class WaterlineEstimate:
    line: tuple[int, int, int, int] | None
    confidence: float
    reason: str
    candidates: tuple[tuple[int, float], ...] = ()


V2_BAND_ABOVE = 16          # rows of board used as the upper reference
V2_BAND_BELOW = 5           # rows of water used as the lower reference
V2_ANCHOR_MARGIN = 10       # rows of clearance below the octagon
V2_ANCHOR_COLS = True       # also restrict the search to octagon columns
V2_VAR_FLOOR = 25.0         # sensor noise floor inside the log
V2_DROP_K = 6.0             # steepness of the variance-drop weight

# ---------------------------------------------------------------------
# LOCAL CONTRAST ENHANCEMENT (CLAHE) on the crop, before features.
#
# The board is not uniformly lit: on some frames V falls ~32 levels down
# the panel ABOVE the waterline, so the real step competes with a lighting
# ramp, and the within-band spread that normalises the score is inflated
# by illumination rather than by texture. CLAHE equalises locally, which
# removes the ramp while preserving the step.
#
# Applied to the L channel of Lab so hue and saturation are untouched --
# the saturation feature has to keep meaning water tint, not be rescaled.
#
# Measured on 11 hand-labelled frames: 6/11 within 6px and a median error
# of 3.0 without it, 10/11 and a median of 1.0 with it. Unlike every other
# candidate tried, this does not trade one frame for another: it fixed the
# heavy-fouling frame (-22 -> 0), a submerged-board frame (+25 -> +2) and
# 07-23 (-7 -> +1) while breaking none. The result is a broad plateau --
# clip 3-5 with 4-12 tiles all score 9-10/11 -- so these are not knife-edge
# constants.
V2_CLAHE_CLIP = 4.0         # 0 disables
V2_CLAHE_TILES = 4

# ---------------------------------------------------------------------
# CANDIDATE SELECTION: not argmax.
#
# The score finds strong edges. It cannot tell which strong edge is the
# surface when there are two, and there are two in both hard regimes:
#   * submerged board: the surface, and a STRONGER edge lower down where
#     the board disappears (lightness keeps attenuating with depth)
#   * heavy fouling: the surface, and a strong edge at the TOP of the
#     fouling band
# Hand-decomposing true and false rows side by side, two properties held
# on every true row and failed on the false ones:
#   1. V FALLS across the surface. Water is darker than the board above
#      it. The spurious edges that are NOT waterlines at all -- brighter
#      water lower down, glare -- fail this.
#   2. Within a frame, the true row has the LARGER saturation rise.
#      Crossing into water applies the water's tint; the board-end edge
#      and the fouling-band top do not (27 vs 7, 50 vs 33, 32 vs 10).
# So: take the distinct edges scoring at least V2_POOL_FRAC of the best,
# keep those where V falls and that are not below the score winner, and
# choose the largest saturation rise.
#
# Measured on 12 hand-marked frames: argmax gave 9/12 within 6px, median
# 1.5, worst 21. This gives 11/12, median 1.0, worst 9, and is flat for
# V2_POOL_FRAC anywhere in 0.3-0.5. It is a selection stage after the
# score, so it changes nothing about how edges are found.
V2_POOL_FRAC = 0.4



# ---------------------------------------------------------------------
# MIRROR RE-RANK
#
# Open water reflects what is above it, so a water surface is an axis of
# symmetry: flip the strip below a candidate row and it should match the
# strip above. A board bottom edge or a fouling line has no reflection
# beneath it. Where fouling is heavy, the band and its reflection form one
# symmetric blob and the waterline runs through the middle of it.
#
# Evidence, from a hand-labelled ground-truth log: the mirror preferred the
# true row over the detector's row 12 times out of 13, where the composite
# score managed 8. But as an OVERRIDE it is not yet separable from the
# cases where the score is already right:
#   2026-07-12  helps  -22 -> -1   (heavy fouling, band plus reflection)
#   2026-05-09  hurts   -4 -> +19  (the score was already rank-1 correct)
# so it is DEFAULT OFF, and it only overrides when the score-winner itself
# looks unlike water (V2_MIRROR_MAX_RTOP).
#
# Two failure modes are guarded:
#   * uniform regions -- flipping blank board against blank board
#     correlates trivially, so a candidate needs a real brightness step
#     across it (0.5 and 4.9 measured at spurious rows, 42-78 at real
#     waterlines)
#   * near-duplicate rows -- every row inside one cluster is a candidate,
#     and within a cluster the symmetry test always prefers a row a few px
#     lower, which biased every result positive. Only distinct edges (local
#     maxima) are re-ranked.
V2_MIRROR_RERANK = False
V2_MIRROR_H = 35            # rows above/below used for the symmetry test
V2_MIRROR_MIN_STEP = 30.0   # min |mean(above) - mean(below)| in grey levels
V2_MIRROR_MAX_RTOP = 0.25   # only override a score-winner below this r


def _mirror(gray: np.ndarray, x0: int, x1: int, y: int,
            half: int = V2_MIRROR_H) -> float:
    """Reflection symmetry about row y: correlation of the strip above
    against the vertically flipped strip below."""
    if y - half < 0 or y + half >= gray.shape[0]:
        return float("nan")
    above = gray[y - half:y, x0:x1]
    below = gray[y:y + half, x0:x1][::-1, :]
    if above.shape != below.shape or above.size == 0:
        return float("nan")
    u = above - above.mean()
    v = below - below.mean()
    denominator = np.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / denominator) if denominator > 1e-6 else float("nan")


def _mirror_rerank(image: np.ndarray, scores: np.ndarray, row: int,
                   y0: int, x0: int, x1: int) -> tuple[int, str]:
    """Prefer the most symmetric DISTINCT edge, when the score-winner does
    not already look like a water surface. Returns (row, note)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    half = V2_MIRROR_H

    distinct: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if scores[index] <= 0:
            break
        if any(abs(int(index) - taken) < V2_BAND_ABOVE for taken in distinct):
            continue
        distinct.append(int(index))

    top_r = _mirror(gray, x0, x1, y0 + row, half)
    picked, picked_r = None, None
    for index in distinct:
        y = y0 + index
        if y - half < 0 or y + half >= gray.shape[0]:
            continue
        above = gray[y - half:y, x0:x1]
        below = gray[y:y + half, x0:x1]
        if above.size == 0 or below.size == 0:
            continue
        if abs(float(above.mean()) - float(below.mean())) < V2_MIRROR_MIN_STEP:
            continue                        # no real transition here
        value = _mirror(gray, x0, x1, y, half)
        if value != value:
            continue
        if picked_r is None or value > picked_r:
            picked, picked_r = index, value

    if picked is None:
        return row, "mirror: no candidate had a real step"
    if picked == row:
        return row, "mirror agrees (r=%.2f)" % picked_r
    if top_r == top_r and top_r >= V2_MIRROR_MAX_RTOP:
        return row, "mirror: kept y=%d (its r=%.2f >= %.2f)" % (
            y0 + row, top_r, V2_MIRROR_MAX_RTOP)
    return picked, "mirror: y=%d (r=%.2f) over y=%d (r=%.2f)" % (
        y0 + picked, picked_r, y0 + row, top_r)


def _enhance(crop: np.ndarray) -> np.ndarray:
    """Local contrast equalisation on lightness only. See the note above."""
    if V2_CLAHE_CLIP <= 0:
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(
        clipLimit=V2_CLAHE_CLIP,
        tileGridSize=(V2_CLAHE_TILES, V2_CLAHE_TILES)).apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]),
                        cv2.COLOR_LAB2BGR)


def _features(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row [value, saturation, log-texture] plus the raw row variance.

# Features are Lab CHROMA (a, b) plus texture -- NOT lightness.
#
# Lightness is the wrong channel where the board continues below the
# surface. It keeps attenuating with depth, so it shows TWO edges: the
# waterline, and a stronger one lower down where the submerged board
# finally disappears. Measured on 2025-09-30: L steps at row 714 and again
# at 736, and the second step is larger, so any lightness-driven score
# picks the wrong one.
#
# Chroma does not attenuate the same way. Submerged board and open water
# are both tinted by the same water, so a and b step ONCE, at the surface,
# and then stay flat. On that frame a goes 133 -> 117 and b goes 126 -> 143
# at row 714 and both hold through 736 and beyond.
#
# Measured over 12 hand-labelled frames: chroma+texture gives a median
# error of 1.0px against 1.5px for value+saturation+texture, at the same
# 10/12 within 6px. Chroma alone fixes every submerged-board frame
# (+7, 0, -1, 0) but costs the others, so texture stays in.

    The noise floor inside the log is not a maths guard.

    With +1, a row
    variance of 1 against 0 -- pure sensor noise on blown-out white board --
    swings the texture feature by 12, while a real board-to-water step of
    1227 -> 472 only moves it 16, so saturated regions outscored the
    waterline. A floor of 25 (std of 5 grey levels) drops that noise swing
    to 0.7. It matters most on narrow ROIs, where fewer columns per row
    push more rows down into the noise.
    """
    crop = _enhance(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    value = hsv[..., 2].mean(axis=1)
    saturation = hsv[..., 1].mean(axis=1)
    variance = gray.var(axis=1)
    texture = np.log10(variance + V2_VAR_FLOOR) * 40.0
    return np.stack([value, saturation, texture], axis=1), variance


def _score(features: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Separation between a LONG band above and a SHORT band below.

    The asymmetry is the point: the board above a waterline is stationary,
    while the water below drifts with reflections, so only the few rows
    immediately under the surface are trustworthy. A symmetric window
    collapses once the lower band starts spanning the reflection gradient.

    The variance drop is a WEIGHT, not a gate. As a hard gate it deleted
    hand-marked true waterlines outright -- one where the drop ratio was
    1.18, and one at 0.76 where variance RISES across the true surface
    because the water below was rougher than the board above. A deleted row
    cannot be recovered downstream; a mis-weighted one can still win.
    """
    count = len(features)
    scores = np.full(count, -1.0, dtype=np.float32)
    for row in range(V2_BAND_ABOVE, count - V2_BAND_BELOW):
        above = float(np.median(variance[row - V2_BAND_ABOVE:row]))
        below = float(np.median(variance[row:row + V2_BAND_BELOW]))
        first = features[row - V2_BAND_ABOVE:row]
        second = features[row:row + V2_BAND_BELOW]
        scale = np.sqrt((first.var(axis=0) + second.var(axis=0)) / 2.0) + 1.0
        separation = float(np.linalg.norm(
            np.abs(first.mean(axis=0) - second.mean(axis=0)) / scale))
        ratio = np.log10(max(above, 1.0) / max(below, 1.0))
        scores[row] = separation * float(1.0 / (1.0 + np.exp(-V2_DROP_K * ratio)))
    return scores
def _select_row(scores, features, w_above, w_below):
    """Choose among strong distinct edges. See the note on V2_POOL_FRAC."""
    n = len(scores)
    order = np.argsort(scores)[::-1]
    distinct = []
    for index in order:
        index = int(index)
        if scores[index] <= 0:
            break
        if any(abs(index - taken) < w_above for taken in distinct):
            continue
        distinct.append(index)
    if not distinct:
        return int(np.argmax(scores))
    best = float(scores[distinct[0]])

    def d_value(j):
        return float(features[j:j + w_below, 0].mean()
                     - features[j - w_above:j, 0].mean())

    def d_sat(j):
        return float(features[j:j + w_below, 1].mean()
                     - features[j - w_above:j, 1].mean())

    strong = [j for j in distinct if scores[j] >= V2_POOL_FRAC * best]
    pool = [j for j in strong if d_value(j) < 0] or strong or distinct[:1]
    # Never choose BELOW the score winner. The strongest edge is either the
    # surface itself or the submerged board disappearing beneath it, and
    # the surface is never below that. Without this bound a deep search
    # polygon let a saturation edge 215px down in the water win.
    above = [j for j in pool if j <= distinct[0]] or pool
    return max(above, key=d_sat)


def estimate_waterline(
    image: np.ndarray,
    roi: Sequence[int],
    anchor: bool = True,
) -> WaterlineEstimate:
    """Confidence-gated waterline estimator based on the supplied waterline_tool."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("waterline ensemble requires a BGR image")
    x0, y0, x1, y1 = map(int, roi)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0, x0), min(image.shape[1] - 1, x1)
    y0, y1 = max(0, y0), min(image.shape[0] - 1, y1)
    # ANCHOR. The octagon is a fixed landmark on the board, so it bounds the
    # search in BOTH directions:
    #   rows    -- start below it, so nothing printed on the board (the
    #              octagon border, the logo) can be taken for the waterline
    #   columns -- use only the octagon's own columns, which are guaranteed
    #              board-over-water with no vegetation, gauge or pipe. Row
    #              statistics are means across columns, so anything else in
    #              the box dilutes the real step: with a ROI where ~60% of
    #              columns were vegetation, full-width columns scored 17/20
    #              and the misses landed BELOW the waterline, while octagon
    #              columns scored 20/20.
    # The octagon is located in the WHOLE FRAME, not inside the ROI:
    # requiring the user's box to contain it only creates a silent failure
    # when the box is drawn below it.
    anchor_note = ""
    if anchor:
        try:
            octagon = np.asarray(detect_octagon_points(image), dtype=np.float32)
            top = int(np.max(octagon[:, 1]) + V2_ANCHOR_MARGIN)
            if top < y1 - V2_BAND_ABOVE - V2_BAND_BELOW - 4:
                y0 = max(y0, top)
                anchor_note = f"octagon anchor: rows from y={y0}"
                if V2_ANCHOR_COLS:
                    left, right = int(np.min(octagon[:, 0])), int(np.max(octagon[:, 0]))
                    new_x0, new_x1 = max(x0, left), min(x1, right)
                    if new_x1 - new_x0 >= 8:
                        x0, x1 = new_x0, new_x1
                        anchor_note += f", cols {x0}-{x1}"
                    else:
                        anchor_note += ", no column overlap"
            else:
                anchor_note = "octagon too low in ROI; anchor ignored"
        except (ValueError, cv2.error):
            anchor_note = "octagon not found; anchor inactive"
    crop = image[y0:y1 + 1, x0:x1 + 1]
    if crop.shape[0] < V2_BAND_ABOVE + V2_BAND_BELOW + 4 or crop.shape[1] < 6:
        return WaterlineEstimate(None, 0.0, "ROI too small")

    features, variance = _features(crop)
    scores = _score(features, variance)
    valid = scores > 0
    if not valid.any():
        return WaterlineEstimate(None, 0.0, "no scorable row in the search range")
    row = _select_row(scores, features, V2_BAND_ABOVE, V2_BAND_BELOW)
    mirror_note = ""
    if V2_MIRROR_RERANK:
        row, mirror_note = _mirror_rerank(image, scores, row, y0, x0, x1)
    best = float(scores[row])
    candidates = []
    for candidate in np.argsort(scores)[::-1]:
        if scores[candidate] <= 0:
            break
        if any(abs(int(candidate) - prior) < V2_BAND_ABOVE for prior, _ in candidates):
            continue
        candidates.append((y0 + int(candidate), float(scores[candidate])))
        if len(candidates) == 10:
            break
    pool = np.sort(scores[valid])[::-1]
    reference = float(pool[min(len(pool) - 1, max(1, int(len(pool) * 0.05)))])
    confidence = min(1.0, max(0.0, best / (reference + 1e-6) - 1.0))
    reason = (f"{anchor_note}; candidate y={y0 + row}, score={best:.2f},"
              f" confidence={confidence:.2f}")
    if mirror_note:
        reason += f" | {mirror_note}"
    return WaterlineEstimate((x0, y0 + row, x1, y0 + row), confidence, reason, tuple(candidates))
