"""
Gate detector using orange HSV colour segmentation.

detect_gate(frame) -> np.ndarray | None
    Returns 4 corner points [TL, TR, BR, BL] of the largest (= closest)
    orange gate in the frame, or None if no gate is found or the gate is
    too close/clipped to estimate a pose from (3+ corners on the frame
    border, see MAX_CLIPPED_CORNERS).

Corner ordering assumes the outer gate boundary (2720 mm × 2720 mm) is
detected.  Callers should use GATE_OUTER_HALF = 1.36 m as their solvePnP
object-point half-size.

Heavily-clipped gates are rejected rather than passed to solvePnP: when a
gate extends past the frame edge, cv2.findContours follows the image border
for the off-screen portion, so the resulting "corner" there is a clipped
border point, not the gate's true physical corner. solvePnP assumes all 4
points are real corners, so each clipped point degrades the pose. Offline
validation (dataset_tools/validate_labels.py) plus live replay against
logged poses showed 1-2 clipped corners still produce useful estimates
(median ~1.1-1.3 m error at medium range), while 3-4 clipped corners
(very close range) produce errors exceeding the gate range itself --
those are rejected.
"""

import cv2
import numpy as np

# ── Tunable constants ─────────────────────────────────────────────────────────

# HSV range for the gate's orange colour.
# OpenCV H: 0–179, S: 0–255, V: 0–255.
# Two bands cover orange (H 0–25) and the red wrap-around (H 160–179).
# Saturation floor lowered to 90 (was 150): at medium/far range the gate's true-red
# pixels commonly measure sat 90-140 (JPEG/lighting), so 150 speckled the mask into
# unusable fragments. Floor-glow rejection is handled by the ring/shape/spotlight
# guards below, not by this floor -- floor glow itself typically reads sat 150-250.
_HSV_LO1 = np.array([  0,  90,  60], dtype=np.uint8)
_HSV_HI1 = np.array([ 25, 255, 255], dtype=np.uint8)
_HSV_LO2 = np.array([160,  90,  60], dtype=np.uint8)
_HSV_HI2 = np.array([179, 255, 255], dtype=np.uint8)

# HSV range for the blue guide ribbon — masked out before gate detection so the
# ribbon doesn't fragment the orange gate contour when it passes through the opening.
_BLUE_LO = np.array([ 95, 120,  60], dtype=np.uint8)
_BLUE_HI = np.array([135, 255, 255], dtype=np.uint8)

# Minimum contour area in pixels to be considered a gate candidate.
MIN_GATE_AREA_PX: int = 200

FLOOR_MASK_FRACTION: float = 0.0

# Inner-hole spotlight trimming: after MORPH_OPEN, we search for the gate's
# hollow interior using hierarchical contour detection. The bottom of that
# inner hole marks where the gate frame ends. Adding GATE_FRAME_MARGIN_PX
# accounts for the frame bar thickness, then everything below is zeroed
# (removing the floor spotlight) before MORPH_CLOSE runs.
# If no inner hole is found (gate too small or distant), trimming is skipped.
GATE_INNER_MIN_AREA_PX: int = 150   # minimum inner hole area to trust
GATE_FRAME_MARGIN_PX:   int = 5     # px below inner hole bottom to include
# Inner hole's parent contour must be at least this fraction of the largest outer
# blob in the scene. Prevents a small background gate's inner hole from being used
# as the trim cutoff when the main (close) gate has no detectable inner hole.
GATE_HOLE_MIN_PARENT_FRACTION: float = 0.30
# If the selected inner hole's bottom exceeds this fraction of frame height, skip
# trimming.  With the adaptive margin (see below) the cutoff is placed at the outer
# gate bottom rather than the inner hole bottom, so the guard can be set high —
# at 90%+ the cutoff naturally falls off the frame edge anyway.
GATE_HOLE_MAX_Y_FRACTION: float = 0.90

# The gate bar thickness / inner-opening ratio from spec: bar = 610 mm, inner = 1500 mm.
# Used to compute an adaptive spotlight-trim margin so the cutoff lands at the outer
# gate bottom (not inside the bar) regardless of distance.
_GATE_BAR_TO_INNER_RATIO: float = 610.0 / 1500.0  # ≈ 0.407

# Kernel size for the ring-validation close pass.  Smaller than the main 9×9 so the
# gate's inner hole is preserved while small ribbon-subtraction gaps are still bridged.
_RING_CLOSE_SIZE: int = 5

# Gate-shape discriminator: a gate frame (ring) has contour_area / bbox_area < this
# threshold because the hollow interior reduces the filled area vs. the bounding rect.
# Solid clutter blobs have fill_ratio near 1.0.  Gate rings (4-sided or C-shaped due
# to ribbon gaps) typically land at 0.45–0.76.
# Only used as a fallback for larger blobs — small blobs must use the RETR_TREE hole
# check to avoid false-accepting noise.
GATE_RING_MAX_FILL:         float = 0.80
GATE_RING_FILL_MIN_AREA_PX: int   = 1000  # minimum contour area to trust fill_ratio alone

# approxPolyDP epsilon as a fraction of contour perimeter.
_POLY_EPS: float = 0.05

# Gate outer half-dimension (metres) — used by pose_estimator object points.
GATE_OUTER_HALF: float = 1.36  # 2720 mm / 2

# A corner within this many pixels of the frame edge is treated as a clipped
# contour point (gate partially out of frame) rather than a true gate corner.
BORDER_MARGIN_PX: float = 2.0

# Reject a detection once this many corners are clipped to the frame edge.
# Empirically (dataset_tools/validate_labels.py + live replay against
# logged poses): 3-4 clipped corners correspond to very-close-range views
# where the polygon is essentially meaningless (reconstruction error exceeds
# the gate range itself). 1-2 clipped corners correspond to medium-range
# partial views that still solvePnP to a useful estimate (~1-1.3 m median
# error -- better than many fully-visible long-range detections), so those
# are kept.
MAX_CLIPPED_CORNERS: int = 2

# Floor-spotlight distortion guard.  At close range the gate's downward orange
# spotlight merges with the gate's bottom bar via MORPH_CLOSE, pulling the BL/BR
# corners outward.  For a real (roughly square) gate the bottom pixel-width
# (BR.x − BL.x) should not be more than this ratio times the top width
# (TR.x − TL.x).  Values > 1.6 reliably indicate spotlight distortion;
# legitimate perspective-tilted gates stay below 1.4.
_SPOTLIGHT_WIDTH_RATIO:    float = 1.6   # bottom_w > top_w * this → spread
_SPOTLIGHT_COLLAPSE_RATIO: float = 0.3   # bottom_w < top_w * this → triangle collapse
# Bottom corners skewed vertically: |BR.y - BL.y| > gate_height * this.
# Floor spotlight can drag one bottom corner down diagonally while the other stays
# near the gate, creating an asymmetric (lopsided) quadrilateral.  Legitimate
# perspective (gate yawed up to ~30°) produces bottom_skew < 0.25; spotlight
# distortion produces 0.4+.
_SPOTLIGHT_SKEW_RATIO:     float = 0.35

# Set True to print per-frame detection diagnostics (HSV at gate centre, area, etc.)
VERBOSE: bool = False


# ── Public API ────────────────────────────────────────────────────────────────

def detect_gate(frame: np.ndarray, diag: dict | None = None) -> np.ndarray | None:
    """
    Detect the closest (largest) orange gate in a BGR frame.

    Returns float32 (4, 2) corners ordered [TL, TR, BR, BL], or None.
    Uses the outer gate boundary (2720 mm); caller should use
    GATE_OUTER_HALF = 1.36 m as solvePnP object-point half-size.

    diag : optional dict, populated (in place, cleared first) with
           pipeline-internals diagnostics for offline audit of the filter
           stages this function silently discards results from -- e.g.
           "was there a gate-sized blob that the aspect/area filter
           rejected?" or "how close did approxPolyDP get to 4 points?".
           Costs nothing extra when left None.
    """
    if diag is not None:
        diag.clear()

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    final_mask, open_mask = _orange_mask_from_hsv(hsv)

    # Ring check: identify blobs in open_mask that look like gate frames (hollow rings).
    # Primary: RETR_TREE hole detection — a gate ring encloses a void; solid clutter
    # does not.  Works at any size for fully-closed rings.
    # Fallback: fill_ratio (contour_area / bbox_area) for large C-shaped rings where
    # the ribbon or clipping breaks the ring and RETR_TREE finds no inner hole.
    # Small blobs (< GATE_RING_FILL_MIN_AREA_PX) must use the primary check only —
    # noise and irregular clutter at small scales also have low fill_ratio.
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_RING_CLOSE_SIZE, _RING_CLOSE_SIZE))
    ring_mask   = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, ring_kernel)
    ring_cnts, ring_hier = cv2.findContours(ring_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ring_data: list of outer bboxes for blobs that pass the ring-shape check.
    # Used to filter candidates: a detection is only accepted if its centroid
    # falls inside one of these ring bboxes.
    ring_data: list[tuple] = []
    if ring_cnts and ring_hier is not None:
        rh = ring_hier[0]
        for i, row in enumerate(rh):
            if row[3] >= 0:
                continue  # inner contour
            rc_area = cv2.contourArea(ring_cnts[i])
            if rc_area < GATE_INNER_MIN_AREA_PX:
                continue

            # Primary: detect closed ring via inner hole (RETR_TREE child contour).
            has_hole = False
            child = row[2]
            while child >= 0:
                if cv2.contourArea(ring_cnts[child]) >= GATE_INNER_MIN_AREA_PX:
                    has_hole = True
                    break
                child = rh[child][0]

            if has_hole:
                ring_data.append(cv2.boundingRect(ring_cnts[i]))
                continue

            # Fallback: large C-shaped/partial ring with low fill_ratio (no closed hole).
            if rc_area < GATE_RING_FILL_MIN_AREA_PX:
                continue
            rx, ry, rw, rh_dim = cv2.boundingRect(ring_cnts[i])
            if rw * rh_dim > 0 and rc_area / (rw * rh_dim) < GATE_RING_MAX_FILL:
                ring_data.append((rx, ry, rw, rh_dim))

    # Main contour detection on the spotlight-trimmed, 9×9-closed final mask.
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if diag is not None:
        diag['n_contours'] = len(contours)
    if not contours:
        if diag is not None:
            diag['detect_result'] = 'no_contours'
        if VERBOSE:
            print('[detector] no orange contours found')
        return None

    areas   = [float(cv2.contourArea(c)) for c in contours]
    aspects = [_aspect_ratio(c) for c in contours]

    # A candidate must: meet minimum area, be roughly square, AND have its centroid
    # inside one of the ring bboxes found on the open mask.
    has_ring = [_centroid_in_any_bbox(c, ring_data) for c in contours]
    passed   = [
        a >= MIN_GATE_AREA_PX and _roughly_square(c) and ring
        for a, c, ring in zip(areas, contours, has_ring)
    ]
    li = int(np.argmax(areas))

    if diag is not None:
        diag['largest_area']    = areas[li]
        diag['largest_aspect']  = aspects[li]
        diag['largest_passed']  = bool(passed[li])
        diag['n_candidates']    = int(sum(passed))
        diag['n_ring']          = int(sum(has_ring))

    candidates = [c for c, p in zip(contours, passed) if p]
    if not candidates:
        if diag is not None:
            diag['detect_result'] = 'no_candidates'
        if VERBOSE:
            print(f'[detector] {len(contours)} contour(s) found but none passed ring+shape filters '
                  f'(largest area={areas[li]:.0f} px², rings={sum(has_ring)}/{len(contours)})')
        return None

    best = max(candidates, key=cv2.contourArea)

    if diag is not None:
        diag['best_area']   = float(cv2.contourArea(best))
        diag['best_aspect'] = _aspect_ratio(best)

    corners, poly_diag = _four_corners(best)
    if corners is None:
        if diag is not None:
            diag['detect_result'] = 'no_4corner'
        if VERBOSE:
            print('[detector] could not reduce best contour to 4 corners')
        return None
    ordered = _order_corners(corners)

    if diag is not None:
        diag.update(poly_diag)

    # Reject corners distorted by the floor spotlight.  Two failure modes:
    #
    # 1. Spread: BL/BR dragged far outward → bottom_w >> top_w.
    # 2. Collapse: BL/BR converge at the spotlight TIP (downward triangle) →
    #    bottom_w ≈ 0 while top_w is normal.  approxPolyDP on the T-shape finds
    #    the floor-glow apex as a single "bottom" vertex shared by both BL & BR.
    #
    # Gate corners are ordered [TL, TR, BR, BL].
    top_w    = ordered[1, 0] - ordered[0, 0]   # TR.x − TL.x
    bottom_w = ordered[2, 0] - ordered[3, 0]   # BR.x − BL.x
    gate_h     = (ordered[2, 1] + ordered[3, 1]) / 2 - (ordered[0, 1] + ordered[1, 1]) / 2
    bottom_skew = abs(ordered[2, 1] - ordered[3, 1])  # |BR.y - BL.y|
    if top_w > 0 and (bottom_w > top_w * _SPOTLIGHT_WIDTH_RATIO
                      or bottom_w < top_w * _SPOTLIGHT_COLLAPSE_RATIO
                      or (gate_h > 0 and bottom_skew > gate_h * _SPOTLIGHT_SKEW_RATIO)):
        if diag is not None:
            diag['detect_result'] = 'spotlight_distortion'
        return None

    h, w = frame.shape[:2]
    n_clipped = _n_border_corners(ordered, w, h)
    if diag is not None:
        diag['n_clipped_corners'] = n_clipped
    if n_clipped > MAX_CLIPPED_CORNERS:
        if diag is not None:
            diag['detect_result'] = 'partial'
        if VERBOSE:
            print(f'[detector] gate too clipped ({n_clipped}/4 corners on border) -- skipping pose estimate')
        return None

    if diag is not None:
        diag['detect_result'] = 'ok'

    if VERBOSE:
        area = cv2.contourArea(best)
        cx   = int(ordered[:, 0].mean())
        cy   = int(ordered[:, 1].mean())
        h, w = frame.shape[:2]
        cx   = max(0, min(cx, w - 1))
        cy   = max(0, min(cy, h - 1))
        hsv_at_centre = hsv[cy, cx]
        print(f'[detector] gate detected  area={area:.0f} px²  '
              f'centre=({cx},{cy})  HSV@centre={hsv_at_centre.tolist()}  '
              f'corners(TL→BR)={ordered.astype(int).tolist()}')

    return ordered


def draw_detection(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw detected corners on a copy of frame (for debugging)."""
    out = frame.copy()
    pts = corners.astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    labels = ['TL', 'TR', 'BR', 'BL']
    for pt, label in zip(pts, labels):
        cv2.circle(out, tuple(pt), 5, (0, 255, 0), -1)
        cv2.putText(out, label, tuple(pt + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ── Internals ─────────────────────────────────────────────────────────────────

def _orange_mask_from_hsv(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (final_mask, open_mask).

    final_mask : spotlight-trimmed, 9×9-closed — use for contour detection.
    open_mask  : spotlight-trimmed, 3×3-opened only — use for ring validation
                 (the 9×9 close fills the gate's inner hole at medium range).
    """
    mask1 = cv2.inRange(hsv, _HSV_LO1, _HSV_HI1)
    mask2 = cv2.inRange(hsv, _HSV_LO2, _HSV_HI2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Blank the bottom FLOOR_MASK_FRACTION of the frame to cut off the gate's
    # downward spotlight before it merges with the gate frame contour.
    if FLOOR_MASK_FRACTION > 0.0:
        cutoff = int(hsv.shape[0] * (1.0 - FLOOR_MASK_FRACTION))
        mask[cutoff:, :] = 0

    # Remove pixels that belong to the blue guide ribbon.
    blue_mask   = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
    blue_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask   = cv2.dilate(blue_mask, blue_kernel, iterations=1)
    mask        = cv2.bitwise_and(mask, cv2.bitwise_not(blue_mask))

    open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # Inner-hole spotlight trim: use RETR_TREE to find the gate's hollow
    # interior, which marks where the gate frame ends. Everything below that
    # + GATE_FRAME_MARGIN_PX is floor spotlight — zero it before the real
    # MORPH_CLOSE bridges them.
    #
    # We search for the inner hole on the MORPH_CLOSE result (not MORPH_OPEN)
    # because CLOSE bridges the small ribbon-subtraction gaps that can break
    # the gate ring at the OPEN stage, making the interior un-detectable.
    mask_close_tmp = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, close_kernel)
    cnts, hier = cv2.findContours(mask_close_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cutoff = -1
    hole_bottom = -1
    if hier is not None and len(cnts) > 0:
        h = hier[0]  # shape (n, 4): [next, prev, first_child, parent]
        # Largest outer blob sets the scale — only use inner holes whose parent
        # is big enough to belong to the dominant gate, not a background gate.
        max_outer_area = max(
            (cv2.contourArea(cnts[i]) for i, hr in enumerate(h) if hr[3] < 0),
            default=0,
        )
        min_parent_area = max_outer_area * GATE_HOLE_MIN_PARENT_FRACTION
        best_hole_area = GATE_INNER_MIN_AREA_PX - 1
        for i, h_row in enumerate(h):
            if h_row[3] < 0:
                continue  # outer contour — skip
            if cv2.contourArea(cnts[h_row[3]]) < min_parent_area:
                continue  # parent is a background gate — skip
            area = cv2.contourArea(cnts[i])
            if area > best_hole_area:
                best_hole_area = area
                _, iy, _, ih = cv2.boundingRect(cnts[i])
                hole_bottom  = iy + ih
                # Adaptive margin: bar thickness scales with inner hole height.
                # bar_thickness_px ≈ ih × (610/1500).  At close range this is
                # 40–80 px; the old fixed GATE_FRAME_MARGIN_PX (5 px) was far
                # too small and left the gate's bottom bar in the spotlight region.
                margin  = max(GATE_FRAME_MARGIN_PX, int(ih * _GATE_BAR_TO_INNER_RATIO))
                cutoff  = hole_bottom + margin
    # Guard: skip only if the hole is so far down that the cutoff would exceed
    # the frame (the spotlight is already off-screen at that range).
    # With the adaptive margin, the cutoff now correctly targets the outer gate
    # bottom, so the old 0.70 guard (which skipped trimming for close gates) is
    # raised to 0.90 — at that range, cutoff > frame_height is harmless anyway.
    # Apply the trim only to a copy so the untrimmed open_mask can be used for
    # ring validation (where the bottom bar must be intact to form a closed ring).
    mask_open_trimmed = mask_open.copy()
    if cutoff > 0 and hole_bottom < int(mask_open.shape[0] * GATE_HOLE_MAX_Y_FRACTION):
        mask_open_trimmed[cutoff:, :] = 0

    mask_final = cv2.morphologyEx(mask_open_trimmed, cv2.MORPH_CLOSE, close_kernel)
    return mask_final, mask_open  # open_mask is pre-trim: bottom bar intact for ring check


def _centroid_in_any_bbox(cnt, bboxes: list) -> bool:
    """True if the contour's centroid falls inside any of the given bounding rects."""
    M = cv2.moments(cnt)
    if M['m00'] == 0 or not bboxes:
        return False
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    for x, y, w, h in bboxes:
        if x <= cx <= x + w and y <= cy <= y + h:
            return True
    return False


def _aspect_ratio(cnt) -> float:
    _, _, w, h = cv2.boundingRect(cnt)
    return float(w / h) if h else float('nan')


def _roughly_square(cnt) -> bool:
    return 0.3 < _aspect_ratio(cnt) < 3.0


def _n_border_corners(corners: np.ndarray, w: int, h: int, margin: float = BORDER_MARGIN_PX) -> int:
    x, y = corners[:, 0], corners[:, 1]
    on_border = (x <= margin) | (x >= w - 1 - margin) | (y <= margin) | (y >= h - 1 - margin)
    return int(np.sum(on_border))


def _four_corners(cnt) -> tuple[np.ndarray | None, dict]:
    eps_factors = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    diag = {'best_hull_pts': int(len(hull.reshape(-1, 2))),
            'best_poly_pts': None, 'four_corner_eps': None}

    for i, eps_factor in enumerate(eps_factors):
        poly = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if i == 0:
            diag['best_poly_pts'] = int(len(poly))
        if len(poly) == 4:
            diag['four_corner_eps'] = eps_factor
            if VERBOSE:
                print(f'[detector] 4-corner poly found at eps={eps_factor:.2f}')
            return poly.reshape(4, 2).astype(np.float32), diag

    if VERBOSE:
        print(f'[detector] could not reduce contour to 4 corners '
              f'(hull has {diag["best_hull_pts"]} pts)')
    return None, diag


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s    = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmax(diff)]
    ordered[3] = pts[np.argmin(diff)]
    return ordered
