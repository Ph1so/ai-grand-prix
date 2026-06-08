"""
Gate detector using orange HSV colour segmentation.

detect_gate(frame) -> np.ndarray | None
    Returns 4 corner points [TL, TR, BR, BL] of the largest (= closest)
    orange gate in the frame, or None if no gate is found.

Corner ordering assumes the outer gate boundary (2720 mm × 2720 mm) is
detected.  Callers should use GATE_OUTER_HALF = 1.36 m as their solvePnP
object-point half-size.
"""

import cv2
import numpy as np

# ── Tunable constants ─────────────────────────────────────────────────────────

# HSV range for the gate's orange colour.
# OpenCV H: 0–179, S: 0–255, V: 0–255.
# Two bands cover orange (H 0–25) and the red wrap-around (H 160–179).
_HSV_LO1 = np.array([  0, 120,  60], dtype=np.uint8)
_HSV_HI1 = np.array([ 25, 255, 255], dtype=np.uint8)
_HSV_LO2 = np.array([160, 120,  60], dtype=np.uint8)
_HSV_HI2 = np.array([179, 255, 255], dtype=np.uint8)

# Minimum contour area in pixels to be considered a gate candidate.
MIN_GATE_AREA_PX: int = 200

# approxPolyDP epsilon as a fraction of contour perimeter.
_POLY_EPS: float = 0.05

# Gate outer half-dimension (metres) — used by pose_estimator object points.
GATE_OUTER_HALF: float = 1.36  # 2720 mm / 2

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
           See PERC.md A5/A7. Costs nothing extra when left None.
    """
    if diag is not None:
        diag.clear()

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _orange_mask_from_hsv(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    passed  = [a >= MIN_GATE_AREA_PX and _roughly_square(c) for a, c in zip(areas, contours)]
    li      = int(np.argmax(areas))

    if diag is not None:
        # The single largest blob, regardless of filter outcome -- answers
        # "was a plausible gate actually there, and did the A5 filters reject it?"
        diag['largest_area']    = areas[li]
        diag['largest_aspect']  = aspects[li]
        diag['largest_passed']  = bool(passed[li])
        diag['n_candidates']    = int(sum(passed))

    candidates = [c for c, p in zip(contours, passed) if p]
    if not candidates:
        if diag is not None:
            diag['detect_result'] = 'no_candidates'
        if VERBOSE:
            print(f'[detector] {len(contours)} contour(s) found but none passed filters '
                  f'(largest area={areas[li]:.0f} px², min={MIN_GATE_AREA_PX})')
        return None

    best = max(candidates, key=cv2.contourArea)
    corners, poly_diag = _four_corners(best)

    if diag is not None:
        diag['best_area']   = float(cv2.contourArea(best))
        diag['best_aspect'] = _aspect_ratio(best)
        diag.update(poly_diag)

    if corners is None:
        if diag is not None:
            diag['detect_result'] = 'no_4corner'
        if VERBOSE:
            print('[detector] could not reduce best contour to 4 corners')
        return None

    ordered = _order_corners(corners)

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

def _orange_mask_from_hsv(hsv: np.ndarray) -> np.ndarray:
    mask1 = cv2.inRange(hsv, _HSV_LO1, _HSV_HI1)
    mask2 = cv2.inRange(hsv, _HSV_LO2, _HSV_HI2)
    mask  = cv2.bitwise_or(mask1, mask2)
    # Close small gaps within the gate frame border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _aspect_ratio(cnt) -> float:
    """Bounding-rect width/height ratio. NaN for degenerate (zero-height) contours."""
    _, _, w, h = cv2.boundingRect(cnt)
    return float(w / h) if h else float('nan')


def _roughly_square(cnt) -> bool:
    """True if the contour's bounding rect has aspect ratio 0.3–3.0.
    (NaN comparisons are always False, so degenerate contours correctly fail.)"""
    return 0.3 < _aspect_ratio(cnt) < 3.0


def _four_corners(cnt) -> tuple[np.ndarray | None, dict]:
    """
    Reduce a contour to exactly 4 corners via approxPolyDP.
    Tries a range of epsilon values; returns (None, diag) if none yield 4 points.
    The bounding-rect fallback is intentionally omitted — it produces
    inflated corners when the gate frame has internal decorations, which
    corrupts the solvePnP result.

    Returns (corners, diag) where corners is float32 (4, 2) or None, and
    diag carries A7 instrumentation (PERC.md): how complex was the raw
    shape (`best_hull_pts`, `best_poly_pts` at the finest epsilon — the
    least-simplified read on the contour's true point count) and which
    epsilon (if any) finally converged to exactly 4 (`four_corner_eps`).
    """
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
    """
    Order 4 points as [top-left, top-right, bottom-right, bottom-left].
    Uses coordinate sums / differences — robust to moderate perspective warp.
    """
    s    = pts.sum(axis=1)          # x + y
    diff = pts[:, 0] - pts[:, 1]   # x - y

    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]     # TL: smallest x+y
    ordered[2] = pts[np.argmax(s)]     # BR: largest  x+y
    ordered[1] = pts[np.argmax(diff)]  # TR: largest  x-y  (large x, small y)
    ordered[3] = pts[np.argmin(diff)]  # BL: smallest x-y  (small x, large y)
    return ordered
