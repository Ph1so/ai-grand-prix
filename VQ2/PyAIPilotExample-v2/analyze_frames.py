"""
Headless frame analysis — saves a 6-panel preprocessing pipeline view per frame.

Panels (2 rows × 3 cols):
  [raw frame + detection]  [1. raw HSV orange mask]      [2. after blue subtract]
  [3. after MORPH_OPEN]    [4. after MORPH_CLOSE/final]  [5. contours on final mask]

Usage:
    python analyze_frames.py logs/run_20260629_201358/frames/ --out analysis_out/
    python analyze_frames.py logs/run_20260629_201358/frames/frame_000110.jpg --out analysis_out/
"""

import sys
import pathlib
import cv2
import numpy as np
from gate_detector import (
    detect_gate, draw_detection,
    _HSV_LO1, _HSV_HI1, _HSV_LO2, _HSV_HI2,
    _BLUE_LO, _BLUE_HI,
    FLOOR_MASK_FRACTION, MIN_GATE_AREA_PX,
    GATE_INNER_MIN_AREA_PX, GATE_FRAME_MARGIN_PX,
    GATE_HOLE_MIN_PARENT_FRACTION, GATE_HOLE_MAX_Y_FRACTION,
)


def _build_pipeline_stages(frame: np.ndarray):
    """Return each intermediate preprocessing mask as a BGR image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Stage 1 — raw orange HSV match (no morphology, no blue subtract)
    s1 = cv2.bitwise_or(
        cv2.inRange(hsv, _HSV_LO1, _HSV_HI1),
        cv2.inRange(hsv, _HSV_LO2, _HSV_HI2),
    )

    # Stage 2 — after floor mask + blue ribbon subtraction
    s2 = s1.copy()
    if FLOOR_MASK_FRACTION > 0.0:
        cutoff = int(hsv.shape[0] * (1.0 - FLOOR_MASK_FRACTION))
        s2[cutoff:, :] = 0
    blue_mask   = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
    blue_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask   = cv2.dilate(blue_mask, blue_kernel, iterations=1)
    s2          = cv2.bitwise_and(s2, cv2.bitwise_not(blue_mask))

    # Stage 3 — after MORPH_OPEN (kills small noise blobs)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    s3 = cv2.morphologyEx(s2, cv2.MORPH_OPEN, open_kernel)

    # Stage 3b — after inner-hole spotlight trimming (search on MORPH_CLOSE result)
    close_kernel_tmp = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    s3_close_tmp = cv2.morphologyEx(s3, cv2.MORPH_CLOSE, close_kernel_tmp)
    s3b = s3.copy()
    cnts_tree, hier_tree = cv2.findContours(s3_close_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier_tree is not None and len(cnts_tree) > 0:
        ht = hier_tree[0]
        max_outer_area = max(
            (cv2.contourArea(cnts_tree[i]) for i, hr in enumerate(ht) if hr[3] < 0),
            default=0,
        )
        min_parent_area = max_outer_area * GATE_HOLE_MIN_PARENT_FRACTION
        best_hole_area = GATE_INNER_MIN_AREA_PX - 1
        cutoff_local   = -1
        hole_bottom    = -1
        for i, h_row in enumerate(ht):
            if h_row[3] < 0:
                continue
            if cv2.contourArea(cnts_tree[h_row[3]]) < min_parent_area:
                continue
            area = cv2.contourArea(cnts_tree[i])
            if area > best_hole_area:
                best_hole_area = area
                _, iy, _, ih = cv2.boundingRect(cnts_tree[i])
                hole_bottom  = iy + ih
                cutoff_local = hole_bottom + GATE_FRAME_MARGIN_PX
        if cutoff_local > 0 and hole_bottom < int(s3.shape[0] * GATE_HOLE_MAX_Y_FRACTION):
            s3b[cutoff_local:, :] = 0

    # Stage 4 — after MORPH_CLOSE (final mask, reconnects gate frame)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    s4 = cv2.morphologyEx(s3b, cv2.MORPH_CLOSE, close_kernel)

    # Stage 5 — contours drawn on final mask
    contours, _ = cv2.findContours(s4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s5_bgr = cv2.cvtColor(s4, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        color = (0, 255, 0) if area >= MIN_GATE_AREA_PX else (0, 0, 180)
        cv2.drawContours(s5_bgr, [cnt], -1, color, 1)
        cx, cy = map(int, cv2.boundingRect(cnt)[:2])
        cv2.putText(s5_bgr, f"{int(area)}", (cx, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def to_bgr_overlay(mask, color_bgr):
        """Overlay a binary mask on the raw frame with a tinted colour."""
        col = np.zeros_like(frame)
        col[:] = color_bgr
        out = frame.copy()
        nonzero = mask > 0
        out[nonzero] = cv2.addWeighted(frame, 0.35, col, 0.65, 0)[nonzero]
        return out

    return (
        to_bgr_overlay(s1,  (0, 100, 255)),
        to_bgr_overlay(s3,  (0, 180, 255)),
        to_bgr_overlay(s3b, (0, 220, 100)),  # green tint = after trimming
        to_bgr_overlay(s4,  (0,  60, 255)),
        s5_bgr,
    )


def analyze(path: pathlib.Path, out_dir: pathlib.Path):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read {path}")
        return

    diag    = {}
    corners = detect_gate(frame, diag=diag)

    if corners is not None:
        annotated = draw_detection(frame, corners)
        status    = f"DETECTED ({diag.get('n_clipped_corners', 0)} clipped)"
    else:
        annotated = frame.copy()
        status    = f"MISS — {diag.get('detect_result', '?')}"

    s1, s3, s3b, s4, s5 = _build_pipeline_stages(frame)

    label_h = 20
    def label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], label_h), (0, 0, 0), -1)
        cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    row1 = np.hstack([
        label(annotated, f"{path.name}  |  {status}"),
        label(s1,        "1. raw orange mask (HSV only)"),
        label(s3,        "2. after MORPH_OPEN 3x3"),
    ])
    row2 = np.hstack([
        label(s3b,       f"3. after inner-hole trim  (margin={GATE_FRAME_MARGIN_PX}px)"),
        label(s4,        f"4. after MORPH_CLOSE 9x9 (final)  cands={diag.get('n_candidates','?')}"),
        label(s5,        "5. contours  (green=candidate  red=below min area)"),
    ])
    grid = np.vstack([row1, row2])

    out_path = out_dir / f"{path.stem}_analysis.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{path.name:30s}  {status:35s}  diag={diag}")


def main():
    args    = sys.argv[1:]
    out_dir = pathlib.Path("analysis_out")
    if "--out" in args:
        i       = args.index("--out")
        out_dir = pathlib.Path(args[i + 1])
        args    = args[:i] + args[i + 2:]

    if not args:
        print("Usage: python analyze_frames.py <folder_or_frame.jpg> [--out <dir>]")
        sys.exit(1)

    target = pathlib.Path(args[0])
    frames = sorted(target.glob("*.jpg")) if target.is_dir() else [target]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving pipeline views to {out_dir}/\n")

    for f in frames:
        analyze(f, out_dir)

    print(f"\nDone — {len(frames)} frame(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
