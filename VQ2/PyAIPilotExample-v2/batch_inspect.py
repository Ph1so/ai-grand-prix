"""
Batch version of inspect_frame.py — saves annotated frames + intermediate
mask stages to disk so the pipeline can be audited without a display.

Usage:
    python batch_inspect.py logs/run_20260629_201358/frames/ [out_dir]

Outputs to out_dir (default: batch_inspect_out/):
    annotated/<frame>.jpg   — original + detected corners drawn
    masks/<frame>.jpg       — final orange mask used for contour detection
    summary.txt             — per-frame detect_result + key diag fields
"""

import sys
import pathlib
import cv2
import numpy as np

# Patch gate_detector to also expose the mask for inspection.
import gate_detector as gd

_orig_orange_mask = gd._orange_mask_from_hsv
_last_mask = {}

def _patched_orange_mask(hsv):
    final_mask, open_mask = _orig_orange_mask(hsv)
    _last_mask['mask']      = final_mask.copy()
    _last_mask['open_mask'] = open_mask.copy()
    return final_mask, open_mask

gd._orange_mask_from_hsv = _patched_orange_mask

from gate_detector import detect_gate, draw_detection


def process_frame(path: pathlib.Path, ann_dir: pathlib.Path, mask_dir: pathlib.Path) -> dict:
    frame = cv2.imread(str(path))
    if frame is None:
        return {'frame': path.name, 'result': 'unreadable'}

    diag = {}
    _last_mask.clear()
    corners = detect_gate(frame, diag=diag)

    # Save annotated image
    if corners is not None:
        ann = draw_detection(frame, corners)
        # Draw corner coords as text
        labels = ['TL', 'TR', 'BR', 'BL']
        for pt, label in zip(corners.astype(int), labels):
            cv2.putText(ann, f"{label}({pt[0]},{pt[1]})", tuple(pt + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        ann = frame.copy()
        reason = diag.get('detect_result', 'unknown')
        cv2.putText(ann, f"NO GATE: {reason}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(ann_dir / path.name), ann)

    # Save final mask and open_mask side-by-side for comparison
    if 'mask' in _last_mask:
        final_bgr = cv2.cvtColor(_last_mask['mask'], cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(_last_mask['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_bgr, cnts, -1, (0, 255, 0), 1)

        if 'open_mask' in _last_mask:
            open_bgr = cv2.cvtColor(_last_mask['open_mask'], cv2.COLOR_GRAY2BGR)
            # Label each half
            cv2.putText(final_bgr, 'FINAL', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
            cv2.putText(open_bgr,  'OPEN',  (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
            combined = np.hstack([open_bgr, final_bgr])
            cv2.imwrite(str(mask_dir / path.name), combined)
        else:
            cv2.imwrite(str(mask_dir / path.name), final_bgr)

    row = {
        'frame': path.stem,
        'result': diag.get('detect_result', 'none'),
        'best_area': diag.get('best_area', ''),
        'best_aspect': round(diag.get('best_aspect', float('nan')), 3) if diag.get('best_aspect') else '',
        'n_clipped': diag.get('n_clipped_corners', ''),
        'eps': diag.get('four_corner_eps', ''),
        'poly_pts': diag.get('best_poly_pts', ''),
        'hull_pts': diag.get('best_hull_pts', ''),
        'n_cands': diag.get('n_candidates', ''),
        'n_ring':  diag.get('n_ring', ''),
        'corners': corners.astype(int).tolist() if corners is not None else None,
    }
    return row


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path('.')
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path('batch_inspect_out')

    ann_dir  = out / 'annotated'
    mask_dir = out / 'masks'
    ann_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(src.glob('*.jpg'))
    print(f"Processing {len(frames)} frames -> {out}/")

    rows = []
    for f in frames:
        r = process_frame(f, ann_dir, mask_dir)
        rows.append(r)
        status = r['result']
        area   = f"area={r['best_area']:.0f}" if isinstance(r['best_area'], float) else ''
        clip   = f"clip={r['n_clipped']}" if r['n_clipped'] != '' else ''
        eps    = f"eps={r['eps']}" if r['eps'] else ''
        print(f"  {r['frame']}  {status:12}  {area:12}  {clip}  {eps}")

    # Write summary
    ok     = [r for r in rows if r['result'] == 'ok']
    clipped= [r for r in rows if r['result'] == 'partial']
    missed = [r for r in rows if r['result'] not in ('ok', 'partial', 'unreadable')]

    summary_lines = [
        f"Total frames : {len(rows)}",
        f"  ok         : {len(ok)}",
        f"  partial    : {len(clipped)}  (clipped >= 3 corners)",
        f"  missed     : {len(missed)}",
        "",
        f"{'Frame':<20} {'Result':<14} {'Area':>8} {'Aspect':>8} {'Clipped':>8} {'Eps':>6} {'PolyPts':>8} {'HullPts':>8}",
        "-" * 90,
    ]
    for r in rows:
        area_s   = f"{r['best_area']:.0f}" if isinstance(r['best_area'], float) else '-'
        aspect_s = f"{r['best_aspect']}" if r['best_aspect'] != '' else '-'
        clip_s   = str(r['n_clipped']) if r['n_clipped'] != '' else '-'
        eps_s    = str(r['eps'])       if r['eps']       != '' else '-'
        poly_s   = str(r['poly_pts'])  if r['poly_pts']  != '' else '-'
        hull_s   = str(r['hull_pts'])  if r['hull_pts']  != '' else '-'
        summary_lines.append(
            f"{r['frame']:<20} {r['result']:<14} {area_s:>8} {aspect_s:>8} {clip_s:>8} {eps_s:>6} {poly_s:>8} {hull_s:>8}"
        )

    summary_path = out / 'summary.txt'
    summary_path.write_text('\n'.join(summary_lines))
    print(f"\nSummary -> {summary_path}")
    print('\n'.join(summary_lines[:8]))


if __name__ == '__main__':
    main()
