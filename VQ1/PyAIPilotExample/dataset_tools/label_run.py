"""
Auto-label a run's saved frame snapshots with YOLO-pose gate annotations.

For each snapshot frame, projects every track gate (from `*_gates.json`) into
image space using the drone pose recorded at capture time
(`snapshots_*.csv`, see gate_verifier.py instrumentation). A gate is kept as
a label instance if its projected bbox falls (at least partly) inside the
image AND the corresponding image region actually contains orange pixels
(occlusion heuristic — no depth buffer is available, so "geometry says a gate
is here but nothing orange renders there" means it's hidden behind terrain).

Writes one YOLO-pose label per frame to `<run_dir>/labels/frame_NNNNNN.txt`:
    0 cx cy w h  kp1x kp1y v1  kp2x kp2y v2  kp3x kp3y v3  kp4x kp4y v4
all spatial values normalized to [0,1]. Keypoints (corners) that fall outside
the image are clipped to the image bounds and marked v=0 (not visible).
Frames with multiple visible gates get multiple lines — this is what captures
co-linear gate-overlap ("merge") cases as multi-instance ground truth.

Usage:
    python label_run.py <run_dir>
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_detector import _orange_mask_from_hsv
from gate_geometry import IMG_W, IMG_H, gate_depth, project_gate
from planner import load_gates

# Minimum fraction of a gate's clipped bbox that must be orange pixels for
# the gate to be considered visible (vs. occluded by terrain/geometry).
MIN_ORANGE_COVERAGE = 0.02

# Minimum clipped-bbox size (px) for a gate to be considered "in frame".
MIN_BBOX_PX = 1.0

# If this much of a (farther) gate's projected bbox is covered by the
# projected bbox of a nearer gate, treat it as occluded by that nearer gate
# and drop it. A close gate fills much of the frame with its own orange,
# which would otherwise pass the orange-coverage check for any gate
# projected "through" it.
OCCLUSION_OVERLAP = 0.8


def _find_one(run_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No '{pattern}' found in {run_dir}")
    return matches[0]


def label_run(run_dir: str) -> None:
    snapshots_path = _find_one(run_dir, 'snapshots_*.csv')
    gates_path     = _find_one(run_dir, 'flight_log_*_gates.json')
    frames_dir     = os.path.join(run_dir, 'frames')
    labels_dir     = os.path.join(run_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    gates = load_gates(gates_path)

    stats = defaultdict(int)
    n_frames = 0

    with open(snapshots_path, newline='') as f:
        for row in csv.DictReader(f):
            frame_id  = int(row['frame_id'])
            drone_pos = np.array([float(row['drone_x']), float(row['drone_y']), float(row['drone_z'])])
            roll, pitch, yaw = float(row['roll']), float(row['pitch']), float(row['yaw'])

            img_path = os.path.join(frames_dir, f'frame_{frame_id:06d}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            n_frames += 1

            hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            orange_mask = _orange_mask_from_hsv(hsv)

            # Project every gate, keep those that land (at least partly) in
            # frame, and sort nearest-first so closer gates get first claim
            # on the orange-coverage check below.
            candidates = []
            for gate in gates:
                corners = project_gate(gate, drone_pos, roll, pitch, yaw)
                depth   = gate_depth(gate, drone_pos, roll, pitch, yaw)
                if corners is None or depth is None:
                    continue

                x_min, y_min = corners[:, 0].min(), corners[:, 1].min()
                x_max, y_max = corners[:, 0].max(), corners[:, 1].max()

                cx_min = max(x_min, 0.0)
                cy_min = max(y_min, 0.0)
                cx_max = min(x_max, float(IMG_W))
                cy_max = min(y_max, float(IMG_H))

                bw = cx_max - cx_min
                bh = cy_max - cy_min
                if bw < MIN_BBOX_PX or bh < MIN_BBOX_PX:
                    continue

                candidates.append((depth, corners, (cx_min, cy_min, cx_max, cy_max)))

            candidates.sort(key=lambda c: c[0])

            lines = []
            accepted_bboxes = []
            for depth, corners, (cx_min, cy_min, cx_max, cy_max) in candidates:
                bw = cx_max - cx_min
                bh = cy_max - cy_min
                own_area = bw * bh

                # Reject if a nearer gate's projected bbox already covers
                # most of this one -- its orange pixels would otherwise be
                # mistaken for this (occluded) gate's.
                occluded = 0.0
                for ax0, ay0, ax1, ay1 in accepted_bboxes:
                    ox0, oy0 = max(cx_min, ax0), max(cy_min, ay0)
                    ox1, oy1 = min(cx_max, ax1), min(cy_max, ay1)
                    if ox1 > ox0 and oy1 > oy0:
                        occluded += (ox1 - ox0) * (oy1 - oy0)
                if min(occluded, own_area) / own_area >= OCCLUSION_OVERLAP:
                    continue

                ix0, iy0 = int(round(cx_min)), int(round(cy_min))
                ix1, iy1 = int(round(cx_max)), int(round(cy_max))
                bbox_area_px = (ix1 - ix0) * (iy1 - iy0)
                if bbox_area_px <= 0:
                    continue

                coverage = float((orange_mask[iy0:iy1, ix0:ix1] > 0).sum()) / bbox_area_px
                if coverage < MIN_ORANGE_COVERAGE:
                    continue

                accepted_bboxes.append((cx_min, cy_min, cx_max, cy_max))

                bcx = (cx_min + cx_max) / 2.0 / IMG_W
                bcy = (cy_min + cy_max) / 2.0 / IMG_H
                bw_n = bw / IMG_W
                bh_n = bh / IMG_H

                kp_fields = []
                for px, py in corners:
                    visible = 0.0 <= px < IMG_W and 0.0 <= py < IMG_H
                    npx = min(max(float(px), 0.0), IMG_W - 1) / IMG_W
                    npy = min(max(float(py), 0.0), IMG_H - 1) / IMG_H
                    v   = 2 if visible else 0
                    kp_fields += [f'{npx:.6f}', f'{npy:.6f}', str(v)]

                lines.append(' '.join([
                    '0', f'{bcx:.6f}', f'{bcy:.6f}', f'{bw_n:.6f}', f'{bh_n:.6f}', *kp_fields,
                ]))

            label_path = os.path.join(labels_dir, f'frame_{frame_id:06d}.txt')
            with open(label_path, 'w') as out:
                if lines:
                    out.write('\n'.join(lines) + '\n')

            stats[len(lines)] += 1

    n0 = stats[0]
    n1 = stats[1]
    n2plus = sum(v for k, v in stats.items() if k >= 2)

    print(f'[label_run] {run_dir}')
    print(f'  frames processed : {n_frames}')
    print(f'  0 instances      : {n0}')
    print(f'  1 instance       : {n1}')
    print(f'  2+ instances     : {n2plus}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='Run directory containing frames/, snapshots_*.csv, *_gates.json')
    args = parser.parse_args()
    label_run(args.run_dir)


if __name__ == '__main__':
    main()
