"""
Copy-paste augmentation: synthesize two-gate-overlap ("merge") training
examples.

General augmentation (lighting/scale/rotation/mosaic/HSV-jitter/flips) is
left to Ultralytics' built-in training-time augmentation (see
train_yolo.py). This script targets one specific underrepresented case:
two gates overlapping/co-linear in the frame (e.g. frame_000192-style
merges), which real flight logs rarely produce in quantity.

Pipeline:
  1. Scan dataset/labels/train for "clean" single-instance, high-orange-
     coverage examples -- these are donor gates.
  2. Extract a feathered cutout (HSV-orange mask, Gaussian-blurred for soft
     edges) of each donor's bbox + its 4 keypoints in local crop coords.
  3. For N composites: pick a random base train image, pick a random donor,
     paste the (randomly scaled) donor near/overlapping one of the base
     image's existing gate instances, alpha-blend, recompute keypoints in
     composite coordinates, and append as an extra label line.

Synthetic composites are written only to dataset/images/train and
dataset/labels/train -- never to val.

Usage:
    python augment_copypaste.py -n 200
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_detector import _orange_mask_from_hsv
from gate_geometry import IMG_H, IMG_W

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PyAIPilotExample/
DATASET_DIR = os.path.join(REPO_ROOT, 'dataset')

DONOR_MARGIN = 4  # px of context kept around a donor's bbox before feathering


def _bbox_px(line: str) -> tuple[float, float, float, float]:
    cx, cy, w, h = (float(x) for x in line.split()[1:5])
    return ((cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H,
            (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H)


def _keypoints_px(line: str) -> list[tuple[float, float, int]]:
    parts = line.split()[5:]
    return [(float(parts[i]) * IMG_W, float(parts[i + 1]) * IMG_H, int(parts[i + 2]))
            for i in range(0, len(parts), 3)]


def build_donor_pool(images_dir: str, labels_dir: str, min_coverage: float) -> list[dict]:
    donors = []
    for label_path in sorted(glob.glob(os.path.join(labels_dir, '*.txt'))):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        if stem.startswith('synth_'):
            continue
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) != 1:
            continue

        img = cv2.imread(os.path.join(images_dir, f'{stem}.jpg'))
        if img is None:
            continue

        x0, y0, x1, y1 = _bbox_px(lines[0])
        ix0 = max(int(np.floor(x0)) - DONOR_MARGIN, 0)
        iy0 = max(int(np.floor(y0)) - DONOR_MARGIN, 0)
        ix1 = min(int(np.ceil(x1)) + DONOR_MARGIN, IMG_W)
        iy1 = min(int(np.ceil(y1)) + DONOR_MARGIN, IMG_H)
        if ix1 - ix0 < 8 or iy1 - iy0 < 8:
            continue

        crop = img[iy0:iy1, ix0:ix1]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = _orange_mask_from_hsv(hsv)
        coverage = float((mask > 0).sum()) / mask.size
        if coverage < min_coverage:
            continue

        alpha = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0) / 255.0
        local_kps = [(kx - ix0, ky - iy0, v) for kx, ky, v in _keypoints_px(lines[0])]

        donors.append({
            'crop': crop, 'alpha': alpha, 'kps': local_kps,
            'w': ix1 - ix0, 'h': iy1 - iy0,
        })
    return donors


def paste_donor(base_img: np.ndarray, base_lines: list[str], donor: dict,
                 rng: random.Random) -> tuple[np.ndarray, str] | None:
    anchor = _bbox_px(rng.choice(base_lines))
    anchor_w, anchor_h = anchor[2] - anchor[0], anchor[3] - anchor[1]
    anchor_cx, anchor_cy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2

    scale = rng.uniform(0.6, 1.3)
    new_w = max(8, int(round(donor['w'] * scale)))
    new_h = max(8, int(round(donor['h'] * scale)))
    sx, sy = new_w / donor['w'], new_h / donor['h']

    crop_resized  = cv2.resize(donor['crop'],  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    alpha_resized = cv2.resize(donor['alpha'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    kps_resized   = [(kx * sx, ky * sy, v) for kx, ky, v in donor['kps']]

    # Place near (overlapping or adjacent to) the anchor instance.
    offset = rng.uniform(0.3, 1.0) * max(anchor_w, anchor_h)
    angle  = rng.uniform(0, 2 * np.pi)
    px = int(round(anchor_cx + offset * np.cos(angle) - new_w / 2))
    py = int(round(anchor_cy + offset * np.sin(angle) - new_h / 2))
    px = int(np.clip(px, -new_w * 0.3, IMG_W - new_w * 0.7))
    py = int(np.clip(py, -new_h * 0.3, IMG_H - new_h * 0.7))

    cx0, cy0 = max(px, 0), max(py, 0)
    cx1, cy1 = min(px + new_w, IMG_W), min(py + new_h, IMG_H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    sx0, sy0 = cx0 - px, cy0 - py
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)

    composite = base_img.copy()
    region = composite[cy0:cy1, cx0:cx1].astype(np.float32)
    patch  = crop_resized[sy0:sy1, sx0:sx1].astype(np.float32)
    a      = alpha_resized[sy0:sy1, sx0:sx1, None]
    composite[cy0:cy1, cx0:cx1] = (region * (1 - a) + patch * a).astype(np.uint8)

    new_kps = []
    for kx, ky, v in kps_resized:
        gx, gy = px + kx, py + ky
        visible = v == 2 and 0 <= gx < IMG_W and 0 <= gy < IMG_H
        gx_c = min(max(gx, 0.0), IMG_W - 1)
        gy_c = min(max(gy, 0.0), IMG_H - 1)
        new_kps.append((gx_c, gy_c, 2 if visible else 0))

    xs = [k[0] for k in new_kps]
    ys = [k[1] for k in new_kps]
    nx0, nx1, ny0, ny1 = min(xs), max(xs), min(ys), max(ys)
    if nx1 - nx0 < 1 or ny1 - ny0 < 1:
        return None

    bcx, bcy = (nx0 + nx1) / 2 / IMG_W, (ny0 + ny1) / 2 / IMG_H
    bw, bh   = (nx1 - nx0) / IMG_W, (ny1 - ny0) / IMG_H

    kp_fields = []
    for gx, gy, v in new_kps:
        kp_fields += [f'{gx / IMG_W:.6f}', f'{gy / IMG_H:.6f}', str(v)]

    new_line = ' '.join(['0', f'{bcx:.6f}', f'{bcy:.6f}', f'{bw:.6f}', f'{bh:.6f}', *kp_fields])
    return composite, new_line


def augment(n: int, min_coverage: float, seed: int) -> None:
    images_dir = os.path.join(DATASET_DIR, 'images', 'train')
    labels_dir = os.path.join(DATASET_DIR, 'labels', 'train')

    donors = build_donor_pool(images_dir, labels_dir, min_coverage)
    if not donors:
        print(f'[augment_copypaste] no donor cutouts found '
              f'(need clean single-instance train examples with >= {min_coverage:.0%} orange coverage)')
        sys.exit(1)

    bases = []
    for label_path in sorted(glob.glob(os.path.join(labels_dir, '*.txt'))):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        if stem.startswith('synth_'):
            continue
        with open(label_path) as f:
            if any(l.strip() for l in f):
                bases.append(stem)
    if not bases:
        print('[augment_copypaste] no base train images with at least one instance found')
        sys.exit(1)

    rng = random.Random(seed)
    n_written, attempts = 0, 0
    while n_written < n and attempts < n * 10:
        attempts += 1
        base_stem = rng.choice(bases)
        base_img = cv2.imread(os.path.join(images_dir, f'{base_stem}.jpg'))
        with open(os.path.join(labels_dir, f'{base_stem}.txt')) as f:
            base_lines = [l.strip() for l in f if l.strip()]

        result = paste_donor(base_img, base_lines, rng.choice(donors), rng)
        if result is None:
            continue
        composite, new_line = result

        out_stem = f'synth_{n_written:05d}'
        cv2.imwrite(os.path.join(images_dir, f'{out_stem}.jpg'), composite)
        with open(os.path.join(labels_dir, f'{out_stem}.txt'), 'w') as f:
            f.write('\n'.join(base_lines + [new_line]) + '\n')
        n_written += 1

    print(f'[augment_copypaste] donor pool: {len(donors)}, base images: {len(bases)}')
    print(f'[augment_copypaste] wrote {n_written}/{n} synthetic composite(s) to '
          f'{images_dir} and {labels_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--num', type=int, default=200,
                         help='Number of synthetic composites to generate (default 200)')
    parser.add_argument('--min-coverage', type=float, default=0.15,
                         help='Min HSV-orange coverage for a donor cutout bbox (default 0.15)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    augment(args.num, args.min_coverage, args.seed)


if __name__ == '__main__':
    main()
