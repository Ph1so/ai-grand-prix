"""
Visualize YOLO-pose auto-labels (label_run.py output) on their source frames:
draws each instance's bbox + 4 corner keypoints (TL/TR/BR/BL, color-coded,
connected in order) so you can sanity-check the auto-labeler's projections
against the actual gate in the image.

Filled dot = visible keypoint (v=2). Hollow dot = clipped / out-of-frame (v=0).

Usage:
    python inspect_labels.py logs/run_20260607_xxxxxx           # sample of labeled frames
    python inspect_labels.py logs/run_20260607_xxxxxx --n 20    # sample 20
    python inspect_labels.py logs/run_20260607_xxxxxx --all     # include 0-instance frames
    python inspect_labels.py logs/run_20260607_xxxxxx --play    # FPV-style playback

    # Inspect the assembled dataset directly:
    python inspect_labels.py --images dataset/images/train --labels dataset/labels/train
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate_geometry import IMG_H, IMG_W

CORNER_ORDER = ['TL', 'TR', 'BR', 'BL']
CORNER_COLORS = {  # BGR
    'TL': (0, 0, 255),
    'TR': (0, 255, 0),
    'BR': (255, 0, 0),
    'BL': (0, 255, 255),
}


def draw_label_line(img: np.ndarray, line: str) -> None:
    parts = line.split()
    cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
    x0, y0 = int(round((cx - w / 2) * IMG_W)), int(round((cy - h / 2) * IMG_H))
    x1, y1 = int(round((cx + w / 2) * IMG_W)), int(round((cy + h / 2) * IMG_H))
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 1)

    kp = parts[5:]
    pts = []
    for i, name in enumerate(CORNER_ORDER):
        kx = float(kp[3 * i]) * IMG_W
        ky = float(kp[3 * i + 1]) * IMG_H
        v = int(kp[3 * i + 2])
        pt = (int(round(kx)), int(round(ky)))
        pts.append(pt)
        color = CORNER_COLORS[name]
        cv2.circle(img, pt, 4, color, -1 if v == 2 else 1)
        cv2.putText(img, name, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=(255, 255, 255), thickness=1)


def _label_lines(labels_dir: str, stem: str) -> list[str]:
    label_path = os.path.join(labels_dir, f'{stem}.txt')
    if not os.path.exists(label_path):
        return []
    with open(label_path) as f:
        return [l.strip() for l in f if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', nargs='?', help='run directory containing frames/ and labels/')
    parser.add_argument('--images', help='explicit images directory (overrides <run_dir>/frames)')
    parser.add_argument('--labels', help='explicit labels directory (overrides <run_dir>/labels)')
    parser.add_argument('--out', help='output directory for overlay images '
                                       '(default: <images_dir>/../label_overlays)')
    parser.add_argument('--n', type=int, default=12, help='max frames to render (default 12)')
    parser.add_argument('--all', action='store_true', help='include frames with 0 instances')
    parser.add_argument('--play', action='store_true', help='FPV-style OpenCV playback instead of saving files')
    parser.add_argument('--fps', type=float, default=5.0)
    args = parser.parse_args()

    if args.images and args.labels:
        images_dir, labels_dir = args.images, args.labels
    elif args.run_dir:
        images_dir = os.path.join(args.run_dir, 'frames')
        labels_dir = os.path.join(args.run_dir, 'labels')
    else:
        parser.error('pass a run_dir or both --images and --labels')

    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    if not image_paths:
        print(f'No images found in {images_dir}')
        sys.exit(1)

    entries = []  # (image_path, lines)
    for p in image_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        lines = _label_lines(labels_dir, stem)
        if lines or args.all:
            entries.append((p, lines))

    if not entries:
        print(f'No labeled frames found in {labels_dir} (use --all to include 0-instance frames)')
        sys.exit(1)

    if not args.play and len(entries) > args.n:
        idx = np.linspace(0, len(entries) - 1, args.n, dtype=int)
        entries = [entries[i] for i in idx]

    n_with_instances = sum(1 for _, lines in entries if lines)
    print(f'[inspect_labels] {len(entries)} frame(s) selected '
          f'({n_with_instances} with >=1 instance, {sum(len(l) for _, l in entries)} instances total)')

    if args.play:
        _play(entries, fps=args.fps)
        return

    out_dir = args.out or os.path.join(os.path.dirname(images_dir.rstrip('\\/')), 'label_overlays')
    os.makedirs(out_dir, exist_ok=True)
    for p, lines in entries:
        img = cv2.imread(p)
        if img is None:
            print(f'[skip] could not read {p}')
            continue
        for line in lines:
            draw_label_line(img, line)
        stem = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f'{stem}_labels.jpg')
        cv2.imwrite(out_path, img)
        print(f'  {stem}: {len(lines)} instance(s) -> {out_path}')


def _play(entries: list[tuple[str, list[str]]], fps: float) -> None:
    delay_ms = max(1, round(1000 / max(fps, 0.1)))
    win = 'label playback  —  space=pause  a/d=step  c=capture  q=quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx, paused = 0, False
    img = None
    while True:
        p, lines = entries[idx]
        img = cv2.imread(p)
        if img is not None:
            for line in lines:
                draw_label_line(img, line)
            tag = f'[{idx + 1}/{len(entries)}] {os.path.basename(p)}  {len(lines)} instance(s)'
            cv2.putText(img, tag, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, img)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            continue
        elif key == ord('d'):
            idx = min(idx + 1, len(entries) - 1)
            paused = True
            continue
        elif key == ord('a'):
            idx = max(idx - 1, 0)
            paused = True
            continue
        elif key == ord('c'):                       # capture exactly what's on screen
            if img is not None:
                stem = os.path.splitext(os.path.basename(p))[0]
                out_path = os.path.join(os.path.dirname(p), f'{stem}_capture.jpg')
                cv2.imwrite(out_path, img)
                print(f'[capture] saved -> {out_path}')
            continue

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if not paused:
            idx = (idx + 1) % len(entries)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
