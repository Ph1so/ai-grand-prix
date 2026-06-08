"""
Visualize gate_detector's 4-corner detection on saved frames.

For each image, draws the detected corner polygon (TL/TR/BR/BL, labelled)
over the frame next to the orange HSV mask the detector segmented from, and
prints the diagnostic dict (contour count, area/aspect filter pass-fail,
polygon-reduction epsilon) -- so a failed detection shows *why* it failed,
not just that it did.

Usage:
    python inspect_corners.py                           # latest saved frames (any run)
    python inspect_corners.py path/to/frame.jpg
    python inspect_corners.py "logs/run_*/frames/*.jpg"
    python inspect_corners.py frame1.jpg frame2.jpg --save
"""
import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate_detector import detect_gate, draw_detection, _orange_mask_from_hsv

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
_IMG_EXTS = ('*.jpg', '*.jpeg', '*.png')


def _resolve_paths(patterns: list[str]) -> list[str]:
    paths = []
    for p in patterns:
        if os.path.isdir(p):
            for ext in _IMG_EXTS:
                paths += glob.glob(os.path.join(p, ext))
        else:
            matches = glob.glob(p)
            paths += matches if matches else [p]
    return sorted(set(paths))


def _auto_find(n: int = 6) -> list[str]:
    candidates = []
    for ext in _IMG_EXTS:
        candidates += glob.glob(os.path.join(_LOG_DIR, 'run_*', '**', ext), recursive=True)
    candidates.sort(key=os.path.getmtime, reverse=True)
    if not candidates:
        print(f'No frames found under {_LOG_DIR}\\run_*\\. Pass an image path explicitly.')
        sys.exit(1)
    return candidates[:n]


def _inspect(path: str, save: bool) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f'[skip] could not read {path}')
        return

    diag = {}
    corners = detect_gate(img, diag=diag)
    mask = _orange_mask_from_hsv(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    overlay = draw_detection(img, corners) if corners is not None else img.copy()

    print(f'\n{path}')
    for k, v in diag.items():
        print(f'  {k}: {v}')
    if corners is not None:
        for label, (x, y) in zip(['TL', 'TR', 'BR', 'BL'], corners):
            print(f'  {label}: ({x:.1f}, {y:.1f})')

    if save:
        out_path = f'{os.path.splitext(path)[0]}_corners.jpg'
        cv2.imwrite(out_path, overlay)
        print(f'  saved -> {out_path}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{os.path.basename(path)}  —  {diag.get("detect_result")}')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('orange mask')
    axes[1].axis('off')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize gate_detector's 4-corner detection on saved frames.")
    parser.add_argument('images', nargs='*',
                        help='image file(s), directories, or glob pattern(s)')
    parser.add_argument('--save', action='store_true',
                        help='also write <name>_corners.jpg next to each source image')
    args = parser.parse_args()

    paths = _resolve_paths(args.images) if args.images else _auto_find()
    if not paths:
        print('No matching images found.')
        sys.exit(1)

    for p in paths:
        _inspect(p, args.save)
