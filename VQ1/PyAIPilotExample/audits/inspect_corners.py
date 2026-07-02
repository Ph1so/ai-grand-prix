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

    # Play a folder of frames back like FPV footage (OpenCV window) instead
    # of stepping through matplotlib plots one by one:
    python inspect_corners.py logs/run_20260607_210000/frames --play
    python inspect_corners.py logs/run_20260607_210000/frames --play --fps 15
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


def _play(paths: list[str], fps: float = 10.0, show_mask: bool = True) -> None:
    """
    Cycle through saved frames in an OpenCV window like FPV playback,
    overlaying the detected gate corners (and orange mask) on each frame —
    much faster than clicking through matplotlib windows one at a time.

    Controls:
        space   pause / resume
        a / d   step back / forward (auto-pauses)
        +  -    speed up / slow down
        c       capture the current frame, pixel-for-pixel as displayed
        q / Esc quit
    """
    fps = max(fps, 0.1)
    delay_ms = max(1, round(1000 / fps))
    win = 'FPV replay  —  space=pause  a/d=step  +/-=speed  c=capture  q=quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f'[play] {len(paths)} frames at {fps:.1f} fps — '
          f'space=pause, a/d=step, +/-=speed, c=capture, q=quit')

    idx = 0
    paused = False
    frame  = None
    while True:
        path = paths[idx]
        img = cv2.imread(path)
        if img is None:
            print(f'[skip] could not read {path}')
            frame = None
        else:
            diag = {}
            corners = detect_gate(img, diag=diag)
            frame = draw_detection(img, corners) if corners is not None else img.copy()
            if show_mask:
                mask = _orange_mask_from_hsv(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                frame = cv2.hconcat([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

            state = 'PAUSED' if paused else f'{1000 / delay_ms:.1f} fps'
            tag = f'[{idx + 1}/{len(paths)}] {os.path.basename(path)}  {diag.get("detect_result")}  {state}'
            cv2.putText(frame, tag, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord('q'), 27):                  # q / Esc — quit
            break
        elif key == ord(' '):                      # pause / resume
            paused = not paused
            continue
        elif key == ord('d'):                      # step forward
            idx = min(idx + 1, len(paths) - 1)
            paused = True
            continue
        elif key == ord('a'):                      # step back
            idx = max(idx - 1, 0)
            paused = True
            continue
        elif key in (ord('+'), ord('=')):           # speed up
            delay_ms = max(1, delay_ms - 10)
            continue
        elif key == ord('-'):                       # slow down
            delay_ms += 10
            continue
        elif key == ord('c'):                       # capture exactly what's on screen
            if frame is not None:
                stem     = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(os.path.dirname(path), f'{stem}_capture.jpg')
                cv2.imwrite(out_path, frame)
                print(f'[capture] saved -> {out_path}')
            continue

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if not paused:
            idx = (idx + 1) % len(paths)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize gate_detector's 4-corner detection on saved frames.")
    parser.add_argument('images', nargs='*',
                        help='image file(s), directories, or glob pattern(s)')
    parser.add_argument('--save', action='store_true',
                        help='also write <name>_corners.jpg next to each source image')
    parser.add_argument('--play', action='store_true',
                        help='play frames sequentially in an OpenCV window like FPV footage, '
                             'instead of stepping through matplotlib plots one by one')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='playback speed in frames/sec for --play (default 10)')
    parser.add_argument('--no-mask', action='store_true',
                        help='hide the orange-mask panel during --play')
    args = parser.parse_args()

    paths = _resolve_paths(args.images) if args.images else _auto_find()
    if not paths:
        print('No matching images found.')
        sys.exit(1)

    if args.play:
        _play(paths, fps=args.fps, show_mask=not args.no_mask)
        sys.exit(0)

    for p in paths:
        _inspect(p, args.save)
