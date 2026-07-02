"""
Run the gate detector on a logged frame and show where the CV model sees corners.

Usage:
    python inspect_frame.py logs/run_20260629_201358/frames/frame_000097.jpg
    python inspect_frame.py logs/run_20260629_201358/frames/  # runs all frames in folder

Press any key to advance to the next frame, or Q to quit.
"""

import sys
import pathlib
import cv2
from gate_detector import detect_gate, draw_detection

def inspect(path: pathlib.Path):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read {path}")
        return

    diag = {}
    corners = detect_gate(frame, diag=diag)

    if corners is not None:
        out = draw_detection(frame, corners)
        title = f"{path.name} — gate detected ({diag.get('n_clipped_corners',0)} clipped)"
    else:
        out = frame.copy()
        reason = diag.get('detect_result', 'unknown')
        title = f"{path.name} — NO gate ({reason})"

    print(title)
    print("  diag:", diag)

    cv2.imshow("Gate detector", out)
    key = cv2.waitKey(0) & 0xFF
    return key != ord('q')

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_frame.py <frame.jpg or folder>")
        sys.exit(1)

    target = pathlib.Path(sys.argv[1])
    if target.is_dir():
        frames = sorted(target.glob("*.jpg"))
    else:
        frames = [target]

    for f in frames:
        if not inspect(f):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
