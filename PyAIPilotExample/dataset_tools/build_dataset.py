"""
Assemble a YOLO-pose gate-detection dataset from auto-labeled run directories.

Scans logs/run_*/ for dirs containing snapshots_*.csv + frames/ +
*_gates.json (produced by gate_verifier.py's snapshot-pose instrumentation),
auto-labels any that aren't already labeled (via label_run.py), then splits
BY RUN -- not by frame, to avoid near-duplicate leakage between adjacent
frames of the same flight -- into train/val and copies images+labels into
dataset/{images,labels}/{train,val}/. Writes dataset/data.yaml for
ultralytics pose training.

Pre-instrumentation runs (no snapshots_*.csv) are skipped -- they have no
recoverable frame_id <-> drone-pose mapping and cannot be auto-labeled.

Usage:
    python build_dataset.py [--val-frac 0.15] [--seed 42]
"""

import argparse
import glob
import os
import random
import shutil
import sys

from label_run import label_run

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PyAIPilotExample/
LOGS_DIR    = os.path.join(REPO_ROOT, 'logs')
DATASET_DIR = os.path.join(REPO_ROOT, 'dataset')


def find_labeled_runs() -> list[str]:
    run_dirs = sorted(glob.glob(os.path.join(LOGS_DIR, 'run_*')))
    labeled = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        has_snapshots = glob.glob(os.path.join(run_dir, 'snapshots_*.csv'))
        has_gates     = glob.glob(os.path.join(run_dir, 'flight_log_*_gates.json'))
        frames_dir    = os.path.join(run_dir, 'frames')
        if not (has_snapshots and has_gates and os.path.isdir(frames_dir)):
            continue

        labels_dir = os.path.join(run_dir, 'labels')
        n_frames = len(glob.glob(os.path.join(frames_dir, 'frame_*.jpg')))
        n_labels = len(glob.glob(os.path.join(labels_dir, 'frame_*.txt'))) if os.path.isdir(labels_dir) else 0
        if n_labels < n_frames:
            label_run(run_dir)

        labeled.append(run_dir)
    return labeled


def _split_runs(runs: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = runs[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        print('[build_dataset] WARNING: only 1 labeled run found -- val split will be empty.')
        return shuffled, []

    n_val = max(1, round(len(shuffled) * val_frac))
    n_val = min(n_val, len(shuffled) - 1)  # keep at least 1 train run
    return shuffled[n_val:], shuffled[:n_val]


def _reset_split_dirs() -> None:
    for split in ('train', 'val'):
        for sub in ('images', 'labels'):
            d = os.path.join(DATASET_DIR, sub, split)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)


def _write_data_yaml() -> None:
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    dataset_path = DATASET_DIR.replace('\\', '/')
    content = f"""path: {dataset_path}
train: images/train
val: images/val

names:
  0: gate

kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
"""
    with open(yaml_path, 'w') as f:
        f.write(content)


def build_dataset(val_frac: float = 0.15, seed: int = 42) -> None:
    runs = find_labeled_runs()
    if not runs:
        print('[build_dataset] No runs with snapshots_*.csv + frames/ + *_gates.json found under logs/.')
        print('  (Pre-instrumentation runs lack snapshots_*.csv and cannot be auto-labeled.)')
        sys.exit(1)

    train_runs, val_runs = _split_runs(runs, val_frac, seed)
    _reset_split_dirs()

    counts = {'train': [0, 0], 'val': [0, 0]}  # [n_images, n_instances]
    for split, split_runs in (('train', train_runs), ('val', val_runs)):
        for run_dir in split_runs:
            run_name   = os.path.basename(run_dir)
            frames_dir = os.path.join(run_dir, 'frames')
            labels_dir = os.path.join(run_dir, 'labels')
            for img_path in sorted(glob.glob(os.path.join(frames_dir, 'frame_*.jpg'))):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, f'{stem}.txt')
                if not os.path.exists(label_path):
                    continue

                dst_name = f'{run_name}_{stem}'
                shutil.copy2(img_path, os.path.join(DATASET_DIR, 'images', split, f'{dst_name}.jpg'))
                shutil.copy2(label_path, os.path.join(DATASET_DIR, 'labels', split, f'{dst_name}.txt'))

                counts[split][0] += 1
                with open(label_path) as f:
                    counts[split][1] += sum(1 for line in f if line.strip())

    _write_data_yaml()

    print(f'[build_dataset] {len(runs)} labeled run(s): {len(train_runs)} train, {len(val_runs)} val')
    for split in ('train', 'val'):
        n_img, n_inst = counts[split]
        print(f'  {split:5s}: {n_img} images, {n_inst} gate instances')
    print(f'[build_dataset] wrote {os.path.join(DATASET_DIR, "data.yaml")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-frac', type=float, default=0.15,
                         help='Fraction of runs assigned to val (default 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                         help='RNG seed for the train/val run split')
    args = parser.parse_args()
    build_dataset(val_frac=args.val_frac, seed=args.seed)


if __name__ == '__main__':
    main()
