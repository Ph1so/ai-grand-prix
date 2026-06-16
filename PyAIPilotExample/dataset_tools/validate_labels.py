"""
Reconstruct global (NED) gate positions from auto-labeled frames + recorded
drone pose, and compare against MAVLink ground truth.

label_run.py auto-labels frames by projecting KNOWN gate geometry (world ->
image) using the drone pose at capture time. This script runs the inverse -
the same image -> world transform the live CV pipeline uses
(pose_estimator.estimate_gate_camera_frame + camera_to_ned) - on those same
label corners and poses. If the auto-labeler's projection and the
pose-estimator's reverse transform agree, the reconstructed gate centers
should land on top of the MAVLink ground-truth centers. Systematic offsets
here point to a bug in one of those two transforms (or their corner
ordering/conventions), independent of any CV detection error.

Usage:
    python validate_labels.py <run_dir>             # only fully-visible (4/4) corners
    python validate_labels.py <run_dir> --include-partial
    python validate_labels.py <run_dir> --no-plot
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_geometry import IMG_W, IMG_H, gate_world_corners
from pose_estimator import estimate_gate_camera_frame, camera_to_ned
from planner import load_gates

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR  = os.path.join(REPO_ROOT, 'logs')


def _find_one(run_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No '{pattern}' found in {run_dir}")
    return matches[0]


def _load_snapshots(path: str) -> dict[int, dict]:
    poses = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            poses[int(row['frame_id'])] = {
                'sim_time_ns': int(row['sim_time_ns']),
                'pos': np.array([float(row['drone_x']), float(row['drone_y']), float(row['drone_z'])]),
                'roll': float(row['roll']),
                'pitch': float(row['pitch']),
                'yaw': float(row['yaw']),
            }
    return poses


def _parse_label_line(line: str) -> tuple[np.ndarray, int]:
    """One YOLO-pose label line -> ((4,2) pixel corners [TL,TR,BR,BL], n_visible)."""
    kp = line.split()[5:]
    corners = np.zeros((4, 2), dtype=np.float64)
    n_visible = 0
    for i in range(4):
        corners[i] = (float(kp[3 * i]) * IMG_W, float(kp[3 * i + 1]) * IMG_H)
        if int(kp[3 * i + 2]) == 2:
            n_visible += 1
    return corners, n_visible


# -- Core reconstruction ------------------------------------------------------

def reconstruct_run(run_dir: str, include_partial: bool = False) -> tuple[list[dict], list[dict], np.ndarray]:
    snapshots_path = _find_one(run_dir, 'snapshots_*.csv')
    gates_path     = _find_one(run_dir, 'flight_log_*_gates.json')
    labels_dir     = os.path.join(run_dir, 'labels')

    poses = _load_snapshots(snapshots_path)
    gates = load_gates(gates_path)
    # True geometric centre (matches what label_run.py projected) - NOT
    # planner.gate_center(), which adds a controller-only altitude fudge.
    gate_centers = np.array([gate_world_corners(g).mean(axis=0) for g in gates])
    gate_ids     = [g['id'] for g in gates]

    label_paths = sorted(glob.glob(os.path.join(labels_dir, 'frame_*.txt')))

    rows = []
    n_no_pose, n_pnp_fail, n_skipped_partial = 0, 0, 0
    for label_path in label_paths:
        frame_id = int(os.path.splitext(os.path.basename(label_path))[0].split('_')[1])
        pose = poses.get(frame_id)
        if pose is None:
            n_no_pose += 1
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for inst_idx, line in enumerate(lines):
            corners, n_visible = _parse_label_line(line)
            if n_visible < 4 and not include_partial:
                n_skipped_partial += 1
                continue

            tvec, _ = estimate_gate_camera_frame(corners)
            if tvec is None:
                n_pnp_fail += 1
                continue

            recon = camera_to_ned(tvec, pose['pos'], pose['roll'], pose['pitch'], pose['yaw'])

            dists = np.linalg.norm(gate_centers - recon, axis=1)
            idx   = int(np.argmin(dists))
            mav   = gate_centers[idx]
            err   = recon - mav

            rows.append({
                'frame_id': frame_id,
                'instance_idx': inst_idx,
                'n_visible': n_visible,
                'sim_time_ns': pose['sim_time_ns'],
                'drone_x': pose['pos'][0], 'drone_y': pose['pos'][1], 'drone_z': pose['pos'][2],
                'recon_x': recon[0], 'recon_y': recon[1], 'recon_z': recon[2],
                'matched_gate_id': gate_ids[idx],
                'mav_x': mav[0], 'mav_y': mav[1], 'mav_z': mav[2],
                'error_x': err[0], 'error_y': err[1], 'error_z': err[2],
                'error_mag': float(np.linalg.norm(err)),
            })

    print(f'[validate_labels] {run_dir}')
    print(f'  label files            : {len(label_paths)}')
    print(f'  frames w/o pose        : {n_no_pose}')
    print(f'  partial instances skipped (<4/4 corners): {n_skipped_partial}')
    print(f'  PnP failures           : {n_pnp_fail}')
    print(f'  reconstructed instances: {len(rows)}')

    return rows, gates, gate_centers


# -- Output --------------------------------------------------------------------

def _write_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[validate_labels] wrote {out_path}')


def _print_stats(rows: list[dict]) -> None:
    if not rows:
        print('No reconstructed instances.')
        return

    emag = np.array([r['error_mag'] for r in rows])
    full = emag[np.array([r['n_visible'] for r in rows]) == 4]

    print(f"\n{'-'*50}")
    print(f"  All reconstructed instances : {len(rows)}")
    print(f"    mean   error_mag : {emag.mean():.3f} m")
    print(f"    median error_mag : {np.median(emag):.3f} m")
    print(f"    max    error_mag : {emag.max():.3f} m")
    print(f"    std    error_mag : {emag.std():.3f} m")
    if len(full) and len(full) != len(emag):
        print(f"  Full-visibility (4/4 corners): {len(full)}")
        print(f"    mean   error_mag : {full.mean():.3f} m")
        print(f"    median error_mag : {np.median(full):.3f} m")
        print(f"    max    error_mag : {full.max():.3f} m")
        print(f"    std    error_mag : {full.std():.3f} m")
    print(f"{'-'*50}\n")


def plot_reconstruction(rows: list[dict], gates: list[dict], gate_centers: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    gate_ids = [g['id'] for g in gates]
    recon    = np.array([[r['recon_x'], r['recon_y'], r['recon_z']] for r in rows])
    mav      = np.array([[r['mav_x'], r['mav_y'], r['mav_z']] for r in rows])
    n_vis    = np.array([r['n_visible'] for r in rows])
    matched  = np.array([r['matched_gate_id'] for r in rows])
    emag     = np.array([r['error_mag'] for r in rows])
    full_mask = n_vis == 4

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('Label-Reconstructed Gate Positions vs. MAVLink Ground Truth', fontsize=13)
    gs = fig.add_gridspec(2, 3, wspace=0.38, hspace=0.42,
                          left=0.05, right=0.97, top=0.91, bottom=0.08)

    ax_3d  = fig.add_subplot(gs[:, 0], projection='3d')
    ax_top = fig.add_subplot(gs[0, 1])
    ax_sid = fig.add_subplot(gs[1, 1])
    ax_vis = fig.add_subplot(gs[0, 2])
    ax_bx  = fig.add_subplot(gs[1, 2])

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(gate_ids), 1)))

    for i, gid in enumerate(gate_ids):
        col  = colors[i % len(colors)]
        mask = matched == gid
        fm   = mask & full_mask
        pm   = mask & ~full_mask

        if fm.any():
            ax_3d.scatter(recon[fm, 0], recon[fm, 1], -recon[fm, 2],
                          color=col, s=10, alpha=0.7, label=f'G{gid} recon (4/4)')
            ax_top.scatter(recon[fm, 0], recon[fm, 1], color=col, s=10, alpha=0.7)
            ax_sid.scatter(recon[fm, 0], -recon[fm, 2], color=col, s=10, alpha=0.7)
        if pm.any():
            ax_3d.scatter(recon[pm, 0], recon[pm, 1], -recon[pm, 2],
                          color=col, s=8, alpha=0.25, marker='x', label=f'G{gid} recon (partial)')
            ax_top.scatter(recon[pm, 0], recon[pm, 1], color=col, s=8, alpha=0.25, marker='x')
            ax_sid.scatter(recon[pm, 0], -recon[pm, 2], color=col, s=8, alpha=0.25, marker='x')

        # MAVLink ground truth: star marker + gate outline
        gc = gate_centers[i]
        ax_3d.scatter(gc[0], gc[1], -gc[2], marker='*', s=160, color=col,
                      edgecolors='k', linewidths=0.6, zorder=5, label=f'G{gid} MAVLink')

        rect = gate_world_corners(gates[i])
        rect_top = np.vstack([rect, rect[:1]])
        ax_top.plot(rect_top[:, 0], rect_top[:, 1], color=col, linewidth=2, alpha=0.8)
        ax_top.scatter(gc[0], gc[1], marker='*', s=120, color=col, edgecolors='k', zorder=5)
        ax_top.annotate(f'G{gid}', (gc[0], gc[1]), fontsize=8, xytext=(4, 4), textcoords='offset points')

        rect_sid = np.vstack([rect, rect[:1]])
        ax_sid.plot(rect_sid[:, 0], -rect_sid[:, 2], color=col, linewidth=2, alpha=0.8)
        ax_sid.scatter(gc[0], -gc[2], marker='*', s=120, color=col, edgecolors='k', zorder=5)
        ax_sid.annotate(f'G{gid}', (gc[0], -gc[2]), fontsize=8, xytext=(4, 4), textcoords='offset points')

    ax_3d.set_xlabel('X / North (m)', fontsize=7)
    ax_3d.set_ylabel('Y / East (m)',  fontsize=7)
    ax_3d.set_zlabel('Alt (m)',       fontsize=7)
    ax_3d.set_title('Reconstructed vs. MAVLink (3D)', fontsize=9)
    ax_3d.legend(fontsize=6, loc='upper right')
    ax_3d.tick_params(labelsize=6)

    ax_top.set_xlabel('X / North (m)')
    ax_top.set_ylabel('Y / East (m)')
    ax_top.set_title('Top-down (X-Y)')
    ax_top.grid(alpha=0.25)
    ax_top.set_aspect('equal', adjustable='datalim')

    ax_sid.set_xlabel('X / North (m)')
    ax_sid.set_ylabel('Altitude (m)')
    ax_sid.set_title('Side view (X-altitude)')
    ax_sid.grid(alpha=0.25)

    # Error vs. corner visibility
    vis_levels = sorted(set(n_vis.tolist()))
    if len(vis_levels) > 1:
        data = [emag[n_vis == v] for v in vis_levels]
        ax_vis.boxplot(data, labels=[str(v) for v in vis_levels], patch_artist=True,
                       boxprops=dict(facecolor='lightskyblue', color='steelblue'),
                       medianprops=dict(color='darkorange', linewidth=1.5))
        ax_vis.set_xlabel('# visible corners')
        ax_vis.set_title('Error vs. corner visibility')
    else:
        ax_vis.hist(emag, bins=30, color='lightskyblue', edgecolor='steelblue')
        ax_vis.set_xlabel('error_mag (m)')
        ax_vis.set_title('Error magnitude distribution')
    ax_vis.set_ylabel('error_mag (m)')
    ax_vis.grid(axis='y', alpha=0.25)

    # Per-gate error box plot (full-visibility only)
    gate_errors = [emag[full_mask & (matched == gid)] for gid in gate_ids]
    ax_bx.boxplot(gate_errors, labels=[f'G{gid}' for gid in gate_ids], patch_artist=True,
                  boxprops=dict(facecolor='lightgreen', color='seagreen'),
                  medianprops=dict(color='darkorange', linewidth=1.5))
    ax_bx.set_xlabel('Gate')
    ax_bx.set_ylabel('error_mag (m)')
    ax_bx.set_title('Error per gate (4/4 visible only)')
    ax_bx.grid(axis='y', alpha=0.25)

    plt.show()


# -- CLI entry point -----------------------------------------------------------

def _auto_find_run_dir() -> str:
    run_dirs = sorted(glob.glob(os.path.join(LOGS_DIR, 'run_*')), reverse=True)
    for run_dir in run_dirs:
        if os.path.isdir(os.path.join(run_dir, 'labels')):
            return run_dir
    print(f'No run_*/labels/ directory found under {LOGS_DIR}.')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', nargs='?', help='run directory containing labels/, snapshots_*.csv, *_gates.json '
                                                     '(default: latest run with a labels/ dir)')
    parser.add_argument('--include-partial', action='store_true',
                         help='also reconstruct from instances with <4/4 visible corners '
                              '(clipped corners are reprojected to image bounds, so these '
                              'are expected to show larger error)')
    parser.add_argument('--no-plot', action='store_true', help='skip the matplotlib visualization')
    parser.add_argument('--csv', help='write per-instance results to this CSV path '
                                       '(default: <run_dir>/label_recon_<timestamp>.csv)')
    args = parser.parse_args()

    run_dir = args.run_dir or _auto_find_run_dir()
    rows, gates, gate_centers = reconstruct_run(run_dir, include_partial=args.include_partial)
    _print_stats(rows)

    csv_path = args.csv
    if csv_path is None:
        gates_path = _find_one(run_dir, 'flight_log_*_gates.json')
        ts = os.path.basename(gates_path).removeprefix('flight_log_').removesuffix('_gates.json')
        csv_path = os.path.join(run_dir, f'label_recon_{ts}.csv')
    _write_csv(rows, csv_path)

    if not args.no_plot:
        plot_reconstruction(rows, gates, gate_centers)


if __name__ == '__main__':
    main()
