"""
Gate position verifier.

Online  : GateVerifier subclasses VisionRX and logs CV-estimated gate
          positions to a CSV during flight.

Offline : Run as a script to plot CV estimates vs. MAVLink ground truth.

Usage (offline):
    python gate_verifier.py                            # auto-picks latest files
    python gate_verifier.py estimates.csv gates.json   # explicit files
"""

import argparse
import csv
import glob
import os
import sys
import time

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from vision_rx import VisionRX
from gate_detector import detect_gate
from pose_estimator import estimate_gate_camera_frame, camera_to_ned
from planner import load_gates, gate_center


# ── Online component ──────────────────────────────────────────────────────────

class GateVerifier(VisionRX):
    """
    Drop-in replacement for VisionRX.  Intercepts camera frames during
    flight, estimates each visible gate's NED position via PnP, and writes
    one row per successful detection to a CSV for post-flight analysis.
    """

    def __init__(self, data, log_path: str | None = None, run_dir: str | None = None):
        super().__init__(data)
        base = run_dir if run_dir is not None else _LOG_DIR
        ts   = time.strftime('%Y%m%d_%H%M%S')
        if log_path is None:
            log_path = os.path.join(base, f"gate_estimates_{ts}.csv")
        self._log_path = log_path
        self._log_file = open(log_path, 'w', newline='', buffering=1)
        self._csv_out  = csv.writer(self._log_file)
        self._csv_out.writerow([
            'sim_time_ns',
            'tvec_x', 'tvec_y', 'tvec_z',
            'rvec_x', 'rvec_y', 'rvec_z',
            'corner_tl_x', 'corner_tl_y', 'corner_tr_x', 'corner_tr_y',
            'corner_br_x', 'corner_br_y', 'corner_bl_x', 'corner_bl_y',
            'roll', 'pitch', 'yaw',
            'est_x', 'est_y', 'est_z',
            'drone_x', 'drone_y', 'drone_z',
            'matched_gate_id',
            'mav_x', 'mav_y', 'mav_z',
            'error_x', 'error_y', 'error_z', 'error_mag',
        ])
        print(f"[verifier] logging estimates to {log_path}", flush=True)

        # ── PERC.md A5/A7 instrumentation: log detection-pipeline internals
        # for EVERY frame attempt (success AND failure) — gate_estimates only
        # ever sees successes, which can't tell you whether a real gate was
        # in view and got silently dropped by a filter or polygon-reduction.
        diag_path = os.path.join(base, f"detection_diag_{ts}.csv")
        self._diag_path = diag_path
        self._diag_file = open(diag_path, 'w', newline='', buffering=1)
        self._diag_out  = csv.writer(self._diag_file)
        self._diag_out.writerow([
            'sim_time_ns', 'drone_x', 'drone_y', 'drone_z', 'roll', 'pitch', 'yaw',
            'detect_result', 'n_contours',
            'largest_area', 'largest_aspect', 'largest_passed', 'n_candidates',
            'best_area', 'best_aspect', 'best_hull_pts', 'best_poly_pts', 'four_corner_eps',
        ])
        print(f"[verifier] logging detection diagnostics to {diag_path}", flush=True)

    def get_thread_for_join(self):
        self._log_file.close()
        self._diag_file.close()
        return super().get_thread_for_join()

    def process_frame(self, frame_id: int, img, sim_time_ns: int = 0):
        pos      = self.data.get('pos')
        attitude = self.data.get('attitude')
        if pos is None or attitude is None:
            return
        roll, pitch, yaw = attitude

        diag    = {}
        corners = detect_gate(img, diag=diag)

        # A5/A7 instrumentation: log detector internals for EVERY frame —
        # success or failure — so "gate was in view but got dropped" is
        # distinguishable from "gate genuinely wasn't visible".
        self._diag_out.writerow([
            sim_time_ns, *pos.tolist(), roll, pitch, yaw,
            diag.get('detect_result'), diag.get('n_contours'),
            diag.get('largest_area'), diag.get('largest_aspect'),
            diag.get('largest_passed'), diag.get('n_candidates'),
            diag.get('best_area'), diag.get('best_aspect'),
            diag.get('best_hull_pts'), diag.get('best_poly_pts'), diag.get('four_corner_eps'),
        ])

        if corners is None:
            return

        tvec, rvec = estimate_gate_camera_frame(corners)
        if tvec is None:
            return

        est = camera_to_ned(tvec, pos, roll, pitch, yaw)

        # Publish live estimate for the controller
        self.data['cv_gate_pos']  = est
        self.data['cv_gate_time'] = time.time()

        # corners is the ORDERED [TL,TR,BR,BL] array fed to solvePnP — the
        # ground truth reproject_corners(rvec, tvec) (PERC.md A2) must match.
        corner_vals = corners.flatten().tolist()

        # Log to CSV; error columns require MAVLink gate map for ground-truth matching
        gates = self.data.get('gates')
        if gates:
            centers = np.array([gate_center(g) for g in gates])
            dists   = np.linalg.norm(centers - est, axis=1)
            idx     = int(np.argmin(dists))
            g       = gates[idx]
            mav     = gate_center(g)
            err     = est - mav
            self._csv_out.writerow([
                sim_time_ns, *tvec.tolist(), *rvec.tolist(), *corner_vals, roll, pitch, yaw,
                *est.tolist(), *pos.tolist(),
                g['id'], *mav.tolist(), *err.tolist(), float(np.linalg.norm(err)),
            ])
        else:
            nan = float('nan')
            self._csv_out.writerow([
                sim_time_ns, *tvec.tolist(), *rvec.tolist(), *corner_vals, roll, pitch, yaw,
                *est.tolist(), *pos.tolist(),
                -1, nan, nan, nan, nan, nan, nan, nan,
            ])


# ── Offline loader ────────────────────────────────────────────────────────────

def _load_estimates(path: str) -> list[dict]:
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                parsed[k] = int(v) if k == 'matched_gate_id' else float(v)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No data in {path}")
    return rows


# ── Offline plot ──────────────────────────────────────────────────────────────

def plot_verification(estimate_path: str, gates_path: str):
    rows  = _load_estimates(estimate_path)
    gates = load_gates(gates_path)

    t    = np.array([r['sim_time_ns'] * 1e-9 for r in rows])
    t   -= t[0]
    ex   = np.array([r['error_x']   for r in rows])
    ey   = np.array([r['error_y']   for r in rows])
    ez   = np.array([r['error_z']   for r in rows])
    emag = np.array([r['error_mag'] for r in rows])

    drone = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in rows])
    est   = np.array([[r['est_x'],   r['est_y'],   r['est_z']  ] for r in rows])
    mav   = np.array([[r['mav_x'],   r['mav_y'],   r['mav_z']  ] for r in rows])
    d2g   = np.linalg.norm(est - drone, axis=1)

    gate_ids = sorted({r['matched_gate_id'] for r in rows})

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('CV Gate Position Estimate vs. MAVLink Ground Truth', fontsize=13)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    gs   = fig.add_gridspec(2, 3, wspace=0.38, hspace=0.42,
                            left=0.06, right=0.97, top=0.91, bottom=0.08)
    ax_xt = fig.add_subplot(gs[0, 0])
    ax_mt = fig.add_subplot(gs[0, 1])
    ax_rg = fig.add_subplot(gs[1, 0])
    ax_bx = fig.add_subplot(gs[1, 1])
    ax_3d = fig.add_subplot(gs[:, 2], projection='3d')

    # ── Error components vs time ───────────────────────────────────────────────
    for arr, label, col in [(ex,'X (N)','tab:red'),
                             (ey,'Y (E)','tab:green'),
                             (ez,'Z (D)','tab:blue')]:
        ax_xt.plot(t, arr, label=label, color=col, linewidth=0.9)
    ax_xt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_xt.set_xlabel('Time (s)')
    ax_xt.set_ylabel('Error (m)')
    ax_xt.set_title('Error components vs. time')
    ax_xt.legend(fontsize=8)
    ax_xt.grid(alpha=0.25)

    # ── Error magnitude vs time ────────────────────────────────────────────────
    ax_mt.plot(t, emag, color='darkorange', linewidth=0.9)
    ax_mt.set_xlabel('Time (s)')
    ax_mt.set_ylabel('‖error‖ (m)')
    ax_mt.set_title('Error magnitude vs. time')
    ax_mt.grid(alpha=0.25)

    # ── Error magnitude vs drone-to-gate distance ──────────────────────────────
    sc = ax_rg.scatter(d2g, emag, c=t, cmap='plasma', s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax_rg, label='Time (s)', shrink=0.8)
    ax_rg.set_xlabel('Drone-to-gate distance (m)')
    ax_rg.set_ylabel('‖error‖ (m)')
    ax_rg.set_title('Error vs. range')
    ax_rg.grid(alpha=0.25)

    # ── Per-gate error box plot ────────────────────────────────────────────────
    gate_errors = [
        [r['error_mag'] for r in rows if r['matched_gate_id'] == gid]
        for gid in gate_ids
    ]
    ax_bx.boxplot(gate_errors, labels=[f'G{int(gid)}' for gid in gate_ids],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightskyblue', color='steelblue'),
                  medianprops=dict(color='darkorange', linewidth=1.5))
    ax_bx.set_xlabel('Gate')
    ax_bx.set_ylabel('‖error‖ (m)')
    ax_bx.set_title('Error distribution per gate')
    ax_bx.grid(axis='y', alpha=0.25)

    # ── 3-D: estimated vs MAVLink gate centres ────────────────────────────────
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(gate_ids), 1)))
    for i, gid in enumerate(gate_ids):
        mask = np.array([r['matched_gate_id'] == gid for r in rows])
        if not mask.any():
            continue
        col = colors[i % len(colors)]
        # CV estimates (scatter cloud)
        ax_3d.scatter(est[mask, 0], est[mask, 1], -est[mask, 2],
                      color=col, s=6, alpha=0.5, label=f'G{int(gid)} CV')
        # MAVLink ground truth (star marker, constant per gate)
        mav_pt = mav[mask][0]
        ax_3d.scatter(mav_pt[0], mav_pt[1], -mav_pt[2],
                      marker='*', s=140, color=col,
                      edgecolors='k', linewidths=0.5, zorder=5,
                      label=f'G{int(gid)} MAVLink')

    ax_3d.set_xlabel('X / N (m)', fontsize=7)
    ax_3d.set_ylabel('Y / E (m)', fontsize=7)
    ax_3d.set_zlabel('Alt (m)',   fontsize=7)
    ax_3d.set_title('Estimated vs. MAVLink (NED→display)', fontsize=9)
    ax_3d.legend(fontsize=6, loc='upper right')
    ax_3d.tick_params(labelsize=6)

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Frames analysed : {len(rows)}")
    print(f"  Mean  ‖error‖   : {emag.mean():.3f} m")
    print(f"  Median‖error‖   : {np.median(emag):.3f} m")
    print(f"  Max   ‖error‖   : {emag.max():.3f} m")
    print(f"  Std   ‖error‖   : {emag.std():.3f} m")
    print(f"{'─'*50}\n")

    plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _auto_find() -> tuple[str, str]:
    csvs  = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'gate_estimates_*.csv')),    reverse=True)
    jsons = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'flight_log_*_gates.json')), reverse=True)
    if not csvs:
        print('No gate_estimates_*.csv found in current directory.')
        sys.exit(1)
    if not jsons:
        print('No flight_log_*_gates.json found in current directory.')
        sys.exit(1)
    return csvs[0], jsons[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot CV gate estimates vs. MAVLink ground truth.')
    parser.add_argument('estimates', nargs='?', help='gate_estimates_*.csv')
    parser.add_argument('gates',     nargs='?', help='flight_log_*_gates.json')
    args = parser.parse_args()

    if args.estimates is None or args.gates is None:
        est_path, gates_path = _auto_find()
    else:
        est_path, gates_path = args.estimates, args.gates

    print(f'Estimates : {est_path}')
    print(f'Gates     : {gates_path}')
    plot_verification(est_path, gates_path)
