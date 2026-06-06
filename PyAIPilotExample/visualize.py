"""
Offline map and trajectory visualizer.

Usage (run from PyAIPilotExample directory):
  python visualize.py                            # auto-picks most recent pair
  python visualize.py flight_log_YYYYMMDD_HHMMSS # explicit stem
  python visualize.py <csv_file> <gates_json>    # explicit files
"""

import csv
import glob
import json
import os
import sys

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from planner import Plan, load_gates, gate_center, smooth_path, LEAD_IN_DIST, TAKEOFF_ALT


# ── Coordinate helpers ────────────────────────────────────────────────────────

def ned_to_display(pts):
    """Negate z so altitude is positive-upward for plotting (NED -> display)."""
    pts = np.asarray(pts, dtype=float).copy()
    pts[..., 2] = -pts[..., 2]
    return pts


def gate_corners_ned(gates, idx):
    """
    4 corners of a gate rectangle in NED world coords.
    Orientation derived from track geometry to avoid sim quaternion convention ambiguity.
    """
    gate   = gates[idx]
    center = gate_center(gate)          # bottom-edge offset already applied
    w, h   = gate['width'], gate['height']

    if idx > 0:
        forward = center - gate_center(gates[idx - 1])
    elif idx < len(gates) - 1:
        forward = gate_center(gates[idx + 1]) - center
    else:
        forward = np.array([1.0, 0.0, 0.0])

    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, -1.0])          # NED: up = -z
    if abs(np.dot(forward, world_up)) > 0.9:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up);  right /= np.linalg.norm(right)
    up    = np.cross(right, forward);     up    /= np.linalg.norm(up)

    c = center
    return np.array([
        c - (w/2)*right - (h/2)*up,
        c + (w/2)*right - (h/2)*up,
        c + (w/2)*right + (h/2)*up,
        c - (w/2)*right + (h/2)*up,
    ])


# ── File loading ──────────────────────────────────────────────────────────────

def load_trajectory(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def find_latest_pair():
    for run_dir in sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*')), reverse=True):
        for csv_path in sorted(glob.glob(os.path.join(run_dir, 'flight_log_*.csv')), reverse=True):
            gates_path = csv_path[:-4] + "_gates.json"
            if os.path.exists(gates_path):
                return csv_path, gates_path
    return None, None


def find_estimates(csv_path: str) -> str | None:
    """Return the gate_estimates_*.csv that shares a timestamp with csv_path, or None."""
    stem = os.path.basename(csv_path).removeprefix('flight_log_').removesuffix('.csv')
    est  = os.path.join(os.path.dirname(csv_path), f'gate_estimates_{stem}.csv')
    return est if os.path.exists(est) else None


def load_estimates(path: str) -> list[dict]:
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                parsed[k] = int(float(v)) if k == 'matched_gate_id' else float(v)
            rows.append(parsed)
    return rows


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_gates_3d(ax, gates):
    gate_colors = plt.cm.Set1(np.linspace(0, 1, max(len(gates), 1)))
    for i, gate in enumerate(gates):
        corners_d = ned_to_display(gate_corners_ned(gates, i))
        poly = Poly3DCollection([corners_d.tolist()],
                                alpha=0.25, facecolor='cyan',
                                edgecolor='deepskyblue', linewidth=1.5)
        ax.add_collection3d(poly)
        c_d = ned_to_display(gate_center(gate))
        ax.text(c_d[0], c_d[1], c_d[2] + 1.2,
                f"G{gate['id']}", color='deepskyblue',
                fontsize=8, ha='center', fontweight='bold')


def draw_gates_2d(ax, gates, xi, yi):
    gate_colors = plt.cm.Set1(np.linspace(0, 1, max(len(gates), 1)))
    for i, gate in enumerate(gates):
        corners_d = ned_to_display(gate_corners_ned(gates, i))
        xs = list(corners_d[:, xi]) + [corners_d[0, xi]]
        ys = list(corners_d[:, yi]) + [corners_d[0, yi]]
        col = gate_colors[i % len(gate_colors)]
        ax.plot(xs, ys, color=col, linewidth=2.5, solid_capstyle='round')
        c_d = ned_to_display(gate_center(gate))
        # small dot at opening center
        ax.scatter(c_d[xi], c_d[yi], color=col, s=30, zorder=4)
        ax.text(c_d[xi], c_d[yi] + 1.2,
                f"G{gate['id']}", fontsize=7, ha='center',
                color=col, fontweight='bold')


def draw_gate_events_2d(ax, pos_d, ag, xi, yi):
    """Mark the point where each gate was confirmed by the sim."""
    transitions = np.where(np.diff(ag) > 0)[0]
    for idx in transitions:
        ax.scatter(pos_d[idx, xi], pos_d[idx, yi],
                   marker='*', s=120, color='gold',
                   zorder=6, linewidths=0.5, edgecolors='darkorange')


def draw_gate_events_3d(ax, pos_d, ag):
    transitions = np.where(np.diff(ag) > 0)[0]
    if len(transitions):
        pts = pos_d[transitions]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   marker='*', s=120, color='gold',
                   zorder=6, depthshade=False, label='Gate confirmed')


# ── CV overlay helpers ────────────────────────────────────────────────────────

def compute_cv_gate_means(rows):
    """
    Group CV estimates by matched_gate_id and return per-gate means in NED.
    Rows with matched_gate_id == -1 (no gate map during flight) are excluded.
    Returns (gate_ids list, means ndarray (G,3)) or ([], None).
    """
    gate_ids = sorted({r['matched_gate_id'] for r in rows if r['matched_gate_id'] >= 0})
    if not gate_ids:
        return [], None
    means = np.array([
        np.nanmean([[r['est_x'], r['est_y'], r['est_z']]
                    for r in rows if r['matched_gate_id'] == gid], axis=0)
        for gid in gate_ids
    ])
    return gate_ids, means


def build_cv_plan(cv_means_ned):
    """
    Build lead-in + gate-center waypoints and a smooth spline from CV-estimated
    gate centers, using the same geometry as the MAVLink planner.
    Returns (waypoints ndarray (W,3), path ndarray (N,3)) in NED.
    """
    start = np.array([0.0, 0.0, TAKEOFF_ALT])
    waypoints = [start]
    prev = start.copy()
    for center in cv_means_ned:
        center = np.asarray(center, dtype=float)
        to_gate = center - prev
        dist = float(np.linalg.norm(to_gate))
        if dist > 1.0:
            waypoints.append(center - (to_gate / dist) * LEAD_IN_DIST)
        waypoints.append(center)
        prev = center
    path = smooth_path(waypoints)
    return np.array(waypoints), path


def draw_cv_overlay_3d(ax, cv_all_d, cv_wps_d, cv_path_d, cv_means_d, gate_ids):
    """Overlay CV estimates and CV-derived plan on a 3-D axis (display coords)."""
    # Raw per-frame estimate cloud
    if cv_all_d is not None:
        ax.scatter(cv_all_d[:, 0], cv_all_d[:, 1], cv_all_d[:, 2],
                   color='orangered', s=3, alpha=0.20, zorder=3, label='CV estimates')
    # CV-derived smooth planned path
    if cv_path_d is not None:
        ax.plot(cv_path_d[:, 0], cv_path_d[:, 1], cv_path_d[:, 2],
                color='orangered', linewidth=1.2, alpha=0.7, linestyle='--',
                label='CV planned path')
    # CV waypoints (lead-ins + gate centers)
    if cv_wps_d is not None:
        ax.scatter(cv_wps_d[:, 0], cv_wps_d[:, 1], cv_wps_d[:, 2],
                   color='orangered', s=22, marker='D', zorder=5, alpha=0.9,
                   label='CV waypoints')
    # CV gate centers highlighted separately
    if cv_means_d is not None and len(cv_means_d):
        ax.scatter(cv_means_d[:, 0], cv_means_d[:, 1], cv_means_d[:, 2],
                   color='orangered', s=90, marker='D', zorder=7,
                   edgecolors='k', linewidths=0.7)


def draw_cv_overlay_2d(ax, cv_all_d, cv_wps_d, cv_path_d, cv_means_d, gate_ids, xi, yi):
    """Overlay CV estimates and CV-derived plan on a 2-D axis (display coords)."""
    if cv_all_d is not None:
        ax.scatter(cv_all_d[:, xi], cv_all_d[:, yi],
                   color='orangered', s=4, alpha=0.20, zorder=3, label='CV estimates')
    if cv_path_d is not None:
        ax.plot(cv_path_d[:, xi], cv_path_d[:, yi],
                color='orangered', linewidth=1.2, linestyle='--',
                alpha=0.7, label='CV planned path', zorder=2)
    if cv_wps_d is not None:
        ax.scatter(cv_wps_d[:, xi], cv_wps_d[:, yi],
                   color='orangered', s=22, marker='D', zorder=5, alpha=0.9,
                   label='CV waypoints')
    if cv_means_d is not None and len(cv_means_d):
        ax.scatter(cv_means_d[:, xi], cv_means_d[:, yi],
                   color='orangered', s=90, marker='D', zorder=7,
                   edgecolors='k', linewidths=0.7)
        for i, gid in enumerate(gate_ids):
            ax.annotate(f'G{int(gid)}',
                        xy=(cv_means_d[i, xi], cv_means_d[i, yi]),
                        xytext=(0, -10), textcoords='offset points',
                        fontsize=6, color='orangered', ha='center')


# ── CV comparison figure ──────────────────────────────────────────────────────

def plot_cv_comparison(estimates_path: str, gates: list):
    """
    Separate figure: CV-estimated gate positions vs MAVLink ground truth.
    Each gate gets a colour; CV estimates form a scatter cloud, MAVLink
    position is a star.  Arrows connect the mean estimate to the truth.
    Bottom panel shows the mean per-axis error per gate as a bar chart.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rows = load_estimates(estimates_path)
    if not rows:
        print(f'[viz] no data in {estimates_path}')
        return

    gate_ids = sorted({r['matched_gate_id'] for r in rows})
    colors   = plt.cm.Set1(np.linspace(0, 1, max(len(gate_ids), 1)))

    est_ned = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in rows])
    est_d   = ned_to_display(est_ned)

    # One MAVLink centre per gate (constant across rows for that gate)
    mav_centers = {}
    for gid in gate_ids:
        r = next(r for r in rows if r['matched_gate_id'] == gid)
        mav_centers[gid] = ned_to_display(np.array([r['mav_x'], r['mav_y'], r['mav_z']]))

    # Per-gate mask and mean estimates
    masks      = {gid: np.array([r['matched_gate_id'] == gid for r in rows]) for gid in gate_ids}
    mean_est_d = {gid: est_d[masks[gid]].mean(axis=0) for gid in gate_ids}
    mean_err   = {
        gid: {
            ax: np.mean([r[f'error_{ax}'] for r in rows if r['matched_gate_id'] == gid])
            for ax in ('x', 'y', 'z')
        }
        for gid in gate_ids
    }

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f'CV Gate Estimates vs. MAVLink Ground Truth\n{os.path.basename(estimates_path)}',
                 fontsize=12)

    gs     = fig.add_gridspec(2, 3, wspace=0.40, hspace=0.45,
                              left=0.06, right=0.97, top=0.90, bottom=0.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_sid = fig.add_subplot(gs[0, 1])
    ax_3d  = fig.add_subplot(gs[0, 2], projection='3d')
    ax_bar = fig.add_subplot(gs[1, :])

    # ── 2-D spatial views ──────────────────────────────────────────────────────
    for view_ax, xi, yi, xlabel, ylabel, title in [
        (ax_top, 0, 1, 'X / North (m)', 'Y / East (m)', 'Top-down (X–Y)'),
        (ax_sid, 0, 2, 'X / North (m)', 'Altitude (m)', 'Side view (X–alt)'),
    ]:
        for i, gid in enumerate(gate_ids):
            col  = colors[i % len(colors)]
            mask = masks[gid]
            mc   = mav_centers[gid]
            me   = mean_est_d[gid]

            # CV estimate scatter
            view_ax.scatter(est_d[mask, xi], est_d[mask, yi],
                            color=col, s=5, alpha=0.35, zorder=2)
            # MAVLink ground-truth star
            view_ax.scatter(mc[xi], mc[yi],
                            marker='*', s=220, color=col,
                            edgecolors='k', linewidths=0.6, zorder=5,
                            label=f'G{int(gid)}')
            # Arrow: mean CV estimate → MAVLink truth
            view_ax.annotate(
                '', xy=(mc[xi], mc[yi]), xytext=(me[xi], me[yi]),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5, alpha=0.85),
                zorder=4,
            )

        view_ax.set_xlabel(xlabel, fontsize=8)
        view_ax.set_ylabel(ylabel, fontsize=8)
        view_ax.set_title(title, fontsize=9)
        view_ax.set_aspect('equal', adjustable='datalim')
        view_ax.grid(alpha=0.25)
        view_ax.legend(title='★ = MAVLink', fontsize=7, loc='upper right', ncol=2)

    # ── 3-D view ───────────────────────────────────────────────────────────────
    for i, gid in enumerate(gate_ids):
        col  = colors[i % len(colors)]
        mask = masks[gid]
        mc   = mav_centers[gid]
        ax_3d.scatter(est_d[mask, 0], est_d[mask, 1], est_d[mask, 2],
                      color=col, s=4, alpha=0.35, zorder=2)
        ax_3d.scatter(mc[0], mc[1], mc[2],
                      marker='*', s=140, color=col,
                      edgecolors='k', linewidths=0.5, zorder=5,
                      label=f'G{int(gid)}')

    ax_3d.set_xlabel('X/N (m)', fontsize=7)
    ax_3d.set_ylabel('Y/E (m)', fontsize=7)
    ax_3d.set_zlabel('Alt (m)', fontsize=7)
    ax_3d.set_title('3-D  (★ = MAVLink, dots = CV)', fontsize=9)
    ax_3d.legend(fontsize=6, loc='upper right')
    ax_3d.tick_params(labelsize=6)

    # ── Per-gate mean error bar chart ──────────────────────────────────────────
    x_pos = np.arange(len(gate_ids))
    w     = 0.25
    ax_bar.bar(x_pos - w, [mean_err[g]['x'] for g in gate_ids], w,
               label='Err X (North)',  color='tab:red',   alpha=0.82)
    ax_bar.bar(x_pos,     [mean_err[g]['y'] for g in gate_ids], w,
               label='Err Y (East)',   color='tab:green', alpha=0.82)
    ax_bar.bar(x_pos + w, [mean_err[g]['z'] for g in gate_ids], w,
               label='Err Z (Down)',   color='tab:blue',  alpha=0.82)
    ax_bar.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f'Gate {int(g)}' for g in gate_ids], fontsize=9)
    ax_bar.set_ylabel('Mean CV − MAVLink error (m)', fontsize=9)
    ax_bar.set_title('Per-gate mean position error  '
                     '(positive = CV estimate is larger in that axis than MAVLink)', fontsize=9)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis='y', alpha=0.25)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 3:
        csv_path, gates_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        stem = sys.argv[1].removesuffix('.csv')
        csv_path, gates_path = stem + ".csv", stem + "_gates.json"
    else:
        csv_path, gates_path = find_latest_pair()
        if not csv_path:
            print("No matching flight_log_* pair found in current directory.")
            sys.exit(1)

    est_path = find_estimates(csv_path)

    print(f"Trajectory : {csv_path}")
    print(f"Gates      : {gates_path}")
    print(f"Estimates  : {est_path or '(none found)'}")

    # ── Load data ──────────────────────────────────────────────────────────────
    traj  = load_trajectory(csv_path)
    gates = load_gates(gates_path)
    plan  = Plan.from_gates(gates)
    plan.summary()

    pos = np.array([[r['pos_x'], r['pos_y'], r['pos_z']] for r in traj])
    vel = np.array([[r['vel_x'], r['vel_y'], r['vel_z']] for r in traj])
    t   = np.array([r['wall_time_s'] for r in traj]);  t -= t[0]
    ag  = np.array([r['active_gate'] for r in traj])

    pos_d  = ned_to_display(pos)
    plan_d = ned_to_display(plan.path)
    wps_d  = ned_to_display(plan.waypoints)

    # MAVLink gate centers (not lead-ins) for direct comparison
    mav_centers_ned = np.array([gate_center(g) for g in gates])
    mav_centers_d   = ned_to_display(mav_centers_ned)

    # CV estimates + derived plan for overlay
    cv_all_d    = None
    cv_means_d  = None
    cv_wps_d    = None
    cv_path_d   = None
    cv_gate_ids = []
    if est_path:
        est_rows = load_estimates(est_path)
        if est_rows:
            cv_all_ned  = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in est_rows])
            cv_all_d    = ned_to_display(cv_all_ned)
            cv_gate_ids, cv_means_ned = compute_cv_gate_means(est_rows)
            if cv_means_ned is not None:
                cv_means_d          = ned_to_display(cv_means_ned)
                cv_wps_ned, cv_path_ned = build_cv_plan(cv_means_ned)
                cv_wps_d            = ned_to_display(cv_wps_ned)
                cv_path_d           = ned_to_display(cv_path_ned)

    title_stem = os.path.basename(csv_path)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'AI Grand Prix  --  {title_stem}', fontsize=13)

    gs = fig.add_gridspec(2, 3,
                          left=0.04, right=0.98,
                          top=0.92, bottom=0.06,
                          wspace=0.35, hspace=0.40)

    ax3    = fig.add_subplot(gs[:, 0], projection='3d')  # 3-D, full height
    ax_top = fig.add_subplot(gs[0, 1])                   # top-down X-Y
    ax_sid = fig.add_subplot(gs[1, 1])                   # side X-alt
    ax_spd = fig.add_subplot(gs[0, 2])                   # speed over time
    ax_alt = fig.add_subplot(gs[1, 2])                   # altitude over time

    # ── 3-D view ───────────────────────────────────────────────────────────────
    # MAVLink planned path + waypoints
    ax3.plot(plan_d[:, 0], plan_d[:, 1], plan_d[:, 2],
             color='dodgerblue', linewidth=1.2, alpha=0.6, linestyle='--', label='MAVLink planned')
    ax3.scatter(wps_d[:, 0], wps_d[:, 1], wps_d[:, 2],
                color='dodgerblue', s=18, zorder=4, alpha=0.8)
    ax3.scatter(mav_centers_d[:, 0], mav_centers_d[:, 1], mav_centers_d[:, 2],
                color='deepskyblue', s=80, marker='s', zorder=5,
                edgecolors='k', linewidths=0.5, label='MAVLink gate')

    # CV overlay
    if cv_all_d is not None:
        draw_cv_overlay_3d(ax3, cv_all_d, cv_wps_d, cv_path_d, cv_means_d, cv_gate_ids)

    # Actual trajectory
    sc = ax3.scatter(pos_d[:, 0], pos_d[:, 1], pos_d[:, 2],
                     c=t, cmap='plasma', s=3, zorder=3)
    plt.colorbar(sc, ax=ax3, label='Elapsed time (s)', shrink=0.50, pad=0.1)
    ax3.plot(pos_d[:, 0], pos_d[:, 1], pos_d[:, 2],
             color='gray', linewidth=0.4, alpha=0.35, zorder=1)

    ax3.scatter(*pos_d[0], color='lime', s=100, marker='^', zorder=5, label='Start')
    draw_gate_events_3d(ax3, pos_d, ag)
    draw_gates_3d(ax3, gates)

    ax3.set_xlabel('X / North (m)', fontsize=7)
    ax3.set_ylabel('Y / East (m)',  fontsize=7)
    ax3.set_zlabel('Altitude (m)',  fontsize=7)
    ax3.legend(loc='upper right', fontsize=7)
    ax3.set_title('3-D view', fontsize=9)
    ax3.tick_params(labelsize=6)

    # ── 2-D spatial views ──────────────────────────────────────────────────────
    for view_ax, xi, yi, xlabel, ylabel, title in [
        (ax_top, 0, 1, 'X / North (m)', 'Y / East (m)',  'Top-down (X-Y)'),
        (ax_sid, 0, 2, 'X / North (m)', 'Altitude (m)',  'Side view (X-alt)'),
    ]:
        # MAVLink planned path + waypoints
        view_ax.plot(plan_d[:, xi], plan_d[:, yi],
                     color='dodgerblue', linewidth=1.2, linestyle='--',
                     alpha=0.7, label='MAVLink planned', zorder=2)
        view_ax.scatter(wps_d[:, xi], wps_d[:, yi],
                        color='dodgerblue', s=20, zorder=3, alpha=0.8)
        view_ax.scatter(mav_centers_d[:, xi], mav_centers_d[:, yi],
                        color='deepskyblue', s=80, marker='s', zorder=5,
                        edgecolors='k', linewidths=0.5, label='MAVLink gate')

        # CV overlay
        if cv_all_d is not None:
            draw_cv_overlay_2d(view_ax, cv_all_d, cv_wps_d, cv_path_d, cv_means_d, cv_gate_ids, xi, yi)

        # Actual trajectory
        view_ax.scatter(pos_d[:, xi], pos_d[:, yi], c=t, cmap='plasma', s=2, zorder=4)
        view_ax.plot(pos_d[:, xi], pos_d[:, yi],
                     color='gray', linewidth=0.4, alpha=0.35, zorder=1)
        view_ax.scatter(pos_d[0, xi], pos_d[0, yi],
                        color='lime', s=80, marker='^', zorder=5)

        draw_gate_events_2d(view_ax, pos_d, ag, xi, yi)
        draw_gates_2d(view_ax, gates, xi, yi)

        view_ax.set_xlabel(xlabel, fontsize=8)
        view_ax.set_ylabel(ylabel, fontsize=8)
        view_ax.set_title(title, fontsize=9)
        view_ax.set_aspect('equal', adjustable='datalim')
        view_ax.grid(True, alpha=0.25)
        view_ax.legend(fontsize=7, loc='upper right')

    # ── Time-series: speed ─────────────────────────────────────────────────────
    speed = np.linalg.norm(vel, axis=1)
    ax_spd.plot(t, speed, color='tomato', linewidth=1.0)
    ax_spd.set_xlabel('Time (s)', fontsize=8)
    ax_spd.set_ylabel('Speed (m/s)', fontsize=8)
    ax_spd.set_title('Speed over time', fontsize=9)
    ax_spd.grid(True, alpha=0.25)
    # Mark gate confirmations
    gate_times = t[np.where(np.diff(ag) > 0)[0]]
    for gt in gate_times:
        ax_spd.axvline(gt, color='gold', linewidth=1.2, linestyle='--', alpha=0.8)

    # ── Time-series: altitude vs planned ──────────────────────────────────────
    # Map each actual timestamp to the closest point on the planned path by x-position
    actual_alt = pos_d[:, 2]   # display altitude (positive up)

    # Planned altitude along x for reference line
    ax_alt.plot(t, actual_alt, color='tomato', linewidth=1.0, label='Actual')

    # Overlay target waypoint altitudes as horizontal steps
    wp_x_display  = wps_d[:, 0]
    wp_alt_display = wps_d[:, 2]
    for gt in gate_times:
        ax_alt.axvline(gt, color='gold', linewidth=1.2, linestyle='--', alpha=0.8)

    ax_alt.set_xlabel('Time (s)', fontsize=8)
    ax_alt.set_ylabel('Altitude (m)', fontsize=8)
    ax_alt.set_title('Altitude over time', fontsize=9)
    ax_alt.grid(True, alpha=0.25)
    ax_alt.legend(fontsize=7)

    # ── Summary ────────────────────────────────────────────────────────────────
    duration   = t[-1]
    total_dist = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    max_speed  = float(speed.max())
    gates_hit  = int(ag.max())

    print(f"\n{'-'*40}")
    print(f"  Duration  : {duration:.1f} s")
    print(f"  Distance  : {total_dist:.1f} m")
    print(f"  Max speed : {max_speed:.1f} m/s")
    print(f"  Gates hit : {gates_hit} / {len(gates)}")
    print(f"{'-'*40}\n")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    fig.savefig(os.path.join(out_dir, f"{stem}_trajectory.png"), dpi=150, bbox_inches='tight')
    print(f"[viz] saved trajectory figure → {out_dir}")

    if est_path:
        plot_cv_comparison(est_path, gates)
        plt.gcf().savefig(
            os.path.join(out_dir, f"{stem}_cv_comparison.png"), dpi=150, bbox_inches='tight')
        print(f"[viz] saved CV comparison figure → {out_dir}")

    plt.show()


if __name__ == '__main__':
    main()
