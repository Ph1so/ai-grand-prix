"""
Interactive 3D flight replay — reconstruct the drone flight from log files.

What it shows
-------------
  3-D world view  : drone body (attitude-correct, colour-coded arms), gate outer
                    frame + inner opening, trajectory trail, planned path,
                    controller target, gate-confirmation events.
  FPV camera view : simulated onboard camera showing projected gate outlines and
                    CV estimates in image-pixel space — lets you verify whether the
                    gate geometry in the log matches what the camera would see.
  Telemetry panel : speed, altitude, attitude, active gate, target mode.
  Controls        : play/pause, 0.25×–8× speed, frame-step, time scrubber.

Usage (from PyAIPilotExample/):
  python replay3d.py                       # auto-picks most recent run with gates
  python replay3d.py run_20260606_160120   # explicit run directory name
  python replay3d.py path/to/flight_log.csv

Flags:
  --save   render all frames and export to <run_dir>/<stem>_replay.mp4 (needs ffmpeg)
"""

import csv
import glob
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from planner import Plan, load_gates, gate_center
from pose_estimator import R_CAM2BODY, _euler_zyx

# ── Physical / camera constants ───────────────────────────────────────────────
GATE_OUTER_HALF = 1.36      # m  (2.72 m outer frame / 2)
GATE_INNER_HALF = 0.75      # m  (1.50 m inner opening / 2)
DRONE_ARM       = 0.14      # m  arm length from body centre
FX = FY         = 320.0
CX, CY          = 320.0, 180.0
IMG_W, IMG_H    = 640, 360
TRAIL_LEN       = 250       # frames in the moving tail (~8 s at 30 Hz)
VIEW_RADIUS     = 30.0      # m  follow-mode half-window around drone
SPEEDS          = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CV_WINDOW       = 0.15      # s  show a CV estimate if it falls within this window

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# Body FRD arm directions (index 0 = forward → drawn RED for orientation cue)
_ARMS_FRD = np.array([
    [ DRONE_ARM,  0.,  0.],   # fwd  → RED
    [-DRONE_ARM,  0.,  0.],   # aft  → dim gray
    [ 0.,  DRONE_ARM,  0.],   # stbd → white
    [ 0., -DRONE_ARM,  0.],   # port → white
], dtype=float)
_ARM_COLORS = ['#ff4444', '#555566', '#ccccdd', '#ccccdd']


# ── Coordinate helpers ────────────────────────────────────────────────────────
def ned2d(p):
    """NED → display: negate z so altitude is positive-upward."""
    p = np.asarray(p, dtype=float).copy()
    p[..., 2] = -p[..., 2]
    return p


def gate_plane(gates, idx):
    """(center_NED, right_unit, up_unit) for gate idx, derived from track geometry."""
    c = gate_center(gates[idx])
    if idx > 0:
        fwd = c - gate_center(gates[idx - 1])
    elif idx < len(gates) - 1:
        fwd = gate_center(gates[idx + 1]) - c
    else:
        fwd = np.array([1., 0., 0.])
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0., 0., -1.])   # NED: up = −z
    if abs(fwd @ world_up) > 0.9:
        world_up = np.array([0., 1., 0.])
    right = np.cross(fwd, world_up);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd);     up    /= np.linalg.norm(up)
    return c, right, up


def gate_rect(gates, idx, half_dim):
    """4 NED corners of a gate rectangle (outer or inner), shape (4, 3)."""
    c, right, up = gate_plane(gates, idx)
    h = half_dim
    return np.array([
        c - h*right - h*up,
        c + h*right - h*up,
        c + h*right + h*up,
        c - h*right + h*up,
    ])


# ── Camera projection ─────────────────────────────────────────────────────────
def project_to_fov(pts_ned, drone_pos, roll, pitch, yaw):
    """
    Project NED world points into camera pixel coordinates.
    Returns (u, v, valid_mask) where valid = Zc > 0.1 (point in front of camera).
    pts_ned may be (3,) or (N, 3).
    """
    R_b2ned = _euler_zyx(roll, pitch, yaw)
    pts = np.atleast_2d(pts_ned)
    body = (R_b2ned.T @ (pts - drone_pos).T).T        # NED → body FRD
    cam  = (R_CAM2BODY.T @ body.T).T                   # body → camera (x=R, y=D, z=fwd)
    valid  = cam[:, 2] > 0.1
    safe_z = np.where(valid, cam[:, 2], 1.0)
    u = FX * cam[:, 0] / safe_z + CX
    v = FY * cam[:, 1] / safe_z + CY
    return u, v, valid


# ── File loading ──────────────────────────────────────────────────────────────
def _find_latest():
    for d in sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*')), reverse=True):
        for c in sorted(glob.glob(os.path.join(d, 'flight_log_*.csv')), reverse=True):
            g = c[:-4] + '_gates.json'
            if os.path.exists(g):
                return d, c, g
    return None, None, None


def _resolve(argv):
    save_mp4 = '--save' in argv
    args = [a for a in argv[1:] if not a.startswith('--')]
    if not args:
        return *_find_latest(), save_mp4
    arg = args[0]
    # Check if it's a run directory (by name or path)
    for candidate in [arg, os.path.join(_LOG_DIR, arg)]:
        if os.path.isdir(candidate):
            run_dir = candidate
            csvs = sorted(glob.glob(os.path.join(run_dir, 'flight_log_*.csv')), reverse=True)
            for c in csvs:
                g = c[:-4] + '_gates.json'
                if os.path.exists(g):
                    return run_dir, c, g, save_mp4
            print(f'No flight_log_*_gates.json pair found in {candidate}')
            sys.exit(1)
    # Treat as CSV path or stem
    stem = arg.removesuffix('.csv')
    c, g = stem + '.csv', stem + '_gates.json'
    return os.path.dirname(os.path.abspath(c)), c, g, save_mp4


def _load_csv(path, str_cols=()):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if k in str_cols:
                    parsed[k] = v.strip()
                else:
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = float('nan')
            rows.append(parsed)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    run_dir, csv_path, gates_path, do_save = _resolve(sys.argv)
    if csv_path is None:
        print('[replay3d] No matching log pair found.')
        sys.exit(1)

    print(f'[replay3d] CSV   : {csv_path}')
    print(f'[replay3d] Gates : {gates_path}')

    # ── Load flight log ────────────────────────────────────────────────────────
    traj = _load_csv(csv_path)
    t0   = traj[0]['wall_time_s']
    t    = np.array([r['wall_time_s'] - t0 for r in traj])
    pos  = np.array([[r['pos_x'], r['pos_y'], r['pos_z']] for r in traj])
    vel  = np.array([[r['vel_x'], r['vel_y'], r['vel_z']] for r in traj])
    att  = np.array([[r['roll'], r['pitch'], r['yaw']] for r in traj])
    ag   = np.array([r['active_gate'] for r in traj])
    N    = len(traj)

    # ── Load gates + plan ──────────────────────────────────────────────────────
    gates = load_gates(gates_path)
    plan  = Plan.from_gates(gates)
    G     = len(gates)
    print(f'[replay3d] {G} gates, {N} flight frames, {t[-1]:.1f} s duration')

    # ── Load CV estimates (optional) ───────────────────────────────────────────
    stem     = os.path.basename(csv_path).removeprefix('flight_log_').removesuffix('.csv')
    est_path = os.path.join(run_dir, f'gate_estimates_{stem}.csv')
    est_t    = np.array([])
    est_pos  = np.zeros((0, 3))
    if os.path.exists(est_path):
        est_rows = _load_csv(est_path)
        # sim_time_ns uses the same epoch as wall_time_s (confirmed by data alignment)
        est_t   = np.array([r['sim_time_ns'] / 1e9 - t0 for r in est_rows])
        est_pos = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in est_rows])
        print(f'[replay3d] {len(est_rows)} CV estimates')

    # ── Load controller log (optional) ────────────────────────────────────────
    ctrl_path = os.path.join(run_dir, 'controller_log.csv')
    ctrl_t    = np.array([])
    ctrl_tgt  = np.zeros((0, 3))
    ctrl_mode = []
    if os.path.exists(ctrl_path):
        ctrl_rows = _load_csv(ctrl_path, str_cols={'state', 'target_mode'})
        ctrl_t    = np.array([r['time_s'] - t0 for r in ctrl_rows])
        ctrl_tgt  = np.array([[r['target_x'], r['target_y'], r['target_z']]
                               for r in ctrl_rows])
        ctrl_mode = [r.get('target_mode') or r.get('state') or '' for r in ctrl_rows]
        print(f'[replay3d] {len(ctrl_rows)} controller log rows')

    # ── Precompute display-frame coords ────────────────────────────────────────
    pos_d      = ned2d(pos)
    plan_d     = ned2d(plan.path)
    wps_d      = ned2d(plan.waypoints)
    centers_d  = ned2d(np.array([gate_center(g) for g in gates]))
    centers_ned = np.array([gate_center(g) for g in gates])

    gouter_ned = [gate_rect(gates, i, GATE_OUTER_HALF) for i in range(G)]
    ginner_ned = [gate_rect(gates, i, GATE_INNER_HALF) for i in range(G)]
    gouter_d   = [ned2d(c) for c in gouter_ned]
    ginner_d   = [ned2d(c) for c in ginner_ned]

    # World bounding box (for overview mode limits)
    all_xyz = np.vstack([pos_d] + gouter_d)
    bx = (all_xyz[:, 0].min(), all_xyz[:, 0].max())
    by = (all_xyz[:, 1].min(), all_xyz[:, 1].max())
    bz = (all_xyz[:, 2].min(), all_xyz[:, 2].max())
    mid = np.array([(bx[0]+bx[1])/2, (by[0]+by[1])/2, (bz[0]+bz[1])/2])
    half = max(bx[1]-bx[0], by[1]-by[0], bz[1]-bz[0]) / 2 * 1.15

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 10), facecolor='#12121e')
    fig.suptitle('AI Grand Prix — Flight Replay 3D', fontsize=13, color='white',
                 fontweight='bold', y=0.98)

    gs = GridSpec(2, 3,
                  figure=fig,
                  left=0.03, right=0.97, top=0.95, bottom=0.13,
                  wspace=0.32, hspace=0.28,
                  height_ratios=[1.2, 1.0])

    ax3   = fig.add_subplot(gs[:, :2], projection='3d')
    ax_fpv = fig.add_subplot(gs[0, 2])
    ax_tel = fig.add_subplot(gs[1, 2])

    # 3D axis aesthetics
    for ax in [ax_fpv, ax_tel]:
        ax.set_facecolor('#0b0b18')
    ax3.set_facecolor('#0b0b18')
    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#2a2a44')
    ax3.tick_params(colors='#9999bb', labelsize=6)
    for lbl in [ax3.xaxis.label, ax3.yaxis.label, ax3.zaxis.label]:
        lbl.set_color('#9999bb')
    ax3.set_xlabel('X / North (m)', fontsize=7)
    ax3.set_ylabel('Y / East (m)',  fontsize=7)
    ax3.set_zlabel('Altitude (m)',  fontsize=7)
    ax3.set_title('3D World  [CAM button: follow ↔ overview]', fontsize=9, color='white', pad=6)

    # ── Static 3D elements ─────────────────────────────────────────────────────
    ax3.plot(plan_d[:, 0], plan_d[:, 1], plan_d[:, 2],
             color='#1e4488', linewidth=1.0, alpha=0.55, linestyle='--')
    ax3.scatter(*pos_d[0], color='#00ff88', s=90, marker='^', zorder=6,
                depthshade=False, label='Start')

    for i, g in enumerate(gates):
        c = centers_d[i]
        ax3.text(c[0], c[1], c[2] + 1.9, f'G{g["id"]}',
                 color='#6699cc', fontsize=7, ha='center', fontweight='bold')

    # Gate poly collections (per-gate so we can colour them individually)
    gate_outer_polys = []
    gate_inner_polys = []
    for i in range(G):
        op = Poly3DCollection([gouter_d[i].tolist()],
                              alpha=0.15, facecolor='#152233', edgecolor='#335566',
                              linewidth=1.0)
        ip = Poly3DCollection([ginner_d[i].tolist()],
                              alpha=0.25, facecolor='#0a2211', edgecolor='#228844',
                              linewidth=1.8)
        ax3.add_collection3d(op)
        ax3.add_collection3d(ip)
        gate_outer_polys.append(op)
        gate_inner_polys.append(ip)

    # ── Dynamic 3D artists ─────────────────────────────────────────────────────
    trail_ln, = ax3.plot([], [], [], color='#6688ff', linewidth=0.9, alpha=0.55)

    arm_lns = [ax3.plot([], [], [], color=c, linewidth=3.0,
                        solid_capstyle='round')[0]
               for c in _ARM_COLORS]

    drone_pt = ax3.scatter([0], [0], [0], color='#ffff00', s=55, zorder=7,
                           depthshade=False)

    tgt_pt  = ax3.scatter([np.nan], [np.nan], [np.nan],
                          color='#ff44ff', s=110, marker='x', linewidths=2.5,
                          zorder=8, depthshade=False)
    tgt_ln, = ax3.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                       color='#ff44ff', linewidth=0.8, linestyle=':', alpha=0.6)

    conf_pts = ax3.scatter([], [], [], color='#ffd700', s=130, marker='*',
                           zorder=9, depthshade=False, label='Gate confirmed')

    ax3.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', edgecolor='#334466',
               labelcolor='white')

    ax3.set_xlim3d(mid[0]-half, mid[0]+half)
    ax3.set_ylim3d(mid[1]-half, mid[1]+half)
    ax3.set_zlim3d(mid[2]-half, mid[2]+half)
    ax3.view_init(elev=28, azim=-110)

    # ── FPV panel ──────────────────────────────────────────────────────────────
    ax_fpv.set_facecolor('#000000')
    ax_fpv.set_xlim(0, IMG_W)
    ax_fpv.set_ylim(IMG_H, 0)   # y-axis: 0=top, IMG_H=bottom (image convention)
    ax_fpv.set_aspect('equal')
    ax_fpv.set_title('Simulated FPV Camera', fontsize=9, color='white', pad=4)
    ax_fpv.tick_params(colors='#667788', labelsize=6)
    for sp in ax_fpv.spines.values():
        sp.set_edgecolor('#333355')

    # Crosshair
    ax_fpv.axhline(CY, color='#224422', linewidth=0.8, alpha=0.8)
    ax_fpv.axvline(CX, color='#224422', linewidth=0.8, alpha=0.8)
    # FOV boundary
    ax_fpv.add_patch(plt.Rectangle((0, 0), IMG_W, IMG_H,
                                   fill=False, edgecolor='#2a2a3a', linewidth=1.5))

    fpv_outer = [ax_fpv.plot([], [], '-', linewidth=1.5, alpha=0.7)[0] for _ in range(G)]
    fpv_inner = [ax_fpv.plot([], [], '-', linewidth=2.2, alpha=0.95)[0] for _ in range(G)]
    fpv_gate_lbl = [ax_fpv.text(0, 0, '', color='#88bbff', fontsize=7,
                                ha='center', va='bottom', visible=False)
                    for _ in range(G)]
    fpv_cv_dot, = ax_fpv.plot([], [], 'x', color='#ff6622',
                               markersize=12, markeredgewidth=2.5)
    fpv_cv_ring, = ax_fpv.plot([], [], 'o', color='#ff6622',
                                markersize=18, markeredgewidth=1.5,
                                fillstyle='none', alpha=0.7)

    # ── Telemetry panel ────────────────────────────────────────────────────────
    ax_tel.set_facecolor('#0b0b18')
    ax_tel.axis('off')
    ax_tel.set_title('Telemetry', fontsize=9, color='white', pad=4)
    tel_txt = ax_tel.text(0.05, 0.95, '', transform=ax_tel.transAxes,
                          color='#00ee88', fontsize=9.5,
                          verticalalignment='top', fontfamily='monospace')
    err_txt = ax_tel.text(0.05, 0.05, '', transform=ax_tel.transAxes,
                          color='#ff8844', fontsize=8,
                          verticalalignment='bottom', fontfamily='monospace')

    # ── Controls ───────────────────────────────────────────────────────────────
    ax_sld = fig.add_axes([0.09, 0.055, 0.68, 0.022], facecolor='#1a1a2e')
    slider = Slider(ax_sld, '', 0.0, float(t[-1]), valinit=0.0, color='#334488')
    slider.label.set_color('white')
    slider.valtext.set_color('#aaaacc')
    slider.valtext.set_fontsize(8)

    def _btn(rect, label, fn):
        bax = fig.add_axes(rect, facecolor='#222238')
        b = Button(bax, label, color='#222238', hovercolor='#3a3a5a')
        b.label.set_color('white'); b.label.set_fontsize(9)
        b.on_clicked(fn)
        return b

    state = {
        'frame': 0, 'playing': True, 'speed_idx': 2,
        'follow': True, 'sim_time': 0.0,
        'last_wall': time.time(), 'lock_slider': False,
    }

    spd_ax = fig.add_axes([0.52, 0.01, 0.055, 0.033])
    spd_ax.axis('off')
    spd_lbl = spd_ax.text(0.5, 0.5, '1.0×', ha='center', va='center',
                          color='#ccccff', fontsize=11, fontfamily='monospace',
                          transform=spd_ax.transAxes)

    def _set_speed(idx):
        state['speed_idx'] = max(0, min(len(SPEEDS)-1, idx))
        spd_lbl.set_text(f'{SPEEDS[state["speed_idx"]]:.2g}×')

    def _step_back(_):
        fi = max(0, state['frame'] - 1)
        state.update({'playing': False, 'frame': fi, 'sim_time': float(t[fi])})

    def _step_fwd(_):
        fi = min(N - 1, state['frame'] + 1)
        state.update({'playing': False, 'frame': fi, 'sim_time': float(t[fi])})

    def _toggle_play(_):
        state['playing']   = not state['playing']
        state['last_wall'] = time.time()

    def _toggle_cam(_):
        state['follow'] = not state['follow']

    _btn([0.09, 0.01, 0.06, 0.033],  '◀◀ slower', lambda _: _set_speed(state['speed_idx'] - 1))
    _btn([0.16, 0.01, 0.055, 0.033], '◀ step',    _step_back)
    _btn([0.22, 0.01, 0.06, 0.033],  '▶/⏸',       _toggle_play)
    _btn([0.29, 0.01, 0.055, 0.033], 'step ▶',    _step_fwd)
    _btn([0.35, 0.01, 0.06, 0.033],  'faster ▶▶', lambda _: _set_speed(state['speed_idx'] + 1))
    _btn([0.60, 0.01, 0.065, 0.033], 'CAM MODE',  _toggle_cam)
    _btn([0.79, 0.01, 0.075, 0.033], '💾 MP4', lambda _: _trigger_save())
    _btn([0.88, 0.01, 0.06, 0.033], 'Restart',
         lambda _: state.update({'frame': 0, 'sim_time': 0.0,
                                 'last_wall': time.time(), 'playing': True}))

    def _trigger_save():
        state['playing'] = False
        print('[replay3d] Rendering MP4 (this will take a while) ...')
        out = os.path.join(run_dir, f'{stem}_replay.mp4')
        try:
            writer = FFMpegWriter(fps=20, bitrate=3000,
                                  metadata={'title': 'AI Grand Prix Replay'})
            save_anim = FuncAnimation(fig, update, frames=N,
                                      cache_frame_data=False, blit=False)
            state['render_mode'] = 'save'
            save_anim.save(out, writer=writer, dpi=120)
            state.pop('render_mode', None)
            print(f'[replay3d] Saved → {out}')
        except Exception as exc:
            state.pop('render_mode', None)
            print(f'[replay3d] MP4 save failed: {exc}')
            print('[replay3d] Ensure ffmpeg is on PATH. Install via: winget install ffmpeg')

    def _on_slider(val):
        if state['lock_slider']:
            return
        state['playing']  = False
        state['sim_time'] = float(val)
        state['frame']    = int(np.clip(np.searchsorted(t, val)-1, 0, N-1))

    slider.on_changed(_on_slider)

    # ── Per-frame update ───────────────────────────────────────────────────────
    def update(tick):
        # Determine current frame index
        if state.get('render_mode') == 'save':
            fi = int(tick)
            state['frame'] = fi
        elif state['playing']:
            now = time.time()
            state['sim_time'] += (now - state['last_wall']) * SPEEDS[state['speed_idx']]
            state['last_wall'] = now
            state['sim_time']  = float(np.clip(state['sim_time'], 0.0, t[-1]))
            fi = int(np.clip(np.searchsorted(t, state['sim_time'])-1, 0, N-1))
            state['frame'] = fi
            if state['sim_time'] >= t[-1]:
                state['playing'] = False
        else:
            state['last_wall'] = time.time()
            fi = state['frame']

        # Update scrubber without re-triggering callback
        state['lock_slider'] = True
        slider.set_val(t[fi])
        state['lock_slider'] = False

        p   = pos[fi]           # NED
        pd  = pos_d[fi]         # display
        roll, pitch, yaw = att[fi]
        speed  = float(np.linalg.norm(vel[fi]))
        active = int(ag[fi])

        # Trail
        lo = max(0, fi - TRAIL_LEN)
        trail_ln.set_data_3d(pos_d[lo:fi+1, 0], pos_d[lo:fi+1, 1], pos_d[lo:fi+1, 2])

        # Drone body (arms)
        R_b2ned = _euler_zyx(roll, pitch, yaw)
        tips_ned = p + (R_b2ned @ _ARMS_FRD.T).T      # (4, 3) NED
        tips_d   = ned2d(tips_ned)
        for k, ln in enumerate(arm_lns):
            ln.set_data_3d([pd[0], tips_d[k, 0]],
                           [pd[1], tips_d[k, 1]],
                           [pd[2], tips_d[k, 2]])
        drone_pt._offsets3d = ([pd[0]], [pd[1]], [pd[2]])

        # Controller target
        if len(ctrl_t) > 0:
            ci = int(np.clip(np.searchsorted(ctrl_t, t[fi])-1, 0, len(ctrl_t)-1))
            tgt_ned = ctrl_tgt[ci]
            if not np.any(np.isnan(tgt_ned)):
                td = ned2d(tgt_ned)
                tgt_pt._offsets3d = ([td[0]], [td[1]], [td[2]])
                tgt_ln.set_data_3d([pd[0], td[0]], [pd[1], td[1]], [pd[2], td[2]])
                mode_str = ctrl_mode[ci] if ctrl_mode else ''
            else:
                tgt_pt._offsets3d = ([np.nan], [np.nan], [np.nan])
                tgt_ln.set_data_3d([np.nan], [np.nan], [np.nan])
                mode_str = ctrl_mode[ci] if ctrl_mode else ''
        else:
            mode_str = ''

        # Gate colours
        for i in range(G):
            if i < active:
                gate_outer_polys[i].set_facecolor('#0d0d1e'); gate_outer_polys[i].set_edgecolor('#222244')
                gate_outer_polys[i].set_alpha(0.08)
                gate_inner_polys[i].set_facecolor('#0a0d0a'); gate_inner_polys[i].set_edgecolor('#1a3322')
                gate_inner_polys[i].set_alpha(0.10)
            elif i == active or i == active + 1:
                gate_outer_polys[i].set_facecolor('#0d2244'); gate_outer_polys[i].set_edgecolor('#3377ff')
                gate_outer_polys[i].set_alpha(0.30)
                gate_inner_polys[i].set_facecolor('#001a0d'); gate_inner_polys[i].set_edgecolor('#00ff55')
                gate_inner_polys[i].set_alpha(0.55)
            else:
                gate_outer_polys[i].set_facecolor('#0d1a2a'); gate_outer_polys[i].set_edgecolor('#224455')
                gate_outer_polys[i].set_alpha(0.14)
                gate_inner_polys[i].set_facecolor('#0a110a'); gate_inner_polys[i].set_edgecolor('#1a4422')
                gate_inner_polys[i].set_alpha(0.18)

        # Gate confirmation stars
        conf_idx = np.where(np.diff(ag[:fi+1]) > 0)[0]
        if len(conf_idx):
            cpts = pos_d[conf_idx + 1]
            conf_pts._offsets3d = (cpts[:, 0].tolist(), cpts[:, 1].tolist(), cpts[:, 2].tolist())
        else:
            conf_pts._offsets3d = ([], [], [])

        # Camera follow / overview
        if state['follow']:
            ax3.set_xlim3d(pd[0] - VIEW_RADIUS, pd[0] + VIEW_RADIUS)
            ax3.set_ylim3d(pd[1] - VIEW_RADIUS, pd[1] + VIEW_RADIUS)
            ax3.set_zlim3d(pd[2] - VIEW_RADIUS * 0.4, pd[2] + VIEW_RADIUS * 0.4)
        else:
            ax3.set_xlim3d(mid[0]-half, mid[0]+half)
            ax3.set_ylim3d(mid[1]-half, mid[1]+half)
            ax3.set_zlim3d(mid[2]-half, mid[2]+half)

        # ── FPV projection ─────────────────────────────────────────────────────
        for i in range(G):
            if i < active:
                ocol, icol = '#1e1e33', '#112211'
            elif i == active or i == active + 1:
                ocol, icol = '#3366ff', '#00ff55'
            else:
                ocol, icol = '#224466', '#1a4422'

            def _draw_poly(corners_ned, line, col):
                # project all 4 corners; only draw if gate center is in front
                cu, cv, cv_ok = project_to_fov(centers_ned[i:i+1], p, roll, pitch, yaw)
                if not cv_ok[0]:
                    line.set_xdata([]); line.set_ydata([]); return
                u, v, _ = project_to_fov(corners_ned, p, roll, pitch, yaw)
                idx = [0, 1, 2, 3, 0]  # close the polygon
                line.set_xdata([u[j] for j in idx])
                line.set_ydata([v[j] for j in idx])
                line.set_color(col)

            _draw_poly(gouter_ned[i], fpv_outer[i], ocol)
            _draw_poly(ginner_ned[i], fpv_inner[i], icol)

            # Gate label in FPV
            cu, cv, cv_ok = project_to_fov(centers_ned[i:i+1], p, roll, pitch, yaw)
            if cv_ok[0] and 0 < cu[0] < IMG_W and 0 < cv[0] < IMG_H:
                fpv_gate_lbl[i].set_position((cu[0], cv[0] - 12))
                fpv_gate_lbl[i].set_text(f'G{gates[i]["id"]}')
                fpv_gate_lbl[i].set_color('#5599ff' if i == active else '#336655')
                fpv_gate_lbl[i].set_visible(True)
            else:
                fpv_gate_lbl[i].set_visible(False)

        # CV estimate overlay in FPV
        if len(est_t) > 0:
            ei = int(np.clip(np.searchsorted(est_t, t[fi])-1, 0, len(est_t)-1))
            if abs(est_t[ei] - t[fi]) < CV_WINDOW:
                eu, ev, ev_ok = project_to_fov(est_pos[ei:ei+1], p, roll, pitch, yaw)
                if ev_ok[0]:
                    fpv_cv_dot.set_xdata([eu[0]]); fpv_cv_dot.set_ydata([ev[0]])
                    fpv_cv_ring.set_xdata([eu[0]]); fpv_cv_ring.set_ydata([ev[0]])
                else:
                    fpv_cv_dot.set_xdata([]); fpv_cv_dot.set_ydata([])
                    fpv_cv_ring.set_xdata([]); fpv_cv_ring.set_ydata([])
            else:
                fpv_cv_dot.set_xdata([]); fpv_cv_dot.set_ydata([])
                fpv_cv_ring.set_xdata([]); fpv_cv_ring.set_ydata([])

        # ── Telemetry ──────────────────────────────────────────────────────────
        alt = -p[2]
        play_icon = '▶' if state['playing'] else '⏸'
        cam_icon  = 'FOL' if state['follow'] else 'OVR'
        tel = (
            f'T       {t[fi]:6.1f} s\n'
            f'Speed   {speed:5.1f} m/s\n'
            f'Alt     {alt:5.1f} m AGL\n'
            f'Pos N   {p[0]:+7.2f} m\n'
            f'Pos E   {p[1]:+7.2f} m\n'
            f'Roll    {np.degrees(roll):+6.1f}°\n'
            f'Pitch   {np.degrees(pitch):+6.1f}°\n'
            f'Yaw     {np.degrees(yaw):+6.1f}°\n'
            f'Gate    {max(0,active)} / {G-1}\n'
            f'Mode    {mode_str or "—"}\n'
            f'{play_icon} {SPEEDS[state["speed_idx"]]:.2g}×  CAM:{cam_icon}'
        )
        tel_txt.set_text(tel)

        # CV error summary in telemetry
        if len(est_t) > 0:
            ei = int(np.clip(np.searchsorted(est_t, t[fi])-1, 0, len(est_t)-1))
            if abs(est_t[ei] - t[fi]) < CV_WINDOW:
                row_idx = ei
                # est_pos vs MAVLink gate center
                mav_g = int(np.clip(active, 0, G-1))
                mav_c = centers_ned[mav_g]
                err   = est_pos[row_idx] - mav_c
                err_txt.set_text(
                    f'CV vs MAVLink gate {mav_g}:\n'
                    f'  ΔX={err[0]:+.2f} ΔY={err[1]:+.2f} ΔZ={err[2]:+.2f}\n'
                    f'  |err|={np.linalg.norm(err):.2f} m'
                )
            else:
                err_txt.set_text('CV: no estimate at this time')
        else:
            err_txt.set_text('(no CV estimates log)')

        return (trail_ln, *arm_lns, drone_pt, tgt_pt, tgt_ln, conf_pts,
                *fpv_outer, *fpv_inner, *fpv_gate_lbl, fpv_cv_dot, fpv_cv_ring,
                tel_txt, err_txt)

    # ── Render or display ──────────────────────────────────────────────────────
    if do_save:
        out_path = os.path.join(run_dir, f'{stem}_replay.mp4')
        print(f'[replay3d] Saving {N} frames → {out_path}')
        state['render_mode'] = 'save'
        writer = FFMpegWriter(fps=20, bitrate=3000)
        anim = FuncAnimation(fig, update, frames=N, cache_frame_data=False, blit=False)
        anim.save(out_path, writer=writer, dpi=120)
        print(f'[replay3d] Done → {out_path}')
    else:
        anim = FuncAnimation(fig, update, interval=40, cache_frame_data=False, blit=False)
        plt.show()


if __name__ == '__main__':
    main()
