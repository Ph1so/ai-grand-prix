"""
VQ2 Interactive 3D flight replay.

VQ2 differences from VQ1:
  - No gate positions from JSON (nulled by sim) → gates shown as CV-estimated clouds
  - No real NED position → trajectory is dead-reckoned from actual_vx/vy/vz
  - No roll/pitch → FPV projection uses yaw-only rotation (roll=pitch=0)
  - All data from controller_log.csv + gate_estimates_*.csv in the run directory

What it shows:
  3D world view  : dead-reckoned trail, drone heading arrow, CV gate estimate clouds,
                   controller target marker, gate confirmation events
  FPV camera     : simulated onboard view projecting CV gate means + current CV estimate
  Telemetry      : speed, yaw, yaw_err, thrust, target mode, state
  Controls       : play/pause, 0.25x–8x speed, frame-step, time scrubber, Save MP4

Usage (from VQ2/debug/ or anywhere):
  python replay3d.py                       # auto-picks latest run
  python replay3d.py run_20260629_120000   # explicit run dir name
  python replay3d.py /full/path/to/run_dir

Flags:
  --save           headless render to <run_dir>/flight_replay_<speed>x.mp4
  --speed <mult>   playback multiplier for exported video (default 1.0)
"""

import csv
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Camera constants (same as sim/pose_estimator) ────────────────────────────
import math
_CAM_TILT = math.radians(-21.6)
_FX = _FY  = 320.0
_CX, _CY   = 320.0, 180.0
_IMG_W, _IMG_H = 640, 360
_DRONE_ARM     = 0.14      # m
# Camera → body rotation matrix (copy of pose_estimator.R_CAM2BODY)
_R_CAM2BODY = np.array([
    [0.,              -math.sin(_CAM_TILT),  math.cos(_CAM_TILT)],
    [1.,               0.,                   0.                  ],
    [0.,               math.cos(_CAM_TILT),  math.sin(_CAM_TILT)],
], dtype=float)

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'PyAIPilotExample-v2', 'logs'))

SPEEDS     = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CV_WINDOW  = 0.25          # s — show CV estimate marker if within this age
VIEW_RADIUS = 25.0         # m — follow-mode half-window

# FRD arm tips for drone body visualisation (forward = +X, coloured red)
_ARMS_FRD = np.array([
    [ _DRONE_ARM,  0.,  0.],
    [-_DRONE_ARM,  0.,  0.],
    [ 0.,  _DRONE_ARM,  0.],
    [ 0., -_DRONE_ARM,  0.],
], dtype=float)
_ARM_COLORS = ['#ff4444', '#555566', '#ccccdd', '#ccccdd']


# ── Helpers ───────────────────────────────────────────────────────────────────

def ned2d(p):
    p = np.asarray(p, dtype=float).copy()
    p[..., 2] = -p[..., 2]
    return p


def euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [   -sp,            cp*sr,            cp*cr  ],
    ], dtype=float)


def project_to_fov(pts_ned, drone_pos, yaw):
    """
    Project NED world points onto camera pixel coordinates.
    VQ2: roll=pitch=0, yaw=integrated.
    Returns (u, v, valid_mask).
    """
    R_b2ned = euler_zyx(0.0, 0.0, yaw)
    pts = np.atleast_2d(pts_ned)
    body = (R_b2ned.T @ (pts - drone_pos).T).T
    cam  = (_R_CAM2BODY.T @ body.T).T
    valid  = cam[:, 2] > 0.1
    safe_z = np.where(valid, cam[:, 2], 1.0)
    u = _FX * cam[:, 0] / safe_z + _CX
    v = _FY * cam[:, 1] / safe_z + _CY
    return u, v, valid


# ── File discovery ────────────────────────────────────────────────────────────

def find_latest_run() -> str | None:
    for d in sorted(glob.glob(os.path.join(_LOGS_DIR, 'run_*')), reverse=True):
        if os.path.isfile(os.path.join(d, 'controller_log.csv')):
            return d
    return None


def resolve_run_dir(argv):
    save_mp4  = '--save' in argv
    cli_speed = 1.0
    consumed  = set()
    for i, a in enumerate(argv):
        if a == '--speed' and i + 1 < len(argv):
            try:   cli_speed = float(argv[i + 1])
            except ValueError: pass
            consumed.update({i, i + 1})
        elif a.startswith('--speed='):
            try:   cli_speed = float(a.split('=', 1)[1])
            except ValueError: pass
            consumed.add(i)
    args = [a for i, a in enumerate(argv[1:], start=1)
            if i not in consumed and not a.startswith('--')]
    if not args:
        d = find_latest_run()
        if d is None:
            print(f'No run_* dir with controller_log.csv in {_LOGS_DIR}')
            sys.exit(1)
        return d, save_mp4, cli_speed
    arg = args[0]
    for candidate in [arg, os.path.join(_LOGS_DIR, arg)]:
        if os.path.isdir(candidate):
            return candidate, save_mp4, cli_speed
    print(f'Directory not found: {arg}')
    sys.exit(1)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_csv(path, str_cols=frozenset()):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if k in str_cols:
                    parsed[k] = v.strip()
                else:
                    try:   parsed[k] = float(v)
                    except (ValueError, TypeError): parsed[k] = float('nan')
            rows.append(parsed)
    return rows


def load_run(run_dir: str):
    ctrl_path = os.path.join(run_dir, 'controller_log.csv')
    ctrl = _load_csv(ctrl_path, str_cols={'state', 'target_mode'})

    est  = []
    for p in sorted(glob.glob(os.path.join(run_dir, 'gate_estimates_*.csv')), reverse=True)[:1]:
        est = _load_csv(p)

    return ctrl, est


def dead_reckon(ctrl: list[dict]) -> np.ndarray:
    n   = len(ctrl)
    pos = np.zeros((n, 3))
    p   = np.array([0.0, 0.0, -0.5])
    pos[0] = p
    for i in range(1, n):
        dt  = float(np.clip(ctrl[i]['time_s'] - ctrl[i-1]['time_s'], 0.0, 0.5))
        vel = np.array([ctrl[i]['actual_vx'], ctrl[i]['actual_vy'], ctrl[i]['actual_vz']])
        if not np.any(np.isnan(vel)):
            p = p + vel * dt
        pos[i] = p
    return pos


def group_cv_estimates(est: list[dict], gap_s: float = 0.8):
    """Cluster CV estimates by time gap → list of (N,3) NED arrays."""
    if not est:
        return []
    times  = np.array([r['sim_time_ns'] * 1e-9 for r in est])
    pts    = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in est])
    et     = np.array([r['sim_time_ns'] * 1e-9 for r in est])
    gaps   = np.concatenate([[np.inf], np.diff(times)])
    labels = np.cumsum(gaps > gap_s)
    groups = []
    for gid in range(int(labels.max()) + 1):
        mask = labels == gid
        if mask.sum() >= 3:
            groups.append((pts[mask], et[mask]))
    return groups


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    run_dir, do_save, cli_speed = resolve_run_dir(sys.argv)
    print(f'[replay3d] Run dir : {run_dir}')

    ctrl, est = load_run(run_dir)
    N = len(ctrl)
    if N == 0:
        print('[replay3d] controller_log.csv is empty')
        sys.exit(1)

    t0  = ctrl[0]['time_s']
    t   = np.array([r['time_s'] - t0 for r in ctrl])
    pos = dead_reckon(ctrl)
    yaw = np.array([r['yaw'] for r in ctrl])
    vel = np.array([[r['actual_vx'], r['actual_vy'], r['actual_vz']] for r in ctrl])
    tgt = np.array([[r['target_x'], r['target_y'], r['target_z']] for r in ctrl])
    thr = np.array([r['cmd_thrust'] for r in ctrl])
    zin = np.array([r['z_integral'] for r in ctrl])
    sat = np.array([r['saturated'] for r in ctrl])
    states = [r.get('state', '') for r in ctrl]
    modes  = [r.get('target_mode', '') for r in ctrl]
    wp_idx = np.array([r['waypoint_idx'] for r in ctrl])

    pos_d = ned2d(pos)
    tgt_d = ned2d(tgt)

    # ── CV data ────────────────────────────────────────────────────────────────
    groups = group_cv_estimates(est)
    G = len(groups)
    gate_means_ned = (np.array([g.mean(axis=0) for g, _ in groups]) if G > 0
                      else np.zeros((0, 3)))
    gate_means_d   = ned2d(gate_means_ned) if G > 0 else np.zeros((0, 3))

    # All CV estimate positions and their relative times (aligned to controller log start by
    # best-effort offset: first estimate time ≈ first FLY log row time)
    all_est_ned   = np.vstack([g for g, _ in groups]) if G > 0 else np.zeros((0, 3))
    all_est_times = np.concatenate([et for _, et in groups]) if G > 0 else np.array([])
    # Shift est times so they align with wall time: assume first est ≈ first FLY row time
    fly_rows = [i for i, s in enumerate(states) if s == 'FLY']
    if fly_rows and len(all_est_times) > 0:
        fly_t0_wall = t[fly_rows[0]]
        fly_t0_sim  = all_est_times[0]
        est_wall_offset = fly_t0_wall - fly_t0_sim
    else:
        est_wall_offset = 0.0
    all_est_wall = all_est_times + est_wall_offset

    print(f'[replay3d] {N} ctrl rows  |  {G} CV gate groups  '
          f'|  {len(all_est_ned)} total estimates  |  {t[-1]:.1f}s duration')

    # ── World bounding box ─────────────────────────────────────────────────────
    all_xyz = np.vstack([pos_d, tgt_d[~np.any(np.isnan(tgt_d), axis=1)]])
    if G > 0:
        all_xyz = np.vstack([all_xyz, ned2d(all_est_ned)])
    bx = (all_xyz[:, 0].min(), all_xyz[:, 0].max())
    by = (all_xyz[:, 1].min(), all_xyz[:, 1].max())
    bz = (all_xyz[:, 2].min(), all_xyz[:, 2].max())
    mid  = np.array([(bx[0]+bx[1])/2, (by[0]+by[1])/2, (bz[0]+bz[1])/2])
    half = max(bx[1]-bx[0], by[1]-by[0], bz[1]-bz[0]) / 2 * 1.2 + 5.0

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10), facecolor='#12121e')
    fig.suptitle(f'VQ2 Replay 3D — {os.path.basename(run_dir)}',
                 fontsize=13, color='white', fontweight='bold', y=0.98)

    gs = GridSpec(2, 2, figure=fig,
                  left=0.015, right=0.99, top=0.965, bottom=0.10,
                  wspace=0.08, hspace=0.12,
                  width_ratios=[1.0, 1.05], height_ratios=[2.8, 0.85])

    ax3    = fig.add_subplot(gs[:, 0], projection='3d')
    ax_fpv = fig.add_subplot(gs[0, 1])
    ax_tel = fig.add_subplot(gs[1, 1])

    # Style
    for ax in [ax_fpv, ax_tel]:
        ax.set_facecolor('#0b0b18')
    ax3.set_facecolor('#0b0b18')
    ax3.grid(False)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('#2a2a44')
        axis._axinfo['grid']['color'] = (0, 0, 0, 0)
    ax3.tick_params(colors='#9999bb', labelsize=6)
    for lbl in [ax3.xaxis.label, ax3.yaxis.label, ax3.zaxis.label]:
        lbl.set_color('#9999bb')
    ax3.set_xlabel('X / North (m)', fontsize=7)
    ax3.set_ylabel('Y / East (m)',  fontsize=7)
    ax3.set_zlabel('Altitude (m)',  fontsize=7)
    ax3.set_title('3D World  [CAM: follow ↔ overview | scroll: zoom]',
                  fontsize=9, color='white', pad=6)

    # ── Static 3D elements ─────────────────────────────────────────────────────
    # Full planned target path (static background)
    valid_tgt = ~np.any(np.isnan(tgt_d), axis=1)
    ax3.plot(tgt_d[valid_tgt, 0], tgt_d[valid_tgt, 1], tgt_d[valid_tgt, 2],
             color='#1e3a77', linewidth=0.8, alpha=0.45, linestyle='--')

    # CV gate estimate clouds (static scatter, coloured per group)
    _GATE_COLORS = ['#ff6633', '#33cc66', '#3399ff', '#ffcc00',
                    '#cc66ff', '#ff3399', '#00cccc', '#ff9900']
    for i, (g, _) in enumerate(groups):
        gd  = ned2d(g)
        col = _GATE_COLORS[i % len(_GATE_COLORS)]
        ax3.scatter(gd[:, 0], gd[:, 1], gd[:, 2],
                    color=col, s=4, alpha=0.25, zorder=2)
        md = gate_means_d[i]
        ax3.scatter(md[0], md[1], md[2], color=col, s=90, marker='D',
                    edgecolors='white', linewidths=0.6, zorder=6, depthshade=False)
        ax3.text(md[0], md[1], md[2] + 2.0, f'G{i}',
                 color=col, fontsize=7, ha='center', fontweight='bold')

    ax3.scatter(*pos_d[0], color='#00ff88', s=90, marker='^', zorder=7,
                depthshade=False, label='Start')
    ax3.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
               edgecolor='#334466', labelcolor='white')
    ax3.set_xlim3d(mid[0]-half, mid[0]+half)
    ax3.set_ylim3d(mid[1]-half, mid[1]+half)
    ax3.set_zlim3d(mid[2]-half, mid[2]+half)
    ax3.view_init(elev=28, azim=-110)

    # ── Dynamic 3D artists ─────────────────────────────────────────────────────
    trail_ln, = ax3.plot([], [], [], color='#6688ff', linewidth=1.0, alpha=0.55)
    arm_lns   = [ax3.plot([], [], [], color=c, linewidth=3.0,
                          solid_capstyle='round')[0] for c in _ARM_COLORS]
    drone_pt  = ax3.scatter([0], [0], [0], color='#ffff00', s=60, zorder=8, depthshade=False)
    tgt_pt    = ax3.scatter([np.nan], [np.nan], [np.nan],
                            color='#ff44ff', s=120, marker='x', linewidths=2.5,
                            zorder=9, depthshade=False)
    tgt_ln,   = ax3.plot([np.nan]*2, [np.nan]*2, [np.nan]*2,
                          color='#ff44ff', linewidth=0.8, linestyle=':', alpha=0.6)
    conf_pts  = ax3.scatter([], [], [], color='#ffd700', s=130, marker='*',
                             zorder=10, depthshade=False)

    # ── FPV panel ──────────────────────────────────────────────────────────────
    ax_fpv.set_facecolor('#000000')
    ax_fpv.set_xlim(0, _IMG_W); ax_fpv.set_ylim(_IMG_H, 0)
    ax_fpv.set_aspect('equal')
    ax_fpv.set_title('Simulated FPV Camera  (CV gate means + current estimate)',
                     fontsize=9, color='white', pad=4)
    ax_fpv.tick_params(colors='#667788', labelsize=6)
    for sp in ax_fpv.spines.values(): sp.set_edgecolor('#333355')
    ax_fpv.axhline(_CY, color='#224422', linewidth=0.8, alpha=0.7)
    ax_fpv.axvline(_CX, color='#224422', linewidth=0.8, alpha=0.7)
    ax_fpv.add_patch(plt.Rectangle((0, 0), _IMG_W, _IMG_H,
                                   fill=False, edgecolor='#2a2a3a', linewidth=1.5))

    # Gate-mean markers in FPV (one per gate group)
    fpv_gate_dots = [ax_fpv.plot([], [], 'D', color=_GATE_COLORS[i % len(_GATE_COLORS)],
                                 markersize=8, markeredgecolor='white',
                                 markeredgewidth=0.8)[0] for i in range(G)]
    fpv_gate_lbls = [ax_fpv.text(0, 0, '', color=_GATE_COLORS[i % len(_GATE_COLORS)],
                                 fontsize=8, ha='center', va='bottom', visible=False)
                     for i in range(G)]
    fpv_cv_dot, = ax_fpv.plot([], [], 'x', color='#ff6622', markersize=14, markeredgewidth=2.5)
    fpv_cv_ring,= ax_fpv.plot([], [], 'o', color='#ff6622', markersize=20,
                               markeredgewidth=1.5, fillstyle='none', alpha=0.7)

    # ── Telemetry panel ────────────────────────────────────────────────────────
    ax_tel.set_facecolor('#0b0b18')
    ax_tel.axis('off')
    ax_tel.set_title('Telemetry', fontsize=9, color='white', pad=4)
    tel_c1 = ax_tel.text(0.03, 0.90, '', transform=ax_tel.transAxes,
                         color='#00ee88', fontsize=9.5, va='top', fontfamily='monospace')
    tel_c2 = ax_tel.text(0.36, 0.90, '', transform=ax_tel.transAxes,
                         color='#00ee88', fontsize=9.5, va='top', fontfamily='monospace')
    tel_c3 = ax_tel.text(0.70, 0.90, '', transform=ax_tel.transAxes,
                         color='#aaaacc', fontsize=9, va='top', fontfamily='monospace')
    tel_cv = ax_tel.text(0.70, 0.55, '', transform=ax_tel.transAxes,
                         color='#ff8844', fontsize=8, va='top', fontfamily='monospace')

    # ── Controls ───────────────────────────────────────────────────────────────
    ax_sld = fig.add_axes([0.09, 0.052, 0.67, 0.022], facecolor='#1a1a2e')
    slider = Slider(ax_sld, '', 0.0, float(t[-1]), valinit=0.0, color='#334488')
    slider.label.set_color('white')
    slider.valtext.set_color('#aaaacc'); slider.valtext.set_fontsize(8)
    _widgets = [slider]

    def _btn(rect, label, fn):
        bax = fig.add_axes(rect, facecolor='#222238')
        b   = Button(bax, label, color='#222238', hovercolor='#3a3a5a')
        b.label.set_color('white'); b.label.set_fontsize(8)
        b.on_clicked(fn); _widgets.append(b); return b

    state = {
        'frame': 0, 'playing': True, 'speed_idx': 2,
        'follow': True, 'sim_time': 0.0,
        'last_wall': time.time(), 'lock_slider': False, 'zoom': 1.0,
    }

    spd_ax = fig.add_axes([0.50, 0.01, 0.06, 0.033]); spd_ax.axis('off')
    spd_lbl = spd_ax.text(0.5, 0.5, '1.0x', ha='center', va='center',
                          color='#ccccff', fontsize=11, fontfamily='monospace',
                          transform=spd_ax.transAxes)

    def _set_speed(idx):
        state['speed_idx'] = max(0, min(len(SPEEDS)-1, idx))
        spd_lbl.set_text(f'{SPEEDS[state["speed_idx"]]:.2g}x')

    def _scroll(ev):
        if ev.inaxes is not ax3: return
        state['zoom'] = float(np.clip(state['zoom'] * (0.85 if ev.button == 'up' else 1/0.85), 0.05, 6.0))
    fig.canvas.mpl_connect('scroll_event', _scroll)

    _btn([0.09, 0.01, 0.055, 0.033], 'Slower',    lambda _: _set_speed(state['speed_idx'] - 1))
    _btn([0.15, 0.01, 0.055, 0.033], '< Step',    lambda _: state.update({'playing': False, 'frame': max(0, state['frame']-1), 'sim_time': float(t[max(0, state['frame']-1)])}))
    _btn([0.21, 0.01, 0.06,  0.033], 'Play/Paus', lambda _: state.update({'playing': not state['playing'], 'last_wall': time.time()}))
    _btn([0.28, 0.01, 0.055, 0.033], 'Step >',    lambda _: state.update({'playing': False, 'frame': min(N-1, state['frame']+1), 'sim_time': float(t[min(N-1, state['frame']+1)])}))
    _btn([0.34, 0.01, 0.055, 0.033], 'Faster',    lambda _: _set_speed(state['speed_idx'] + 1))
    _btn([0.58, 0.01, 0.065, 0.033], 'Cam Mode',  lambda _: state.update({'follow': not state['follow']}))
    _btn([0.78, 0.01, 0.075, 0.033], 'Save MP4',  lambda _: _export_mp4())
    _btn([0.87, 0.01, 0.06,  0.033], 'Restart',   lambda _: state.update({'frame': 0, 'sim_time': 0.0, 'playing': True, 'last_wall': time.time()}))

    stem = 'flight'

    def _export_mp4(speed=None, fps=24, progress_every=40):
        if state.get('saving'):
            print('[replay3d] Already rendering — wait for it to finish.')
            return
        spd       = float(speed) if speed is not None else SPEEDS[state['speed_idx']]
        was_play  = state['playing']
        state['saving'] = True; state['playing'] = False
        live = state.get('anim')
        if live: live.event_source.stop()

        n_frames = max(1, int(np.ceil(t[-1] / spd * fps)))
        out = os.path.join(run_dir, f'{stem}_replay_{spd:g}x.mp4')
        print(f'[replay3d] Rendering {n_frames} frames @ {fps}fps → {out}')
        try:
            writer = FFMpegWriter(fps=fps, bitrate=4000,
                                  metadata={'title': 'AI Grand Prix VQ2 Replay'})
            with writer.saving(fig, out, dpi=120):
                for k in range(n_frames):
                    sim_t = min(k / fps * spd, t[-1])
                    fi = int(np.clip(np.searchsorted(t, sim_t)-1, 0, N-1))
                    state['frame'] = fi; state['sim_time'] = float(sim_t)
                    _render(fi)
                    state['lock_slider'] = True; slider.set_val(t[fi]); state['lock_slider'] = False
                    fig.canvas.draw(); writer.grab_frame()
                    if k % progress_every == 0 or k == n_frames - 1:
                        print(f'[replay3d]   frame {k+1}/{n_frames}')
            print(f'[replay3d] Saved → {out}')
        except Exception as exc:
            print(f'[replay3d] MP4 save failed: {exc}')
            print('[replay3d] Ensure ffmpeg is on PATH: winget install ffmpeg')
        finally:
            state['saving'] = False; state['playing'] = was_play; state['last_wall'] = time.time()
            if live: live.event_source.start()

    def _on_slider(val):
        if state['lock_slider']: return
        state['playing'] = False
        state['sim_time'] = float(val)
        state['frame'] = int(np.clip(np.searchsorted(t, val)-1, 0, N-1))
    slider.on_changed(_on_slider)

    # ── Per-frame renderer ────────────────────────────────────────────────────

    def _render(fi: int):
        p      = pos[fi]
        pd     = pos_d[fi]
        yw     = float(yaw[fi])
        spd    = float(np.linalg.norm(vel[fi]))
        state_ = states[fi]
        mode_  = modes[fi]

        # ── 3D: trail + drone body ──────────────────────────────────────────
        lo = max(0, fi - 300)
        trail_ln.set_data_3d(pos_d[lo:fi+1, 0], pos_d[lo:fi+1, 1], pos_d[lo:fi+1, 2])

        R_b2ned = euler_zyx(0.0, 0.0, yw)
        tips_ned = p + (R_b2ned @ _ARMS_FRD.T).T
        tips_d   = ned2d(tips_ned)
        for k, ln in enumerate(arm_lns):
            ln.set_data_3d([pd[0], tips_d[k,0]], [pd[1], tips_d[k,1]], [pd[2], tips_d[k,2]])
        drone_pt._offsets3d = ([pd[0]], [pd[1]], [pd[2]])

        # Controller target
        td_now = tgt_d[fi]
        if not np.any(np.isnan(td_now)):
            tgt_pt._offsets3d = ([td_now[0]], [td_now[1]], [td_now[2]])
            tgt_ln.set_data_3d([pd[0], td_now[0]], [pd[1], td_now[1]], [pd[2], td_now[2]])
        else:
            tgt_pt._offsets3d = ([np.nan], [np.nan], [np.nan])
            tgt_ln.set_data_3d([np.nan], [np.nan], [np.nan])

        # Gate confirmation stars (where waypoint_idx was first seen at each even→odd transition)
        past_transitions = np.where(np.diff((wp_idx[:fi+1] % 2 == 1).astype(int)) > 0)[0]
        if len(past_transitions):
            cpts = pos_d[past_transitions + 1]
            conf_pts._offsets3d = (cpts[:,0].tolist(), cpts[:,1].tolist(), cpts[:,2].tolist())
        else:
            conf_pts._offsets3d = ([], [], [])

        # Camera follow / overview
        z = state['zoom']
        if state['follow']:
            r = VIEW_RADIUS * z
            ax3.set_xlim3d(pd[0]-r, pd[0]+r)
            ax3.set_ylim3d(pd[1]-r, pd[1]+r)
            ax3.set_zlim3d(pd[2]-r*0.5, pd[2]+r*0.5)
        else:
            h = half * z
            ax3.set_xlim3d(mid[0]-h, mid[0]+h)
            ax3.set_ylim3d(mid[1]-h, mid[1]+h)
            ax3.set_zlim3d(mid[2]-h, mid[2]+h)

        # ── FPV: project gate means + current CV estimate ──────────────────
        for i in range(G):
            gm_ned = gate_means_ned[i:i+1]
            u, v, ok = project_to_fov(gm_ned, p, yw)
            if ok[0] and 0 < u[0] < _IMG_W and 0 < v[0] < _IMG_H:
                fpv_gate_dots[i].set_xdata([u[0]]); fpv_gate_dots[i].set_ydata([v[0]])
                fpv_gate_lbls[i].set_position((u[0], v[0] - 14))
                fpv_gate_lbls[i].set_text(f'G{i}')
                fpv_gate_lbls[i].set_visible(True)
            else:
                fpv_gate_dots[i].set_xdata([]); fpv_gate_dots[i].set_ydata([])
                fpv_gate_lbls[i].set_visible(False)

        # Current CV estimate (most recent within CV_WINDOW)
        cv_txt = 'CV: no estimate'
        if len(all_est_wall) > 0:
            ei = int(np.clip(np.searchsorted(all_est_wall, t[fi])-1, 0, len(all_est_wall)-1))
            if abs(all_est_wall[ei] - t[fi]) < CV_WINDOW:
                eu, ev, ev_ok = project_to_fov(all_est_ned[ei:ei+1], p, yw)
                if ev_ok[0]:
                    fpv_cv_dot.set_xdata([eu[0]]); fpv_cv_dot.set_ydata([ev[0]])
                    fpv_cv_ring.set_xdata([eu[0]]); fpv_cv_ring.set_ydata([ev[0]])
                    cv_txt = f'CV est: ({all_est_ned[ei,0]:.1f}, {all_est_ned[ei,1]:.1f}, {all_est_ned[ei,2]:.1f}) m'
                else:
                    fpv_cv_dot.set_xdata([]); fpv_cv_dot.set_ydata([])
                    fpv_cv_ring.set_xdata([]); fpv_cv_ring.set_ydata([])
            else:
                fpv_cv_dot.set_xdata([]); fpv_cv_dot.set_ydata([])
                fpv_cv_ring.set_xdata([]); fpv_cv_ring.set_ydata([])

        # ── Telemetry ─────────────────────────────────────────────────────────
        yaw_err_v = float(ctrl[fi].get('yaw_err', float('nan')))
        play_icon = '>>' if state['playing'] else '||'
        cam_icon  = 'FOL' if state['follow'] else 'OVR'
        thr_now   = float(thr[fi])
        zin_now   = float(zin[fi])
        sat_now   = int(sat[fi]) if not math.isnan(sat[fi]) else 0

        tel_c1.set_text(
            f'T        {t[fi]:6.1f} s\n'
            f'Speed    {spd:5.1f} m/s\n'
            f'Alt (DR) {-p[2]:5.1f} m\n'
            f'DR pos N {p[0]:+7.2f} m\n'
            f'DR pos E {p[1]:+7.2f} m'
        )
        tel_c2.set_text(
            f'Yaw      {math.degrees(yw):+6.1f}°\n'
            f'Yaw err  {math.degrees(yaw_err_v):+6.1f}° ({"nan" if math.isnan(yaw_err_v) else "ok"})\n'
            f'Thrust   {thr_now:.3f}  zi={zin_now:+.3f}\n'
            f'State    {state_}\n'
            f'Mode     {mode_ or "—"}'
        )
        tel_c3.set_text(f'{play_icon} {SPEEDS[state["speed_idx"]]:.2g}×  CAM:{cam_icon}'
                        + (f'\n⚠ SATURATED' if sat_now else ''))
        tel_cv.set_text(cv_txt)

        return (trail_ln, *arm_lns, drone_pt, tgt_pt, tgt_ln, conf_pts,
                *fpv_gate_dots, *fpv_gate_lbls, fpv_cv_dot, fpv_cv_ring,
                tel_c1, tel_c2, tel_c3, tel_cv)

    # ── Live playback driver ──────────────────────────────────────────────────

    def update(tick):
        if state['playing']:
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

        state['lock_slider'] = True; slider.set_val(t[fi]); state['lock_slider'] = False
        return _render(fi)

    # ── Render or display ─────────────────────────────────────────────────────

    if do_save:
        _export_mp4(speed=cli_speed, fps=24)
    else:
        anim = FuncAnimation(fig, update, interval=40, cache_frame_data=False, blit=False)
        state['anim'] = anim
        plt.show()


if __name__ == '__main__':
    main()
