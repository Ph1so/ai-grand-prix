"""
VQ2 Post-flight visualizer.

VQ2 differences from VQ1:
  - No flight_log_*.csv (LOCAL_POSITION_NED is blocked)
  - No _gates.json (gate positions are nulled by sim)
  - Drone position is FIXED at [0,0,-0.5] in controller_log — NOT a real trajectory
  - 'Drone trajectory' here is dead-reckoned by integrating actual_vx/vy/vz from controller_log
  - Gate positions are inferred from CV estimates, grouped by time-gap clustering

Usage (from VQ2/debug/ or anywhere):
  python visualize.py                       # auto-picks latest run
  python visualize.py run_20260629_120000   # run dir name inside logs/
  python visualize.py /full/path/to/run_dir

Produces three PNGs in the run directory:
  *_overview.png     — 6-panel main overview
  *_cv_gates.png     — CV estimate scatter per gate group
  *_path_clean.png   — clean target path + CV gate means
"""

import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; plt.show() still works if a display is available
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'PyAIPilotExample-v2', 'logs'))


# ── File discovery ────────────────────────────────────────────────────────────

def find_latest_run() -> str | None:
    dirs = sorted(glob.glob(os.path.join(_LOGS_DIR, 'run_*')), reverse=True)
    for d in dirs:
        if os.path.isfile(os.path.join(d, 'controller_log.csv')):
            return d
    return None


def resolve_run_dir(argv: list[str]) -> str:
    args = [a for a in argv[1:] if not a.startswith('--')]
    if not args:
        d = find_latest_run()
        if d is None:
            print(f'No run_* directory with controller_log.csv found in {_LOGS_DIR}')
            sys.exit(1)
        return d
    arg = args[0]
    for candidate in [arg, os.path.join(_LOGS_DIR, arg)]:
        if os.path.isdir(candidate):
            return candidate
    print(f'Directory not found: {arg}')
    sys.exit(1)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_csv_numeric(path: str, str_cols: set = frozenset()) -> list[dict]:
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


def load_controller_log(run_dir: str) -> list[dict] | None:
    p = os.path.join(run_dir, 'controller_log.csv')
    if not os.path.isfile(p):
        return None
    return _load_csv_numeric(p, str_cols={'state', 'target_mode'})


def load_gate_estimates(run_dir: str) -> list[dict] | None:
    paths = sorted(glob.glob(os.path.join(run_dir, 'gate_estimates_*.csv')), reverse=True)
    if not paths:
        return None
    return _load_csv_numeric(paths[0])


def load_detection_diag(run_dir: str) -> list[dict] | None:
    paths = sorted(glob.glob(os.path.join(run_dir, 'detection_diag_*.csv')), reverse=True)
    if not paths:
        return None
    return _load_csv_numeric(paths[0], str_cols={'detect_result'})


# ── Coordinate helpers ────────────────────────────────────────────────────────

def ned2d(pts: np.ndarray) -> np.ndarray:
    """NED → display coords (negate Z so altitude is positive upward)."""
    p = np.asarray(pts, dtype=float).copy()
    p[..., 2] = -p[..., 2]
    return p


def dead_reckon(ctrl: list[dict]) -> np.ndarray:
    """
    Integrate actual_vx/vy/vz from controller_log to estimate NED trajectory.
    The controller's fixed pos=[0,0,-0.5] is useless as a real trajectory, but
    the CV-derived velocity in actual_vx/vy/vz gives a reasonable dead-reckoned path
    when CV is actively detecting gates.
    """
    n = len(ctrl)
    pos = np.zeros((n, 3))
    p = np.array([0.0, 0.0, -0.5])
    pos[0] = p
    for i in range(1, n):
        dt = ctrl[i]['time_s'] - ctrl[i-1]['time_s']
        dt = float(np.clip(dt, 0.0, 0.5))  # guard against log gaps
        vel = np.array([ctrl[i]['actual_vx'], ctrl[i]['actual_vy'], ctrl[i]['actual_vz']])
        if not np.any(np.isnan(vel)):
            p = p + vel * dt
        pos[i] = p
    return pos


# ── CV gate grouping ──────────────────────────────────────────────────────────

def group_cv_estimates(est: list[dict], gap_s: float = 0.8) -> list[np.ndarray]:
    """
    Cluster CV estimates into gate groups by time gap.
    Each continuous window with no gap > gap_s is one gate group.
    Returns list of (N, 3) arrays, one per group, in NED.
    """
    if not est:
        return []
    times = np.array([r['sim_time_ns'] * 1e-9 for r in est])
    pts   = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in est])
    gaps  = np.concatenate([[np.inf], np.diff(times)])
    labels = np.cumsum(gaps > gap_s)
    groups = []
    for gid in range(int(labels.max()) + 1):
        mask = labels == gid
        if mask.sum() >= 3:
            groups.append(pts[mask])
    return groups


def cv_gate_means(groups: list[np.ndarray]) -> np.ndarray | None:
    """Return (G, 3) NED array of per-group means."""
    if not groups:
        return None
    return np.array([g.mean(axis=0) for g in groups])


# ── Drawing helpers ───────────────────────────────────────────────────────────

GATE_COLORS = [
    '#ff6633', '#33cc66', '#3399ff', '#ffcc00',
    '#cc66ff', '#ff3399', '#00cccc', '#ff9900',
]

def _gate_color(i: int) -> str:
    return GATE_COLORS[i % len(GATE_COLORS)]


def draw_cv_gates_2d(ax, groups: list[np.ndarray], xi: int, yi: int,
                     alpha_scatter: float = 0.25, label_prefix: str = 'G'):
    """Draw per-gate CV estimate clouds and means on a 2D axis."""
    means = cv_gate_means(groups)
    for i, g in enumerate(groups):
        col  = _gate_color(i)
        gd   = ned2d(g)
        ax.scatter(gd[:, xi], gd[:, yi], color=col, s=5, alpha=alpha_scatter, zorder=2)
        if means is not None:
            md = ned2d(means[i:i+1])[0]
            ax.scatter(md[xi], md[yi], color=col, s=130, marker='D',
                       edgecolors='white', linewidths=0.8, zorder=6)
            ax.text(md[xi], md[yi] + 1.5,
                    f'{label_prefix}{i}', fontsize=7, ha='center',
                    color=col, fontweight='bold', clip_on=True)


def draw_gate_confirmations_2d(ax, t: np.ndarray, ctrl_tgt_d: np.ndarray,
                                state_arr, xi: int, yi: int):
    """Mark where the controller switched waypoint (inferred gate confirmation)."""
    wp = np.array([r if isinstance(r, float) else float('nan')
                   for r in [row for row in [None]]])  # placeholder


def mark_gate_events_vline(ax, t: np.ndarray, wp_idx: np.ndarray):
    """Draw vertical lines at every even-to-odd waypoint_idx transition (gate waypoint entries)."""
    is_gate_wp = (wp_idx % 2 == 1)
    transitions = np.where(np.diff(is_gate_wp.astype(int)) > 0)[0]
    for idx in transitions:
        ax.axvline(t[idx], color='gold', linewidth=1.0, linestyle='--', alpha=0.75)


# ── Detection rate ────────────────────────────────────────────────────────────

def detection_rate_per_second(diag: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_centers, hit_rate, frames_per_bin) where hit_rate is fraction of
    frames with detect_result=='ok' per 1-second bin using relative time from first diag row.
    """
    t0 = diag[0]['sim_time_ns'] * 1e-9
    times   = np.array([r['sim_time_ns'] * 1e-9 - t0 for r in diag])
    is_ok   = np.array([r['detect_result'] == 'ok' for r in diag], dtype=float)
    dur     = times[-1] - times[0]
    if dur < 0.5:
        return np.array([0.5]), np.array([is_ok.mean()]), np.array([len(diag)])

    bins = int(np.ceil(dur))
    centers = np.arange(0.5, bins + 0.5)
    hits    = np.zeros(bins)
    counts  = np.zeros(bins)
    for i, (ti, ok) in enumerate(zip(times, is_ok)):
        b = min(int(ti), bins - 1)
        hits[b]   += ok
        counts[b] += 1.0
    with np.errstate(invalid='ignore'):
        rate = np.where(counts > 0, hits / counts, float('nan'))
    return centers, rate, counts


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(ctrl: list[dict], groups: list[np.ndarray]):
    fly = [r for r in ctrl if r.get('state') == 'FLY']
    if not fly:
        print('[viz] No FLY rows in controller_log.')
        return
    t0   = fly[0]['time_s']
    dur  = fly[-1]['time_s'] - t0
    speeds = [np.linalg.norm([r['actual_vx'], r['actual_vy'], r['actual_vz']]) for r in fly]
    speeds = [s for s in speeds if not np.isnan(s)]
    n_gates_conf = 0
    last_wp = -1
    for r in fly:
        wi = int(r.get('waypoint_idx', 0))
        if wi % 2 == 1 and wi > last_wp:
            last_wp = wi
            n_gates_conf += 1

    print(f"\n{'─'*42}")
    print(f"  FLY duration : {dur:.1f} s")
    print(f"  Avg speed    : {np.mean(speeds):.1f} m/s  Max: {np.max(speeds):.1f} m/s" if speeds else "  Speed: n/a")
    print(f"  CV gate grps : {len(groups)}")
    print(f"  WP advances  : {n_gates_conf}")
    print(f"{'─'*42}\n")


# ── Figures ───────────────────────────────────────────────────────────────────

def make_overview_figure(ctrl: list[dict], groups: list[np.ndarray],
                          dead_pos_d: np.ndarray, run_dir: str) -> plt.Figure:
    t0     = ctrl[0]['time_s']
    t      = np.array([r['time_s'] - t0 for r in ctrl])
    tgt    = np.array([[r['target_x'], r['target_y'], r['target_z']] for r in ctrl])
    vel    = np.array([[r['actual_vx'], r['actual_vy'], r['actual_vz']] for r in ctrl])
    yaw    = np.array([r['yaw'] for r in ctrl])
    tgt_yaw = np.array([r['target_yaw'] for r in ctrl])
    thrust = np.array([r['cmd_thrust'] for r in ctrl])
    zint   = np.array([r['z_integral'] for r in ctrl])
    wp_idx = np.array([r['waypoint_idx'] for r in ctrl])

    tgt_d = ned2d(tgt)
    speed = np.linalg.norm(vel, axis=1)

    # Only show FLY phase rows meaningfully
    fly_mask = np.array([r.get('state') == 'FLY' for r in ctrl])

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'VQ2 Flight Overview — {os.path.basename(run_dir)}', fontsize=12)
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.92, bottom=0.07,
                          wspace=0.38, hspace=0.42)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_sid = fig.add_subplot(gs[1, 0])
    ax_spd = fig.add_subplot(gs[0, 1])
    ax_yaw = fig.add_subplot(gs[1, 1])
    ax_thr = fig.add_subplot(gs[0, 2])
    ax_det = fig.add_subplot(gs[1, 2])

    # ── Top-down: dead-reckoned path + targets + CV gate clouds ───────────────
    for view_ax, xi, yi, xlabel, ylabel, title in [
        (ax_top, 0, 1, 'X / North (m)', 'Y / East (m)', 'Top-down  (dead-reckoned + CV gates)'),
        (ax_sid, 0, 2, 'X / North (m)', 'Altitude (m)', 'Side view  (dead-reckoned + CV gates)'),
    ]:
        # Dead-reckoned trajectory (colored by time)
        sc = view_ax.scatter(dead_pos_d[fly_mask, xi], dead_pos_d[fly_mask, yi],
                             c=t[fly_mask], cmap='plasma', s=3, zorder=4)
        view_ax.plot(dead_pos_d[fly_mask, xi], dead_pos_d[fly_mask, yi],
                     color='gray', linewidth=0.5, alpha=0.3, zorder=1)

        # Controller targets (where it was commanding toward)
        if fly_mask.sum() > 1:
            valid_tgt = ~np.any(np.isnan(tgt_d[fly_mask]), axis=1)
            view_ax.scatter(tgt_d[fly_mask][valid_tgt, xi],
                            tgt_d[fly_mask][valid_tgt, yi],
                            c=t[fly_mask][valid_tgt], cmap='cool',
                            s=2, alpha=0.45, zorder=3)

        # CV gate clouds
        draw_cv_gates_2d(view_ax, groups, xi, yi)

        # Start marker
        view_ax.scatter(dead_pos_d[0, xi], dead_pos_d[0, yi],
                        color='lime', s=100, marker='^', zorder=7, label='Start')

        view_ax.set_xlabel(xlabel, fontsize=8)
        view_ax.set_ylabel(ylabel, fontsize=8)
        view_ax.set_title(title, fontsize=9)
        view_ax.set_aspect('equal', adjustable='datalim')
        view_ax.grid(True, alpha=0.22)

    fig.colorbar(sc, ax=[ax_top, ax_sid], label='Elapsed time (s)', shrink=0.55, pad=0.01)

    # ── Speed over time ────────────────────────────────────────────────────────
    ax_spd.plot(t[fly_mask], speed[fly_mask], color='tomato', linewidth=0.9)
    mark_gate_events_vline(ax_spd, t[fly_mask], wp_idx[fly_mask])
    ax_spd.set_xlabel('Time (s)', fontsize=8)
    ax_spd.set_ylabel('Speed (m/s)', fontsize=8)
    ax_spd.set_title('Speed  (gold = gate WP entered)', fontsize=9)
    ax_spd.grid(True, alpha=0.22)

    # ── Yaw tracking ──────────────────────────────────────────────────────────
    yaw_d    = np.degrees(yaw)
    tgt_yaw_d = np.degrees(tgt_yaw)
    ax_yaw.plot(t[fly_mask], yaw_d[fly_mask], color='#44aaff', linewidth=0.9, label='yaw (IMU)')
    valid_ty = fly_mask & ~np.isnan(tgt_yaw)
    ax_yaw.plot(t[valid_ty], tgt_yaw_d[valid_ty], color='orange', linewidth=0.9,
                linestyle='--', label='target_yaw', alpha=0.8)
    mark_gate_events_vline(ax_yaw, t[fly_mask], wp_idx[fly_mask])
    ax_yaw.set_xlabel('Time (s)', fontsize=8)
    ax_yaw.set_ylabel('Yaw (°)', fontsize=8)
    ax_yaw.set_title('Yaw tracking', fontsize=9)
    ax_yaw.legend(fontsize=7)
    ax_yaw.grid(True, alpha=0.22)

    # ── Thrust + integral ─────────────────────────────────────────────────────
    ax_thr.plot(t[fly_mask], thrust[fly_mask], color='tomato', linewidth=0.9, label='thrust')
    ax_thr.plot(t[fly_mask], zint[fly_mask], color='#aaaaff', linewidth=0.9,
                linestyle=':', label='z_integral')
    ax_thr.axhline(0.28, color='gray', linewidth=0.7, linestyle='--', alpha=0.5,
                   label='hover (0.28)')
    mark_gate_events_vline(ax_thr, t[fly_mask], wp_idx[fly_mask])
    ax_thr.set_xlabel('Time (s)', fontsize=8)
    ax_thr.set_ylabel('Thrust / integral', fontsize=8)
    ax_thr.set_title('Thrust control', fontsize=9)
    ax_thr.legend(fontsize=7)
    ax_thr.grid(True, alpha=0.22)

    # ── Detection rate (loaded externally, empty bars if no diag) ─────────────
    diag = load_detection_diag(run_dir)
    if diag:
        centers, rate, counts = detection_rate_per_second(diag)
        ok_counts = np.array([r['detect_result'] == 'ok' for r in diag], dtype=float)
        all_results = [r['detect_result'] for r in diag]
        unique, cnts = np.unique(all_results, return_counts=True)
        labels = list(unique)
        ax_det.bar(np.arange(len(labels)), cnts, color='#4488cc', edgecolor='white',
                   linewidth=0.5)
        ax_det.set_xticks(np.arange(len(labels)))
        ax_det.set_xticklabels(labels, rotation=25, fontsize=7, ha='right')
        ax_det.set_ylabel('Frame count', fontsize=8)
        ax_det.set_title(f'Detection outcomes  ({len(diag)} frames total)', fontsize=9)

        # twin axis: detection rate per second
        ax2 = ax_det.twinx()
        ax2.plot(centers - 0.5, rate, color='orange', linewidth=1.2,
                 label='hit rate/s', zorder=3)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Detection rate', fontsize=7, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange', labelsize=7)
    else:
        ax_det.text(0.5, 0.5, 'No detection_diag_*.csv found',
                    transform=ax_det.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
        ax_det.set_title('CV detection rate', fontsize=9)
    ax_det.grid(True, alpha=0.22)

    return fig


def make_cv_gate_figure(groups: list[np.ndarray], run_dir: str) -> plt.Figure | None:
    if not groups:
        return None
    G   = len(groups)
    cols = min(G, 4)
    rows = (G + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 0.8))
    fig.suptitle(f'CV Gate Estimates by Group — {os.path.basename(run_dir)}', fontsize=12)
    axes = np.array(axes).flatten()
    means = cv_gate_means(groups)

    for i, g in enumerate(groups):
        ax  = axes[i]
        col = _gate_color(i)
        gd  = ned2d(g)
        m   = ned2d(means[i:i+1])[0]
        n   = len(g)

        # XY scatter
        ax.scatter(gd[:, 0], gd[:, 1], color=col, s=8, alpha=0.5, label='estimates')
        ax.scatter(m[0], m[1], color=col, s=150, marker='D', edgecolors='white',
                   linewidths=0.8, zorder=5, label='mean')
        # Error bars (std ellipse approximation — just std X and Y separately)
        sx = float(np.std(gd[:, 0]))
        sy = float(np.std(gd[:, 1]))
        ax.errorbar(m[0], m[1], xerr=sx, yerr=sy, fmt='none', color=col,
                    capsize=5, linewidth=1.5, alpha=0.8)

        ax.set_title(f'Gate group {i}  (n={n})\n'
                     f'mean=({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f} m)\n'
                     f'std=(X:{sx:.2f}, Y:{sy:.2f}) m',
                     fontsize=8)
        ax.set_xlabel('X/N (m)', fontsize=7)
        ax.set_ylabel('Y/E (m)', fontsize=7)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=6)

    # Hide unused subplots
    for j in range(G, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_clean_path_figure(ctrl: list[dict], groups: list[np.ndarray],
                            dead_pos_d: np.ndarray, run_dir: str) -> plt.Figure:
    t0     = ctrl[0]['time_s']
    t      = np.array([r['time_s'] - t0 for r in ctrl])
    fly_mask = np.array([r.get('state') == 'FLY' for r in ctrl])

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f'Dead-reckoned Path + CV Gate Means — {os.path.basename(run_dir)}',
                 fontsize=11)
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.06, right=0.97,
                          top=0.88, bottom=0.10)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_sid = fig.add_subplot(gs[0, 1])
    means  = cv_gate_means(groups)

    for view_ax, xi, yi, xlabel, ylabel, title in [
        (ax_top, 0, 1, 'X / North (m)', 'Y / East (m)',  'Top-down'),
        (ax_sid, 0, 2, 'X / North (m)', 'Altitude (m)', 'Side view'),
    ]:
        sc = view_ax.scatter(dead_pos_d[fly_mask, xi], dead_pos_d[fly_mask, yi],
                             c=t[fly_mask], cmap='plasma', s=4, zorder=3)
        view_ax.plot(dead_pos_d[fly_mask, xi], dead_pos_d[fly_mask, yi],
                     color='gray', linewidth=0.6, alpha=0.3, zorder=1)
        view_ax.scatter(dead_pos_d[0, xi], dead_pos_d[0, yi],
                        color='lime', s=100, marker='^', zorder=5, label='Start')
        if means is not None:
            means_d = ned2d(means)
            for i, md in enumerate(means_d):
                col = _gate_color(i)
                view_ax.scatter(md[xi], md[yi], color=col, s=200, marker='D',
                                edgecolors='white', linewidths=0.9, zorder=6)
                view_ax.annotate(f'G{i}',
                                 xy=(md[xi], md[yi]),
                                 xytext=(0, 12), textcoords='offset points',
                                 fontsize=8, color=col, ha='center', fontweight='bold')

        view_ax.set_xlabel(xlabel, fontsize=9)
        view_ax.set_ylabel(ylabel, fontsize=9)
        view_ax.set_title(title, fontsize=10)
        view_ax.set_aspect('equal', adjustable='datalim')
        view_ax.grid(True, alpha=0.22)
        view_ax.legend(fontsize=8)

    fig.colorbar(sc, ax=[ax_top, ax_sid], label='Elapsed time (s)', shrink=0.6, pad=0.02)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    run_dir = resolve_run_dir(sys.argv)
    print(f'[viz] Run dir : {run_dir}')

    ctrl = load_controller_log(run_dir)
    if ctrl is None:
        print(f'[viz] ERROR: no controller_log.csv in {run_dir}')
        sys.exit(1)
    print(f'[viz] controller_log: {len(ctrl)} rows')

    est = load_gate_estimates(run_dir)
    if est:
        print(f'[viz] gate_estimates: {len(est)} rows')
    else:
        print('[viz] gate_estimates: none found')

    # Cluster CV estimates by gate
    groups = group_cv_estimates(est or [])
    print(f'[viz] CV gate groups: {len(groups)}  sizes={[len(g) for g in groups]}')

    dead_pos     = dead_reckon(ctrl)
    dead_pos_d   = ned2d(dead_pos)

    print_summary(ctrl, groups)

    stem = 'flight'

    # Figure 1: Overview
    fig1 = make_overview_figure(ctrl, groups, dead_pos_d, run_dir)
    out1 = os.path.join(run_dir, f'{stem}_overview.png')
    fig1.savefig(out1, dpi=140, bbox_inches='tight')
    print(f'[viz] saved → {out1}')

    # Figure 2: CV gate scatter
    fig2 = make_cv_gate_figure(groups, run_dir)
    if fig2 is not None:
        out2 = os.path.join(run_dir, f'{stem}_cv_gates.png')
        fig2.savefig(out2, dpi=140, bbox_inches='tight')
        print(f'[viz] saved → {out2}')

    # Figure 3: Clean path
    fig3 = make_clean_path_figure(ctrl, groups, dead_pos_d, run_dir)
    out3 = os.path.join(run_dir, f'{stem}_path_clean.png')
    fig3.savefig(out3, dpi=140, bbox_inches='tight')
    print(f'[viz] saved → {out3}')

    try:
        matplotlib.use('TkAgg')
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
