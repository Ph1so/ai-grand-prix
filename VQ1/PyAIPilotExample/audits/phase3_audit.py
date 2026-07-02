"""
Phase 3 audit -- score the calibration flight (PERC.md Sec.6 Phase 3) purely
from completed log files. No live interaction; designed to run after the
fact on whatever legs the flight actually completed.

Alignment-free design: calibration_log.csv (wall-clock `time_s`) and
gate_estimates_*.csv (`sim_time_ns`) run on different clocks (the A12
concern). Rather than aligning them by time, we match each CV-estimate row
to the nearest calibration_log tick in *state space* -- (drone position, yaw)
-- and inherit that tick's leg/phase label. Each leg occupies a geometrically
distinct region by design, so nearest-neighbour state matching is unambiguous
and sidesteps cross-clock alignment entirely.

Usage:
    python phase3_audit.py                                  # auto-picks latest run
    python phase3_audit.py <calibration_log.csv> <gate_estimates.csv>
"""
import csv
import glob
import os
import sys

import numpy as np

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')


def _load_cal_log(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'time_s':   float(row['time_s']),
                'leg_idx':  int(row['leg_idx']),
                'leg_name': row['leg_name'],
                'phase':    row['phase'],
                'settled':  int(row['settled']),
                'pos':      np.array([float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])]),
                'yaw':      float(row['yaw']),
            })
    rows.sort(key=lambda r: r['time_s'])
    return rows


def _load_estimates(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            r = {k: v for k, v in row.items()}
            r['sim_time_ns']     = int(r['sim_time_ns'])
            r['matched_gate_id'] = int(r['matched_gate_id'])
            for k in ('tvec_x', 'tvec_y', 'tvec_z', 'roll', 'pitch', 'yaw',
                      'est_x', 'est_y', 'est_z', 'drone_x', 'drone_y', 'drone_z',
                      'mav_x', 'mav_y', 'mav_z',
                      'error_x', 'error_y', 'error_z', 'error_mag'):
                r[k] = float(r[k])
            rows.append(r)
    rows.sort(key=lambda r: r['sim_time_ns'])
    return rows


def _dedup_by_frame(rows):
    """Phase 0 lesson: gate_estimates_*.csv re-logs the same camera frame ~17x.
    Keep one row per unique sim_time_ns (first-seen)."""
    seen, out = set(), []
    for r in rows:
        if r['sim_time_ns'] not in seen:
            seen.add(r['sim_time_ns'])
            out.append(r)
    return out


def _wrap(a):
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _label_estimates(estimates, cal_rows):
    """
    Nearest-neighbour state-space match: each CV row inherits the leg/phase
    label of the closest calibration_log tick in (position, yaw) space.
    Position dominates (metres); yaw difference is scaled into comparable
    units (rad * 3m/rad =~ how far a 1-rad yaw error moves the camera's view
    of a gate ~3m away -- rough but sufficient to break ties consistently).
    """
    cal_pos = np.array([r['pos'] for r in cal_rows])               # (N, 3)
    cal_yaw = np.array([r['yaw'] for r in cal_rows])               # (N,)

    YAW_WEIGHT = 3.0
    for e in estimates:
        d_pos = cal_pos - np.array([e['drone_x'], e['drone_y'], e['drone_z']])
        d_yaw = np.array([_wrap(y - e['yaw']) for y in cal_yaw]) * YAW_WEIGHT
        dist  = np.sqrt(np.sum(d_pos**2, axis=1) + d_yaw**2)
        idx   = int(np.argmin(dist))
        e['_leg_idx']  = cal_rows[idx]['leg_idx']
        e['_leg_name'] = cal_rows[idx]['leg_name']
        e['_phase']    = cal_rows[idx]['phase']
        e['_settled']  = cal_rows[idx]['settled']
        e['_match_dist'] = float(dist[idx])
    return estimates


def _gate_range(rows):
    drone = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in rows])
    est   = np.array([[r['est_x'],   r['est_y'],   r['est_z']]   for r in rows])
    return np.linalg.norm(est - drone, axis=1)


def _stats(label, arr):
    arr = np.asarray(arr, dtype=float)
    return (f"{label:10s} n={len(arr):4d}  mean={arr.mean():+7.3f}  "
            f"median={np.median(arr):+7.3f}  std={arr.std():6.3f}  "
            f"min={arr.min():+7.3f}  max={arr.max():+7.3f}")


# ── Per-leg scoring ──────────────────────────────────────────────────────────

def score_stare(rows):
    """A11 -- camera-tilt sign: at dead-on range/angle, any persistent
    component-wise bias is pure rotation-chain error (no range/angle confound)."""
    print(f"\n{'='*72}\nLEG: stare  (A11 -- camera-tilt sign)\n{'='*72}")
    if not rows:
        print("  No settled CV detections for this leg -- cannot score.")
        return
    ex = np.array([r['error_x'] for r in rows])
    ey = np.array([r['error_y'] for r in rows])
    ez = np.array([r['error_z'] for r in rows])
    print(_stats('error_x', ex))
    print(_stats('error_y', ey))
    print(_stats('error_z', ez))
    # one-sample sign test: is the mean far enough from 0 relative to spread to call it systematic?
    for name, arr in (('x', ex), ('y', ey), ('z', ez)):
        bias_ratio = abs(arr.mean()) / max(arr.std(), 1e-6)
        verdict = 'SYSTEMATIC BIAS' if bias_ratio > 1.0 and abs(arr.mean()) > 0.15 else 'within noise'
        print(f"  -> error_{name}: |mean|/std = {bias_ratio:5.2f}   [{verdict}]")
    print(f"  Verdict: error_z mean = {ez.mean():+.3f} m. "
          f"{'Negative => CV places gate too HIGH (less negative/shallower z than truth) -> ' if ez.mean() < 0 else 'Positive => CV places gate too LOW -> '}"
          f"check R_CAM2BODY tilt sign in pose_estimator.py if |bias| is large relative to std.")


def score_yaw_sweep(rows):
    """A13 -- yaw / rotation-chain coupling: position held constant, yaw swept.
    Any error correlation with yaw means the rotation chain mistracks attitude."""
    print(f"\n{'='*72}\nLEG: yaw_sweep  (A13 -- yaw / rotation-chain coupling)\n{'='*72}")
    if not rows:
        print("  No settled CV detections for this leg -- cannot score.")
        return
    yaw  = np.array([r['yaw'] for r in rows])
    emag = np.array([r['error_mag'] for r in rows])
    ex   = np.array([r['error_x'] for r in rows])
    ey   = np.array([r['error_y'] for r in rows])
    ez   = np.array([r['error_z'] for r in rows])
    print(f"  yaw range observed: {np.degrees(yaw.min()):.1f} .. {np.degrees(yaw.max()):.1f} deg  (n={len(rows)})")
    print(_stats('error_mag', emag))
    for name, arr in (('mag', emag), ('x', ex), ('y', ey), ('z', ez)):
        if np.std(yaw) > 1e-6 and np.std(arr) > 1e-6:
            corr = float(np.corrcoef(yaw, arr)[0, 1])
        else:
            corr = float('nan')
        flag = '  <-- correlated with yaw (rotation-chain coupling present)' if abs(corr) > 0.4 else ''
        print(f"  corr(yaw, error_{name}) = {corr:+.2f}{flag}")
    print(f"  Verdict: |corr| > ~0.4 indicates the rotation chain (R_b2ned @ R_CAM2BODY) "
          f"is not fully compensating attitude -- error tracks where the camera points, "
          f"not just where the gate is.")


def score_range_sweep(rows):
    """A1/A9/A10 -- range-dependent error growth: continuous approach across the working range."""
    print(f"\n{'='*72}\nLEG: range_sweep  (A1/A9/A10 -- range-error growth)\n{'='*72}")
    if not rows:
        print("  No settled CV detections for this leg -- cannot score.")
        return
    rng  = _gate_range(rows)
    emag = np.array([r['error_mag'] for r in rows])
    ex   = np.array([r['error_x'] for r in rows])
    ey   = np.array([r['error_y'] for r in rows])
    ez   = np.array([r['error_z'] for r in rows])
    print(f"  range observed: {rng.min():.1f} .. {rng.max():.1f} m  (n={len(rows)})")
    print(_stats('error_mag', emag))
    corr_mag = float(np.corrcoef(rng, emag)[0, 1]) if np.std(rng) > 1e-6 else float('nan')
    print(f"  corr(range, error_mag) = {corr_mag:+.2f}"
          f"{'  <-- error grows with range (perspective/focal-length mismatch likely)' if corr_mag > 0.4 else ''}")
    # bucket into near/mid/far thirds for a clearer trend readout
    order = np.argsort(rng)
    third = max(1, len(order) // 3)
    buckets = [('near', order[:third]), ('mid', order[third:2*third]), ('far', order[2*third:])]
    for name, idx in buckets:
        if len(idx) == 0:
            continue
        print(f"    {name:5s}: range {rng[idx].mean():5.1f}m (n={len(idx):4d})  "
              f"mean|err|={emag[idx].mean():.3f}m  ex={ex[idx].mean():+.2f}  ey={ey[idx].mean():+.2f}  ez={ez[idx].mean():+.2f}")
    print(f"  Verdict: corr > ~0.4 and a clear near->far growth in the bucket table together "
          f"indicate range-dependent error (check focal length fx/fy=320 and gate half-dimension 1.36m).")


def _recover_oblique_angle(rows, ref_bearing):
    """Oblique angle = bearing(gate -> drone) measured relative to the dead-on
    reference bearing (where, by the calibration plan's station geometry, the
    angle is exactly 0). Recovered purely from logged drone/gate positions --
    no dependence on the Leg objects, so this works offline from the CSVs alone."""
    out = []
    for r in rows:
        bearing = float(np.arctan2(r['drone_y'] - r['mav_y'], r['drone_x'] - r['mav_x']))
        out.append(np.degrees(_wrap(bearing - ref_bearing)))
    return np.array(out)


def _geometric_dead_on_bearing(cal_rows, oblique_rows):
    """Exact dead-on reference bearing -- the same `bearing_from_gate`
    geometry calibration_plan.build_calibration_plan uses to lay out every
    leg (direction from the gate back toward the drone's pre-CALIBRATE
    position, i.e. the lead-in side every leg orbits from). Needs no `stare`
    leg: only the drone's first-tick position (calibration_log) and the
    matched gate's MAVLink centre (already logged on every CV row as
    mav_x/y/z) -- both exact, neither carries CV noise, so this is a strictly
    better reference than any circular-mean-of-noisy-bearings approach."""
    start = cal_rows[0]['pos']
    g0    = np.array([oblique_rows[0]['mav_x'], oblique_rows[0]['mav_y'], oblique_rows[0]['mav_z']])
    approach = (g0 - start).copy()
    approach[2] = 0.0
    norm = float(np.linalg.norm(approach))
    approach_dir = approach / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    boresight_yaw = float(np.arctan2(approach_dir[1], approach_dir[0]))
    return _wrap(boresight_yaw + np.pi)


def score_oblique_sweep(rows, cal_rows):
    """A6/A5/A7 -- oblique-angle robustness: continuous traverse from -40deg to
    +40deg off-axis. Oblique angle is recovered geometrically from logged
    drone/gate positions (relative to the dead-on bearing derived purely from
    course geometry -- see _geometric_dead_on_bearing), so this scores purely
    from the CSVs and needs no `stare` leg or Leg-object dependency."""
    print(f"\n{'='*72}\nLEG: oblique_sweep  (A6/A5/A7 -- oblique-angle robustness, continuous -40..+40 deg)\n{'='*72}")
    if not rows:
        print("  No settled CV detections for this leg -- cannot score.")
        return
    ref_bearing = _geometric_dead_on_bearing(cal_rows, rows)
    angle = _recover_oblique_angle(rows, ref_bearing)
    emag  = np.array([r['error_mag'] for r in rows])
    abs_angle = np.abs(angle)

    print(f"  oblique angle observed: {angle.min():+.1f} .. {angle.max():+.1f} deg  (n={len(rows)})")
    print(_stats('error_mag', emag))
    corr = float(np.corrcoef(abs_angle, emag)[0, 1]) if np.std(abs_angle) > 1e-6 else float('nan')
    print(f"  corr(|oblique_angle|, error_mag) = {corr:+.2f}"
          f"{'  <-- error grows with viewing obliqueness (PnP/detection degrades off-axis)' if corr > 0.4 else ''}")

    order = np.argsort(abs_angle)
    third = max(1, len(order) // 3)
    buckets = [('axis', order[:third]), ('mid', order[third:2*third]), ('oblique', order[2*third:])]
    for name, idx in buckets:
        if len(idx) == 0:
            continue
        print(f"    {name:8s}: |angle| {abs_angle[idx].mean():5.1f} deg (n={len(idx):4d})  mean|err|={emag[idx].mean():.3f}m")
    print(f"  Verdict: corr > ~0.4 and growth from 'axis' to 'oblique' buckets together indicate the "
          f"gate's apparent shape skew at high viewing angles degrades PnP/detection accuracy "
          f"(check gate_detector corner ordering and solvePnP behaviour on foreshortened quads).")


# ── Settling/overshoot diagnosis (why a leg never went 'active') ─────────────

def diagnose_unsettled_leg(cal_rows, leg_idx, leg_name):
    sub = [r for r in cal_rows if r['leg_idx'] == leg_idx]
    if not sub:
        print(f"\n  Leg {leg_idx} '{leg_name}': no controller-log rows at all (never reached).")
        return
    settled = [r for r in sub if r['settled'] == 1]
    print(f"\n  Leg {leg_idx} '{leg_name}': {len(sub)} ticks logged, "
          f"{'reached its active window' if settled else 'NEVER settled (stuck in settling/transit)'}")
    if settled:
        return
    # Distance-to-target over time, to characterise the failure (orbit vs slow convergence vs divergence)
    # We don't have leg.pos_start here without recomputing the plan; horiz_dist column was logged though
    # -- but _load_cal_log doesn't carry it. Re-derive from the dataclass would need the plan; instead
    # report on the raw tick stream's position spread, which is enough to characterise an orbit.
    pos = np.array([r['pos'] for r in sub])
    centroid = pos.mean(axis=0)
    radii = np.linalg.norm(pos[:, :2] - centroid[:2], axis=1)
    print(f"    Position spread around its own centroid: mean radius={radii.mean():.2f}m  "
          f"std={radii.std():.3f}m  (low std + nonzero mean radius = stable ORBIT, "
          f"the position-PID -> velocity cascade overshot and entered a limit cycle "
          f"around the target instead of converging onto it)")


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        cal_path, est_path = args[0], args[1]
    else:
        cal_logs = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'calibration_log.csv')), reverse=True)
        ests     = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'gate_estimates_*.csv')), reverse=True)
        if not cal_logs:
            print('No calibration_log.csv found -- has the calibration flight been run?')
            sys.exit(1)
        cal_path, est_path = cal_logs[0], ests[0]

    print(f"Calibration log : {cal_path}")
    print(f"Gate estimates  : {est_path}")

    cal_rows = _load_cal_log(cal_path)
    raw      = _load_estimates(est_path)
    deduped  = _dedup_by_frame(raw)
    print(f"\ncalibration_log.csv : {len(cal_rows)} ticks")
    print(f"gate_estimates.csv  : {len(raw)} logged rows -> {len(deduped)} genuine frames "
          f"(dedup x{len(raw)/max(1,len(deduped)):.1f}, see Phase 0)")

    labeled = _label_estimates(deduped, cal_rows)
    settled = [e for e in labeled if e['_settled'] == 1]
    print(f"State-space-matched & filtered to 'settled' windows: {len(settled)} CV detections usable for scoring")
    print(f"  (median match distance: {np.median([e['_match_dist'] for e in labeled]):.3f} -- "
          f"should be small; large values would mean the nearest-neighbour match is unreliable)")

    by_leg = {}
    for e in settled:
        by_leg.setdefault(e['_leg_idx'], []).append(e)

    leg_names = {r['leg_idx']: r['leg_name'] for r in cal_rows}
    print(f"\nLegs present in calibration_log: " +
          ", ".join(f"{idx}:{name}" for idx, name in sorted(leg_names.items())))
    print(f"Legs with usable ('settled') CV data: " +
          (", ".join(f"{idx}:{leg_names[idx]} (n={len(rows)})" for idx, rows in sorted(by_leg.items())) or "none"))

    # ── Score whatever legs completed (matched by NAME, not index -- a trimmed
    # plan via CAL_LEG_FILTER can put any leg at idx 0, so index-based lookups
    # would silently score the wrong leg's data under the wrong verdict logic) ──
    def _idx_for(name):
        return next((idx for idx, n in leg_names.items() if n == name), None)

    stare_idx = _idx_for('stare')
    if stare_idx is not None:
        score_stare(by_leg.get(stare_idx, []))

    yaw_idx = _idx_for('yaw_sweep')
    if yaw_idx is not None:
        score_yaw_sweep(by_leg.get(yaw_idx, []))

    range_idx = _idx_for('range_sweep')
    if range_idx is not None:
        score_range_sweep(by_leg.get(range_idx, []))

    oblique_idx = _idx_for('oblique_sweep')
    if oblique_idx is not None:
        score_oblique_sweep(by_leg.get(oblique_idx, []), cal_rows)

    # ── Diagnose any leg that never reached its active window ────────────────
    incomplete = [idx for idx in leg_names if idx not in by_leg]
    if incomplete:
        print(f"\n{'='*72}\nDIAGNOSIS: legs that never produced usable data\n{'='*72}")
        for idx in sorted(incomplete):
            diagnose_unsettled_leg(cal_rows, idx, leg_names[idx])


if __name__ == '__main__':
    main()
