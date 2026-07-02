"""
Phase 1.1 audit -- mathematical error decomposition (PERC.md Sec.6 Phase 1.1).

Splits the end-to-end CV error into its two independent stages WITHOUT the
circularity problem the original Phase-1.1 framing had (you can't "solve for
R_CAM2BODY" by assuming R_b2ned is exact, when Phase 3 already proved
R_CAM2BODY itself is biased -- any such solve would just be fitting one
wrong assumption to compensate for the other). Instead this uses two
*rotation-invariant* and *rotation-chain-independent* checks:

  (a) SCALE check (A1, detection/PnP-stage):
      |tvec| (PnP's estimated camera-to-gate distance) vs.
      |mav - drone_pos| (true camera-to-gate distance).
      Rotations preserve vector norms, so this ratio is completely
      independent of R_CAM2BODY / R_b2ned -- a clean, assumption-free probe
      of whether solvePnP's distance estimate is scale-correct (i.e.
      whether OBJ_PTS' 1.36 m outer-boundary assumption (A1) matches what
      the detector actually traces).

  (b) ANGLE fit (A11, rotation-chain-stage):
      Given the (now scale-checked) tvec and the logged attitude, what
      single tilt angle `t` in a `R_CAM2BODY(t)` of the *same form* as the
      coded matrix makes `R_CAM2BODY(t) @ tvec` best match
      `R_b2ned^T @ (mav - drone_pos)` (the body-frame vector that *would*
      produce the correct world position)? A 1-D least-squares fit across
      every settled sample in the calibration flight -- cross-checked
      against Phase 3's stare-derived ~33 deg estimate.

  (c) VALIDATION: apply the fitted angle, recompute `est`, and report how
      much the end-to-end error collapses -- turns "here's a number" into
      "here's what happens if you change _t in pose_estimator.py".

Usage:
    python phase1_1_audit.py                                  # auto-picks latest run
    python phase1_1_audit.py <calibration_log.csv> <gate_estimates.csv>
"""
import glob
import os
import sys

import numpy as np

# Reuse Phase 3's loaders/labelers -- same alignment-free, dedup'd, settled-window
# pipeline; no point re-deriving it.
from phase3_audit import _load_cal_log, _load_estimates, _dedup_by_frame, _label_estimates, _wrap

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Mirrors pose_estimator.R_CAM2BODY's coded value, for "before" comparison.
_T_CODED = np.radians(20.0)


def _euler_zyx(roll, pitch, yaw):
    """ZYX Euler -> R_b2ned. Identical formula to pose_estimator._euler_zyx,
    reproduced here so this script has no dependency on pose_estimator
    beyond what it's actively probing (R_CAM2BODY)."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [   -sp,            cp*sr,            cp*cr  ],
    ])


def _r_cam2body(t):
    """Same single-tilt-parameter form as pose_estimator.R_CAM2BODY, with `t`
    free. (Camera tilts about the body-Y/'right' axis; column 0 = (0,1,0)
    is the fixed rotation axis -- this is genuinely a 1-DOF family.)"""
    return np.array([
        [0.,         -np.sin(t),  np.cos(t)],
        [1.,          0.,          0.       ],
        [0.,          np.cos(t),  np.sin(t)],
    ])


def _load_settled(cal_path, est_path):
    cal_rows = _load_cal_log(cal_path)
    raw      = _load_estimates(est_path)
    deduped  = _dedup_by_frame(raw)
    labeled  = _label_estimates(deduped, cal_rows)
    settled  = [e for e in labeled if e['_settled'] == 1]
    return settled


def _vectors(rows):
    """For each row, return (tvec, mav-drone_pos, R_b2ned) as arrays."""
    tvec  = np.array([[r['tvec_x'], r['tvec_y'], r['tvec_z']] for r in rows])
    delta = np.array([[r['mav_x'] - r['drone_x'],
                       r['mav_y'] - r['drone_y'],
                       r['mav_z'] - r['drone_z']] for r in rows])
    R = np.array([_euler_zyx(r['roll'], r['pitch'], r['yaw']) for r in rows])
    return tvec, delta, R


def _scale_check(rows):
    print(f"\n{'='*72}\n(a) SCALE CHECK -- A1 (object-point scale): |tvec| vs. |mav - drone_pos|\n"
          f"    Rotation-invariant -- needs NO assumption about R_CAM2BODY or R_b2ned.\n{'='*72}")
    tvec, delta, _ = _vectors(rows)
    pnp_range  = np.linalg.norm(tvec, axis=1)
    true_range = np.linalg.norm(delta, axis=1)
    ratio = pnp_range / true_range
    print(f"  n={len(rows)}")
    print(f"  |tvec|/|mav-drone_pos| ratio: mean={ratio.mean():.4f}  median={np.median(ratio):.4f}  "
          f"std={ratio.std():.4f}")
    corr = np.corrcoef(true_range, ratio)[0, 1]
    print(f"  corr(true_range, ratio) = {corr:+.3f}  "
          f"{'<-- ratio drifts with range -- detector likely traces a different boundary at different ranges' if abs(corr) > 0.3 else '(no meaningful range-dependence -- ratio is a fixed scale factor, if any)'}")
    order = np.argsort(true_range)
    third = max(1, len(order)//3)
    for name, idx in [('near', order[:third]), ('mid', order[third:2*third]), ('far', order[2*third:])]:
        print(f"    {name:4s}: true_range={true_range[idx].mean():5.1f}m (n={len(idx):4d})  "
              f"ratio={ratio[idx].mean():.4f}  std={ratio[idx].std():.4f}")
    if abs(ratio.mean() - 1.0) > 0.05 and abs(corr) < 0.3:
        implied = 1.36 * ratio.mean()
        print(f"  Verdict: ratio is ~constant but != 1 (mean {ratio.mean():.3f}) -- consistent with the "
              f"detector consistently tracing a boundary of ~{implied:.2f}m half-dimension "
              f"(OBJ_PTS assumes 1.36m outer; inner opening is 0.75m) -- check A1.")
    elif abs(corr) > 0.3:
        print(f"  Verdict: ratio drifts with range -- NOT a simple fixed object-point-scale error; "
              f"points at a range-dependent detection effect (motion blur / sub-pixel corner noise "
              f"/ HSV-mask boundary creep at small apparent sizes) rather than a wrong constant in OBJ_PTS.")
    else:
        print(f"  Verdict: ratio ~= 1.0 and range-independent -- PnP's distance estimate is "
              f"scale-correct. A1 (outer-boundary assumption) is NOT a meaningful error source; "
              f"detection/PnP-stage contributes negligible *radial* error. Any remaining error is "
              f"angular in nature (rotation-chain and/or corner-ordering/direction noise -- see (b)).")
    return ratio, true_range


def _angle_fit(rows):
    print(f"\n{'='*72}\n(b) ANGLE FIT -- A11 (rotation-chain tilt): best single-parameter t in\n"
          f"    R_CAM2BODY(t) such that R_CAM2BODY(t) @ tvec ~= R_b2ned^T @ (mav - drone_pos)\n{'='*72}")
    tvec, delta, R = _vectors(rows)
    # Body-frame vector that *would* produce the correct world position, given
    # the *actual logged* attitude -- i.e. "what R_CAM2BODY @ tvec should equal".
    v_ideal = np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), delta)

    def residual_norm(t):
        Rcb = _r_cam2body(t)
        v_actual = (Rcb @ tvec.T).T
        return np.linalg.norm(v_actual - v_ideal, axis=1)

    # 1-D grid search (the family is a simple, smooth, single-minimum cost in t
    # over the physically-plausible range -- a fine grid is exact enough and
    # avoids any solver-convergence assumptions).
    grid = np.radians(np.arange(-90.0, 90.01, 0.05))
    costs = np.array([np.sum(residual_norm(t)**2) for t in grid])
    t_fit = grid[np.argmin(costs)]

    rms_coded = np.sqrt(np.mean(residual_norm(_T_CODED)**2))
    rms_fit   = np.sqrt(np.mean(residual_norm(t_fit)**2))

    print(f"  n={len(rows)}")
    print(f"  Currently coded tilt          : {np.degrees(_T_CODED):+7.2f} deg   "
          f"(RMS body-frame residual = {rms_coded:.3f} m)")
    print(f"  Best-fit tilt (this dataset)  : {np.degrees(t_fit):+7.2f} deg   "
          f"(RMS body-frame residual = {rms_fit:.3f} m)")
    print(f"  Phase 3 'stare'-derived est.  : ~+33    deg   (from constant error_z/range ~= 0.66 = tan 33 deg)")
    print(f"  -> {'CONSISTENT' if abs(np.degrees(t_fit) - 33) < 5 else 'CHECK'}: independent methods "
          f"({'global least-squares fit across all legs' if True else ''} vs. single-leg ratio-based "
          f"estimate) {'agree to within a few degrees' if abs(np.degrees(t_fit)-33) < 5 else 'disagree -- worth investigating why'}.")
    return t_fit


def _validate(rows, t_fit):
    print(f"\n{'='*72}\n(c) VALIDATION -- recompute `est` with the fitted tilt; how much does error collapse?\n{'='*72}")
    tvec, delta, R = _vectors(rows)
    mav   = np.array([[r['mav_x'], r['mav_y'], r['mav_z']] for r in rows])
    drone = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in rows])

    def scored(t, label):
        Rcb = _r_cam2body(t)
        est = drone + np.einsum('nij,nj->ni', R, (Rcb @ tvec.T).T)
        err = est - mav
        emag = np.linalg.norm(err, axis=1)
        print(f"  {label:32s}  mean|err|={emag.mean():6.3f}m  median={np.median(emag):6.3f}m  "
              f"std={emag.std():6.3f}m  max={emag.max():6.3f}m")
        print(f"    {'':32s}  mean error_x={err[:,0].mean():+6.3f}  "
              f"error_y={err[:,1].mean():+6.3f}  error_z={err[:,2].mean():+6.3f}")
        return emag

    e_before = scored(_T_CODED, f"Current code  (t={np.degrees(_T_CODED):+.1f} deg)")
    e_after  = scored(t_fit,    f"Fitted tilt   (t={np.degrees(t_fit):+.1f} deg)")
    reduction = 100 * (1 - e_after.mean() / e_before.mean())
    print(f"\n  Mean |error| reduction from re-tuning _t alone: {reduction:.1f}%")
    print(f"  Verdict: {'a single-constant change to pose_estimator._t would collapse the large majority of the systematic error -- the rotation chain IS the dominant error source, exactly as A11 suspected.' if reduction > 50 else 're-tuning the tilt alone does not collapse most of the error -- a meaningful fraction comes from elsewhere (detection/PnP-stage, see (a), or higher-order rotation-chain terms beyond a single tilt angle).'}")


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

    rows = _load_settled(cal_path, est_path)
    # Need tvec/roll/pitch/yaw -- only present in Phase-2-instrumented runs.
    if not rows or 'tvec_x' not in rows[0]:
        print(f"\nNo settled rows with tvec/attitude columns (Phase 2 logging) -- "
              f"this run predates Phase 2, or no leg ever settled.")
        sys.exit(1)
    print(f"\n{len(rows)} settled CV detections with full Phase-2 instrumentation (tvec + attitude)")

    ratio, true_range = _scale_check(rows)
    t_fit = _angle_fit(rows)
    _validate(rows, t_fit)


if __name__ == '__main__':
    main()
