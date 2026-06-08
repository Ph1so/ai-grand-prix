"""
A12 audit -- pos/attitude synchronicity with the camera frame (PERC.md A12).

A12's worry: `process_frame` reads `shared_data['pos']` / `['attitude']`
at whatever instant it happens to run -- the *latest* MAVLink sample, not
necessarily the telemetry sample at the camera's actual capture instant.
TimeSync exists but isn't consumed (known gap #5), so there's no correction.

This is fully testable from EXISTING logs, no new flight required, because
the Phase-2-instrumented `gate_estimates_*.csv` already records BOTH halves
of the comparison for every CV detection:

  - `sim_time_ns`            : the camera frame's TRUE capture instant
                               (stamped by the simulator -- ground truth)
  - `roll/pitch/yaw`,
    `drone_x/y/z`            : the telemetry `process_frame` ACTUALLY used
                               (snapshotted from `shared_data` at processing
                               time -- exactly the inputs fed to `camera_to_ned`)

`flight_log_*.csv` is an independent, dense (~88 Hz) telemetry time series
in the SAME epoch (`wall_time_s` ~= `sim_time_ns/1e9`, established in Phase
4's A4/A8 alignment). Interpolating it AT each frame's `sim_time_ns` gives
the telemetry that *should* have been used -- "ground truth at capture
instant". The difference between "used" and "should-have-used" IS the
synchronization error A12 worries about, by definition -- independent of
*why* it's nonzero (decode/reassembly latency, telemetry-sample staleness,
network jitter, ...).

Propagating that delta through the SAME rotation chain pose_estimator uses
(`drone_pos + R_b2ned(roll,pitch,yaw) @ R_CAM2BODY @ tvec`) turns "telemetry
was off by X degrees / Y metres" into "and that alone moved `est` by Z
metres" -- directly comparable to the actual logged `error_mag`, so we can
say whether A12 is a meaningful contributor to the error budget or noise.

Usage:
    python a12_sync_audit.py                                  # auto-picks latest calibration run
    python a12_sync_audit.py <flight_log.csv> <gate_estimates.csv>
"""
import csv
import glob
import os
import sys

import numpy as np

from phase3_audit import _load_estimates, _dedup_by_frame, _wrap
from phase1_1_audit import _euler_zyx, _r_cam2body, _T_CODED

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')


def _load_flight_log_dense(path):
    """Dense (~88 Hz) telemetry time series, sorted by time, ready for
    interpolation. Returns (t, pos[N,3], att[N,3]=[roll,pitch,yaw])."""
    with open(path, newline='') as f:
        rows = [
            (float(r['wall_time_s']),
             float(r['pos_x']), float(r['pos_y']), float(r['pos_z']),
             float(r['roll']), float(r['pitch']), float(r['yaw']))
            for r in csv.DictReader(f)
        ]
    rows.sort(key=lambda r: r[0])
    t   = np.array([r[0] for r in rows])
    pos = np.array([r[1:4] for r in rows])
    att = np.array([r[4:7] for r in rows])
    return t, pos, att


def _interp_telemetry(t_query, flog_t, flog_pos, flog_att):
    """Linear interpolation of dense telemetry at arbitrary query instants.
    Angles are unwrapped first -- naive interpolation across a +-pi
    wraparound would otherwise produce huge spurious deltas."""
    pos_i = np.column_stack([np.interp(t_query, flog_t, flog_pos[:, k]) for k in range(3)])
    att_unwrapped = np.unwrap(flog_att, axis=0)
    att_i = np.column_stack([np.interp(t_query, flog_t, att_unwrapped[:, k]) for k in range(3)])
    att_i = np.vectorize(_wrap)(att_i)
    return pos_i, att_i


def _sync_check(rows, flog_t, flog_pos, flog_att):
    print(f"\n{'='*72}\nA12 -- pos/attitude synchronicity with the camera frame\n"
          f"  (does `process_frame` reading the LATEST telemetry -- instead of\n"
          f"   telemetry time-aligned to sim_time_ns -- meaningfully corrupt `est`?)\n{'='*72}")

    t_capture = np.array([r['sim_time_ns'] for r in rows]) / 1e9
    print(f"  n={len(rows)} CV detections")
    print(f"  capture-time span: [{t_capture.min():.1f}, {t_capture.max():.1f}]  "
          f"flight_log span: [{flog_t.min():.1f}, {flog_t.max():.1f}]  (same epoch per Phase 4)")

    pos_used = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in rows])
    att_used = np.array([[r['roll'], r['pitch'], r['yaw']] for r in rows])
    tvec     = np.array([[r['tvec_x'], r['tvec_y'], r['tvec_z']] for r in rows])
    mav      = np.array([[r['mav_x'], r['mav_y'], r['mav_z']] for r in rows])
    emag_actual = np.array([r['error_mag'] for r in rows])

    pos_true, att_true = _interp_telemetry(t_capture, flog_t, flog_pos, flog_att)

    d_pos = pos_used - pos_true
    d_att = np.vectorize(_wrap)(att_used - att_true)            # rad, wrap-safe
    d_pos_mag = np.linalg.norm(d_pos, axis=1)
    d_att_deg = np.degrees(d_att)

    print(f"\n  -- Telemetry mismatch: 'what process_frame used' vs.\n"
          f"     'what flight_log interpolated AT sim_time_ns says was true' --")
    print(f"  |drone_pos used - pos at capture instant| : "
          f"mean={d_pos_mag.mean():.4f}m  median={np.median(d_pos_mag):.4f}m  max={d_pos_mag.max():.4f}m")
    for i, name in enumerate(('roll', 'pitch', 'yaw')):
        a = d_att_deg[:, i]
        print(f"  {name:5s} used - {name} at capture instant       : "
              f"mean={a.mean():+7.4f} deg  median={np.median(a):+7.4f}  std={a.std():.4f}  "
              f"max|.|={np.max(np.abs(a)):.4f}")

    # -- Propagate the mismatch through the SAME rotation chain pose_estimator
    # uses, to convert "telemetry was off by X" into "and `est` moved by Y":
    #   est_used   = pos_used + R_b2ned(att_used) @ R_CAM2BODY @ tvec   (== logged est, by construction)
    #   est_synced = pos_true + R_b2ned(att_true) @ R_CAM2BODY @ tvec   (what a perfectly time-aligned
    #                                                                     pipeline would have produced)
    # |est_used - est_synced| is the sync-error contribution ALONE, with
    # every other error source (tilt bias, scale bias, PnP noise, ...) held
    # fixed -- a clean isolation of exactly what A12 worries about.
    Rcb = _r_cam2body(_T_CODED)          # use the CURRENTLY CODED tilt -- we're isolating
                                         # the sync question, not re-litigating A11
    R_used = np.array([_euler_zyx(*a) for a in att_used])
    R_true = np.array([_euler_zyx(*a) for a in att_true])
    cam_term = (Rcb @ tvec.T).T                                  # R_CAM2BODY @ tvec, shared

    est_used   = pos_used + np.einsum('nij,nj->ni', R_used, cam_term)
    est_synced = pos_true + np.einsum('nij,nj->ni', R_true, cam_term)

    sync_err = np.linalg.norm(est_used - est_synced, axis=1)
    # sanity: est_used should equal the logged `est` (both computed the same way)
    est_logged = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in rows])
    recompute_check = np.linalg.norm(est_used - est_logged, axis=1)

    print(f"\n  -- Propagated through the rotation chain (R_b2ned @ R_CAM2BODY @ tvec) --")
    print(f"  sanity: |recomputed est_used - logged est|  : max={recompute_check.max():.2e}m  "
          f"(should be ~0 -- confirms 'used' telemetry matches what was logged)")
    print(f"  |est_used - est_synced|  (sync-error alone)  : "
          f"mean={sync_err.mean():.4f}m  median={np.median(sync_err):.4f}m  "
          f"p90={np.percentile(sync_err,90):.4f}m  max={sync_err.max():.4f}m")
    print(f"  actual logged |est - mav| (TOTAL error)      : "
          f"mean={emag_actual.mean():.4f}m  median={np.median(emag_actual):.4f}m")
    frac = 100 * sync_err.mean() / emag_actual.mean()
    print(f"\n  Sync-error alone accounts for ~{frac:.2f}% of the mean total error magnitude.")

    corr = np.corrcoef(sync_err, emag_actual)[0, 1]
    ratio = sync_err.mean() / emag_actual.mean()
    print(f"  corr(sync_err, total_error_mag) = {corr:+.3f}  but  mean(sync_err)/mean(total_error) = {100*ratio:.2f}%  "
          f"{'-- the correlation is real but a red herring at this scale: both simply rise together when the drone maneuvers harder (more rotation -> both more telemetry staleness AND more rotation-chain/oblique-angle error per Phase 3) -- it is NOT evidence that sync error meaningfully *drives* the total, which it could not at ' + f'{100*ratio:.2f}% of its magnitude.' if corr > 0.3 and ratio < 0.05 else '-- no meaningful relationship; sync error looks independent of (and dwarfed by) the dominant error sources'}")

    print(f"\n  Verdict: {'sync error is NEGLIGIBLE (sub-millimetre to low-centimetre, ~' + f'{100*ratio:.1f}% of the metres-scale total error Phases 1.1/3 already explained) -- A12 is confirmed as a non-issue ON THIS COURSE (current speeds / ~90Hz telemetry rate are fast enough that staleness never accumulates into anything visible). It would be the first thing to revisit if future legs fly substantially faster or telemetry rate drops.' if sync_err.mean() < 0.05 else ('sync error is SMALL but not negligible (~centimetres) -- a minor contributor, dwarfed by the tilt/scale biases Phase 1.1 already found, but real and would compound at higher speed.' if sync_err.mean() < 0.5 else 'sync error is SUBSTANTIAL -- a real, independent contributor to the error budget that deserves a TimeSync-based fix (known gap #5), not just a footnote.')}")


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        flog_path, est_path = args[0], args[1]
    else:
        flogs = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'flight_log_*.csv')), reverse=True)
        flogs = [p for p in flogs if not p.endswith('_gates.json')]
        ests  = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'gate_estimates_*.csv')), reverse=True)
        # Need a Phase-2-instrumented run (tvec/roll/pitch/yaw columns) -- pair
        # each candidate flight_log with the gate_estimates from the SAME run dir.
        flog_path = est_path = None
        for e in ests:
            run_dir = os.path.dirname(e)
            cand = [f for f in flogs if os.path.dirname(f) == run_dir]
            if cand:
                flog_path, est_path = cand[0], e
                break
        if flog_path is None:
            print('No paired flight_log_*.csv + gate_estimates_*.csv found.')
            sys.exit(1)

    print(f"Flight log      : {flog_path}")
    print(f"Gate estimates  : {est_path}")

    raw = _load_estimates(est_path)
    if 'tvec_x' not in raw[0]:
        print(f"\n{est_path} predates Phase 2 instrumentation (no tvec/attitude columns) -- "
              f"can't isolate the sync-error term without them.")
        sys.exit(1)
    rows = _dedup_by_frame(raw)
    print(f"CV detections (deduped): {len(rows)}")

    flog_t, flog_pos, flog_att = _load_flight_log_dense(flog_path)
    print(f"Flight log telemetry samples: {len(flog_t)}  (~{1/np.median(np.diff(np.sort(flog_t))):.0f} Hz)")

    _sync_check(rows, flog_t, flog_pos, flog_att)


if __name__ == '__main__':
    main()
