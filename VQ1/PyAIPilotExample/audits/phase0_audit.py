"""
Phase 0 audit -- calibrate the measuring instrument before trusting per-gate
CV-error statistics (see PERC.md Sec.6 Phase 0 / new Sec.7 finding).

Root-cause finding (this script's reason for existing): gate_estimates_*.csv
contains ~17x more rows than there are genuinely distinct camera frames.
Grouping by exact-integer `sim_time_ns` shows the full detect->PnP->NED
pipeline is being re-run repeatedly for what is, optically, the SAME frame
(same tvec/attitude -- confirmed by `est - drone_pos` staying constant within
a group), each repeat picking up a freshly-updated `drone_pos` and logging a
"new" row. This single mechanism explains nearly every oddity in PERC.md Sec.1:
inflated per-gate counts, the "Gate 0 = 56%" stat, exact-duplicate blocks
during static periods, and the A14 matched_gate_id flips (the smeared `est`
drifts enough between repeats that nearest-centroid attribution can change
mid-group).

This script:
  1. Groups rows by exact integer sim_time_ns; keeps first-seen row per group
     -> the deduplicated, "one row per genuine measurement" dataset.
  2. Recomputes the Sec.1 error table on RAW vs DEDUPED data side by side.
  3. Further splits DEDUPED into in-flight vs static-drone (speed proxy
     derived purely from consecutive deduped rows -- no cross-file
     timestamp alignment needed).
  4. Re-runs the A14 sanity check (sequential, non-overlapping matched_gate_id
     windows) on RAW vs DEDUPED vs DEDUPED+in-flight.

Usage:
    python phase0_audit.py <gate_estimates.csv> <flight_log_gates.json>
    python phase0_audit.py                # auto-picks latest run
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner import load_gates

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

STATIC_SPEED_THRESHOLD = 0.3  # m/s -- below this, treat the drone as "parked"


def _load(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            r = {k: v for k, v in row.items()}
            r['sim_time_ns']     = int(r['sim_time_ns'])       # exact int -- avoid float64 rounding
            r['matched_gate_id'] = int(r['matched_gate_id'])
            for k in ('est_x', 'est_y', 'est_z', 'drone_x', 'drone_y', 'drone_z',
                      'error_x', 'error_y', 'error_z', 'error_mag'):
                r[k] = float(r[k])
            rows.append(r)
    rows.sort(key=lambda r: r['sim_time_ns'])
    return rows


def _dedup_by_frame(rows):
    """One row per unique sim_time_ns -- the first logged (freshest-at-detect-time)."""
    seen = {}
    out = []
    for r in rows:
        if r['sim_time_ns'] not in seen:
            seen[r['sim_time_ns']] = True
            out.append(r)
    return out


def _speed_proxy(rows):
    pos = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in rows])
    t   = np.array([r['sim_time_ns'] for r in rows], dtype=np.float64) * 1e-9
    speed = np.zeros(len(rows))
    if len(rows) > 1:
        dt = np.diff(t)
        dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(dt > 1e-9, dp / dt, 0.0)
        speed[1:] = v
        speed[0]  = v[0]
    return speed


def _per_gate_table(rows, title):
    gate_ids = sorted({r['matched_gate_id'] for r in rows})
    print(f"\n-- {title} (n={len(rows)}) " + "-" * max(0, 56 - len(title)))
    print(f"{'Gate':<5}{'n':>7}{'mean ex':>9}{'mean ey':>9}{'mean ez':>9}"
          f"{'mean|err|':>11}{'mean rng':>10}{'corr(rng,err)':>15}")
    for gid in gate_ids:
        sub = [r for r in rows if r['matched_gate_id'] == gid]
        if not sub:
            continue
        ex   = np.array([r['error_x'] for r in sub])
        ey   = np.array([r['error_y'] for r in sub])
        ez   = np.array([r['error_z'] for r in sub])
        emag = np.array([r['error_mag'] for r in sub])
        drone = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in sub])
        est   = np.array([[r['est_x'],   r['est_y'],   r['est_z']]   for r in sub])
        rng   = np.linalg.norm(est - drone, axis=1)
        corr  = np.corrcoef(rng, emag)[0, 1] if len(sub) > 1 else float('nan')
        print(f"{gid:<5}{len(sub):>7}{ex.mean():>9.2f}{ey.mean():>9.2f}{ez.mean():>9.2f}"
              f"{emag.mean():>11.2f}{rng.mean():>10.1f}{corr:>15.2f}")
    overall = np.array([r['error_mag'] for r in rows])
    print(f"  Overall: mean={overall.mean():.2f}  median={np.median(overall):.2f}  "
          f"max={overall.max():.2f}")


def _a14_check(rows, title):
    """
    Sequential-course sanity check (no flight_log alignment needed): genuine
    in-order detections should produce non-overlapping, monotonically
    increasing matched_gate_id windows over time. Overlap / out-of-sequence
    matches => nearest-centroid misattribution (A14) is materially present.
    """
    print(f"\n-- A14 check: {title} --")
    if not rows:
        print("  (no rows)")
        return None
    t0 = rows[0]['sim_time_ns']
    spans, seq = {}, []
    for r in rows:
        gid = r['matched_gate_id']
        t = (r['sim_time_ns'] - t0) * 1e-9
        if gid not in spans:
            spans[gid] = [t, t]
            seq.append(gid)
        else:
            spans[gid][1] = t
    print("  First-seen order:", seq)
    overlaps, prev_end, prev_gid = 0, None, None
    for gid in sorted(spans):
        s, e = spans[gid]
        flag = ''
        if prev_end is not None and s < prev_end:
            flag = f'  <-- overlaps gate {prev_gid} (ran to {prev_end:.1f}s)'
            overlaps += 1
        print(f"    gate {gid}: {s:7.2f}s -> {e:7.2f}s{flag}")
        prev_end, prev_gid = e, gid
    monotonic = seq == sorted(set(seq))
    print(f"  Sequential first-seen order: {monotonic}   Overlapping spans: {overlaps}")
    return overlaps == 0 and monotonic


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        est_path, gates_path = args[0], args[1]
    else:
        csvs  = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'gate_estimates_*.csv')), reverse=True)
        jsons = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'flight_log_*_gates.json')), reverse=True)
        est_path, gates_path = csvs[0], jsons[0]

    print(f"Estimates : {est_path}")
    print(f"Gates     : {gates_path}")
    rows  = _load(est_path)
    load_gates(gates_path)  # validates the file; not otherwise needed below

    deduped = _dedup_by_frame(rows)
    speed   = _speed_proxy(deduped)
    static_mask = speed < STATIC_SPEED_THRESHOLD
    flight  = [r for r, m in zip(deduped, static_mask) if not m]
    static  = [r for r, m in zip(deduped, static_mask) if m]

    print(f"\n{'='*72}")
    print(f"  RAW logged rows                 : {len(rows)}")
    print(f"  Unique camera frames (sim_time_ns) : {len(deduped)}   "
          f"(avg {len(rows)/len(deduped):.1f} repeat log-rows per frame)")
    print(f"  Of those genuine frames:")
    print(f"    static-drone (< {STATIC_SPEED_THRESHOLD} m/s)  : {len(static)} "
          f"({100*len(static)/len(deduped):.1f}%)")
    print(f"    in-flight    (>= {STATIC_SPEED_THRESHOLD} m/s) : {len(flight)} "
          f"({100*len(flight)/len(deduped):.1f}%)")
    print(f"{'='*72}")

    _per_gate_table(rows,    "RAW (unfiltered) -- what PERC.md Sec.1 was computed from")
    _per_gate_table(deduped, "DEDUPED (one row per genuine camera frame)")
    _per_gate_table(flight,  "DEDUPED + in-flight only (speed >= threshold)")

    print(f"\n-- Per-gate repeat factor (raw_n / deduped_n) --")
    from collections import Counter
    raw_c = Counter(r['matched_gate_id'] for r in rows)
    ddp_c = Counter(r['matched_gate_id'] for r in deduped)
    for gid in sorted(ddp_c):
        rf = raw_c.get(gid, 0) / ddp_c[gid]
        print(f"    gate {gid}: raw={raw_c.get(gid,0):6d}  deduped={ddp_c[gid]:5d}  repeat factor x{rf:.1f}")

    a_raw = _a14_check(rows,    "RAW data")
    a_ddp = _a14_check(deduped, "DEDUPED data")
    a_flt = _a14_check(flight,  "DEDUPED + in-flight data")

    print(f"\n{'='*72}")
    print(f"  VERDICT")
    print(f"  Repeat-processing inflation : x{len(rows)/len(deduped):.1f}  "
          f"({len(rows)} logged rows / {len(deduped)} genuine frames)")
    print(f"  A14 clean on RAW data       : {a_raw}")
    print(f"  A14 clean on DEDUPED data   : {a_ddp}")
    print(f"  A14 clean on DEDUPED+flight : {a_flt}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
