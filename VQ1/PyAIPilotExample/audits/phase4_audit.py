"""
Phase 4 audit -- robustness characterization (PERC.md Sec.6 Phase 4).

A8 (HSV universality / false positives) and A4 (largest-contour / multi-gate
ambiguity) were originally framed as needing fresh data ("characterize where
HSV bands break", "probe scenarios where >1 gate is visible"). But the
*original full-course race run* (`run_20260607_012243`, 6 gates strung along
a ~140 m corridor, 43k logged CV detections) already contains everything
needed for a first-order geometric pass on both -- no new flight required:

  - The simulator confirms which gate the controller is *currently chasing*
    at every instant (`flight_log_*.csv: active_gate`). Time-aligning that
    onto each CV detection (`gate_estimates_*.csv: sim_time_ns`) gives a
    ground-truth answer to "what should the camera reasonably be seeing
    right now?" -- independent of (and a sharper test than) raw nearest-
    gate-by-distance, which doesn't know which way the camera is pointed.

  - Phase 3's `range_sweep` already established the empirical CV detection
    envelope (~22 m: clean detections inside, none confirmed beyond). Any
    detection that confidently reports a gate *substantially farther than
    that* is, by that same empirical standard, geometrically implausible --
    a candidate false positive (A8) or gate-confusion event (A4).

This script mines exactly those two signals from the existing race-run logs.

Usage:
    python phase4_audit.py                       # auto-picks latest non-CALIBRATE run
    python phase4_audit.py <run_dir>
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Empirical CV detection envelope from Phase 3's range_sweep (PERC.md Sec.6
# Phase 3 PASS, A1/A9/A10): clean, confirmed detections out to ~22m, none
# beyond. A confidently-reported detection of a gate *substantially* farther
# than this is implausible on its face.
DETECTION_ENVELOPE_M = 22.0
# A "confident" detection -- low enough error_mag that it's very unlikely to
# be PnP noise on the right gate; i.e. it really did see *something* clearly
# shaped like a gate, just (per the mismatch) possibly the wrong one.
CONFIDENT_ERROR_MAG_M = 5.0


def _find_run_dir():
    runs = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*')), reverse=True)
    for r in runs:
        # Skip calibration runs (no full 6-gate course; they orbit gate 0 only).
        if glob.glob(os.path.join(r, 'calibration_log.csv')):
            continue
        if glob.glob(os.path.join(r, '*_gates.json')):
            return r
    raise SystemExit('No full-course (non-calibration) run found.')


def _load_gates(run_dir):
    paths = glob.glob(os.path.join(run_dir, '*_gates.json'))
    with open(paths[0]) as f:
        gates = json.load(f)
    gates.sort(key=lambda g: g['id'])
    return np.array([g['pos'] for g in gates])          # (G, 3)


def _load_flight_log(run_dir):
    paths = glob.glob(os.path.join(run_dir, 'flight_log_*.csv'))
    paths = [p for p in paths if not p.endswith('_gates.json')]
    with open(paths[0], newline='') as f:
        rows = [
            (float(r['wall_time_s']), int(r['active_gate']))
            for r in csv.DictReader(f)
        ]
    rows.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in rows])
    ag = np.array([r[1] for r in rows])
    return t, ag


def _load_estimates_dedup(run_dir):
    paths = glob.glob(os.path.join(run_dir, 'gate_estimates_*.csv'))
    with open(paths[0], newline='') as f:
        raw = list(csv.DictReader(f))
    seen, rows = set(), []
    for r in raw:
        st = int(r['sim_time_ns'])
        if st in seen:                      # Phase 0 lesson: ~17x re-logged per frame
            continue
        seen.add(st)
        rows.append({
            'sim_time_ns': st,
            'wall_time_s': st / 1e9,         # confirmed same epoch as flight_log wall_time_s
            'est':   np.array([float(r['est_x']),   float(r['est_y']),   float(r['est_z'])]),
            'drone': np.array([float(r['drone_x']), float(r['drone_y']), float(r['drone_z'])]),
            'mav':   np.array([float(r['mav_x']),   float(r['mav_y']),   float(r['mav_z'])]),
            'matched_gate_id': int(r['matched_gate_id']),
            'error_mag': float(r['error_mag']),
        })
    rows.sort(key=lambda r: r['sim_time_ns'])
    return rows


def _attach_active_gate(rows, flog_t, flog_ag):
    """Nearest-neighbour time alignment onto the sim-confirmed active_gate
    series -- ground truth for "what should the camera be chasing right now"."""
    times = np.array([r['wall_time_s'] for r in rows])
    idx = np.searchsorted(flog_t, times)
    idx = np.clip(idx, 1, len(flog_t) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(flog_t[left] - times) <= np.abs(flog_t[right] - times)
    chosen = np.where(use_left, left, right)
    for r, c in zip(rows, chosen):
        r['active_gate'] = int(flog_ag[c])


def _a4_multi_gate_ambiguity(rows, gate_pos):
    print(f"\n{'='*72}\nA4 -- largest-contour / multi-gate ambiguity\n"
          f"  (does CV ever confidently report a DIFFERENT gate than the one\n"
          f"   the simulator confirms the controller is currently chasing?)\n{'='*72}")

    drone = np.array([r['drone'] for r in rows])
    mid   = np.array([r['matched_gate_id'] for r in rows])
    ag    = np.array([r['active_gate'] for r in rows])
    emag  = np.array([r['error_mag'] for r in rows])

    dists = np.linalg.norm(drone[:, None, :] - gate_pos[None, :, :], axis=2)   # (N, G)
    n_plausible = (dists < DETECTION_ENVELOPE_M).sum(axis=1)

    mismatch = (mid != ag)
    confident_mismatch = mismatch & (emag < CONFIDENT_ERROR_MAG_M)

    print(f"  n={len(rows)} (deduped, unique-frame CV detections)")
    print(f"  active_gate==-1 (not started) / =={len(gate_pos)} (finished, past last gate) excluded")
    valid = (ag >= 0) & (ag < len(gate_pos))
    print(f"    -> {valid.sum()} valid rows (race in progress, valid gate index)")

    mismatch, confident_mismatch, n_plausible = mismatch[valid], confident_mismatch[valid], n_plausible[valid]
    mid_v, ag_v, emag_v = mid[valid], ag[valid], emag[valid]

    print(f"\n  matched_gate_id != active_gate           : {mismatch.sum():5d} / {valid.sum()} ({100*mismatch.mean():.1f}%)")
    print(f"    of which CONFIDENT (error_mag < {CONFIDENT_ERROR_MAG_M:.0f}m)    : {confident_mismatch.sum():5d}  "
          f"({'<-- real ambiguity events: CV clearly saw a gate, just not the active one' if confident_mismatch.sum() else '(none -- mismatches are all noisy/low-confidence)'})")
    print(f"  mean error_mag | matched==active          : {emag_v[~mismatch].mean():.2f} m")
    print(f"  mean error_mag | matched!=active          : {emag_v[mismatch].mean():.2f} m")

    if confident_mismatch.sum():
        pairs = Counter(zip(ag_v[confident_mismatch].tolist(), mid_v[confident_mismatch].tolist()))
        print(f"\n  Confident-mismatch (active_gate -> matched_gate_id) pairings:")
        for (a, m), c in pairs.most_common(10):
            sep = float(np.linalg.norm(gate_pos[a] - gate_pos[m]))
            rel = 'NEXT' if m == a + 1 else ('PREV' if m == a - 1 else f'{m-a:+d}')
            print(f"    gate {a} -> gate {m}  ({rel}, {sep:5.1f}m apart)   n={c:4d}   "
                  f"mean error_mag={emag_v[(ag_v==a)&(mid_v==m)&confident_mismatch].mean():.2f}m")
        print(f"\n  Verdict: CV repeatedly, confidently locks onto a NEIGHBOURING gate while the\n"
              f"  controller is still chasing the current one -- exactly the 'largest contour\n"
              f"  picks whichever blob looks biggest, not the one the controller wants' failure\n"
              f"  mode A4 worried about. {'It is concentrated on adjacent-gate pairs (the natural confusion: seeing the next gate through/around the current one), not random.' if all(abs(m-a)==1 for (a,m),_ in pairs.most_common(len(pairs))) else 'It is NOT confined to adjacent gates -- worth a closer look at why.'}")
    else:
        print(f"\n  Verdict: raw mismatches exist ({mismatch.sum()}), but essentially none are\n"
              f"  *confident* (error_mag < {CONFIDENT_ERROR_MAG_M:.0f}m) -- they're noisy detections of the\n"
              f"  active gate that happen to nearest-match a different gate's MAVLink position\n"
              f"  by coincidence, not genuine 'CV locked onto the wrong gate' events. A4 looks\n"
              f"  like a non-issue *in this course's geometry* (a single-file corridor where\n"
              f"  gates rarely overlap in the frame) -- though that's itself the finding: this\n"
              f"  course never actually exercises the failure mode A4 worries about.")

    multi = n_plausible >= 2
    print(f"\n  Rows with >=2 gates within the {DETECTION_ENVELOPE_M:.0f}m detection envelope: "
          f"{multi.sum()} / {valid.sum()} ({100*multi.mean():.1f}%)")
    if multi.sum():
        print(f"    mismatch rate there      : {100*mismatch[multi].mean():.1f}%   "
              f"(vs. {100*mismatch[~multi].mean():.1f}% when only one gate is plausible)")
        print(f"    confident-mismatch rate  : {100*confident_mismatch[multi].mean():.2f}%   "
              f"(vs. {100*confident_mismatch[~multi].mean():.2f}% when only one gate is plausible)")


def _a8_false_positives(rows, gate_pos):
    print(f"\n{'='*72}\nA8 -- HSV universality / false-positive characterization\n"
          f"  (detections claiming a gate well beyond the empirically-established\n"
          f"   ~{DETECTION_ENVELOPE_M:.0f}m detection envelope are geometrically implausible --\n"
          f"   candidates for 'the detector keyed onto something that wasn't a gate')\n{'='*72}")

    drone = np.array([r['drone'] for r in rows])
    mav   = np.array([r['mav']   for r in rows])
    est   = np.array([r['est']   for r in rows])
    mid   = np.array([r['matched_gate_id'] for r in rows])
    ag    = np.array([r['active_gate'] for r in rows])
    emag  = np.array([r['error_mag'] for r in rows])
    t     = np.array([r['wall_time_s'] for r in rows])

    valid = (ag >= 0) & (ag < len(gate_pos))
    drone, mav, est, mid, ag, emag, t = (a[valid] for a in (drone, mav, est, mid, ag, emag, t))

    range_to_matched = np.linalg.norm(drone - mav, axis=1)
    implausible = range_to_matched > DETECTION_ENVELOPE_M

    print(f"  n={valid.sum()} valid rows")
    print(f"  range_to_matched_gate: mean={range_to_matched.mean():.1f}m  "
          f"median={np.median(range_to_matched):.1f}m  p90={np.percentile(range_to_matched,90):.1f}m  "
          f"max={range_to_matched.max():.1f}m")
    print(f"\n  Implausible (range_to_matched > {DETECTION_ENVELOPE_M:.0f}m): "
          f"{implausible.sum()} / {valid.sum()} ({100*implausible.mean():.1f}%)")
    print(f"    mean error_mag  : {emag[implausible].mean():.2f}m  (vs. {emag[~implausible].mean():.2f}m for plausible detections)")
    print(f"    mean range      : {range_to_matched[implausible].mean():.1f}m")

    # How many of these are "the camera is just looking down a long straight at
    # a gate that's genuinely far ahead/behind, and CV got a (noisy) lock on it"
    # vs. "the est position doesn't correspond to ANY real gate nearby" (true FP)?
    dists_est_to_gates = np.linalg.norm(est[:, None, :] - gate_pos[None, :, :], axis=2)
    nearest_gate_to_est = dists_est_to_gates.min(axis=1)
    orphan_est = nearest_gate_to_est > 5.0   # est itself doesn't sit near ANY real gate

    print(f"\n  Of the implausible ones, how many also have an `est` that sits nowhere\n"
          f"  near ANY real gate (>5m from every gate centre -- a TRUE orphan, the\n"
          f"  closest thing to direct evidence of a non-gate detection)?")
    orphan_implausible = implausible & orphan_est
    print(f"    {orphan_implausible.sum()} / {implausible.sum()} implausible rows "
          f"({100*orphan_implausible[implausible].mean() if implausible.sum() else 0:.1f}%)")

    if orphan_implausible.sum():
        # Where do these cluster -- by active_gate segment (i.e. which leg of
        # the course), and spatially (do `est` positions cluster -- a fixed
        # decoy -- or scatter -- pure noise)?
        seg_counts = Counter(ag[orphan_implausible].tolist())
        print(f"\n    By course segment (active_gate at the time):")
        for g, c in sorted(seg_counts.items()):
            print(f"      active_gate={g}: {c:4d} orphan detections")
        clusters = est[orphan_implausible]
        centroid = clusters.mean(axis=0)
        spread = clusters.std(axis=0)
        print(f"\n    Orphan `est` centroid = ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})  "
              f"spread (std) = ({spread[0]:.1f}, {spread[1]:.1f}, {spread[2]:.1f})")
        tight = np.all(spread < 5.0)
        print(f"    {'TIGHT cluster -- consistent with a fixed real-world decoy object (e.g. an orange prop/wall feature) being repeatedly mis-detected.' if tight else 'Diffuse / scattered -- consistent with transient noise (motion blur, partial occlusion, lighting flicker) rather than one consistent decoy object.'}")
    else:
        print(f"\n  Verdict: every 'implausible-range' detection still resolves to an `est`\n"
              f"  that sits near a REAL gate -- i.e. these aren't false positives on\n"
              f"  non-gate objects, they're just badly-biased-but-real detections (likely\n"
              f"  the same long-range/oblique-angle effects Phases 1.1 and 3 already\n"
              f"  quantified, pushing `est` far enough from `mav` that the distance from\n"
              f"  `drone` to `mav` alone looks implausible). A8 (HSV false-positives on\n"
              f"  non-gate orange objects) does not manifest anywhere in this course --\n"
              f"  consistent with VQ1 being the friendliest environment for colour-keyed\n"
              f"  segmentation (PERC.md Sec.5.2.e); the real test will be VQ2's "
              f"3D-scanned scenery.")


def main():
    args = sys.argv[1:]
    run_dir = args[0] if args else _find_run_dir()
    print(f"Run: {run_dir}")

    gate_pos = _load_gates(run_dir)
    print(f"Course: {len(gate_pos)} gates")

    flog_t, flog_ag = _load_flight_log(run_dir)
    rows = _load_estimates_dedup(run_dir)
    _attach_active_gate(rows, flog_t, flog_ag)
    print(f"CV detections (deduped): {len(rows)}")

    _a4_multi_gate_ambiguity(rows, gate_pos)
    _a8_false_positives(rows, gate_pos)


if __name__ == '__main__':
    main()
