"""
Phase 1.2 audit -- visual + quantitative scoring of A2/A5/A7 (PERC.md Sec.6
Phase 1.2), using the Phase-2 instrumentation (detection_diag_*.csv, the
extended gate_estimates_*.csv with rvec/corners, and saved frames/).

Three concrete targets handed down from Phases 1.1/3/4 (see PERC.md "suggested
execution order", item 5):

  (a) A6/A5/A7 rotation-chain side: the +15 deg -> +25 deg oblique-angle
      transition where Phase 3 found CV detections cut off sharply.
  (b) A1 detection side: long range (>= 15 m) -- does the detected quad
      visibly "fatten" relative to the gate's true outer edge as range grows?
  (c) A8 "the big one": pull frames from inside a burst of physically-
      impossible detections and see directly what gate_detector locks onto.

Plus the two filter/convergence audits the new diag CSV makes possible
straight from the numbers, no visual inspection required:

  A5 -- aspect/area filter calibration: among frames with a gate-sized blob,
        how often does the aspect filter reject it, and by how much?
  A7 -- approxPolyDP convergence: when a gate-sized blob IS found, how many
        points does its hull/poly actually have, and at what epsilon (if any)
        does it reduce to 4?

Usage:
    python phase1_2_audit.py                 # auto-picks latest instrumented run
    python phase1_2_audit.py <run_dir>
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate_detector import draw_detection, MIN_GATE_AREA_PX
from pose_estimator import reproject_corners
from planner import load_gates, gate_center, gate_approach_dir
from phase3_audit import _wrap, _stats

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

_FLOAT_COLS = (
    'tvec_x', 'tvec_y', 'tvec_z', 'rvec_x', 'rvec_y', 'rvec_z',
    'corner_tl_x', 'corner_tl_y', 'corner_tr_x', 'corner_tr_y',
    'corner_br_x', 'corner_br_y', 'corner_bl_x', 'corner_bl_y',
    'roll', 'pitch', 'yaw',
    'est_x', 'est_y', 'est_z', 'drone_x', 'drone_y', 'drone_z',
    'mav_x', 'mav_y', 'mav_z', 'error_x', 'error_y', 'error_z', 'error_mag',
)

_DIAG_FLOAT_COLS = (
    'drone_x', 'drone_y', 'drone_z', 'roll', 'pitch', 'yaw',
    'largest_area', 'largest_aspect', 'best_area', 'best_aspect', 'four_corner_eps',
)
_DIAG_INT_COLS = ('n_contours', 'n_candidates', 'best_hull_pts', 'best_poly_pts')


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_estimates_full(path):
    """Like phase3_audit._load_estimates, but also parses the Phase-2 columns
    (rvec/corners) this audit needs and weren't there for earlier phases."""
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            r = dict(row)
            r['sim_time_ns']     = int(r['sim_time_ns'])
            r['matched_gate_id'] = int(r['matched_gate_id'])
            for k in _FLOAT_COLS:
                r[k] = float(r[k])
            rows.append(r)
    rows.sort(key=lambda r: r['sim_time_ns'])
    seen, out = set(), []
    for r in rows:
        if r['sim_time_ns'] not in seen:
            seen.add(r['sim_time_ns'])
            out.append(r)
    return out


def _load_diag(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            r = dict(row)
            r['sim_time_ns'] = int(r['sim_time_ns'])
            for k in _DIAG_FLOAT_COLS:
                r[k] = float(r[k]) if r[k] not in ('', None) else float('nan')
            for k in _DIAG_INT_COLS:
                r[k] = int(float(r[k])) if r[k] not in ('', None) else None
            rows.append(r)
    rows.sort(key=lambda r: r['sim_time_ns'])
    return rows


def _corners_array(r):
    return np.array([
        [r['corner_tl_x'], r['corner_tl_y']],
        [r['corner_tr_x'], r['corner_tr_y']],
        [r['corner_br_x'], r['corner_br_y']],
        [r['corner_bl_x'], r['corner_bl_y']],
    ], dtype=np.float32)


# ── Course geometry: per-gate "dead-on" reference bearing ────────────────────
#
# Phase 3's _geometric_dead_on_bearing was scoped to gate 0 and the
# calibration flight's pre-CALIBRATE start position. A racing flight passes
# every gate, so we need the SAME idea -- "the bearing from which this gate is
# normally approached, derived purely from course geometry" -- generalised to
# every gate. gate_approach_dir() already encodes exactly that (course
# quaternion + previous-gate direction, with a centre-line fallback); chaining
# it down the gate sequence gives an exact, CV-independent reference for each
# gate with no new geometry to derive.

def _per_gate_ref_bearing(gates):
    centers = [gate_center(g) for g in gates]
    ref = {}
    prev = centers[0] - gate_approach_dir(gates[0], centers[0] - np.array([1.0, 0.0, 0.0]), centers[0]) * 12.0
    for g, c in zip(gates, centers):
        approach = gate_approach_dir(g, prev, c)
        # Dead-on bearing = direction FROM the gate back toward its approach
        # origin (mirrors Phase 3's `boresight_yaw + pi`); oblique angle is
        # then "how far off that line is the drone, seen from the gate".
        ref[g['id']] = _wrap(float(np.arctan2(-approach[1], -approach[0])))
        prev = c
    return ref


def _oblique_angle(r, ref_bearing):
    bearing = float(np.arctan2(r['drone_y'] - r['mav_y'], r['drone_x'] - r['mav_x']))
    return float(np.degrees(_wrap(bearing - ref_bearing[r['matched_gate_id']])))


# ── A5: aspect/area filter calibration ───────────────────────────────────────

def audit_filter_calibration(diag_rows):
    print(f"\n{'='*72}\nA5 -- aspect-ratio / min-area filter calibration\n"
          f"  (0.3 < w/h < 3.0 and area >= {MIN_GATE_AREA_PX} px^2 -- do they match real gate silhouettes?)\n{'='*72}")

    plausible = [r for r in diag_rows if not np.isnan(r['largest_area']) and r['largest_area'] >= MIN_GATE_AREA_PX]
    if not plausible:
        print("  No frames with a gate-sized blob -- cannot calibrate the filter.")
        return
    rejected = [r for r in plausible if r['detect_result'] == 'no_candidates']
    passed   = [r for r in plausible if r['detect_result'] != 'no_candidates']

    print(f"  Frames with a gate-sized blob present (area >= {MIN_GATE_AREA_PX} px^2): {len(plausible)}")
    print(f"    -- passed the aspect filter            : {len(passed):5d}  ({100*len(passed)/len(plausible):.1f}%)")
    print(f"    -- REJECTED by the aspect filter alone : {len(rejected):5d}  ({100*len(rejected)/len(plausible):.1f}%)")

    if passed:
        a = np.array([r['largest_aspect'] for r in passed])
        print(f"\n  Aspect ratio of PASSING gate-sized blobs:")
        print(f"  {_stats('w/h', a)}")
    if rejected:
        a = np.array([r['largest_aspect'] for r in rejected])
        print(f"\n  Aspect ratio of REJECTED gate-sized blobs (the filter's near-misses):")
        print(f"  {_stats('w/h', a)}")
        near = np.sum((a > 0.15) & (a < 0.3)) + np.sum((a > 3.0) & (a < 6.0))
        far  = np.sum((a <= 0.15) | (a >= 6.0))
        print(f"    -- just outside the [0.3, 3.0] band (0.15-0.3 or 3-6)   : {near:4d}  "
              f"{'<-- the filter may be too strict; these are plausibly real gates at oblique angles' if near > 0.3*len(rejected) else ''}")
        print(f"    -- wildly outside (<0.15 or >6)                          : {far:4d}  "
              f"{'<-- almost certainly NOT gates; filter correctly rejects these' if far > 0.5*len(rejected) else ''}")
        verdict = ('the filter band looks WELL-CALIBRATED: rejected blobs are mostly wildly non-square '
                   '(true non-gate clutter), not marginal gate views the filter is wrongly excluding'
                   if far > 2 * max(near, 1) else
                   'a meaningful fraction of rejections are MARGINAL (just outside the band) -- '
                   'worth checking visually whether these are real gates at steep oblique angles being dropped')
        print(f"\n  Verdict: {verdict}")
    else:
        print(f"\n  Verdict: every gate-sized blob this flight passed the aspect filter -- "
              f"no evidence of the filter rejecting plausible gates (A5 non-issue on this course).")


# ── A7: approxPolyDP convergence ─────────────────────────────────────────────

def audit_polygon_convergence(diag_rows):
    print(f"\n{'='*72}\nA7 -- approxPolyDP convergence to exactly 4 points\n"
          f"  (no fallback exists -- known gap #8; does the lack of one cost real detections?)\n{'='*72}")

    attempted = [r for r in diag_rows if r['detect_result'] in ('ok', 'no_4corner')]
    if not attempted:
        print("  No frames reached the polygon-reduction stage -- cannot score.")
        return
    ok    = [r for r in attempted if r['detect_result'] == 'ok']
    failed = [r for r in attempted if r['detect_result'] == 'no_4corner']
    print(f"  Candidate blobs that reached approxPolyDP: {len(attempted)}")
    print(f"    -- converged to 4 points  : {len(ok):5d}  ({100*len(ok)/len(attempted):.1f}%)")
    print(f"    -- FAILED to converge     : {len(failed):5d}  ({100*len(failed)/len(attempted):.1f}%)")

    if ok:
        eps = np.array([r['four_corner_eps'] for r in ok])
        print(f"\n  Epsilon at which converging shapes reduced to 4 points (smaller = cleaner silhouette):")
        for val in sorted(set(eps.tolist())):
            n = int(np.sum(eps == val))
            print(f"    eps={val:.2f} : {n:5d}  ({100*n/len(ok):.1f}%)")
        frac_clean = float(np.mean(eps <= 0.06))
        print(f"  -- {100*frac_clean:.1f}% converged at the two finest epsilons (0.04/0.06): "
              f"{'silhouettes are mostly clean, near-quadrilateral shapes' if frac_clean > 0.7 else 'a substantial share need heavy simplification -- noisier silhouettes than a clean gate view'}")

    if failed:
        hp = np.array([r['best_hull_pts'] for r in failed])
        pp = np.array([r['best_poly_pts'] for r in failed])
        print(f"\n  Shapes that NEVER reduced to 4 points -- how far off were they?")
        print(f"  {_stats('hull_pts', hp)}")
        print(f"  {_stats('poly_pts', pp)}  (at the finest epsilon, 0.04)")
        close = int(np.sum((pp == 3) | (pp == 5)))
        print(f"    -- within 1 point of 4 (poly_pts in {{3,5}}) at finest eps: {close:4d}  "
              f"{'<-- a fallback (e.g. accept the best 4-of-N or relax eps further) would likely recover these' if close > 0.3*len(failed) else ''}")
        print(f"\n  Verdict: {'most non-convergent shapes are CLOSE to 4 points -- a fallback (known gap #8) would recover a meaningful number of dropped detections' if close > 0.3 * len(failed) else 'most non-convergent shapes are FAR from 4 points (genuinely degenerate clutter) -- a fallback would mostly just admit garbage; the lack of one is not costing many real detections'}")
    else:
        print(f"\n  Verdict: every gate-sized candidate this flight converged to 4 points -- "
              f"no evidence approxPolyDP is silently dropping real detections (A7 non-issue on this course).")


# ── Range / "fattening" relationship (A1 visual confirmation, target b) ──────

def audit_range_fattening(est_rows):
    print(f"\n{'='*72}\nA1 visual confirmation -- does the detected quad 'fatten' / does z-error\n"
          f"grow at long range? (Phase 1.2 target b: range >= 15 m)\n{'='*72}")

    rng  = np.array([float(np.linalg.norm([r['tvec_x'], r['tvec_y'], r['tvec_z']])) for r in est_rows])
    bbox = []
    for r in est_rows:
        c = _corners_array(r)
        bbox.append(float((c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min())) ** 0.5)
    bbox = np.array(bbox)
    ez   = np.array([r['error_z'] for r in est_rows])
    emag = np.array([r['error_mag'] for r in est_rows])

    long_range = rng >= 15.0
    print(f"  n={len(est_rows)} detections; {long_range.sum()} ({100*long_range.mean():.1f}%) at range >= 15 m")
    if long_range.sum() < 5:
        print("  Too few long-range detections to characterise -- skipping.")
        return

    print(f"\n  Apparent quad size (sqrt(bbox area), px) vs. range:")
    for lo, hi, name in [(0, 8, '<8m'), (8, 15, '8-15m'), (15, 25, '15-25m'), (25, 999, '>=25m')]:
        m = (rng >= lo) & (rng < hi)
        if m.sum() < 3:
            continue
        print(f"    range {name:7s} (n={m.sum():4d}): mean px-size={bbox[m].mean():6.1f}  "
              f"mean|error_z|={np.abs(ez[m]).mean():6.2f}m  mean|error|={emag[m].mean():6.2f}m")

    corr_size = float(np.corrcoef(rng, bbox)[0, 1])
    corr_ez   = float(np.corrcoef(rng[long_range], np.abs(ez[long_range]))[0, 1]) if long_range.sum() > 2 else float('nan')
    print(f"\n  corr(range, apparent_px_size)        = {corr_size:+.2f}  (expected strongly negative -- farther = smaller)")
    print(f"  corr(range, |error_z|) at >=15 m      = {corr_ez:+.2f}  "
          f"{'<-- z-error keeps growing with range even past 15 m: A1 range-scaled bias, not a fixed offset' if corr_ez > 0.3 else ''}")
    print(f"\n  Verdict: {'CONFIRMS the range-scaled segmentation-bias hypothesis behind A1 -- depth error grows with range because PnP infers depth from apparent (px) size, and pixel-level segmentation noise is a near-constant absolute quantity that becomes a LARGER proportional error as the silhouette shrinks.' if corr_ez > 0.3 else 'no clear range-scaling trend in this sample -- A1 bias may be closer to a fixed offset than a range-scaled one on this course.'}")


# ── "Impossible" detections: range-bias tail vs. fixed-feature lock-on ───────

def audit_impossible_detections(est_rows, gates):
    print(f"\n{'='*72}\nA8 cross-check -- physically-impossible detections on THIS course:\n"
          f"  range-bias tail (A1, real gate, badly-estimated depth) or\n"
          f"  fixed-feature lock-on (Phase 4's mechanism: est barely moves while drone does)?\n{'='*72}")

    centers_z = np.array([gate_center(g)[2] for g in gates])
    lo, hi = float(centers_z.min()), float(centers_z.max())
    margin = 5.0
    print(f"  Course gate-centre z envelope (NED): [{lo:.1f}, {hi:.1f}] m  "
          f"({'descends' if centers_z[-1] > centers_z[0] else 'climbs'} as the course progresses)")
    print(f"  Flagging detections with est_z outside [{lo-margin:.1f}, {hi+margin:.1f}] (envelope +/- {margin:.0f} m margin) "
          f"as 'impossible' -- NOTE: this bound is COURSE-SPECIFIC, recomputed from this run's own gate map "
          f"(Phase 4's fixed '+10 m' bound does not transfer between courses with different altitude profiles).")

    est_z = np.array([r['est_z'] for r in est_rows])
    impossible = (est_z < lo - margin) | (est_z > hi + margin)
    print(f"\n  n={len(est_rows)}  impossible: {impossible.sum()} ({100*impossible.mean():.1f}%)")
    if not impossible.any():
        print("  No impossible detections this flight -- nothing to characterise.")
        return

    idx = np.where(impossible)[0]
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev)); start = prev = i
    runs.append((start, prev))
    runs = [r for r in runs if r[1] - r[0] + 1 >= 20]
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    print(f"  bursts of >=20 consecutive impossible detections: {len(runs)}")

    for s, e in runs[:5]:
        sub = est_rows[s:e + 1]
        est   = np.array([[r['est_x'], r['est_y'], r['est_z']] for r in sub])
        drone = np.array([[r['drone_x'], r['drone_y'], r['drone_z']] for r in sub])
        rng   = np.array([float(np.linalg.norm([r['tvec_x'], r['tvec_y'], r['tvec_z']])) for r in sub])
        d_est   = float(np.linalg.norm(est[-1] - est[0]))
        d_drone = float(np.linalg.norm(drone[-1] - drone[0]))
        ratio = d_est / d_drone if d_drone > 1e-6 else float('inf')
        t0, t1 = sub[0]['sim_time_ns'] / 1e9, sub[-1]['sim_time_ns'] / 1e9
        mech = ('FIXED-FEATURE LOCK-ON (Phase-4-style): est barely moves while the drone does -- '
                'the pipeline is re-estimating the pose of one fixed real-world object'
                if ratio < 0.3 else
                'RANGE-BIAS TAIL (A1-style): est moves roughly with the drone and converges toward the '
                'matched gate as range closes -- a real (likely correctly-identified) gate, badly '
                'mis-ranged by PnP at long apparent distance')
        print(f"\n    burst t=[{t0:.1f},{t1:.1f}]s  n={e-s+1}  matched_gate={sub[0]['matched_gate_id']}")
        print(f"      est moved {d_est:6.1f} m  |  drone moved {d_drone:6.1f} m  |  ratio={ratio:.2f}")
        print(f"      range (camera->detected obj): {rng.min():.1f}-{rng.max():.1f} m")
        print(f"      mechanism: {mech}")

    print(f"\n  Verdict: {'at least one burst shows the FIXED-FEATURE signature -- A8 (non-gate lock-on) is reproduced on this course too.' if any(np.linalg.norm(np.array([est_rows[s]['est_x'],est_rows[s]['est_y'],est_rows[s]['est_z']])-np.array([est_rows[e]['est_x'],est_rows[e]['est_y'],est_rows[e]['est_z']])) / max(np.linalg.norm(np.array([est_rows[s]['drone_x'],est_rows[s]['drone_y'],est_rows[s]['drone_z']])-np.array([est_rows[e]['drone_x'],est_rows[e]['drone_y'],est_rows[e]['drone_z']])),1e-6) < 0.3 for s,e in runs[:5]) else 'NONE of the bursts on this course show the fixed-feature signature -- every long impossible run here is the A1 range-bias tail on a real (correctly-identified) gate, not a non-gate lock-on. A8 still stands (Phase 4 reproduced it cleanly elsewhere), but it is evidently NOT the only -- or even the dominant -- source of out-of-envelope detections; a z-bound sanity filter would catch both mechanisms regardless of which produced them.'}")


# ── Visual frame selection + overlay (target a, b, c) ────────────────────────

def _annotate(img, r, ref_bearing, gates_by_id):
    out = img.copy()
    corners = _corners_array(r)
    out = draw_detection(out, corners)
    rvec = np.array([r['rvec_x'], r['rvec_y'], r['rvec_z']])
    tvec = np.array([r['tvec_x'], r['tvec_y'], r['tvec_z']])
    try:
        proj = reproject_corners(tvec, rvec)
        pts = proj.astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    except cv2.error:
        pass
    rng = float(np.linalg.norm(tvec))
    angle = _oblique_angle(r, ref_bearing)
    lines = [
        f"gate {r['matched_gate_id']}  range={rng:5.1f}m  oblique={angle:+5.1f} deg",
        f"err_mag={r['error_mag']:5.2f}m  err_z={r['error_z']:+5.2f}m",
        f"est_z={r['est_z']:+5.1f}  mav_z={r['mav_z']:+5.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (4, 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (4, 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def select_and_save_frames(est_rows, gates, ref_bearing, frames_dir, out_dir):
    print(f"\n{'='*72}\nA2 visual audit -- selecting representative frames for the three Phase 1.2\n"
          f"targets and overlaying detected corners (green) + reproject_corners()\n"
          f"(magenta) for visual inspection\n{'='*72}")

    available = {}
    for f in glob.glob(os.path.join(frames_dir, '*_ok.jpg')):
        ts = int(os.path.basename(f).split('_')[0])
        available[ts] = f
    if not available:
        print(f"  No saved 'ok' frames in {frames_dir} -- cannot produce overlays.")
        return

    angle = np.array([_oblique_angle(r, ref_bearing) for r in est_rows])
    rng   = np.array([float(np.linalg.norm([r['tvec_x'], r['tvec_y'], r['tvec_z']])) for r in est_rows])

    targets = {
        'a_oblique_15_25': [r for r, a in zip(est_rows, angle) if 15.0 <= abs(a) <= 25.0 and r['sim_time_ns'] in available],
        'b_long_range':    [r for r, d in zip(est_rows, rng)   if d >= 15.0 and r['sim_time_ns'] in available],
    }

    os.makedirs(out_dir, exist_ok=True)
    n_saved = 0
    for name, cands in targets.items():
        if not cands:
            print(f"  [{name}] no saved frames matched this criterion -- nothing to overlay.")
            continue
        # Spread the picks across the available range/angle span, not just time
        if name == 'a_oblique_15_25':
            cands.sort(key=lambda r: abs(_oblique_angle(r, ref_bearing)))
        else:
            cands.sort(key=lambda r: float(np.linalg.norm([r['tvec_x'], r['tvec_y'], r['tvec_z']])))
        picks = [cands[i] for i in np.linspace(0, len(cands) - 1, min(6, len(cands))).astype(int)]
        print(f"  [{name}] {len(cands)} candidates with saved frames -- writing {len(picks)} annotated samples")
        for r in picks:
            img = cv2.imread(available[r['sim_time_ns']])
            if img is None:
                continue
            ann = _annotate(img, r, ref_bearing, {g['id']: g for g in gates})
            path = os.path.join(out_dir, f"{name}_{r['sim_time_ns']}.jpg")
            cv2.imwrite(path, ann)
            n_saved += 1
    print(f"\n  Wrote {n_saved} annotated frames to {out_dir}")
    print(f"  -- inspect these by eye: does the magenta reproject_corners() outline trace the same\n"
          f"     real-world edge the green detected silhouette does, and does that edge visibly\n"
          f"     correspond to the gate's TRUE outer boundary (vs. inner opening, side panel, or\n"
          f"     something else entirely)? That visual judgement is what scores A2 -- no script\n"
          f"     can substitute for looking at the images.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _auto_find_run():
    diags = sorted(glob.glob(os.path.join(_LOG_DIR, 'run_*', 'detection_diag_*.csv')), reverse=True)
    if not diags:
        print('No detection_diag_*.csv found -- need a Phase-2-instrumented flight (see PERC.md Phase 1.2).')
        sys.exit(1)
    return os.path.dirname(diags[0])


def main():
    args = sys.argv[1:]
    run_dir = args[0] if args else _auto_find_run()
    print(f"Run directory: {run_dir}")

    diag_path = sorted(glob.glob(os.path.join(run_dir, 'detection_diag_*.csv')))[-1]
    est_path  = sorted(glob.glob(os.path.join(run_dir, 'gate_estimates_*.csv')))[-1]
    gates_path = sorted(glob.glob(os.path.join(run_dir, 'flight_log_*_gates.json')))[-1]
    frames_dir = os.path.join(run_dir, 'frames')
    out_dir    = os.path.join(run_dir, 'phase1_2_frames')

    print(f"Detection diag : {diag_path}")
    print(f"Gate estimates : {est_path}")
    print(f"Gate map       : {gates_path}")

    diag_rows = _load_diag(diag_path)
    est_rows  = _load_estimates_full(est_path)
    gates     = load_gates(gates_path)
    print(f"Loaded {len(diag_rows)} detection-diag rows, {len(est_rows)} unique CV detections, "
          f"{len(gates)} gates")

    if 'corner_tl_x' not in csv.DictReader(open(est_path)).fieldnames:
        print(f"\n{est_path} predates Phase 2 corner/rvec instrumentation -- cannot run this audit.")
        sys.exit(1)

    ref_bearing = _per_gate_ref_bearing(gates)

    audit_filter_calibration(diag_rows)
    audit_polygon_convergence(diag_rows)
    audit_range_fattening(est_rows)
    audit_impossible_detections(est_rows, gates)
    select_and_save_frames(est_rows, gates, ref_bearing, frames_dir, out_dir)


if __name__ == '__main__':
    main()
