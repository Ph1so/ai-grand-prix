"""
Phase 3 calibration-flight trajectory (see PERC.md Sec.6 Phase 3).

Builds a short, fully-scripted flight plan around gate 0 that the controller
flies autonomously in GUIDANCE='CALIBRATE' mode (controller._handle_calibrate).
Every leg is logged to calibration_log.csv, and every CV detection during the
flight lands in gate_estimates_*.csv carrying attitude + raw tvec (Phase 2
logging) — so the whole flight is scoreable offline from those two files
alone. Run it once; analyse later.

Course geometry: gate 0 sits on the "lead-in" side relative to the arm point —
the same safe stretch the drone already flies through every race. Every leg
here orbits gate 0 from that side, oriented by the *actual* arm->gate0
direction (not a hardcoded compass bearing), so the plan self-adapts to
whatever course is loaded.

Leg sequence (~2.5 min of data-collection time, well inside the 8-min limit):
  1. stare        hold dead-on,  R=12m, facing the gate squarely
                    -> resolves A11 (camera-tilt sign): any residual
                       systematic z-error here is pure rotation-chain bias,
                       isolated from range/angle confounds
  2. yaw_sweep    hold position, sweep yaw +-30 deg through boresight
                    -> resolves A13 (yaw / rotation-chain coupling): position
                       is constant, so any error correlation with yaw is the
                       rotation chain mis-tracking attitude
  3. range_sweep  fly the boresight line from R=22m down to R=5m
                    -> resolves A1/A9/A10 (range-dependent error growth):
                       continuous detections across the whole working range
  4. oblique_sweep  continuous traverse from -40deg to +40deg off-axis at
                    R=12m, camera continuously re-aimed at the gate centre
                    -> resolves A6/A5/A7 (oblique-angle detection robustness):
                       the gate's apparent shape skews increasingly with angle.
                       Linear interpolation between the two endpoint stations
                       traces a chord whose closest approach to the gate is
                       R*cos(40deg) =~ 9.2m (farther than range_sweep's 5m
                       closest approach -- no collision risk) and which, by
                       the symmetric station geometry, passes exactly through
                       the dead-on configuration at its midpoint -- giving
                       smooth, continuous coverage of the whole +-40deg span
                       in one motion. (An earlier design used four discrete
                       "hold and settle" stations; each one entered a stable
                       orbit around its target instead of converging -- a
                       textbook case of a central position-restoring force
                       being the wrong tool to kill tangential momentum. This
                       sweep sidesteps the problem entirely by never asking
                       the controller to stop: it only ever needs to do what
                       range_sweep already proved it does flawlessly --
                       continuous tracking of a slowly-moving target.)
"""
from dataclasses import dataclass

import numpy as np

from planner import gate_center

# ── Geometry ─────────────────────────────────────────────────────────────────
STARE_RANGE     = 12.0    # m  — matches LEAD_IN_DIST; the drone already flies this distance from gates every race
SWEEP_RANGE     = 12.0    # m
RANGE_FAR       = 22.0    # m  — far end of the range sweep (still well inside detection range per existing logs)
RANGE_NEAR      = 5.0     # m  — near end; close enough to approach without risking collision
OBLIQUE_RANGE   = 12.0    # m
OBLIQUE_SWEEP_ENDPOINTS = (-40.0, 40.0)   # degrees off boresight -- sweep traverses continuously between these
YAW_SWEEP_HALF  = np.radians(30.0)             # +-30 deg sweep around the boresight yaw

# ── Timing (data-collection windows; transit between legs is extra, but slow) ─
HOLD_DURATION    = 8.0    # s — static legs (stare, oblique holds)
SWEEP_DURATION   = 16.0   # s — yaw sweep, ~3.75 deg/s
RANGE_DURATION   = 30.0   # s — range sweep, ~0.57 m/s end-to-end
OBLIQUE_SWEEP_DURATION = 30.0  # s — continuous -40deg->+40deg traverse, ~0.5 m/s end-to-end (mirrors range_sweep)

# ── Settle detection (controller waits for these before starting a leg's clock) ─
CAL_SPEED_CAP   = 1.5             # m/s — deliberately slow; "we know the drone can follow waypoints slowly and controlled"
SETTLE_RADIUS   = 1.0             # m   — must be within this of the leg's start config...
SETTLE_YAW_TOL  = np.radians(8.0) # ...and within this much yaw...
SETTLE_HOLD_S   = 1.0             # ...continuously for this long (debounces sim noise) before data collection starts


@dataclass
class Leg:
    """One calibration leg: linear interpolation of (position, yaw) over `duration` seconds."""
    name:      str
    pos_start: np.ndarray
    pos_end:   np.ndarray
    yaw_start: float
    yaw_end:   float
    duration:  float


def _wrap(angle: float) -> float:
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def build_calibration_plan(gates: list[dict], start: np.ndarray | None = None) -> list[Leg]:
    """
    Build the Phase-3 leg sequence around gate 0, oriented by the drone's
    actual start position so every leg sits on the safe (already-flown)
    lead-in side regardless of course layout.
    """
    if start is None:
        start = np.array([0.0, 0.0, -0.5])

    g0 = gate_center(sorted(gates, key=lambda g: g['id'])[0])

    # Direction the drone naturally approaches gate 0 from (start -> gate),
    # projected to horizontal — the "well-trodden" line.
    approach = g0 - np.asarray(start, dtype=float)
    approach[2] = 0.0
    approach_norm = float(np.linalg.norm(approach))
    approach_dir  = approach / approach_norm if approach_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    boresight_yaw     = float(np.arctan2(approach_dir[1], approach_dir[0]))  # yaw that points the nose at the gate from the lead-in side
    bearing_from_gate = _wrap(boresight_yaw + np.pi)                          # bearing FROM the gate TO the lead-in side (where every leg orbits)

    def stand_off(range_m: float, angle_deg: float = 0.0):
        """
        Station point `range_m` from the gate, offset `angle_deg` around the
        gate from the boresight line (still on the lead-in side). Returns
        (position, yaw-to-face-the-gate-centre).
        """
        bearing = bearing_from_gate + np.radians(angle_deg)
        offset  = np.array([np.cos(bearing), np.sin(bearing), 0.0]) * range_m
        pos     = g0 + offset
        pos[2]  = g0[2]
        yaw     = float(np.arctan2(-offset[1], -offset[0]))   # face back toward the gate centre
        return pos, yaw

    legs: list[Leg] = []

    # 1. Stare — dead-on, fixed (resolves A11: tilt sign)
    p, y = stand_off(STARE_RANGE)
    legs.append(Leg('stare', p, p.copy(), y, y, HOLD_DURATION))

    # 2. Yaw sweep — same spot, sweep yaw through boresight (resolves A13: yaw/rotation-chain coupling)
    p, y = stand_off(SWEEP_RANGE)
    legs.append(Leg('yaw_sweep', p, p.copy(),
                    _wrap(y - YAW_SWEEP_HALF), _wrap(y + YAW_SWEEP_HALF), SWEEP_DURATION))

    # 3. Range sweep — straight-on approach, far to near (resolves A1/A9/A10: range-error growth)
    p_far,  y_far  = stand_off(RANGE_FAR)
    p_near, y_near = stand_off(RANGE_NEAR)
    legs.append(Leg('range_sweep', p_far, p_near, y_far, y_near, RANGE_DURATION))

    # 4. Oblique sweep — continuous traverse from -40deg to +40deg off-axis (resolves A6/A5/A7).
    # A single straight-line interpolation between the two endpoint stations: by the symmetric
    # geometry, its midpoint sits exactly on the dead-on boresight line at range R*cos(40deg)
    # =~ 9.2m (farther from the gate than range_sweep ever gets), so the whole traverse stays
    # safely clear while smoothly sweeping the viewing angle from -40deg through 0deg to +40deg.
    angle_lo, angle_hi = OBLIQUE_SWEEP_ENDPOINTS
    p_lo, y_lo = stand_off(OBLIQUE_RANGE, angle_lo)
    p_hi, y_hi = stand_off(OBLIQUE_RANGE, angle_hi)
    legs.append(Leg('oblique_sweep', p_lo, p_hi, y_lo, y_hi, OBLIQUE_SWEEP_DURATION))

    return legs


def print_plan(legs: list[Leg]) -> None:
    total = sum(l.duration for l in legs)
    print(f"[calplan] {len(legs)} legs, {total:.0f}s of data-collection time "
          f"(plus slow inter-leg transit — total flight stays well inside the 8-min limit)")
    for i, l in enumerate(legs):
        print(f"  [{i}] {l.name:12s}  "
              f"({l.pos_start[0]:7.1f},{l.pos_start[1]:7.1f},{l.pos_start[2]:6.1f}) "
              f"yaw={np.degrees(l.yaw_start):6.1f} deg  -->  "
              f"({l.pos_end[0]:7.1f},{l.pos_end[1]:7.1f},{l.pos_end[2]:6.1f}) "
              f"yaw={np.degrees(l.yaw_end):6.1f} deg   {l.duration:5.1f}s")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob
    import os
    import sys

    from planner import load_gates

    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        paths = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'logs', 'run_*', 'flight_log_*_gates.json')), reverse=True)
        if not paths:
            print('No *_gates.json found.')
            sys.exit(1)
        path = paths[0]

    print(f'Gates : {path}')
    gates = load_gates(path)
    legs  = build_calibration_plan(gates)
    print_plan(legs)
