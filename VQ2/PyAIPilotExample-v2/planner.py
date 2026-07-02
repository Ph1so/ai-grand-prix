"""
Race trajectory planner.

Given a gate map, computes:
  - waypoints : discrete lead-in / gate-center sequence the controller follows
  - path      : smooth cubic-spline trajectory through all waypoints (for viz / analysis)

VQ2 note: MAVLink gate positions are nulled (zeroed) by the simulator.
In VQ2 the controller always uses GUIDANCE='CV_PLAN', which calls
smooth_path() and _rebuild_cv_plan() directly with CV-derived gate means —
Plan.from_gates() and build_waypoints() are not used at runtime.
"""

import json
import os
import numpy as np
from scipy.interpolate import CubicSpline

LEAD_IN_DIST  = 18.0  # m — approach point before each gate; must match controller usage
TAKEOFF_ALT   = -0.5  # m NED — hover altitude before racing begins
GATE_Z_BIAS   = -0.1  # m NED — upward nudge applied to every gate center waypoint

PATH_VARIANT_OFFSET = float(os.environ.get('PATH_VARIANT_OFFSET', '0.0'))


# ── Gate helpers ─────────────────────────────────────────────────────────────

def gate_center(gate: dict) -> np.ndarray:
    pos = np.array(gate['pos'], dtype=float)
    pos[2] = -pos[2] - gate.get('height', 2.7) / 2.0 + GATE_Z_BIAS
    return pos


def _quat_rotate(q: list[float] | np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in q]
    q_vec = np.array([qx, qy, qz], dtype=float)
    return (
        v
        + 2.0 * qw * np.cross(q_vec, v)
        + 2.0 * np.cross(q_vec, np.cross(q_vec, v))
    )


def gate_approach_dir(gate: dict, prev: np.ndarray, center: np.ndarray) -> np.ndarray:
    quat = gate.get('quat')
    if quat is not None:
        normal = _quat_rotate(quat, np.array([0.0, 1.0, 0.0]))
        normal[2] = 0.0
        norm = float(np.linalg.norm(normal))
        if norm > 1e-6:
            approach = normal / norm
            to_gate = center - prev
            to_gate[2] = 0.0
            if float(np.dot(approach, to_gate)) < 0.0:
                approach = -approach
            return approach

    fallback = center - prev
    fallback[2] = 0.0
    norm = float(np.linalg.norm(fallback))
    if norm > 1e-6:
        return fallback / norm
    return np.array([1.0, 0.0, 0.0])


# ── Waypoint builder ─────────────────────────────────────────────────────────

def build_waypoints(gates: list[dict],
                    start: np.ndarray | None = None
                    ) -> tuple[list[np.ndarray], list[str]]:
    if start is None:
        start = np.array([0.0, 0.0, TAKEOFF_ALT])

    waypoints: list[np.ndarray] = [start]
    labels:    list[str]        = ['start']
    prev = start.copy()

    for g in sorted(gates, key=lambda x: x['id']):
        center   = gate_center(g)
        approach_dir = gate_approach_dir(g, prev, center)
        lead_in = center - approach_dir * LEAD_IN_DIST
        lead_in[2] = center[2]

        if PATH_VARIANT_OFFSET:
            sign  = 1.0 if g['id'] % 2 == 0 else -1.0
            right = np.array([approach_dir[1], -approach_dir[0], 0.0])
            offset = right * (PATH_VARIANT_OFFSET * sign)
            offset[2] = -PATH_VARIANT_OFFSET * sign
            lead_in = lead_in + offset
            center  = center + offset * 0.3

        if float(np.linalg.norm(lead_in - prev)) > 1.0:
            waypoints.append(lead_in)
            labels.append(f'lead-in-{g["id"]}')
        waypoints.append(center)
        labels.append(f'gate-{g["id"]}')
        prev = center

    return waypoints, labels


# ── Smooth path ───────────────────────────────────────────────────────────────

def smooth_path(waypoints: list[np.ndarray],
                samples_per_segment: int = 80) -> np.ndarray:
    """
    Fit a C2-continuous cubic spline through waypoints parameterised by
    cumulative chord length. Returns array of shape (N, 3) in NED coords.
    """
    pts    = np.array(waypoints, dtype=float)
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    chords = np.where(chords < 1e-6, 1e-6, chords)
    t      = np.concatenate([[0.0], np.cumsum(chords)])
    t     /= t[-1]

    cs      = CubicSpline(t, pts, bc_type='not-a-knot')
    t_dense = np.linspace(0.0, 1.0, len(waypoints) * samples_per_segment)
    return cs(t_dense)


# ── Plan object ───────────────────────────────────────────────────────────────

class Plan:
    def __init__(self,
                 gates:     list[dict],
                 waypoints: list[np.ndarray],
                 labels:    list[str],
                 path:      np.ndarray):
        self.gates     = gates
        self.waypoints = np.array(waypoints)
        self.labels    = labels
        self.path      = path

    @classmethod
    def from_gates_file(cls, gates_path: str,
                        start: np.ndarray | None = None) -> 'Plan':
        gates = load_gates(gates_path)
        return cls.from_gates(gates, start)

    @classmethod
    def from_gates(cls, gates: list[dict],
                   start: np.ndarray | None = None) -> 'Plan':
        waypoints, labels = build_waypoints(gates, start)
        path = smooth_path(waypoints)
        return cls(gates, waypoints, labels, path)

    @property
    def total_length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.path, axis=0), axis=1)))

    def summary(self):
        print(f'[plan] {len(self.gates)} gates -> {len(self.waypoints)} waypoints, '
              f'smooth path {self.total_length:.1f} m')
        for i, (wp, lbl) in enumerate(zip(self.waypoints, self.labels)):
            print(f'  [{i:2d}] {lbl:15s}  ({wp[0]:8.1f}, {wp[1]:6.1f}, {wp[2]:6.1f})')


# ── Offline loader ────────────────────────────────────────────────────────────

def load_gates(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    return [
        {
            'id':     g['id'],
            'pos':    np.array(g['pos']),
            'width':  g['width'],
            'height': g['height'],
            'quat':   g['quat'],
        }
        for g in sorted(raw, key=lambda x: x['id'])
    ]


if __name__ == '__main__':
    import glob
    import sys

    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        paths = sorted(glob.glob('flight_log_*_gates.json'), reverse=True)
        if not paths:
            print('No *_gates.json found in current directory.')
            sys.exit(1)
        path = paths[0]

    plan = Plan.from_gates_file(path)
    plan.summary()
