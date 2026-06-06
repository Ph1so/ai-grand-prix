"""
Race trajectory planner.

Given a gate map, computes:
  - waypoints : discrete lead-in / gate-center sequence the controller follows
  - path      : smooth cubic-spline trajectory through all waypoints (for viz / analysis)

Can be used offline (load from _gates.json) or at runtime (pass gates from shared_data).
"""

import json
import numpy as np
from scipy.interpolate import CubicSpline

LEAD_IN_DIST = 12.0  # m — approach point before each gate; must match controller usage
TAKEOFF_ALT  = -0.5  # m NED — hover altitude before racing begins


# ── Gate helpers ─────────────────────────────────────────────────────────────

def gate_center(gate: dict) -> np.ndarray:
    """
    Center of a gate opening in NED coords.
    The sim gives gate pos z as the bottom edge of the opening; shift up by half height.
    """
    pos = np.array(gate['pos'], dtype=float)
    pos[2] -= gate.get('height', 2.7) / 2.0  # more negative NED = higher altitude
    return pos


# ── Waypoint builder ─────────────────────────────────────────────────────────

def build_waypoints(gates: list[dict],
                    start: np.ndarray | None = None
                    ) -> tuple[list[np.ndarray], list[str]]:
    """
    Build the ordered waypoint list the controller targets:
      start -> [lead-in-0 -> gate-0] -> [lead-in-1 -> gate-1] -> ...

    Parameters
    ----------
    gates : list of gate dicts (from shared_data or load_gates())
    start : drone start position in NED; defaults to origin at takeoff altitude

    Returns
    -------
    waypoints : list of np.ndarray, shape (3,)
    labels    : matching human-readable labels
    """
    if start is None:
        start = np.array([0.0, 0.0, TAKEOFF_ALT])

    waypoints: list[np.ndarray] = [start]
    labels:    list[str]        = ['start']
    prev = start.copy()

    for g in sorted(gates, key=lambda x: x['id']):
        center   = gate_center(g)
        # center[2] = -center[2]          # flip altitude experiment
        to_gate  = center - prev
        dist     = float(np.linalg.norm(to_gate))
        if dist > 1.0:
            lead_in = center - (to_gate / dist) * LEAD_IN_DIST
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
    cumulative chord length.

    Returns array of shape (N, 3) in NED coords.
    """
    pts    = np.array(waypoints, dtype=float)
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # Guard against duplicate points (e.g. start == first lead-in)
    chords = np.where(chords < 1e-6, 1e-6, chords)
    t      = np.concatenate([[0.0], np.cumsum(chords)])
    t     /= t[-1]

    cs      = CubicSpline(t, pts, bc_type='not-a-knot')
    t_dense = np.linspace(0.0, 1.0, len(waypoints) * samples_per_segment)
    return cs(t_dense)


# ── Plan object ───────────────────────────────────────────────────────────────

class Plan:
    """Full planned trajectory for a race course."""

    def __init__(self,
                 gates:     list[dict],
                 waypoints: list[np.ndarray],
                 labels:    list[str],
                 path:      np.ndarray):
        self.gates     = gates
        self.waypoints = np.array(waypoints)   # (W, 3)
        self.labels    = labels
        self.path      = path                  # (N, 3) smooth NED path

    @classmethod
    def from_gates_file(cls, gates_path: str,
                        start: np.ndarray | None = None) -> 'Plan':
        """Build a Plan from a _gates.json file saved by mavlink_rx."""
        gates = load_gates(gates_path)
        return cls.from_gates(gates, start)

    @classmethod
    def from_gates(cls, gates: list[dict],
                   start: np.ndarray | None = None) -> 'Plan':
        """Build a Plan from a runtime gate list (e.g. shared_data['gates'])."""
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
    """Load gate list from a _gates.json file, returning numpy arrays for pos."""
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


# ── CLI entry point ───────────────────────────────────────────────────────────

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
