"""
Project known 3-D gate geometry (from flight_log_*_gates.json) into image
space, given a drone pose at capture time.

This is the inverse of pose_estimator.camera_to_ned() + projectPoints(OBJ_PTS),
generalized to arbitrary world points (any gate, not just a PnP-fitted one).
Used by dataset_tools/label_run.py to auto-generate YOLO-pose training labels
from ground-truth gate positions + logged drone poses — no manual annotation.
"""

import numpy as np

from gate_detector import _order_corners
from planner import _quat_rotate
from pose_estimator import K, R_CAM2BODY, _euler_zyx

IMG_W, IMG_H = 640, 360

# Corners with camera-frame depth at or below this (incl. behind the camera)
# cannot be projected.
NEAR_CLIP = 0.05  # m


def gate_world_corners(gate: dict) -> np.ndarray:
    """
    4 outer-frame corners of `gate` in NED world coordinates, shape (4,3).

    True opening center = [pos_x, pos_y, -pos_z - height/2] (NED). Note: this
    deliberately omits planner.GATE_Z_BIAS, which is a controller waypoint
    fudge factor, not part of the gate's true geometry.

    Corner order is arbitrary here — project_gate() re-orders by image-space
    position (TL/TR/BR/BL) after projection.
    """
    pos    = np.asarray(gate['pos'], dtype=float)
    width  = float(gate.get('width', 2.7))
    height = float(gate.get('height', 2.7))
    quat   = gate['quat']

    center = np.array([pos[0], pos[1], -pos[2] - height / 2.0])
    hw, hh = width / 2.0, height / 2.0

    local_corners = np.array([
        [-hw, 0.,  -hh],
        [ hw, 0.,  -hh],
        [ hw, 0.,   hh],
        [-hw, 0.,   hh],
    ])
    return np.array([center + _quat_rotate(quat, c) for c in local_corners])


def project_point(world_pt, drone_pos, roll: float, pitch: float, yaw: float):
    """
    Project a world-frame (NED) point into the camera image.

    Returns (u, v, depth) pixel coordinates and the point's distance along
    the camera's optical axis, or None if the point is behind the camera
    (depth <= NEAR_CLIP).
    """
    R_b2ned  = _euler_zyx(roll, pitch, yaw)
    body_vec = R_b2ned.T @ (np.asarray(world_pt, dtype=float) - np.asarray(drone_pos, dtype=float))
    cam_vec  = R_CAM2BODY.T @ body_vec

    depth = float(cam_vec[2])
    if depth <= NEAR_CLIP:
        return None

    px = K @ (cam_vec / depth)
    return float(px[0]), float(px[1]), depth


def gate_depth(gate: dict, drone_pos, roll: float, pitch: float, yaw: float):
    """
    Mean camera-frame depth (m) of `gate`'s 4 corners, for depth-ordering
    multiple gates that project into overlapping image regions (see
    dataset_tools/label_run.py occlusion handling). Returns None if any
    corner is behind the camera.
    """
    depths = []
    for wc in gate_world_corners(gate):
        proj = project_point(wc, drone_pos, roll, pitch, yaw)
        if proj is None:
            return None
        depths.append(proj[2])
    return float(np.mean(depths))


def project_gate(gate: dict, drone_pos, roll: float, pitch: float, yaw: float):
    """
    Project all 4 corners of `gate` into the image.

    Returns (4,2) float32 pixel coordinates ordered [TL,TR,BR,BL]
    (gate_detector._order_corners convention — same as the live PnP
    pipeline), or None if any corner is behind the camera. Pixel coordinates
    may fall outside [0,IMG_W)x[0,IMG_H) — clipping/visibility filtering is
    the caller's responsibility (see dataset_tools/label_run.py).
    """
    world_corners = gate_world_corners(gate)
    pixels = []
    for wc in world_corners:
        proj = project_point(wc, drone_pos, roll, pitch, yaw)
        if proj is None:
            return None
        pixels.append([proj[0], proj[1]])
    return _order_corners(np.array(pixels, dtype=np.float32))
