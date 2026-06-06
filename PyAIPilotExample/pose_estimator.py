"""
Gate pose estimation: image corners → gate centre in NED world frame.

estimate_gate_camera_frame(corners) -> tvec (3,) | None
    PnP solve: gate centre position in camera frame (metres).

camera_to_ned(tvec, drone_pos, roll, pitch, yaw) -> np.ndarray (3,)
    Transform camera-frame position to NED world frame using drone attitude.
"""

import cv2
import numpy as np

# ── Camera intrinsics (640 × 360 pinhole, no distortion) ─────────────────────

K = np.array(
    [[320.,   0., 320.],
     [  0., 320., 180.],
     [  0.,   0.,   1.]],
    dtype=np.float64,
)
DIST = np.zeros(4, dtype=np.float64)

# ── Gate 3-D object points (gate-local frame, z = 0 plane) ───────────────────
# Origin at gate centre opening; x-right, y-down (matches camera convention).
# Outer boundary: 2720 mm → half = 1.36 m.
# Corner order: [TL, TR, BR, BL] — must match gate_detector._order_corners.

_H = 1.36  # outer half-dimension in metres

OBJ_PTS = np.array(
    [[-_H, -_H, 0.],   # top-left
     [ _H, -_H, 0.],   # top-right
     [ _H,  _H, 0.],   # bottom-right
     [-_H,  _H, 0.]],  # bottom-left
    dtype=np.float64,
)

# ── Camera → body (FRD) rotation ─────────────────────────────────────────────
# Per spec (VADR-TS-002 §3.8): camera is tilted 20° upward relative to body-x.
# Optical axis therefore points (cos20°, 0, −sin20°) in body FRD (forward+up).
# OpenCV camera frame: x-right, y-down, z-forward.
#
# Camera axes in body FRD:
#   cam-z (optical axis):  (cos t,  0, -sin t)   — forward and up
#   cam-x (right):         (0,      1,  0     )   — body-y, unchanged
#   cam-y (down in image): (sin t,  0,  cos t )   — from right-hand rule: cam-x × cam-y = cam-z
#
# Columns of R_CAM2BODY are the camera basis vectors expressed in body frame.

_t = np.radians(20.0)   # camera tilt magnitude (spec §3.8: 20°)
# Spec says "upwards" but data shows the correction must go in the opposite
# direction — the camera is effectively looking 20° below body-x.
# cam-z (optical) → (cos t, 0, +sin t) in FRD  (forward + downward)
# cam-x (right)   → (0, 1, 0)
# cam-y (down)    → (−sin t, 0, cos t)  from right-hand rule: cam-x × cam-y = cam-z

R_CAM2BODY = np.array(
    [[0.,         -np.sin(_t),  np.cos(_t)],
     [1.,          0.,          0.        ],
     [0.,          np.cos(_t),  np.sin(_t)]],
    dtype=np.float64,
)


# ── Public API ────────────────────────────────────────────────────────────────

def estimate_gate_camera_frame(corners: np.ndarray) -> np.ndarray | None:
    """
    Run solvePnP to find the gate centre in camera frame.

    Parameters
    ----------
    corners : float32 (4, 2) — [TL, TR, BR, BL] image-space corners
              from gate_detector.detect_gate().

    Returns
    -------
    tvec : float64 (3,) — gate centre in camera frame (x-right, y-down,
           z-forward), in metres.  None if PnP fails.
    """
    try:
        ok, _, tvec = cv2.solvePnP(
            OBJ_PTS.reshape(-1, 1, 3),
            corners.astype(np.float64).reshape(-1, 1, 2),
            K,
            DIST,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    return tvec.flatten() if ok else None


def camera_to_ned(
    tvec: np.ndarray,
    drone_pos: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Transform gate position from camera frame to NED world frame.

    Parameters
    ----------
    tvec      : (3,) gate centre in camera frame (from estimate_gate_camera_frame).
    drone_pos : (3,) drone NED position in metres.
    roll, pitch, yaw : ZYX Euler angles in radians from MAVLink ATTITUDE.

    Returns
    -------
    (3,) gate centre in NED world frame (metres).
    """
    R_b2ned = _euler_zyx(roll, pitch, yaw)
    return drone_pos + R_b2ned @ R_CAM2BODY @ tvec


def reproject_corners(tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """
    Reproject the 3-D gate corners back to image space (for debug overlays).
    Returns (4, 2) float32 pixel coordinates.
    """
    pts, _ = cv2.projectPoints(OBJ_PTS, rvec, tvec.reshape(3, 1), K, DIST)
    return pts.reshape(4, 2).astype(np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    ZYX Euler angles → 3×3 rotation matrix that converts body vectors to NED.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [   -sp,            cp*sr,            cp*cr  ],
    ], dtype=np.float64)
