"""
Gate pose estimation: image corners → gate centre in NED world frame.

estimate_gate_camera_frame(corners) -> (tvec (3,), rvec (3,)) | (None, None)
    PnP solve: gate centre position in camera frame (metres) + rotation vector.

camera_to_ned(tvec, drone_pos, roll, pitch, yaw) -> np.ndarray (3,)
    Transform camera-frame position to NED world frame using drone attitude.

VQ2 note: ATTITUDE telemetry is blocked. Call with roll=0.0, pitch=0.0,
yaw=shared_data['integrated_yaw']. The key property used by the VQ2
controller is that (camera_to_ned(tvec, pos, 0, 0, yaw) - pos) equals the
gate's NED offset from the drone — independent of what pos is — so the
direction and range to any visible gate are always correct regardless of
accumulated position drift.
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
# CAM_TILT = -21.6° fitted from VQ1 data (2237 detections, least-squares).
# Negative angle = upward tilt ~21.6°, matching spec §3.8 "20° upward".
# Phase 0 outcome: copy VQ1 matrix as-is; residual ~1m z-bias is noted and
# managed at the controller level (see PLAN.md Phase 0 outcome notes).
CAM_TILT = np.radians(-21.6)

R_CAM2BODY = np.array(
    [[0.,         -np.sin(CAM_TILT),  np.cos(CAM_TILT)],
     [1.,          0.,                0.              ],
     [0.,          np.cos(CAM_TILT),  np.sin(CAM_TILT)]],
    dtype=np.float64,
)


# ── Public API ────────────────────────────────────────────────────────────────

def estimate_gate_camera_frame(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Run solvePnP to find the gate centre in camera frame.

    Parameters
    ----------
    corners : float32 (4, 2) — [TL, TR, BR, BL] image-space corners
              from gate_detector.detect_gate().

    Returns
    -------
    (tvec, rvec) : float64 (3,), float64 (3,) — gate centre in camera frame
           (x-right, y-down, z-forward, metres). (None, None) if PnP fails.
    """
    try:
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_PTS.reshape(-1, 1, 3),
            corners.astype(np.float64).reshape(-1, 1, 2),
            K,
            DIST,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None, None
    if not ok:
        return None, None
    return tvec.flatten(), rvec.flatten()


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
    roll, pitch, yaw : ZYX Euler angles in radians.
                       In VQ2: roll=0.0, pitch=0.0, yaw=integrated_yaw.

    Returns
    -------
    (3,) gate centre in NED world frame (metres).
    """
    R_b2ned = _euler_zyx(roll, pitch, yaw)
    return drone_pos + R_b2ned @ R_CAM2BODY @ tvec


def reproject_corners(tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """Reproject 3-D gate corners back to image space (for debug overlays)."""
    pts, _ = cv2.projectPoints(OBJ_PTS, rvec, tvec.reshape(3, 1), K, DIST)
    return pts.reshape(4, 2).astype(np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [   -sp,            cp*sr,            cp*cr  ],
    ], dtype=np.float64)
