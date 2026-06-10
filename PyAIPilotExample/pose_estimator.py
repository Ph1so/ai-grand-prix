"""
Gate pose estimation: image corners → gate centre in NED world frame.

estimate_gate_camera_frame(corners) -> (tvec (3,), rvec (3,)) | (None, None)
    PnP solve: gate centre position in camera frame (metres) + rotation vector.

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
# Rotation about the fixed body-Y axis by angle `t` (matrix convention: a
# *positive* `t` tilts the optical axis downward, i.e. toward +z in FRD/NED;
# negative `t` tilts it upward). OpenCV camera frame: x-right, y-down,
# z-forward.
#
# Camera axes in body FRD:
#   cam-x (right):         (0,      1,  0     )  — body-y, unchanged
#   cam-y (down in image): (−sin t, 0,  cos t )
#   cam-z (optical axis):  ( cos t, 0,  sin t )  — forward; +sin t = downward
#
# Columns of R_CAM2BODY are the camera basis vectors expressed in body frame.

# `t` empirically fitted (PERC.md A11 / Phase 1.1): a global least-squares
# minimization of `R_CAM2BODY(t)·tvec ≈ R_b2nedᵀ·(mav − drone_pos)` across
# 2237 independent CV detections from two datasets/geometries (dead-on
# stare+yaw+range and a continuous oblique sweep) converges on t ≈ −21.6°,
# reproducible to within ~1° across three independent fits — collapsing
# `error_z` to ~0 and cutting mean CV position error 66.7% (10.24 m ->
# 3.41 m) versus the previously-coded +20°.
#
# Per *this* matrix's sign convention, t ≈ −21.6° is an UPWARD optical-axis
# tilt of ~21.6° — i.e. it actually lands close to spec §3.8's original
# "20° upward" claim (just ~1.6° larger), not in the "opposite, effectively
# downward" direction the previously-coded +20° assumed. That earlier
# "flip" traced back to an angle recovered via `arctan` of an error/range
# ratio — a shortcut PERC.md's Phase 1.1 separately found uses the wrong
# geometric transform for this matrix's structure (see write-up).
# Public so other modules (e.g. controller.py's "look toward target" pitch
# trim) can reuse the fitted mount angle without re-deriving or duplicating it.
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
           (x-right, y-down, z-forward, metres) and the corresponding
           rotation vector. `rvec` is what `reproject_corners()` needs to
           project OBJ_PTS back into image space for a visual ground-truth
           check (PERC.md A2) — previously solved for and discarded.
           (None, None) if PnP fails.
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
