"""
VQ2 Gate position verifier / CV pipeline.

GateVerifier subclasses VisionRX, overrides process_frame, and:
  - Runs HSV gate detection + solvePnP each frame
  - Publishes cv_gate_pos and cv_gate_time to shared_data (controller reads these)
  - Estimates drone velocity from frame-to-frame gate position change → shared_data['vel']
  - Rejects outlier estimates (Step 1.1: >3 m jump within a 150 ms window)
  - Logs all detections to gate_estimates_*.csv

VQ2 adaptations vs VQ1:
  - ATTITUDE is blocked: uses roll=0, pitch=0, yaw=shared_data['integrated_yaw']
  - LOCAL_POSITION_NED is blocked: reads shared_data['pos'] (pre-set by main.py,
    fixed at takeoff origin). The NED offset (est - pos) = R@tvec is always the
    correct gate direction and range regardless of absolute position.
  - Gate map positions are nulled: no ground-truth matching; mav_* columns are nan.
"""

import csv
import os
import time

import cv2
import numpy as np

from vision_rx import VisionRX
from gate_detector import detect_gate
from pose_estimator import estimate_gate_camera_frame, camera_to_ned

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

# Step 1.1 — Outlier rejection threshold.
# Reject a new estimate if it is more than this far from the previous estimate
# within the last CV_OUTLIER_WINDOW seconds (inter-gate gaps are longer than
# this window, so a new gate's first detection is never incorrectly rejected).
CV_OUTLIER_DIST_M   = 3.0   # metres
CV_OUTLIER_WINDOW_S = 0.15  # seconds (~4.5 frames at 30 Hz)


class GateVerifier(VisionRX):
    """
    Drop-in replacement for VisionRX for VQ2.
    Override process_frame to run the CV pipeline and publish gate estimates.
    """

    def __init__(self, data, log_path: str | None = None, run_dir: str | None = None,
                 snapshot_hz: float = 5.0):
        super().__init__(data)
        base = run_dir if run_dir is not None else _LOG_DIR
        ts   = time.strftime('%Y%m%d_%H%M%S')

        if log_path is None:
            log_path = os.path.join(base, f"gate_estimates_{ts}.csv")
        self._log_path = log_path
        self._log_file = open(log_path, 'w', newline='', buffering=1)
        self._csv_out  = csv.writer(self._log_file)
        self._csv_out.writerow([
            'sim_time_ns',
            'tvec_x', 'tvec_y', 'tvec_z',
            'rvec_x', 'rvec_y', 'rvec_z',
            'corner_tl_x', 'corner_tl_y', 'corner_tr_x', 'corner_tr_y',
            'corner_br_x', 'corner_br_y', 'corner_bl_x', 'corner_bl_y',
            'roll', 'pitch', 'yaw',
            'est_x', 'est_y', 'est_z',
            'drone_x', 'drone_y', 'drone_z',
            'matched_gate_id',
            'mav_x', 'mav_y', 'mav_z',
            'error_x', 'error_y', 'error_z', 'error_mag',
            'outlier_rejected',
        ])
        print(f"[verifier] logging estimates to {log_path}", flush=True)

        # Detection diagnostics CSV — every frame attempt, success or failure
        diag_path = os.path.join(base, f"detection_diag_{ts}.csv")
        self._diag_file = open(diag_path, 'w', newline='', buffering=1)
        self._diag_out  = csv.writer(self._diag_file)
        self._diag_out.writerow([
            'sim_time_ns', 'drone_x', 'drone_y', 'drone_z', 'yaw',
            'detect_result', 'n_contours',
            'largest_area', 'largest_aspect', 'largest_passed', 'n_candidates',
            'best_area', 'best_aspect', 'best_hull_pts', 'best_poly_pts',
            'four_corner_eps', 'n_clipped_corners',
        ])

        # Periodic raw-frame snapshots for offline CV inspection
        self._frames_dir = os.path.join(base, 'frames')
        os.makedirs(self._frames_dir, exist_ok=True)
        self._snapshot_period_ns = int(1e9 / snapshot_hz)
        self._last_snapshot_ns   = None

        # Try to load YOLO weights if present; fall back to HSV
        self._yolo = None
        _yolo_pt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'runs', 'pose', 'gate_yolo', 'weights', 'best.pt')
        if os.path.exists(_yolo_pt):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(_yolo_pt)
                print(f"[verifier] YOLO detector loaded ({_yolo_pt})", flush=True)
            except Exception as exc:
                print(f"[verifier] YOLO load failed ({exc}), using HSV detector", flush=True)
        else:
            print("[verifier] no YOLO weights found, using HSV detector", flush=True)

        # State for velocity estimation and outlier rejection
        self._prev_cv_pos  = None
        self._prev_cv_time = 0.0

    def get_thread_for_join(self):
        self._log_file.close()
        self._diag_file.close()
        return super().get_thread_for_join()

    def _yolo_detect(self, img) -> np.ndarray | None:
        results = self._yolo.predict(img, verbose=False, conf=0.25)
        r = results[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        best = int(r.boxes.conf.argmax())
        corners = r.keypoints.xy[best].cpu().numpy().astype(np.float32)
        if r.keypoints.conf is not None:
            if float(r.keypoints.conf[best].cpu().numpy().min()) < 0.2:
                return None
        return corners

    def process_frame(self, frame_id: int, img, sim_time_ns: int = 0):
        # VQ2: ATTITUDE blocked → use integrated yaw; roll/pitch assumed 0
        yaw  = float(self.data.get('integrated_yaw', 0.0))
        pos  = self.data.get('pos', np.zeros(3))
        roll = 0.0
        pitch = 0.0

        # Periodic frame snapshots
        if (self._last_snapshot_ns is None
                or sim_time_ns - self._last_snapshot_ns >= self._snapshot_period_ns):
            cv2.imwrite(os.path.join(self._frames_dir, f"frame_{frame_id:06d}.jpg"), img)
            self._last_snapshot_ns = sim_time_ns

        diag    = {}
        corners = self._yolo_detect(img) if self._yolo is not None else detect_gate(img, diag=diag)

        # Log diagnostics for every frame (success AND failure)
        self._diag_out.writerow([
            sim_time_ns, *pos.tolist(), yaw,
            diag.get('detect_result'), diag.get('n_contours'),
            diag.get('largest_area'), diag.get('largest_aspect'),
            diag.get('largest_passed'), diag.get('n_candidates'),
            diag.get('best_area'), diag.get('best_aspect'),
            diag.get('best_hull_pts'), diag.get('best_poly_pts'),
            diag.get('four_corner_eps'), diag.get('n_clipped_corners'),
        ])

        if corners is None:
            return

        tvec, rvec = estimate_gate_camera_frame(corners)
        if tvec is None:
            return

        # VQ2: roll=0, pitch=0; yaw from IMU integration
        est = camera_to_ned(tvec, pos, roll, pitch, yaw)

        now = time.time()
        corner_vals = corners.flatten().tolist()
        nan = float('nan')

        # ── Step 1.1: Outlier rejection ───────────────────────────────────────
        # Reject estimates that jump >3m from the previous estimate within the
        # last 150ms. Larger time gaps (inter-gate transitions) skip the check
        # so the first detection of a new gate is never rejected.
        outlier = False
        if (self._prev_cv_pos is not None
                and (now - self._prev_cv_time) < CV_OUTLIER_WINDOW_S):
            dist = float(np.linalg.norm(est - self._prev_cv_pos))
            if dist > CV_OUTLIER_DIST_M:
                outlier = True

        # Log the estimate regardless (with outlier flag)
        self._csv_out.writerow([
            sim_time_ns, *tvec.tolist(), *rvec.tolist(), *corner_vals,
            roll, pitch, yaw,
            *est.tolist(), *pos.tolist(),
            -1, nan, nan, nan, nan, nan, nan, nan,
            int(outlier),
        ])

        if outlier:
            return

        # ── CV velocity estimation ─────────────────────────────────────────────
        # d(est)/dt = -v_drone_NED  (gate is stationary; drone motion shifts it)
        # Only compute when same-gate-window (time gap < CV_OUTLIER_WINDOW_S covers
        # 4-5 consecutive frames at 30Hz; inter-gate gaps are much longer).
        if (self._prev_cv_pos is not None
                and 0.01 < (now - self._prev_cv_time) < CV_OUTLIER_WINDOW_S):
            dt = now - self._prev_cv_time
            cv_vel = -(est - self._prev_cv_pos) / dt
            self.data['vel']    = cv_vel
            self.data['cv_vel'] = cv_vel

        self._prev_cv_pos  = est.copy()
        self._prev_cv_time = now

        # Publish for controller
        self.data['cv_gate_pos']  = est
        self.data['cv_gate_time'] = now
        self.data['last_cv_pos']  = est.copy()
        self.data['last_cv_time'] = now
