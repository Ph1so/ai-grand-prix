"""
VQ2 Racing Controller

Adapted from VQ1 controller with the following changes for VQ2 constraints:

  ATTITUDE blocked   → yaw integrated from HIGHRES_IMU zgyro (Step 1.4).
                       roll=0, pitch=0 assumed (sufficient for Phase 1).
  LOCAL_POSITION_NED → pos fixed at [0, 0, TAKEOFF_ALT] in shared_data.
  blocked              The key invariant: (cv_gate_pos - pos) == R@tvec, the
                       gate's NED offset from the drone, independent of absolute
                       pos. Direction and range are always correct.
  Velocity           → estimated by gate_verifier from CV frame-to-frame change.
  Gate map nulled    → GUIDANCE='CV_PLAN' only. Plan built from CV estimates.
  CONTROL_HZ         → 99 (spec requires <100 Hz; example had 250).
"""

import csv
import os
import time

import numpy as np
from pymavlink import mavutil

from planner import smooth_path, TAKEOFF_ALT as TAKEOFF_ALT_NED, LEAD_IN_DIST
from pose_estimator import CAM_TILT

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

MAVLINK_CMD_SIM_RESET = 31000

# ── Tuning constants ─────────────────────────────────────────────────────────
HOVER_THRUST        = 0.28
KP_THRUST           = 0.15
KI_ALT              = 0.02
CRUISE_SPEED        = 8.0
GATE_APPROACH_SPEED = 2.0
BRAKE_MARGIN        = 5.0
TANGENT_SAMPLES     = 30
MAX_SPEED           = 10.0
KP_POS              = 0.25
KP_CROSS            = 0.8
KP_POS_Z            = 0.50
MAX_Z_VEL           = 3.5
KP_VEL_ANGLE        = 0.12
MAX_TILT_ANGLE      = 0.45
KP_ANGLE            = 2.0
KD_VEL              = 0.04
KP_VEL_Z            = 0.08
KP_LEVEL            = 2.0
MIN_FLIGHT_THRUST   = 0.18
MAX_RATE            = 0.5
WAYPOINT_RADIUS     = 1.5
TAKEOFF_DELAY       = 3.0       # s on the ground before climbing
VQ2_CLIMB_TIME      = 2.5       # s of climb thrust before transitioning to FLY
GATE_WAIT_TIMEOUT   = 5.0
CONTROL_HZ          = 99        # Step 1.2: was 250, spec requires <100
CV_STALE_TIMEOUT    = 0.3
LOOKAHEAD_DIST      = 12.0
GATE_DIRECT_DIST    = 25.0
KP_YAW              = 1.0       # Step 1.4e: start conservative (VQ1 used 1.5)
MAX_YAW_RATE        = 1.5
MAX_VEL_SLEW        = 3.0
K_BRAKE             = 1.0
CV_RANGE_FLOOR      = 1.0
CV_OUTLIER_MIN_N    = 5
CV_OUTLIER_DIST     = 15.0
CV_MAX_PITCH        = 0.035
LOOK_GAIN           = 0.2

# VQ2 only uses CV_PLAN — MAVLINK gate positions are nulled by the simulator
GUIDANCE = 'CV_PLAN'

RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def _wrap(angle):
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def _lerp_angle(a, b, frac):
    return _wrap(a + _wrap(b - a) * frac)


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms, run_dir=None):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        self.state = 'INIT'
        self.waypoints = []
        self.path = None
        self._path_idx = 0
        self.current_idx = 0
        self._cv_gate_idx = 0
        self._cv_estimates  = {}
        self._cv_weights    = {}
        self._cv_gate_means = {}
        self._z_integral = 0.0
        self._settle_start_time = None
        self._prev_vel_error = np.zeros(3)
        self._vel_error_dot  = np.zeros(3)
        self._prev_desired_vel = np.zeros(3)

        self._init_pos_time  = None
        self._last_diag_time = 0.0
        self._idle_seen_not_started = False
        self._idle_last_boot = 0
        self._idle_start_time = None
        self._takeoff_entry_time = None

        # Step 1.4: IMU yaw integration state
        self._integrated_yaw  = 0.0
        self._last_imu_time_us = None

        if run_dir is None:
            run_dir = _LOG_DIR
            os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, 'controller_log.csv')
        self._cmd_log = open(log_path, 'w', newline='', buffering=1)
        self._cmd_csv = csv.writer(self._cmd_log)
        self._cmd_csv.writerow([
            'time_s', 'state', 'target_mode',
            'waypoint_idx', 'path_idx',
            'target_x', 'target_y', 'target_z',
            'pos_x', 'pos_y', 'pos_z',
            'error_x', 'error_y', 'error_z',
            'horiz_dist', 'gate_dist', 'speed_cap',
            'desired_vx', 'desired_vy', 'desired_vz',
            'actual_vx', 'actual_vy', 'actual_vz',
            'vel_error_fwd', 'vel_error_right', 'vel_error_z',
            'desired_pitch', 'desired_roll', 'actual_pitch', 'actual_roll',
            'cmd_roll_rate', 'cmd_pitch_rate', 'cmd_yaw_rate', 'cmd_thrust',
            'z_integral', 'yaw', 'target_yaw', 'yaw_err', 'saturated',
        ])
        print(f"[ctrl] command log → {log_path}", flush=True)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def update(self):
        # Step 1.4: Integrate yaw from HIGHRES_IMU zgyro
        imu_time_us  = self.data.get('highres_imu_time_us', 0)
        gyro_yaw_rate = float(self.data.get('gyro_yaw_rate', 0.0))
        if self._last_imu_time_us is not None and imu_time_us != self._last_imu_time_us:
            dt_imu = (imu_time_us - self._last_imu_time_us) * 1e-6
            if 0 < dt_imu < 0.1:
                self._integrated_yaw = _wrap(self._integrated_yaw + gyro_yaw_rate * dt_imu)
                self.data['integrated_yaw'] = self._integrated_yaw
        self._last_imu_time_us = imu_time_us

        # Race reset detection: sim confirmed reset while mid-flight
        if self.state in ('FLY', 'FINISHED', 'TAKEOFF'):
            pos = self.data.get('pos')
            race_started = self.data.get('race_status', {}).get('race_started', True)
            if pos is not None and np.linalg.norm(pos) < 2.0 and not race_started:
                print("[ctrl] race reset detected — returning to IDLE", flush=True)
                self.current_idx        = 0
                self._cv_gate_idx       = 0
                self._cv_estimates      = {}
                self._cv_weights        = {}
                self._cv_gate_means     = {}
                self._z_integral        = 0.0
                self._path_idx          = 0
                self._settle_start_time = None
                self._prev_vel_error    = np.zeros(3)
                self._vel_error_dot     = np.zeros(3)
                self._prev_desired_vel  = np.zeros(3)
                self._integrated_yaw    = 0.0
                self._idle_seen_not_started = False
                self._idle_start_time   = None
                self._takeoff_entry_time = None
                self._transition('IDLE')

        if self.state == 'INIT':
            self._handle_init()
        elif self.state == 'IDLE':
            self._handle_idle()
        elif self.state == 'TAKEOFF':
            self._handle_takeoff()
        elif self.state == 'FLY':
            self._handle_fly()
        elif self.state == 'FINISHED':
            self._send_attitude_rates(0.0, 0.0, 0.0, HOVER_THRUST + self._z_integral)
        time.sleep(1.0 / CONTROL_HZ)

    # ── State handlers ────────────────────────────────────────────────────────

    def _handle_init(self):
        # In VQ2: pos is pre-populated, gate map is always nulled.
        # With GUIDANCE='CV_PLAN', we don't need a gate map — transition to IDLE
        # as soon as the sim is connected (first time_boot_ms received).
        if self.data.get('time_boot_ms', 0) == 0:
            return
        if 'pos' not in self.data:
            return
        self._transition('IDLE')

    def _accumulate_cv_estimate(self, gate_idx: int, pos_ned: np.ndarray, drone_pos: np.ndarray) -> bool:
        """
        Record a CV estimate for gate_idx and update a range-weighted running mean.
        Returns True when the plan should be rebuilt.
        """
        bucket  = self._cv_estimates.setdefault(gate_idx, [])
        weights = self._cv_weights.setdefault(gate_idx, [])

        if len(bucket) >= CV_OUTLIER_MIN_N:
            median = np.median(np.asarray(bucket), axis=0)
            if float(np.linalg.norm(pos_ned - median)) > CV_OUTLIER_DIST:
                return False

        rng    = float(np.linalg.norm(pos_ned - drone_pos))
        weight = 1.0 / max(rng, CV_RANGE_FLOOR) ** 2

        bucket.append(pos_ned.copy())
        weights.append(weight)
        self._cv_gate_means[gate_idx] = np.average(
            np.asarray(bucket), axis=0, weights=np.asarray(weights))
        n = len(bucket)
        return n == 1 or n % 30 == 0

    def _rebuild_cv_plan(self):
        """Build waypoints and spline from accumulated per-gate CV means."""
        gate_ids = sorted(self._cv_gate_means)
        if not gate_ids:
            return
        start = np.array([0.0, 0.0, TAKEOFF_ALT_NED])
        wps   = [start]
        prev  = start.copy()
        for gid in gate_ids:
            center  = self._cv_gate_means[gid].copy()
            to_gate = center - prev
            dist    = float(np.linalg.norm(to_gate))
            if dist > 1.0:
                lead_in    = center - (to_gate / dist) * LEAD_IN_DIST
                lead_in[2] = center[2]
                wps.append(lead_in)
            wps.append(center)
            prev = center
        self.waypoints  = wps[1:]
        self.path       = smooth_path(wps)
        self._path_idx  = 0
        counts = {gid: len(self._cv_estimates[gid]) for gid in gate_ids}
        print(f"[ctrl] CV plan rebuilt: {len(gate_ids)} gate(s) → "
              f"{len(self.waypoints)} waypoints  samples={counts}", flush=True)

    def _lookahead_target(self, pos: np.ndarray) -> np.ndarray:
        path = self.path
        n = len(path)
        lo = max(0, self._path_idx - 50)
        hi = min(n, self._path_idx + 300)
        self._path_idx = lo + int(np.argmin(np.linalg.norm(path[lo:hi] - pos, axis=1)))
        acc, idx = 0.0, self._path_idx
        while idx < n - 1:
            seg = path[idx + 1] - path[idx]
            seg_len = float(np.linalg.norm(seg))
            if acc + seg_len >= LOOKAHEAD_DIST:
                return path[idx] + seg * ((LOOKAHEAD_DIST - acc) / seg_len)
            acc += seg_len
            idx += 1
        return path[-1]

    def _path_tracking_desired_vel(self, pos: np.ndarray, cruise_speed: float, yaw: float
                                   ) -> tuple[np.ndarray, np.ndarray]:
        path = self.path
        n    = len(path)
        lo   = max(0, self._path_idx - 50)
        hi   = min(n, self._path_idx + 300)
        self._path_idx = lo + int(np.argmin(np.linalg.norm(path[lo:hi] - pos, axis=1)))

        look         = min(self._path_idx + TANGENT_SAMPLES, n - 1)
        tangent      = path[look] - path[self._path_idx]
        t_len        = float(np.linalg.norm(tangent))
        if t_len > 1e-6:
            tangent_unit = tangent / t_len
        else:
            tangent_unit = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        nearest    = path[self._path_idx]
        to_nearest = nearest - pos
        along      = np.dot(to_nearest, tangent_unit)
        cross      = to_nearest - along * tangent_unit

        cross_vel = KP_CROSS * cross
        cross_spd = float(np.linalg.norm(cross_vel))
        if cross_spd > 1.0:
            cross_vel = cross_vel * (1.0 / cross_spd)
        desired_vel = cruise_speed * tangent_unit + cross_vel

        return desired_vel, tangent_unit

    def _handle_idle(self):
        pos = self.data.get('pos')
        idle_thrust = HOVER_THRUST if (pos is not None and pos[2] < -1.0) else 0.0
        self._send_attitude_rates(0.0, 0.0, 0.0, idle_thrust)

        race_started   = self.data.get('race_status', {}).get('race_started', False)
        current_boot   = self.data.get('time_boot_ms', 0)
        time_advancing = current_boot != self._idle_last_boot
        self._idle_last_boot = current_boot

        if not race_started:
            self._idle_seen_not_started = True

        if race_started and self._idle_seen_not_started and time_advancing:
            if self._idle_start_time is None:
                self._idle_start_time = time.time()
                print("[ctrl] start signal received, holding 0.5s...", flush=True)
            elif time.time() - self._idle_start_time >= 0.5:
                self._transition('TAKEOFF')

    def _handle_takeoff(self):
        vel = self.data.get('vel', np.zeros(3))

        if self._takeoff_entry_time is None:
            self._takeoff_entry_time = time.time()
            print(f"[ctrl] holding {TAKEOFF_DELAY:.1f}s on the ground before climb...", flush=True)

        elapsed = time.time() - self._takeoff_entry_time

        if elapsed < TAKEOFF_DELAY:
            # Sit on the ground — no thrust so race-start trigger doesn't cause early lift
            self._send_attitude_rates(0.0, 0.0, 0.0, 0.0)
            return

        # Climb phase: no altitude feedback in VQ2, use fixed above-hover thrust
        climb_thrust = float(np.clip(HOVER_THRUST + 0.10, 0.0, 1.0))
        self._send_attitude_rates(0.0, 0.0, 0.0, climb_thrust)

        # Time-based transition to FLY (no pos feedback to detect altitude)
        if elapsed >= TAKEOFF_DELAY + VQ2_CLIMB_TIME:
            self._prev_desired_vel = vel.copy()
            self._transition('FLY')

    def _handle_fly(self):
        pos = self.data.get('pos', np.zeros(3))
        vel = self.data.get('vel', np.zeros(3))
        nan = float('nan')

        race_status = self.data.get('race_status', {})
        active_gate = race_status.get('active_gate', 0)

        # Step 1.4: Use integrated yaw; roll/pitch not available in VQ2
        yaw   = self._integrated_yaw
        roll  = 0.0
        pitch = 0.0

        if race_status.get('race_finished', False):
            self._transition('FINISHED')
            return

        # Sync gate counters with sim's active_gate
        if active_gate > self._cv_gate_idx:
            print(f"[ctrl] gate {self._cv_gate_idx} confirmed, now targeting gate {active_gate}",
                  flush=True)
            self._cv_gate_idx = active_gate

        if self.waypoints and active_gate * 2 > self.current_idx:
            prev_idx = self.current_idx
            self.current_idx = active_gate * 2
            print(f"[ctrl] wp sync: {prev_idx} -> {self.current_idx}", flush=True)

        if self.waypoints and self.current_idx >= len(self.waypoints):
            self._transition('FINISHED')
            return

        # ── CV check ─────────────────────────────────────────────────────────
        cv_pos   = self.data.get('cv_gate_pos')
        cv_age   = time.time() - self.data.get('cv_gate_time', 0.0)
        cv_fresh = cv_pos is not None and cv_age < CV_STALE_TIMEOUT
        if cv_fresh:
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            if np.dot(cv_pos - pos, forward) < 0.0:
                cv_fresh = False

        # Accumulate CV estimates (pitch gate disabled in VQ2 — no pitch telemetry)
        if cv_pos is not None and cv_age < CV_STALE_TIMEOUT:
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            if np.dot(cv_pos - pos, forward) > 0.0:
                if self._accumulate_cv_estimate(active_gate, cv_pos, pos):
                    self._rebuild_cv_plan()

        # ── Gate advancement ──────────────────────────────────────────────────
        gate_dist    = 999.0
        next_gate_wp = None
        if self.waypoints and self.current_idx < len(self.waypoints):
            wp        = self.waypoints[self.current_idx]
            is_leadin = (self.current_idx % 2 == 0)
            wp_dist   = float(np.linalg.norm(wp - pos))

            if is_leadin:
                past_plane = False
                if wp_dist >= WAYPOINT_RADIUS and self.current_idx + 1 < len(self.waypoints):
                    next_wp      = self.waypoints[self.current_idx + 1]
                    approach     = next_wp - wp
                    approach_mag = float(np.linalg.norm(approach))
                    if approach_mag > 1e-6:
                        past_plane = float(np.dot(pos - wp, approach / approach_mag)) > 0.0
                if wp_dist < WAYPOINT_RADIUS or past_plane:
                    reason = 'plane' if past_plane else 'radius'
                    print(f"[ctrl] lead-in wp {self.current_idx}: advancing ({reason})", flush=True)
                    self.current_idx += 1
                    if self.current_idx >= len(self.waypoints):
                        self._transition('FINISHED')
                        return
                    is_leadin = False

            if not is_leadin and not cv_fresh:
                gate_idx      = self.current_idx // 2
                sim_confirmed = active_gate > gate_idx
                if self.current_idx > 0:
                    leadin_wp = self.waypoints[self.current_idx - 1]
                    approach = wp - leadin_wp
                    approach_mag = float(np.linalg.norm(approach))
                    if approach_mag > 1e-6:
                        approach_dir = approach / approach_mag
                        drone_past_gate = float(np.dot(pos - wp, approach_dir)) > 1.0
                    else:
                        drone_past_gate = False
                else:
                    drone_past_gate = False
                if sim_confirmed or drone_past_gate:
                    reason = '(sim)' if sim_confirmed else '(past plane+aligned)'
                    print(f"[ctrl] gate wp {self.current_idx} passed {reason}", flush=True)
                    self.current_idx += 1
                    if self.current_idx >= len(self.waypoints):
                        self._transition('FINISHED')
                        return

            next_gate_idx = self.current_idx if (self.current_idx % 2 == 1) else self.current_idx + 1
            next_gate_idx = min(next_gate_idx, len(self.waypoints) - 1)
            next_gate_wp  = self.waypoints[next_gate_idx]
            gate_dist     = float(np.linalg.norm(next_gate_wp - pos))
        elif cv_fresh:
            gate_dist = float(np.linalg.norm(cv_pos - pos))

        _v_now      = float(np.linalg.norm(vel[:2]))
        _a_brake    = MAX_VEL_SLEW
        _brake_dist = (max(_v_now, GATE_APPROACH_SPEED) ** 2 - GATE_APPROACH_SPEED ** 2) / (2.0 * _a_brake) + BRAKE_MARGIN
        speed_cap   = GATE_APPROACH_SPEED + (CRUISE_SPEED - GATE_APPROACH_SPEED) * min(1.0, gate_dist / max(_brake_dist, 1.0))

        # ── Target selection ─────────────────────────────────────────────────
        tangent_unit = None

        if cv_fresh:
            target      = cv_pos
            error       = target - pos
            horiz_dist  = float(np.linalg.norm(error[:2]))
            desired_vel = np.zeros(3)
            if horiz_dist > 0.1:
                dir2d = error[:2] / horiz_dist
                desired_vel[:2] = dir2d * speed_cap
                tangent_unit = np.array([dir2d[0], dir2d[1], 0.0])
            desired_vel[2] = float(np.clip(KP_POS_Z * error[2], -MAX_Z_VEL, MAX_Z_VEL))
            src = f'CV(age={cv_age*1000:.0f}ms)'
        elif (self.current_idx % 2 == 1) and next_gate_wp is not None and gate_dist < GATE_DIRECT_DIST:
            target      = next_gate_wp
            error       = target - pos
            horiz_dist  = float(np.linalg.norm(error[:2]))
            desired_vel = np.zeros(3)
            if horiz_dist > 0.1:
                desired_vel[:2] = error[:2] / horiz_dist * speed_cap
                tangent_unit = np.array([error[0] / horiz_dist, error[1] / horiz_dist, 0.0])
            desired_vel[2] = float(np.clip(KP_POS_Z * error[2], -MAX_Z_VEL, MAX_Z_VEL))
            src = f'gate[{self.current_idx // 2}]'
        elif self.path is not None and len(self.path) > 1:
            desired_vel, tangent_unit = self._path_tracking_desired_vel(pos, speed_cap, yaw)
            if self.current_idx < len(self.waypoints):
                wp_z = self.waypoints[self.current_idx][2]
                if pos[2] < wp_z - 0.3:
                    desired_vel[2] = max(desired_vel[2], KP_POS_Z * (wp_z - pos[2]))
            desired_vel[2] = float(np.clip(desired_vel[2], -MAX_Z_VEL, MAX_Z_VEL))
            target     = self.path[self._path_idx]
            error      = target - pos
            horiz_dist = float(np.linalg.norm(error[:2]))
            src        = 'spline'
        elif self.waypoints and self.current_idx < len(self.waypoints):
            target      = self.waypoints[self.current_idx]
            error       = target - pos
            horiz_dist  = float(np.linalg.norm(error[:2]))
            spd         = min(speed_cap, KP_POS * horiz_dist)
            desired_vel = np.zeros(3)
            if horiz_dist > 0.1:
                desired_vel[:2] = error[:2] / horiz_dist * spd
            desired_vel[2] = float(np.clip(KP_POS_Z * error[2], -MAX_Z_VEL, MAX_Z_VEL))
            src = f'wp[{self.current_idx}]'
        else:
            # Creep forward: no waypoints or CV — pitch forward slowly
            creep_thrust = HOVER_THRUST + self._z_integral
            self._send_attitude_rates(0.0, -0.2, 0.0, creep_thrust)
            self._cmd_csv.writerow([
                time.time(), 'FLY', 'creep',
                self.current_idx, self._path_idx,
                nan, nan, nan,
                pos[0], pos[1], pos[2],
                nan, nan, nan,
                nan, gate_dist, speed_cap,
                nan, nan, nan,
                vel[0], vel[1], vel[2],
                nan, nan, nan,
                nan, nan, 0.0, 0.0,
                0.0, -0.2, 0.0, creep_thrust,
                self._z_integral, yaw, nan, nan, 0,
            ])
            return

        now = time.time()
        if now - self._last_diag_time >= 2.0:
            self._last_diag_time = now
            speed = float(np.linalg.norm(vel))
            gate_label = self._cv_gate_idx if cv_fresh else self.current_idx // 2
            print(f"[fly] gate {gate_label} src={src}: gate_dist={gate_dist:.1f}m  "
                  f"spd={speed:.1f}m/s  pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  "
                  f"yaw={np.degrees(yaw):.1f}°  zi={self._z_integral:.3f}  sim_gate={active_gate}",
                  flush=True)

        # ── Velocity slew ────────────────────────────────────────────────────
        vel_delta = desired_vel - self._prev_desired_vel
        max_delta = MAX_VEL_SLEW / CONTROL_HZ
        d_xy = vel_delta[:2]
        mag_xy = float(np.linalg.norm(d_xy))
        if mag_xy > max_delta:
            desired_vel[:2] = self._prev_desired_vel[:2] + d_xy * (max_delta / mag_xy)
        mag_z = abs(vel_delta[2])
        if mag_z > max_delta:
            desired_vel[2] = self._prev_desired_vel[2] + vel_delta[2] * (max_delta / mag_z)
        self._prev_desired_vel = desired_vel.copy()

        # ── Speed brake ──────────────────────────────────────────────────────
        actual_spd_xy = float(np.linalg.norm(vel[:2]))
        if actual_spd_xy > speed_cap and actual_spd_xy > 0.1:
            excess = actual_spd_xy - speed_cap
            vel_xy_unit = vel[:2] / actual_spd_xy
            desired_vel[:2] -= vel_xy_unit * excess * K_BRAKE

        # ── Velocity error → attitude rates ───────────────────────────────────
        vel_error = desired_vel - vel

        d_vel               = (vel_error - self._prev_vel_error) * CONTROL_HZ
        self._vel_error_dot = 0.3 * d_vel + 0.7 * self._vel_error_dot
        self._prev_vel_error = vel_error.copy()
        damped_err          = vel_error + KD_VEL * self._vel_error_dot

        z_vel_err = vel_error[2]
        self._z_integral -= z_vel_err / CONTROL_HZ * KI_ALT
        self._z_integral = float(np.clip(self._z_integral, -0.25, 0.25))
        # roll=pitch=0 → tilt_comp = 1.0
        thrust = float(np.clip(HOVER_THRUST + self._z_integral - KP_VEL_Z * z_vel_err,
                               MIN_FLIGHT_THRUST, 1.0))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        fwd   =  cos_yaw * damped_err[0] + sin_yaw * damped_err[1]
        right = -sin_yaw * damped_err[0] + cos_yaw * damped_err[1]

        desired_pitch = float(np.clip( KP_VEL_ANGLE * fwd,   -MAX_TILT_ANGLE, MAX_TILT_ANGLE))
        desired_roll  = float(np.clip( KP_VEL_ANGLE * right, -MAX_TILT_ANGLE, MAX_TILT_ANGLE))

        # Look toward target: bias pitch toward centering gate on optical axis
        if horiz_dist > 1.0:
            to_tgt   = target - pos
            fwd_comp = cos_yaw * to_tgt[0] + sin_yaw * to_tgt[1]
            if abs(fwd_comp) > 1e-6 or abs(to_tgt[2]) > 1e-6:
                target_elev   = float(np.arctan2(-to_tgt[2], fwd_comp))
                pitch_to_look = target_elev + CAM_TILT
                desired_pitch = float(np.clip(
                    (1.0 - LOOK_GAIN) * desired_pitch + LOOK_GAIN * pitch_to_look,
                    -MAX_TILT_ANGLE, MAX_TILT_ANGLE))

        pitch_rate = float(np.clip( KP_ANGLE * (pitch - desired_pitch), -MAX_RATE, MAX_RATE))
        roll_rate  = float(np.clip(-KP_ANGLE * (roll  - desired_roll),  -MAX_RATE, MAX_RATE))

        actual_speed = float(np.linalg.norm(vel))
        if actual_speed > MAX_SPEED:
            scale = actual_speed / MAX_SPEED
            pitch_rate = float(np.clip(pitch_rate * scale, -MAX_RATE, MAX_RATE))
            roll_rate  = float(np.clip(roll_rate  * scale, -MAX_RATE, MAX_RATE))

        # ── Yaw toward path tangent (Step 1.4e) ───────────────────────────────
        target_yaw = nan
        yaw_err    = nan
        yaw_rate   = 0.0
        if tangent_unit is not None:
            target_yaw = float(np.arctan2(tangent_unit[1], tangent_unit[0]))
            yaw_err    = float(((target_yaw - yaw + np.pi) % (2 * np.pi)) - np.pi)
            yaw_rate   = float(np.clip(-KP_YAW * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE))
        elif horiz_dist > 1.0:
            h_norm  = float(np.linalg.norm(error[:2]))
            heading = error / h_norm if h_norm > 1e-6 else None
            if heading is not None:
                target_yaw = float(np.arctan2(heading[1], heading[0]))
                yaw_err    = float(((target_yaw - yaw + np.pi) % (2 * np.pi)) - np.pi)
                yaw_rate   = float(np.clip(-KP_YAW * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE))

        saturated = int(abs(pitch_rate) >= MAX_RATE - 1e-6 or abs(roll_rate) >= MAX_RATE - 1e-6)
        self._send_attitude_rates(roll_rate, pitch_rate, yaw_rate, thrust)
        self._cmd_csv.writerow([
            now, 'FLY', src,
            self.current_idx, self._path_idx,
            target[0], target[1], target[2],
            pos[0], pos[1], pos[2],
            error[0], error[1], error[2],
            horiz_dist, gate_dist, speed_cap,
            desired_vel[0], desired_vel[1], desired_vel[2],
            vel[0], vel[1], vel[2],
            fwd, right, z_vel_err,
            desired_pitch, desired_roll, 0.0, 0.0,
            roll_rate, pitch_rate, yaw_rate, thrust,
            self._z_integral, yaw, target_yaw, yaw_err, saturated,
        ])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _send_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            RATES_MASK,
            [1, 0, 0, 0],
            roll_rate,
            pitch_rate,
            yaw_rate,
            thrust,
        )

    def close_log(self):
        if not self._cmd_log.closed:
            self._cmd_log.close()

    def _transition(self, new_state):
        print(f"[ctrl] {self.state} --> {new_state}", flush=True)
        self.state = new_state

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0, 0, 0, 0, 0, 0, 0
        )
