import csv
import os
import time

import numpy as np
from pymavlink import mavutil

from planner import Plan, TAKEOFF_ALT as TAKEOFF_ALT_NED, smooth_path, LEAD_IN_DIST
from calibration_plan import (
    build_calibration_plan, print_plan,
    CAL_SPEED_CAP, SETTLE_RADIUS, SETTLE_YAW_TOL, SETTLE_HOLD_S,
)

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

MAVLINK_CMD_SIM_RESET = 31000

# ── Tuning constants ────────────────────────────────────────────────────────
HOVER_THRUST        = 0.28   # estimated actual hover thrust (derived from equilibrium data)
KP_THRUST           = 0.15   # extra thrust per metre of altitude error during climb
KI_ALT              = 0.02   # integral gain — fine-tunes hover estimate over time
CRUISE_SPEED        = 3.0    # m/s — target speed on open stretches
GATE_APPROACH_SPEED = 1.2    # m/s — reduced speed within GATE_APPROACH_DIST of each gate
GATE_APPROACH_DIST  = 12.0   # m  — distance where the gate approach ramp reaches cruise speed
TANGENT_SAMPLES     = 20     # path samples ahead to compute tangent (smooths spline kinks at waypoints)
MAX_SPEED           = 10.0   # m/s — hard cap; above this, scale rates for deceleration
KP_POS              = 0.25   # position error (m) → desired speed (m/s), used in fallback modes
KP_CROSS            = 0.8    # cross-track error (m) → desired lateral velocity (m/s)
KP_POS_Z            = 0.50   # altitude error (m) → desired climb rate (m/s)
MAX_Z_VEL           = 3.5    # m/s — max climb / descend rate
KP_VEL_ANGLE        = 0.12   # velocity error (m/s) → desired tilt angle (rad) — outer/cascade loop
MAX_TILT_ANGLE      = 0.30   # rad (~17°) — caps outer-loop tilt request; g*tan(0.30)≈2.96 m/s² max accel
KP_ANGLE            = 2.0    # tilt-angle error (rad) → attitude rate (rad/s) — inner/cascade loop
KD_VEL              = 0.04   # velocity error derivative gain (filtered damping)
KP_VEL_Z            = 0.08   # z velocity error (m/s) → thrust delta
KP_LEVEL            = 2.0    # attitude angle (rad) → levelling rate (rad/s) used in takeoff
MIN_FLIGHT_THRUST   = 0.18   # lower bound during flight — prevents Z overcorrection causing crash
MAX_RATE            = 0.4    # rad/s — max pitch/roll rate command
WAYPOINT_RADIUS     = 1.5    # m — switch to next waypoint when within this distance
TAKEOFF_DELAY       = 3.0    # seconds to hold on the ground after entering TAKEOFF before climbing
GATE_WAIT_TIMEOUT   = 5.0    # seconds to wait for gate map before flying blind
CONTROL_HZ          = 100    # Hz
CV_STALE_TIMEOUT    = 0.3    # seconds — treat CV estimate as lost after this gap
LOOKAHEAD_DIST      = 12.0   # m — kept for reference
KP_YAW              = 1.5    # yaw error (rad) → yaw rate (rad/s)
MAX_YAW_RATE        = 1.5    # rad/s — yaw rate limit
MAX_VEL_SLEW        = 2.5    # m/s² — max rate of change of velocity setpoint (limits oscillation)
K_BRAKE             = 0.8    # over-speed correction blended into the slewed velocity target

# ── Guidance mode ────────────────────────────────────────────────────────────
# 'MAVLINK'   — follow pre-planned MAVLink waypoints (default)
# 'CV_LIVE'   — track live CV pose estimate each frame; MAVLink waypoints as fallback
# 'CV_PLAN'   — accumulate CV estimates per gate, build + follow a CV-derived plan; no MAVLink fallback
# 'CALIBRATE' — fly the scripted Phase-3 calibration sequence around gate 0 (see calibration_plan.py); no racing
GUIDANCE = 'MAVLINK'

# CALIBRATE-only: restrict the flight to legs whose name starts with one of
# these prefixes (None = run the full plan). Lets a re-run target only the
# legs that didn't complete last time, without re-flying the ones that did.
CAL_LEG_FILTER = ('oblique',) 

RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def _wrap(angle):
    """Wrap an angle (rad) to (-pi, pi]."""
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def _lerp_angle(a, b, frac):
    """Shortest-path angle interpolation from a to b (rad)."""
    return _wrap(a + _wrap(b - a) * frac)


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms, run_dir=None):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        self.state = 'INIT'
        self.waypoints = []        # flat list: [lead-in-0, gate-0, lead-in-1, gate-1, ...]
        self.path = None           # (N, 3) smooth spline — None until gate map arrives
        self._path_idx = 0         # current progress index along path
        self.current_idx = 0
        self._cv_gate_idx = 0      # gates confirmed passed in CV-guided mode
        self._cv_estimates  = {}   # CV_PLAN: gate_idx -> list[np.ndarray] raw NED estimates
        self._cv_gate_means = {}   # CV_PLAN: gate_idx -> np.ndarray running mean
        self._cal_legs            = []     # CALIBRATE: list[Leg] built once gate map arrives
        self._cal_idx             = 0      # CALIBRATE: index of the leg currently flying
        self._cal_leg_start_time  = None   # CALIBRATE: wall time the current leg's data window opened (None = still settling)
        self._z_integral = 0.0    # altitude integral — accumulates hover thrust error
        self._settle_start_time = None  # stop-and-go: when drone first entered settle window
        self._prev_vel_error = np.zeros(3)
        self._vel_error_dot  = np.zeros(3)
        self._prev_desired_vel = np.zeros(3)

        self._init_pos_time  = None
        self._last_diag_time = 0.0
        self._idle_seen_not_started = False
        self._idle_last_boot = 0
        self._idle_start_time = None
        self._takeoff_entry_time = None

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

        self._cal_log = None
        self._cal_csv = None
        if GUIDANCE == 'CALIBRATE':
            cal_log_path = os.path.join(run_dir, 'calibration_log.csv')
            self._cal_log = open(cal_log_path, 'w', newline='', buffering=1)
            self._cal_csv = csv.writer(self._cal_log)
            self._cal_csv.writerow([
                'time_s', 'leg_idx', 'leg_name', 'phase', 'frac',
                'target_x', 'target_y', 'target_z', 'target_yaw',
                'pos_x', 'pos_y', 'pos_z', 'yaw',
                'error_x', 'error_y', 'error_z', 'horiz_dist', 'yaw_err',
                'desired_vx', 'desired_vy', 'desired_vz',
                'actual_vx', 'actual_vy', 'actual_vz',
                'cmd_roll_rate', 'cmd_pitch_rate', 'cmd_yaw_rate', 'cmd_thrust',
                'settled',
            ])
            print(f"[ctrl] calibration log → {cal_log_path}", flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    def update(self):
        # Detect race restart: if sim resets us back near the origin while we're
        # mid-flight or finished, re-enter the start sequence cleanly.
        if self.state in ('FLY', 'FINISHED', 'TAKEOFF', 'CALIBRATE'):
            pos = self.data.get('pos')
            race_started = self.data.get('race_status', {}).get('race_started', True)
            if pos is not None and np.linalg.norm(pos) < 2.0 and not race_started:
                print("[ctrl] race reset detected — returning to IDLE", flush=True)
                self.current_idx    = 0
                self._cv_gate_idx   = 0
                self._cv_estimates      = {}
                self._cv_gate_means     = {}
                self._cal_idx            = 0
                self._cal_leg_start_time = None
                self._z_integral        = 0.0
                self._path_idx          = 0
                self._settle_start_time = None
                self._prev_vel_error    = np.zeros(3)
                self._vel_error_dot     = np.zeros(3)
                self._prev_desired_vel  = np.zeros(3)
                self._idle_seen_not_started = False
                self._idle_start_time = None
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
        elif self.state == 'CALIBRATE':
            self._handle_calibrate()
        elif self.state == 'FINISHED':
            self._send_attitude_rates(0.0, 0.0, 0.0, HOVER_THRUST + self._z_integral)
        time.sleep(1.0 / CONTROL_HZ)

    # ── State handlers ───────────────────────────────────────────────────────

    def _handle_init(self):
        if 'pos' not in self.data:
            return
        now = time.time()
        # CALIBRATE needs the gate map too (it locates gate 0), so it waits like MAVLINK does
        needs_gate_map = GUIDANCE in ('MAVLINK', 'CALIBRATE')
        if self._init_pos_time is None:
            self._init_pos_time = now
            msg = "waiting for gate map..." if needs_gate_map else "CV — not waiting for gate map"
            print(f"[ctrl] position acquired, {msg}", flush=True)

        if 'gates' in self.data and GUIDANCE != 'CV_PLAN':
            if GUIDANCE == 'CALIBRATE':
                self._build_calibration_plan()
            else:
                self._build_waypoints()
            self._transition('IDLE')
        elif not needs_gate_map or now - self._init_pos_time >= GATE_WAIT_TIMEOUT:
            if needs_gate_map:
                print("[ctrl] WARNING: no gate map — cannot proceed meaningfully", flush=True)
            self._transition('IDLE')
        elif now - self._last_diag_time >= 2.0:
            self._last_diag_time = now
            print(f"[ctrl] INIT {now - self._init_pos_time:.1f}s — waiting for gate map...", flush=True)

    def _build_waypoints(self):
        plan = Plan.from_gates(self.data['gates'])
        plan.summary()
        # Skip the synthetic 'start' waypoint — controller handles takeoff separately
        self.waypoints = list(plan.waypoints[1:])
        self.path = plan.path
        self._path_idx = 0

    def _build_calibration_plan(self):
        pos = self.data.get('pos', np.array([0.0, 0.0, TAKEOFF_ALT_NED]))
        legs = build_calibration_plan(self.data['gates'], start=pos)
        if CAL_LEG_FILTER is not None:
            legs = [l for l in legs if l.name.startswith(CAL_LEG_FILTER)]
        self._cal_legs = legs
        print_plan(self._cal_legs)
        self._cal_idx            = 0
        self._cal_leg_start_time = None

    def _accumulate_cv_estimate(self, gate_idx: int, pos_ned: np.ndarray) -> bool:
        """
        Record a CV estimate for gate_idx and update the running mean.
        Returns True when the plan should be rebuilt (first sighting of a gate,
        or every 30 samples thereafter as the mean refines).
        """
        bucket = self._cv_estimates.setdefault(gate_idx, [])
        bucket.append(pos_ned.copy())
        self._cv_gate_means[gate_idx] = np.mean(bucket, axis=0)
        n = len(bucket)
        return n == 1 or n % 30 == 0

    def _rebuild_cv_plan(self):
        """
        Build self.waypoints and self.path from accumulated per-gate CV means.
        Uses the same lead-in geometry as the MAVLink planner.
        current_idx is preserved so in-flight rebuilds don't reset progress.
        """
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
                wps.append(center - (to_gate / dist) * LEAD_IN_DIST)
            wps.append(center)
            prev = center
        self.waypoints  = wps[1:]          # skip start; matches current_idx convention
        self.path       = smooth_path(wps)
        self._path_idx  = 0
        counts = {gid: len(self._cv_estimates[gid]) for gid in gate_ids}
        print(f"[ctrl] CV plan rebuilt: {len(gate_ids)} gate(s) → "
              f"{len(self.waypoints)} waypoints  samples={counts}", flush=True)

    def _lookahead_target(self, pos: np.ndarray) -> np.ndarray:
        """Pure pursuit: find the point LOOKAHEAD_DIST ahead on the smooth spline."""
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

    def _path_tracking_desired_vel(self, pos: np.ndarray, cruise_speed: float
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """
        Feed-forward velocity from path tangent + cross-track error feedback.

        Returns (desired_vel_NED, tangent_unit_NED).  Updates _path_idx.
        The caller is responsible for clamping desired_vel[2] to MAX_Z_VEL.
        """
        path = self.path
        n    = len(path)
        lo   = max(0, self._path_idx - 50)
        hi   = min(n, self._path_idx + 300)
        self._path_idx = lo + int(np.argmin(np.linalg.norm(path[lo:hi] - pos, axis=1)))

        # Tangent from a lookahead point to avoid spline kinks right at waypoints
        look         = min(self._path_idx + TANGENT_SAMPLES, n - 1)
        tangent      = path[look] - path[self._path_idx]
        t_len        = float(np.linalg.norm(tangent))
        tangent_unit = tangent / t_len if t_len > 1e-6 else np.array([1.0, 0.0, 0.0])

        # Cross-track error: component of (nearest - pos) perpendicular to path
        nearest    = path[self._path_idx]
        to_nearest = nearest - pos
        along      = np.dot(to_nearest, tangent_unit)
        cross      = to_nearest - along * tangent_unit

        # Cap cross-track correction independently so it never reduces forward progress.
        # Old approach capped the combined vector, which let lateral correction eat into
        # forward speed and caused the drone to slow down on every curve.
        cross_vel = KP_CROSS * cross
        cross_spd = float(np.linalg.norm(cross_vel))
        if cross_spd > 1.0:
            cross_vel = cross_vel * (1.0 / cross_spd)
        desired_vel = cruise_speed * tangent_unit + cross_vel

        return desired_vel, tangent_unit

    def _handle_idle(self):
        # Keep minimum thrust so the drone doesn't free-fall if it's airborne at reset
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
        pos = self.data.get('pos')
        if pos is None:
            return
        vel = self.data.get('vel', np.zeros(3))

        if self._takeoff_entry_time is None:
            self._takeoff_entry_time = time.time()
            print(f"[ctrl] holding {TAKEOFF_DELAY:.1f}s on the ground before climb...", flush=True)

        if time.time() - self._takeoff_entry_time < TAKEOFF_DELAY:
            # Hold on the ground without moving. The race-start signal fires
            # slightly before the official start; climbing immediately on that
            # signal was triggering an early-start disqualification.
            roll, pitch, yaw = self.data.get('attitude', (0.0, 0.0, 0.0))
            hold_thrust = HOVER_THRUST if pos[2] < -1.0 else 0.0
            self._send_attitude_rates(0.0, 0.0, 0.0, hold_thrust)

            nan = float('nan')
            self._cmd_csv.writerow([
                time.time(), 'TAKEOFF', 'hold',
                nan, nan,
                nan, nan, TAKEOFF_ALT_NED,
                pos[0], pos[1], pos[2],
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan,
                vel[0], vel[1], vel[2],
                nan, nan, nan,
                0.0, 0.0, pitch, roll,
                0.0, 0.0, 0.0, hold_thrust,
                self._z_integral, yaw, nan, nan, 0,
            ])
            return

        alt_error = TAKEOFF_ALT_NED - pos[2]
        thrust = float(np.clip(HOVER_THRUST - KP_THRUST * alt_error, 0.0, 1.0))

        # Actively level any residual pitch/roll from the previous run's attitude.
        # Sending zero rates would lock in whatever angle the sim reset with (~18°),
        # causing uncontrolled horizontal acceleration throughout the climb.
        roll, pitch, yaw = self.data.get('attitude', (0.0, 0.0, 0.0))
        level_pitch = float(np.clip( KP_LEVEL * pitch, -MAX_RATE, MAX_RATE))
        level_roll  = float(np.clip(-KP_LEVEL * roll,  -MAX_RATE, MAX_RATE))
        self._send_attitude_rates(level_roll, level_pitch, 0.0, thrust)

        nan = float('nan')
        self._cmd_csv.writerow([
            time.time(), 'TAKEOFF', '',
            nan, nan,
            nan, nan, TAKEOFF_ALT_NED,
            pos[0], pos[1], pos[2],
            nan, nan, alt_error,
            nan, nan, nan,
            nan, nan, nan,
            vel[0], vel[1], vel[2],
            nan, nan, nan,
            0.0, 0.0, pitch, roll,
            level_roll, level_pitch, 0.0, thrust,
            self._z_integral, yaw, nan, nan, 0,
        ])

        if pos[2] <= TAKEOFF_ALT_NED + 0.1:
            # Seed slew from current velocity so the next state doesn't brake from zero on entry
            self._prev_desired_vel = vel.copy()
            self._transition('CALIBRATE' if GUIDANCE == 'CALIBRATE' else 'FLY')

    def _handle_fly(self):
        pos = self.data.get('pos')
        vel = self.data.get('vel', np.zeros(3))
        if pos is None:
            return
        nan = float('nan')

        race_status        = self.data.get('race_status', {})
        active_gate        = race_status.get('active_gate', 0)
        roll, pitch, yaw   = self.data.get('attitude', (0.0, 0.0, 0.0))

        if race_status.get('race_finished', False):
            self._transition('FINISHED')
            return

        # Late-arriving gate map — build waypoints and spline on the fly (MAVLINK / CV_LIVE only)
        if GUIDANCE != 'CV_PLAN' and not self.waypoints and 'gates' in self.data:
            self._build_waypoints()

        # Sync gate counters with sim's authoritative active_gate signal
        if active_gate > self._cv_gate_idx:
            print(f"[ctrl] gate {self._cv_gate_idx} confirmed, now targeting gate {active_gate}", flush=True)
            self._cv_gate_idx = active_gate

        if self.waypoints and active_gate * 2 > self.current_idx:
            prev_idx = self.current_idx
            self.current_idx = active_gate * 2
            print(f"[ctrl] wp sync: {prev_idx} -> {self.current_idx}", flush=True)

        if self.waypoints and self.current_idx >= len(self.waypoints):
            self._transition('FINISHED')
            return

        # ── CV check ──────────────────────────────────────────────────────────
        cv_pos   = self.data.get('cv_gate_pos')
        cv_age   = time.time() - self.data.get('cv_gate_time', 0.0)
        cv_fresh = GUIDANCE == 'CV_LIVE' and cv_pos is not None and cv_age < CV_STALE_TIMEOUT
        if cv_fresh:
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            if np.dot(cv_pos - pos, forward) < 0.0:
                cv_fresh = False

        # CV_PLAN: accumulate fresh in-front estimates and keep plan up to date
        if GUIDANCE == 'CV_PLAN' and cv_pos is not None and cv_age < CV_STALE_TIMEOUT:
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            if np.dot(cv_pos - pos, forward) > 0.0:
                if self._accumulate_cv_estimate(active_gate, cv_pos):
                    self._rebuild_cv_plan()

        # ── Gate advancement and distance (computed before target selection) ────
        gate_dist = 999.0
        next_gate_wp = None
        if self.waypoints and self.current_idx < len(self.waypoints):
            wp        = self.waypoints[self.current_idx]
            is_leadin = (self.current_idx % 2 == 0)
            wp_dist   = float(np.linalg.norm(wp - pos))

            if is_leadin and wp_dist < WAYPOINT_RADIUS:
                print(f"[ctrl] lead-in wp {self.current_idx}: advancing", flush=True)
                self.current_idx += 1
                if self.current_idx >= len(self.waypoints):
                    self._transition('FINISHED')
                    return
                is_leadin = False

            if not is_leadin and not cv_fresh:
                gate_idx      = self.current_idx // 2
                sim_confirmed = active_gate > gate_idx
                # Project drone position onto the approach direction (lead-in → gate).
                # The old X-axis check only worked for North-facing gates; this works
                # for any gate orientation.
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

        # Linear ramp from GATE_APPROACH_SPEED (at gate) to CRUISE_SPEED (at GATE_APPROACH_DIST).
        # Avoids the hard setpoint step that caused the drone to overshoot into backward flight.
        speed_cap = GATE_APPROACH_SPEED + (CRUISE_SPEED - GATE_APPROACH_SPEED) * min(1.0, gate_dist / GATE_APPROACH_DIST)

        # ── Position target and desired velocity ──────────────────────────────────
        tangent_unit = None

        if cv_fresh:
            target      = cv_pos
            error       = target - pos
            horiz_dist  = float(np.linalg.norm(error[:2]))
            spd         = min(GATE_APPROACH_SPEED, KP_POS * horiz_dist)
            desired_vel = np.zeros(3)
            if horiz_dist > 0.1:
                desired_vel[:2] = error[:2] / horiz_dist * spd
            desired_vel[2] = float(np.clip(KP_POS_Z * error[2], -MAX_Z_VEL, MAX_Z_VEL))
            src = f'CV(age={cv_age*1000:.0f}ms)'
        elif self.path is not None and len(self.path) > 1:
            # Primary mode: feed-forward along spline tangent + cross-track correction
            desired_vel, tangent_unit = self._path_tracking_desired_vel(pos, speed_cap)
            desired_vel[2] = float(np.clip(desired_vel[2], -MAX_Z_VEL, MAX_Z_VEL))
            target     = self.path[self._path_idx]   # nearest spline point — for logging
            error      = target - pos
            horiz_dist = float(np.linalg.norm(error[:2]))
            src        = 'spline'
        elif self.waypoints and self.current_idx < len(self.waypoints):
            # Fallback: proportional waypoint-chasing (no spline available yet)
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
                nan, nan, nan, nan,
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
                  f"zi={self._z_integral:.3f}  sim_gate={active_gate}", flush=True)

        # ── Speed brake ────────────────────────────────────────────────────────
        # Blend over-speed correction into the desired velocity before slew limiting,
        # so braking cannot introduce a one-frame setpoint jump.
        actual_spd_xy = float(np.linalg.norm(vel[:2]))
        if actual_spd_xy > speed_cap and actual_spd_xy > 0.1:
            excess = actual_spd_xy - speed_cap
            vel_xy_unit = vel[:2] / actual_spd_xy
            desired_vel = desired_vel.copy()
            desired_vel[:2] -= vel_xy_unit * excess * K_BRAKE

        # ── Velocity setpoint slew rate ────────────────────────────────────────
        # Prevents sudden velocity target jumps (e.g. speed_cap changing near gate)
        # from causing underdamped pitch/roll oscillations.
        vel_delta = desired_vel - self._prev_desired_vel
        max_delta = MAX_VEL_SLEW / CONTROL_HZ
        delta_mag = float(np.linalg.norm(vel_delta))
        if delta_mag > max_delta:
            desired_vel = self._prev_desired_vel + vel_delta * (max_delta / delta_mag)
        self._prev_desired_vel = desired_vel.copy()

        # ── Velocity error → attitude rates ────────────────────────────────────
        vel_error = desired_vel - vel

        # Filtered derivative: damps velocity overshoot without amplifying sensor noise
        d_vel               = (vel_error - self._prev_vel_error) * CONTROL_HZ
        self._vel_error_dot = 0.3 * d_vel + 0.7 * self._vel_error_dot
        self._prev_vel_error = vel_error.copy()
        damped_err          = vel_error + KD_VEL * self._vel_error_dot

        z_vel_err = vel_error[2]
        self._z_integral -= z_vel_err / CONTROL_HZ * KI_ALT
        self._z_integral = float(np.clip(self._z_integral, -0.25, 0.25))
        # Scale hover thrust to compensate for lost vertical component when tilted
        tilt_comp = 1.0 / max(0.5, float(np.cos(roll) * np.cos(pitch)))
        thrust = float(np.clip(HOVER_THRUST * tilt_comp + self._z_integral - KP_VEL_Z * z_vel_err, MIN_FLIGHT_THRUST, 1.0))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        fwd   =  cos_yaw * damped_err[0] + sin_yaw * damped_err[1]
        right = -sin_yaw * damped_err[0] + cos_yaw * damped_err[1]

        # Outer loop: velocity error → desired tilt angle.
        # NOTE: pitch's sign here is *not* the same as the old direct mapping
        # (-KP_VEL * fwd). The sim applies pitch_rate with INVERTED effect on pitch
        # (d(pitch)/dt ~ -pitch_rate_cmd, empirically confirmed), so the old
        # single-stage formula's net pitch->fwd relationship was actually
        # pitch ~ +fwd. Inserting an explicit angle stage removes that inversion
        # from the loop, so desired_pitch must carry the *net* sign directly:
        # +fwd -> +desired_pitch (flight-tested 2026-06-07: -fwd sign sent the
        # drone the opposite way down the course, confirmed via fwd staying
        # persistently positive while pos_x diverged 85 m the wrong direction).
        # Roll has no such inversion (roll_rate acts on roll with the SAME sign),
        # so its outer-loop sign is unchanged from the old direct mapping.
        desired_pitch = float(np.clip( KP_VEL_ANGLE * fwd,   -MAX_TILT_ANGLE, MAX_TILT_ANGLE))
        desired_roll  = float(np.clip( KP_VEL_ANGLE * right, -MAX_TILT_ANGLE, MAX_TILT_ANGLE))

        # Inner loop: tilt-angle error → attitude rate. This is the missing damping term
        # — it commands "stop rotating, you're at the angle that gives the acceleration
        # you want" instead of letting vel_error drive the rate (and therefore the angle)
        # indefinitely. Mirrors _handle_takeoff's leveling formula (KP_LEVEL * pitch /
        # -KP_LEVEL * roll, lines ~291-292) generalized from a target of zero to a
        # non-zero desired_pitch/desired_roll — same proven sign convention.
        pitch_rate = float(np.clip( KP_ANGLE * (pitch - desired_pitch), -MAX_RATE, MAX_RATE))
        roll_rate  = float(np.clip(-KP_ANGLE * (roll  - desired_roll),  -MAX_RATE, MAX_RATE))

        actual_speed = float(np.linalg.norm(vel))
        if actual_speed > MAX_SPEED:
            scale = actual_speed / MAX_SPEED
            pitch_rate = float(np.clip(pitch_rate * scale, -MAX_RATE, MAX_RATE))
            roll_rate  = float(np.clip(roll_rate  * scale, -MAX_RATE, MAX_RATE))

        # ── Yaw: point nose toward path tangent (or error direction as fallback) ─
        # Simulator applies yaw_rate counterclockwise from above (opposite NED),
        # so negate to get the intended clockwise (North→East) rotation.
        target_yaw = nan
        yaw_err    = nan
        yaw_rate   = 0.0
        if tangent_unit is not None:
            # Spline mode: always yaw from tangent regardless of cross-track distance.
            # Skipping this when cross-track is small causes yaw to drift uncorrected,
            # which misaligns the body frame and makes all velocity corrections go in the
            # wrong NED direction.
            target_yaw = float(np.arctan2(tangent_unit[1], tangent_unit[0]))
            yaw_err    = float(((target_yaw - yaw + np.pi) % (2 * np.pi)) - np.pi)
            yaw_rate   = float(np.clip(-KP_YAW * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE))
        elif horiz_dist > 1.0:
            # Waypoint/CV fallback: only yaw when direction is meaningful
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
            desired_pitch, desired_roll, pitch, roll,
            roll_rate, pitch_rate, yaw_rate, thrust,
            self._z_integral, yaw, target_yaw, yaw_err, saturated,
        ])

    def _handle_calibrate(self):
        """
        Fly the scripted Phase-3 calibration sequence (calibration_plan.py):
        for each leg, settle into its starting (position, yaw) configuration,
        then hold/sweep it for `duration` seconds while CV logs every detection
        with attitude + raw tvec — making the resulting gate_estimates_*.csv
        self-sufficient for offline scoring of every Phase-3 question (A11
        tilt sign, A13 yaw coupling, A1/A9/A10 range error, A6/A5/A7 oblique
        robustness). No racing — `self.waypoints`/`self.path` stay untouched.

        Reuses _handle_fly's proven velocity-PID -> tilt-angle -> attitude-rate
        cascade verbatim (duplicated rather than shared, per project convention
        of not risking the flight-tuned racing path) but drives it from an
        explicit per-leg target instead of waypoints/spline/CV.
        """
        pos = self.data.get('pos')
        vel = self.data.get('vel', np.zeros(3))
        if pos is None:
            return
        now = time.time()
        roll, pitch, yaw = self.data.get('attitude', (0.0, 0.0, 0.0))

        race_status = self.data.get('race_status', {})
        if race_status.get('race_finished', False):
            self._transition('FINISHED')
            return

        if not self._cal_legs:
            if now - self._last_diag_time >= 2.0:
                self._last_diag_time = now
                print("[cal] WARNING: no calibration plan (no gate map arrived) — hovering", flush=True)
            self._send_attitude_rates(0.0, 0.0, 0.0, HOVER_THRUST + self._z_integral)
            return

        if self._cal_idx >= len(self._cal_legs):
            print("[cal] all legs complete", flush=True)
            self._transition('FINISHED')
            return

        leg = self._cal_legs[self._cal_idx]

        # ── Target selection: settle into the leg's start config, then sweep it ──
        if self._cal_leg_start_time is None:
            phase      = 'settling'
            frac       = 0.0
            target     = leg.pos_start
            target_yaw = leg.yaw_start

            pos_err = float(np.linalg.norm(pos - leg.pos_start))
            yaw_err_settle = abs(_wrap(leg.yaw_start - yaw))
            if pos_err < SETTLE_RADIUS and yaw_err_settle < SETTLE_YAW_TOL:
                if self._settle_start_time is None:
                    self._settle_start_time = now
                elif now - self._settle_start_time >= SETTLE_HOLD_S:
                    self._cal_leg_start_time = now
                    self._settle_start_time  = None
                    print(f"[cal] leg {self._cal_idx} '{leg.name}' settled "
                          f"(pos_err={pos_err:.2f}m yaw_err={np.degrees(yaw_err_settle):.1f} deg) "
                          f"— starting {leg.duration:.0f}s data window", flush=True)
            else:
                self._settle_start_time = None
        else:
            t_leg = now - self._cal_leg_start_time
            frac  = float(np.clip(t_leg / leg.duration, 0.0, 1.0))
            phase = 'active'
            target     = leg.pos_start + (leg.pos_end - leg.pos_start) * frac
            target_yaw = _lerp_angle(leg.yaw_start, leg.yaw_end, frac)
            if t_leg >= leg.duration:
                print(f"[cal] leg {self._cal_idx} '{leg.name}' complete — advancing", flush=True)
                self._cal_idx            += 1
                self._cal_leg_start_time  = None
                self._settle_start_time   = None

        settled = int(self._cal_leg_start_time is not None)

        # ── Position error -> desired velocity (slow, deliberate proportional chase) ──
        error      = target - pos
        horiz_dist = float(np.linalg.norm(error[:2]))
        spd        = min(CAL_SPEED_CAP, KP_POS * horiz_dist)
        desired_vel = np.zeros(3)
        if horiz_dist > 0.1:
            desired_vel[:2] = error[:2] / horiz_dist * spd
        desired_vel[2] = float(np.clip(KP_POS_Z * error[2], -MAX_Z_VEL, MAX_Z_VEL))

        # ── Speed brake (same over-speed correction as _handle_fly, capped to CAL_SPEED_CAP) ──
        actual_spd_xy = float(np.linalg.norm(vel[:2]))
        if actual_spd_xy > CAL_SPEED_CAP and actual_spd_xy > 0.1:
            excess = actual_spd_xy - CAL_SPEED_CAP
            vel_xy_unit = vel[:2] / actual_spd_xy
            desired_vel = desired_vel.copy()
            desired_vel[:2] -= vel_xy_unit * excess * K_BRAKE

        # ── Velocity setpoint slew (same anti-oscillation limiter as _handle_fly) ──
        vel_delta = desired_vel - self._prev_desired_vel
        max_delta = MAX_VEL_SLEW / CONTROL_HZ
        delta_mag = float(np.linalg.norm(vel_delta))
        if delta_mag > max_delta:
            desired_vel = self._prev_desired_vel + vel_delta * (max_delta / delta_mag)
        self._prev_desired_vel = desired_vel.copy()

        # ── Velocity error -> attitude rates (identical cascade to _handle_fly) ──
        vel_error = desired_vel - vel
        d_vel               = (vel_error - self._prev_vel_error) * CONTROL_HZ
        self._vel_error_dot = 0.3 * d_vel + 0.7 * self._vel_error_dot
        self._prev_vel_error = vel_error.copy()
        damped_err          = vel_error + KD_VEL * self._vel_error_dot

        z_vel_err = vel_error[2]
        self._z_integral -= z_vel_err / CONTROL_HZ * KI_ALT
        self._z_integral = float(np.clip(self._z_integral, -0.25, 0.25))
        tilt_comp = 1.0 / max(0.5, float(np.cos(roll) * np.cos(pitch)))
        thrust = float(np.clip(HOVER_THRUST * tilt_comp + self._z_integral - KP_VEL_Z * z_vel_err,
                               MIN_FLIGHT_THRUST, 1.0))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        fwd   =  cos_yaw * damped_err[0] + sin_yaw * damped_err[1]
        right = -sin_yaw * damped_err[0] + cos_yaw * damped_err[1]

        desired_pitch = float(np.clip( KP_VEL_ANGLE * fwd,   -MAX_TILT_ANGLE, MAX_TILT_ANGLE))
        desired_roll  = float(np.clip( KP_VEL_ANGLE * right, -MAX_TILT_ANGLE, MAX_TILT_ANGLE))
        pitch_rate = float(np.clip( KP_ANGLE * (pitch - desired_pitch), -MAX_RATE, MAX_RATE))
        roll_rate  = float(np.clip(-KP_ANGLE * (roll  - desired_roll),  -MAX_RATE, MAX_RATE))

        # ── Yaw: track the leg's commanded yaw directly — this IS the test for yaw_sweep ──
        yaw_err  = _wrap(target_yaw - yaw)
        yaw_rate = float(np.clip(-KP_YAW * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE))

        self._send_attitude_rates(roll_rate, pitch_rate, yaw_rate, thrust)
        self._cal_csv.writerow([
            now, self._cal_idx, leg.name, phase, frac,
            target[0], target[1], target[2], target_yaw,
            pos[0], pos[1], pos[2], yaw,
            error[0], error[1], error[2], horiz_dist, yaw_err,
            desired_vel[0], desired_vel[1], desired_vel[2],
            vel[0], vel[1], vel[2],
            roll_rate, pitch_rate, yaw_rate, thrust,
            settled,
        ])

        if now - self._last_diag_time >= 2.0:
            self._last_diag_time = now
            print(f"[cal] leg {self._cal_idx} '{leg.name}' [{phase}] frac={frac:.2f}  "
                  f"pos_err={horiz_dist:.2f}m  yaw_err={np.degrees(yaw_err):+.1f} deg  "
                  f"speed={float(np.linalg.norm(vel)):.2f}m/s", flush=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

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
        if self._cal_log is not None and not self._cal_log.closed:
            self._cal_log.close()

    def _transition(self, new_state):
        print(f"[ctrl] {self.state} --> {new_state}", flush=True)
        self.state = new_state

    # ── Commands ─────────────────────────────────────────────────────────────

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
