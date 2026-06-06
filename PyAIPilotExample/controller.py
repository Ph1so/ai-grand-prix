import csv
import os
import time

import numpy as np
from pymavlink import mavutil

from planner import Plan, TAKEOFF_ALT as TAKEOFF_ALT_NED, smooth_path, LEAD_IN_DIST

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

MAVLINK_CMD_SIM_RESET = 31000

# ── Tuning constants ────────────────────────────────────────────────────────
HOVER_THRUST      = 0.28   # estimated actual hover thrust (derived from equilibrium data)
KP_THRUST         = 0.15   # extra thrust per metre of altitude error during climb
KI_ALT            = 0.02   # integral gain — fine-tunes hover estimate over time
CRUISE_SPEED      = 2.5    # m/s — max speed toward a waypoint
GATE_APPROACH_SPEED = 1.0  # m/s — speed cap on final approach to gate center
GATE_APPROACH_DIST  = 30.0 # m — distance at which to start slowing for gate
MAX_SPEED         = 6.0    # m/s — hard cap; above this, scale rates for deceleration
KP_POS            = 0.4    # position error (m) → desired speed (m/s)
KP_POS_Z          = 0.60   # altitude error (m) → desired climb rate (m/s), independent of XY
MAX_Z_VEL         = 3.5    # m/s — max climb / descend rate
KP_VEL            = 0.16   # velocity error (m/s) → attitude rate (rad/s)
KP_VEL_Z          = 0.08   # z velocity error (m/s) → thrust delta
KP_LEVEL          = 2.0    # attitude angle (rad) → levelling rate (rad/s) used in takeoff
MIN_FLIGHT_THRUST = 0.18   # lower bound during flight — prevents Z overcorrection causing crash
MAX_RATE          = 0.6    # rad/s — max pitch/roll rate command
WAYPOINT_RADIUS   = 1.5    # m — switch to next waypoint when within this distance
GATE_WAIT_TIMEOUT = 5.0    # seconds to wait for gate map before flying blind
CONTROL_HZ        = 100    # Hz
CV_STALE_TIMEOUT  = 0.3    # seconds — treat CV estimate as lost after this gap
LOOKAHEAD_DIST    = 12.0   # m — pure-pursuit look-ahead on smooth spline
KP_YAW            = 0.8    # rad/s per rad of yaw error
MAX_YAW_RATE      = 0.8    # rad/s

# ── Guidance mode ────────────────────────────────────────────────────────────
# 'MAVLINK'  — follow pre-planned MAVLink waypoints (default)
# 'CV_LIVE'  — track live CV pose estimate each frame; MAVLink waypoints as fallback
# 'CV_PLAN'  — accumulate CV estimates per gate, build + follow a CV-derived plan; no MAVLink fallback
GUIDANCE = 'CV_PLAN'

RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


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
        self._z_integral = 0.0    # altitude integral — accumulates hover thrust error

        self._init_pos_time  = None
        self._last_diag_time = 0.0
        self._idle_seen_not_started = False
        self._idle_last_boot = 0
        self._idle_start_time = None

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
            'cmd_roll_rate', 'cmd_pitch_rate', 'cmd_yaw_rate', 'cmd_thrust',
            'z_integral', 'yaw', 'target_yaw', 'yaw_err', 'saturated',
        ])
        print(f"[ctrl] command log → {log_path}", flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    def update(self):
        # Detect race restart: if sim resets us back near the origin while we're
        # mid-flight or finished, re-enter the start sequence cleanly.
        if self.state in ('FLY', 'FINISHED', 'TAKEOFF'):
            pos = self.data.get('pos')
            race_started = self.data.get('race_status', {}).get('race_started', True)
            if pos is not None and np.linalg.norm(pos) < 2.0 and not race_started:
                print("[ctrl] race reset detected — returning to IDLE", flush=True)
                self.current_idx    = 0
                self._cv_gate_idx   = 0
                self._cv_estimates  = {}
                self._cv_gate_means = {}
                self._z_integral    = 0.0
                self._path_idx      = 0
                self._idle_seen_not_started = False
                self._idle_start_time = None
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

    # ── State handlers ───────────────────────────────────────────────────────

    def _handle_init(self):
        if 'pos' not in self.data:
            return
        now = time.time()
        if self._init_pos_time is None:
            self._init_pos_time = now
            msg = "CV — not waiting for gate map" if GUIDANCE != 'MAVLINK' else "waiting for gate map..."
            print(f"[ctrl] position acquired, {msg}", flush=True)

        if 'gates' in self.data and GUIDANCE != 'CV_PLAN':
            self._build_waypoints()
            self._transition('IDLE')
        elif GUIDANCE != 'MAVLINK' or now - self._init_pos_time >= GATE_WAIT_TIMEOUT:
            if GUIDANCE == 'MAVLINK':
                print("[ctrl] WARNING: no gate map — flying without waypoints", flush=True)
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
            level_roll, level_pitch, 0.0, thrust,
            self._z_integral, yaw, nan, nan, 0,
        ])

        if pos[2] <= TAKEOFF_ALT_NED + 0.1:
            self._transition('FLY')

    def _handle_fly(self):
        pos = self.data.get('pos')
        vel = self.data.get('vel', np.zeros(3))
        if pos is None:
            return
        nan = float('nan')

        race_status = self.data.get('race_status', {})
        active_gate = race_status.get('active_gate', 0)
        _, _, yaw   = self.data.get('attitude', (0.0, 0.0, 0.0))

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
                print(f"[ctrl] lead-in wp {self.current_idx} reached", flush=True)
                self.current_idx += 1
                if self.current_idx >= len(self.waypoints):
                    self._transition('FINISHED')
                    return
                is_leadin = False

            if not is_leadin and not cv_fresh:
                gate_idx      = self.current_idx // 2
                sim_confirmed = active_gate > gate_idx
                drone_past_gate = (pos[0] < wp[0] - 1.0 and
                                   abs(pos[1] - wp[1]) < 1.0 and
                                   abs(pos[2] - wp[2]) < 1.0)
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

        if gate_dist < GATE_APPROACH_DIST:
            t = gate_dist / GATE_APPROACH_DIST
            speed_cap = GATE_APPROACH_SPEED + (CRUISE_SPEED - GATE_APPROACH_SPEED) * t
        else:
            speed_cap = CRUISE_SPEED

        # ── Position target: CV > gate center (close approach) > spline > wp ───
        # Within gate approach distance the look-ahead overshoots the gate altitude;
        # target the gate center directly so the drone aligns with the opening.
        if cv_fresh:
            target = cv_pos
            src    = f'CV(age={cv_age*1000:.0f}ms)'
        elif next_gate_wp is not None and gate_dist < GATE_APPROACH_DIST:
            target = next_gate_wp
            src    = 'gate'
        elif self.path is not None:
            target = self._lookahead_target(pos)
            src    = 'spline'
        elif self.waypoints and self.current_idx < len(self.waypoints):
            target = self.waypoints[self.current_idx]
            src    = f'wp[{self.current_idx}]'
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
                0.0, -0.2, 0.0, creep_thrust,
                self._z_integral, yaw, nan, nan, 0,
            ])
            return

        # ── Control errors ─────────────────────────────────────────────────────
        error      = target - pos
        dist       = float(np.linalg.norm(error))
        horiz_dist = float(np.linalg.norm(error[:2]))

        now = time.time()
        if now - self._last_diag_time >= 2.0:
            self._last_diag_time = now
            speed = float(np.linalg.norm(vel))
            gate_label = self._cv_gate_idx if cv_fresh else self.current_idx // 2
            print(f"[fly] gate {gate_label} src={src}: dist={dist:.1f}m  "
                  f"target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})  "
                  f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  "
                  f"spd={speed:.1f}m/s  zi={self._z_integral:.3f}  sim_gate={active_gate}", flush=True)

        # ── Position → desired velocity (NED) ─────────────────────────────────
        if horiz_dist > 0.1:
            desired_horiz_speed = min(speed_cap, KP_POS * horiz_dist)
            desired_vel = np.array([
                error[0] / horiz_dist * desired_horiz_speed,
                error[1] / horiz_dist * desired_horiz_speed,
                0.0,
            ])
        else:
            desired_vel = np.zeros(3)

        z_cap = GATE_APPROACH_SPEED if gate_dist < GATE_APPROACH_DIST else MAX_Z_VEL
        desired_vel[2] = float(np.clip(KP_POS_Z * error[2], -z_cap, z_cap))

        # ── Velocity error → attitude rates ────────────────────────────────────
        vel_error = desired_vel - vel

        z_vel_err = vel_error[2]
        self._z_integral -= z_vel_err / CONTROL_HZ * KI_ALT
        self._z_integral = float(np.clip(self._z_integral, -0.25, 0.25))
        thrust = float(np.clip(HOVER_THRUST + self._z_integral - KP_VEL_Z * z_vel_err, MIN_FLIGHT_THRUST, 1.0))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        fwd   =  cos_yaw * vel_error[0] + sin_yaw * vel_error[1]
        right = -sin_yaw * vel_error[0] + cos_yaw * vel_error[1]

        pitch_rate = float(np.clip(-KP_VEL * fwd,   -MAX_RATE, MAX_RATE))
        roll_rate  = float(np.clip( KP_VEL * right,  -MAX_RATE, MAX_RATE))

        actual_speed = float(np.linalg.norm(vel))
        if actual_speed > MAX_SPEED:
            scale = actual_speed / MAX_SPEED
            pitch_rate = float(np.clip(pitch_rate * scale, -MAX_RATE, MAX_RATE))
            roll_rate  = float(np.clip(roll_rate  * scale, -MAX_RATE, MAX_RATE))

        # ── Yaw control — point nose toward look-ahead target ──────────────────
        # Simulator applies yaw_rate counterclockwise from above (opposite of NED convention),
        # so negate to get the intended clockwise (North→East) rotation.
        target_yaw = nan
        yaw_err    = nan
        yaw_rate   = 0.0
        if horiz_dist > 1.0:
            target_yaw = float(np.arctan2(error[1], error[0]))
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
            roll_rate, pitch_rate, yaw_rate, thrust,
            self._z_integral, yaw, target_yaw, yaw_err, saturated,
        ])

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
