# Drone Control System — Complete Technical Fact Sheet

## 1. System Architecture

### 1.1 Threading model
- `main.py` runs a single-threaded **100 Hz control loop** calling `controller.update()` in a `while True`
- `MAVLinkRX` runs in a **separate daemon thread**, writing to `shared_data` dict (no locks)
- `GateVerifier` (subclass of `VisionRX`) runs in a **separate daemon thread**, writing CV estimates to `shared_data`
- `TimeSyncLoop` runs in a **separate daemon thread** at 10 Hz
- The control loop reads `shared_data` at the start of each `update()` call; values may have been written by any thread at any time between calls

### 1.2 MAVLink interface
- Simulator → client at UDP :14550
- Client → simulator via `SET_ATTITUDE_TARGET` with `type_mask = ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` (value 0x80)
- This mask tells the sim to use body rates and thrust, and ignore the quaternion field
- The quaternion field is hard-coded to `[1, 0, 0, 0]` (identity) in every send
- Telemetry received: `ATTITUDE` (roll, pitch, yaw in radians), `LOCAL_POSITION_NED` (pos xyz m, vel xyz m/s), `ENCAPSULATED_DATA` (race status, gate map)
- Sim sends `ENCAPSULATED_DATA` type=1 as race status; type=2 as track info (multi-packet, reassembled)
- `active_gate` in race status is the index of the next gate the drone must pass (0-indexed), starting at 0 when race begins; increments when sim confirms passage
- `race_started` is `True` when `race_start_boot_time_ms >= 0`
- `race_finished` is `True` when `race_finish_time_ns >= 0`

### 1.3 Coordinate system
- All position, velocity, and waypoints are in **MAV_FRAME_LOCAL_NED**: origin at arm point on ground, X=North, Y=East, Z=Down (positive Z = below takeoff altitude)
- Negative Z = above takeoff altitude; e.g. pos_z = −1.5 means drone is 1.5 m AGL
- Gate pos from the sim: simulator sends Z as altitude-above-ground (Z-up). `mavlink_rx.on_track_data()` negates Z on receipt, storing `pos[2] = -z_sim`. This means a gate at 2 m AGL is stored with `pos[2] = -2.0` in `shared_data['gates']`
- Gate center altitude computation in `planner.gate_center()`: `pos[2] = -pos[2] - height/2 + GATE_Z_BIAS`. Undoes the MAVLink negation, subtracts half the gate height to reach center, adds `GATE_Z_BIAS = -0.5 m` (upward adjustment)

### 1.4 Attitude representation
- `shared_data['attitude']` is a 3-tuple `(roll, pitch, yaw)` in radians, written by `on_attitude()`
- Convention: yaw = 0 → North, yaw = π/2 → East, yaw = ±π → South
- The controller reads it as `roll, pitch, yaw = self.data.get('attitude', (0.0, 0.0, 0.0))`

---

## 2. State Machine

States: `INIT → IDLE → TAKEOFF → FLY → FINISHED`

### 2.1 INIT
- Waits until `'pos'` appears in `shared_data`
- If `'gates'` arrives and `GUIDANCE != 'CV_PLAN'`: calls `_build_waypoints()` then transitions to IDLE
- Falls through to IDLE after `GATE_WAIT_TIMEOUT = 5.0 s` even without gate map (prints warning)

### 2.2 IDLE
- Sends zero rates, thrust = `HOVER_THRUST` if pos_z < −1.0 m (airborne at reset), else thrust = 0
- Waits for race_started to flip True (with hysteresis: must have seen `race_started=False` first, then True, then waits 0.5 s)
- Transition to TAKEOFF requires: `race_started=True`, `_idle_seen_not_started=True`, `time_boot_ms` advancing (sim is running), and 0.5 s elapsed since start signal

### 2.3 TAKEOFF
- Target altitude: `TAKEOFF_ALT_NED = -0.5 m`
- Thrust: `clip(HOVER_THRUST - KP_THRUST * alt_error, 0, 1)` where `alt_error = TAKEOFF_ALT_NED - pos[2]`
- Attitude levelling: `level_pitch = clip(KP_LEVEL * pitch, -MAX_RATE, MAX_RATE)` and `level_roll = clip(-KP_LEVEL * roll, -MAX_RATE, MAX_RATE)` — note the sign inversion on roll
- Transitions to FLY when `pos[2] <= TAKEOFF_ALT_NED + 0.1 m` (i.e. pos_z ≤ −0.4 m)

### 2.4 FLY
- The main racing state. Described in full in Section 4.
- Transitions to FINISHED when `race_finished=True` or `current_idx >= len(waypoints)`

### 2.5 FINISHED
- Sends zero rates, thrust = `HOVER_THRUST + _z_integral` indefinitely

### 2.6 Race reset detection
- In FLY, FINISHED, or TAKEOFF: if `norm(pos) < 2.0 m` and `race_started=False`, resets all state and returns to IDLE
- Resets: `current_idx`, `_cv_gate_idx`, `_cv_estimates`, `_cv_gate_means`, `_z_integral`, `_path_idx`, `_settle_start_time`, `_prev_vel_error`, `_vel_error_dot`, `_idle_seen_not_started`, `_idle_start_time`

---

## 3. Planner

### 3.1 Inputs
- Gate list from `shared_data['gates']`, sorted by gate `id` (ascending)
- Start point defaults to `[0, 0, TAKEOFF_ALT]` = `[0, 0, -0.5]` NED

### 3.2 Waypoint structure
- Flat alternating list: `[lead-in-0, gate-0, lead-in-1, gate-1, ...]`
- The `start` waypoint is index 0 in the plan but **skipped** by the controller (controller uses `plan.waypoints[1:]`)
- Even indices in controller's list = lead-in waypoints
- Odd indices = gate center waypoints
- Total waypoints = 2 × number of gates (after removing start)

### 3.3 Lead-in computation
- `lead_in = gate_center - (unit_vector_from_prev_to_gate) × LEAD_IN_DIST`
- `LEAD_IN_DIST = 12.0 m`
- Direction computed from the **previous waypoint** (start or last gate center) to the current gate center
- Placed 12 m before each gate along the approach axis

### 3.4 Gate center altitude
- Raw from sim: Z-up AGL, e.g. gate at 2 m AGL sends z=2.0
- After `on_track_data()` negation: stored as pos[2] = −2.0
- After `gate_center()`: `pos[2] = −(−2.0) − height/2 + GATE_Z_BIAS = 2.0 − 1.36 − 0.5 = 0.14 m NED`
- `GATE_Z_BIAS = −0.5 m` is a constant upward nudge applied uniformly to all gate centers

### 3.5 Spline
- Type: C2-continuous cubic spline via `scipy.interpolate.CubicSpline` with `bc_type='not-a-knot'`
- Parameterized by cumulative chord length, normalized to [0, 1]
- Density: 80 samples per waypoint-to-waypoint segment
- For a 6-gate course with 13 waypoints (start + 12): 1040 spline points total
- The spline includes the `start` point at `[0, 0, −0.5]`; this is the point before the controller's waypoint list begins
- Spline used exclusively in FLY state for `_path_tracking_desired_vel()`; the `_lookahead_target()` method exists but is **never called** during flight

---

## 4. FLY State — Full Control Flow

### 4.1 Gate advancement logic
Runs every control loop iteration before velocity computation.

**Lead-in advancement** (even `current_idx`):
`wp_dist < WAYPOINT_RADIUS (1.5 m)` → advance `current_idx` by 1

**Gate advancement** (odd `current_idx`):
Two triggers, either sufficient:
1. `sim_confirmed`: `active_gate > current_idx // 2` (sim says next gate is now active)
2. `drone_past_gate`: `pos[0] < wp[0] − 1.0` AND `abs(pos[1] − wp[1]) < 1.0` AND `abs(pos[2] − wp[2]) < 1.0`

**Sim sync override**: if `active_gate * 2 > current_idx`, jumps `current_idx = active_gate * 2` (skips ahead to match sim)

`gate_dist` is the 3D Euclidean distance to the **next odd-index waypoint** (gate center), not the next waypoint the controller is currently targeting.

### 4.2 Speed cap
```python
speed_cap = GATE_APPROACH_SPEED + (CRUISE_SPEED - GATE_APPROACH_SPEED) * min(1.0, gate_dist / GATE_APPROACH_DIST)
```
- `CRUISE_SPEED = 3.0 m/s`, `GATE_APPROACH_SPEED = 1.2 m/s`, `GATE_APPROACH_DIST = 12.0 m`
- Linear ramp: at gate_dist ≥ 12 m → `speed_cap = 3.0`; at gate_dist = 0 → `speed_cap = 1.2`
- At gate_dist = 6 m → `speed_cap = 2.1 m/s`

### 4.3 Target selection priority
1. **CV** (active only in `GUIDANCE = 'CV_LIVE'`): uses `cv_gate_pos` if fresh (age < 0.3 s) and in front of drone
2. **Spline** (`self.path is not None`): calls `_path_tracking_desired_vel(pos, speed_cap)`
3. **Waypoint fallback** (`self.waypoints` available, no spline): proportional chase of `waypoints[current_idx]`
4. **Creep**: if nothing available, sends pitch_rate = −0.2 rad/s, thrust = `HOVER_THRUST + _z_integral`

In practice, `GUIDANCE = 'MAVLINK'` and a gate map is always received before flight starts, so the spline path is always available. Mode 2 (spline) is the normal operating mode.

### 4.4 Spline path tracking — `_path_tracking_desired_vel(pos, cruise_speed)`

**Step 1: Find nearest path index**
```python
lo = max(0, _path_idx - 50)
hi = min(n, _path_idx + 300)
_path_idx = lo + argmin(norm(path[lo:hi] - pos, axis=1))
```
Search window: 50 points behind, 300 points ahead of last known index. Total window: 350 points.

**Step 2: Compute tangent**
```python
look = min(_path_idx + TANGENT_SAMPLES, n-1)   # TANGENT_SAMPLES = 20
tangent = path[look] - path[_path_idx]
tangent_unit = tangent / norm(tangent)
```
Tangent computed as chord from `_path_idx` to `_path_idx + 20`. At 80 samples/segment and typical segment ≈ 10–15 m, 20 samples ≈ 2.5–3.75 m lookahead for tangent direction.

**Step 3: Cross-track error**
```python
nearest = path[_path_idx]
to_nearest = nearest - pos
along = dot(to_nearest, tangent_unit)
cross = to_nearest - along * tangent_unit
```
`cross` is the component of `(nearest_point − drone_pos)` perpendicular to the tangent. Units: metres. Positive direction: toward the path from the drone.

**Step 4: Desired velocity**
```python
desired_vel = cruise_speed * tangent_unit + KP_CROSS * cross
if norm(desired_vel) > cruise_speed:
    desired_vel = desired_vel * (cruise_speed / norm(desired_vel))
```
Feed-forward: `cruise_speed × tangent_unit`. Cross-track feedback: `KP_CROSS × cross` where `KP_CROSS = 0.8`. Total magnitude capped at `cruise_speed`. The cap is applied **after** computing the combined vector; when cross-track is large, it reduces the forward feed-forward component.

The Z component of `desired_vel` comes from both the spline tangent Z and the cross-track Z, and is then clamped separately to `±MAX_Z_VEL = 3.5 m/s` by the caller.

### 4.5 Velocity error and derivative
```python
vel_error = desired_vel - vel
d_vel = (vel_error - _prev_vel_error) * CONTROL_HZ
_vel_error_dot = 0.3 * d_vel + 0.7 * _vel_error_dot
_prev_vel_error = vel_error.copy()
damped_err = vel_error + KD_VEL * _vel_error_dot
```
- `vel_error` is 3D NED velocity error
- `d_vel` is a finite-difference derivative, scaled by `CONTROL_HZ = 100` → units m/s²
- Low-pass filter: α = 0.3 (exponential moving average with τ ≈ 3 samples = 30 ms)
- `KD_VEL = 0.04`: derivative contribution in rad/s = `0.04 × _vel_error_dot` (units: m/s²)
- `damped_err` is used for pitch/roll rate computation; raw `vel_error` is used for Z

### 4.6 Thrust computation
```python
z_vel_err = vel_error[2]
_z_integral -= z_vel_err / CONTROL_HZ * KI_ALT
_z_integral = clip(_z_integral, -0.25, 0.25)
tilt_comp = 1.0 / max(0.5, cos(roll) * cos(pitch))
thrust = clip(HOVER_THRUST * tilt_comp + _z_integral - KP_VEL_Z * z_vel_err, MIN_FLIGHT_THRUST, 1.0)
```
- `z_vel_err` = desired Z velocity − actual Z velocity (NED, positive = error is downward)
- Positive `z_vel_err` (drone ascending less than desired or descending): integral decreases → thrust decreases over time
- `KI_ALT = 0.02`: integral accumulates at `0.02 / 100 = 0.0002` per sample per m/s of Z error
- `_z_integral` saturates at ±0.25; this limits total integral correction to ±0.25 thrust units
- `tilt_comp`: multiplies `HOVER_THRUST` by `1 / (cos(roll) × cos(pitch))` to compensate for reduced vertical thrust when tilted; clamped so denominator ≥ 0.5 (prevents division by zero above 60° tilt)
- `KP_VEL_Z = 0.08`: proportional Z velocity error to thrust; sign: positive `z_vel_err` → reduce thrust
- `MIN_FLIGHT_THRUST = 0.18`: hard floor
- Thrust is on the range [0, 1] where 1 = maximum motor output

### 4.7 Body-rate pitch and roll computation
```python
fwd   =  cos(yaw) * damped_err[0] + sin(yaw) * damped_err[1]
right = -sin(yaw) * damped_err[0] + cos(yaw) * damped_err[1]
pitch_rate = clip(-KP_VEL * fwd,  -MAX_RATE, MAX_RATE)
roll_rate  = clip( KP_VEL * right, -MAX_RATE, MAX_RATE)
```
- `fwd` and `right` are NED velocity error projected into body forward/right axes using yaw angle
- Negative sign on pitch_rate: positive `fwd` error (need more forward velocity) → negative pitch_rate → nose pitches down → forward acceleration
- Positive sign on roll_rate: positive `right` error (need more rightward velocity) → positive roll_rate → rolls right → rightward acceleration
- `KP_VEL = 0.14`: 1 m/s velocity error → 0.14 rad/s rate command
- `MAX_RATE = 0.4 rad/s`: saturation point = `0.4 / 0.14 = 2.86 m/s` velocity error before saturation
- Over-speed scaling: if `norm(vel) > MAX_SPEED (10.0)`, pitch and roll rates are multiplied by `norm(vel) / MAX_SPEED` before re-clipping to MAX_RATE (has no practical effect below MAX_SPEED)

### 4.8 Yaw computation
```python
# Spline mode (tangent_unit is not None):
target_yaw = arctan2(tangent_unit[1], tangent_unit[0])
yaw_err = ((target_yaw - yaw + π) % (2π)) - π
yaw_rate = clip(-KP_YAW * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE)

# CV/waypoint fallback (only if horiz_dist > 1.0):
target_yaw = arctan2(error[1], error[0])
yaw_err = ((target_yaw - yaw + π) % (2π)) - π
yaw_rate = clip(-KP_YAW * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE)
```
- In spline mode: yaw target is the XY direction of `tangent_unit`, applied **unconditionally** (no minimum distance guard)
- `KP_YAW = 1.5`, `MAX_YAW_RATE = 1.5 rad/s`; saturation at `yaw_err = 1.0 rad` (57°)
- Negative sign: positive `yaw_err` (target is CW from current) → negative `yaw_rate`
- The simulator applies yaw_rate counterclockwise from above; negating the gain makes the drone rotate clockwise (NED convention) toward the target
- The tangent_unit Z component is ignored for yaw (uses only `[1]` and `[0]` components)

### 4.9 MAVLink send
```python
set_attitude_target_send(
    time_boot_ms,
    target_system, target_component,
    RATES_MASK,       # 0x80 = ignore attitude quaternion
    [1, 0, 0, 0],    # identity quaternion (ignored)
    roll_rate, pitch_rate, yaw_rate,
    thrust
)
```
Sent every control loop iteration at up to 100 Hz. `time_boot_ms = int(time.time()*1000) - system_boot_ms` where `system_boot_ms` is set once at program start.

---

## 5. Control Loop Structure (Signal Chain)

```
shared_data['pos'], ['vel'], ['attitude']
          │
          ▼
    gate_dist  →  speed_cap (linear ramp, 1.2–3.0 m/s)
          │
          ▼
 _path_tracking_desired_vel(pos, speed_cap)
   ├── Find nearest spline point  (_path_idx)
   ├── Tangent = path[idx+20] - path[idx]
   ├── Cross-track = (nearest - pos) ⊥ tangent
   ├── desired_vel = speed_cap × tangent + 0.8 × cross
   └── Cap: if |desired_vel| > speed_cap → rescale
          │
          ▼
   desired_vel (NED, 3D)
          │
          ├──→  Z branch:
          │         z_vel_err → integral update → KI_ALT accumulates
          │         tilt_comp = 1/(cos(roll)×cos(pitch))
          │         thrust = HOVER_THRUST×tilt_comp + integral - KP_VEL_Z×z_vel_err
          │         clipped to [0.18, 1.0]
          │
          └──→  XY branch:
                    derivative: d_vel_error (LP filtered, α=0.3)
                    damped_err = vel_error + 0.04 × d_vel_error
                    project to body frame via yaw rotation
                    fwd, right = rotation(damped_err_XY, yaw)
                    pitch_rate = clip(-0.14 × fwd, ±0.4)
                    roll_rate  = clip( 0.14 × right, ±0.4)

   yaw_err = arctan2(tangent[1], tangent[0]) - yaw (wrapped to ±π)
   yaw_rate = clip(-1.5 × yaw_err, ±1.5)
          │
          ▼
   SET_ATTITUDE_TARGET → simulator
   [roll_rate, pitch_rate, yaw_rate, thrust]
```

---

## 6. Observed Behavior from Log Data

The following facts are from logged flight data at time of writing.

### run_20260606_210814 (most recent)
- **Total FLY rows**: 1589 (≈ 15.9 seconds in FLY state)
- **Waypoint at end**: `current_idx = 1` throughout entire FLY phase — drone never advanced past gate 0
- **Speed in cruise zone** (gate_dist > 12 m): oscillates between ≈ 0.82 m/s and 5.32 m/s; first peak reaches 5.32 m/s at row ≈ 120 (1.2 s into FLY)
- **Stuck behavior**: drone oscillated near gate 0 (gate_dist between 1.5–12 m) for the entire 15.9-second run; final state: gate_dist = 0.3 m, speed ≈ 0.01 m/s, yaw_err = 1.71 rad (98°) — drone hovering stationary next to gate frame

### run_20260606_205408 (previous run)
- **Hard threshold transition** (pre smooth-ramp): at gate_dist crossing 12 m, `desired_vx` stepped from −2.987 to −1.196 m/s in one sample; `vel_error_fwd` jumped from −1.76 to −3.98 m/s; pitch_rate hit MAX_RATE immediately and stayed there for ~80 rows; drone velocity went from −4.8 m/s to +0.292 m/s (reversed direction) within 80 rows
- **Yaw drift** (pre always-on yaw fix): between rows 875–975, `target_yaw = nan` (no yaw command, horiz_dist ≤ 1 m); actual yaw drifted from 165° to 113° (~52° drift in ≈ 1 s); yaw error reached 0.907 rad (52°) before correction
- **Lateral instability**: after gate 0 transit at 9.74–10.81 m/s, cross-track error grew to 4+ m over 200 rows; `desired_vy` grew to −3.79 m/s (cross-track correction); roll_rate saturated at MAX_RATE continuously from row 1040 onward; pos_z went from −0.46 m to +0.02 m (ground impact) at row 1122
- **Cruise oscillation**: acceleration from FLY start: 0.61 m/s → 5.32 m/s in 120 rows (1.2 s); deceleration: 5.32 → 1.02 m/s in 120 rows; second overshoot: 1.02 → 3.77 m/s in 100 rows

---

## 7. Physical and Timing Constants

| Parameter | Value | Source |
|---|---|---|
| Drone body size | 280 × 280 × 160 mm | Spec |
| Gate outer | 2700 × 2700 × 260 mm | Spec |
| Gate inner (opening) | 1500 × 1500 × 260 mm | Spec |
| Lateral clearance per side | ~600 mm | Derived: (1500 − 280) / 2 |
| Camera resolution | 640 × 360 px | Spec |
| Camera focal length | fx = fy = 320 px | Spec |
| Camera principal point | cx = 320, cy = 180 | Spec |
| Camera tilt | 20° (sign empirically determined) | Spec + code comment |
| Simulator physics rate | 120 Hz | Spec |
| Vision stream | 30 Hz, 640×360 JPEG | Spec |
| Control loop target | 100 Hz | `CONTROL_HZ = 100` |
| `time.sleep()` per loop | 10 ms | controller.py |
| MAVLink port | UDP :14550 | main.py |
| Vision port | UDP :5600 | setup.py |
| HOVER_THRUST | 0.28 | Tuning constant |
| TAKEOFF_ALT_NED | −0.5 m | planner.py |
| LEAD_IN_DIST | 12.0 m | planner.py |
| GATE_Z_BIAS | −0.5 m | planner.py |
| Spline samples/segment | 80 | planner.py |

---

## 8. All Current Tuning Constants

| Constant | Value | Meaning |
|---|---|---|
| `HOVER_THRUST` | 0.28 | Baseline thrust for level hover |
| `KP_THRUST` | 0.15 | Thrust per metre altitude error during TAKEOFF climb |
| `KI_ALT` | 0.02 | Altitude integral accumulation rate |
| `CRUISE_SPEED` | 3.0 m/s | Feed-forward speed in open stretches |
| `GATE_APPROACH_SPEED` | 1.2 m/s | Feed-forward speed at the gate (ramp floor) |
| `GATE_APPROACH_DIST` | 12.0 m | Distance at which ramp reaches CRUISE_SPEED |
| `TANGENT_SAMPLES` | 20 | Path samples ahead for tangent direction |
| `MAX_SPEED` | 10.0 m/s | Hard cap; above this, rates are scaled up |
| `KP_POS` | 0.25 | Position error → speed (fallback/CV mode only) |
| `KP_CROSS` | 0.8 | Cross-track error (m) → lateral velocity (m/s) |
| `KP_POS_Z` | 0.50 | Altitude error → climb rate (CV/fallback only) |
| `MAX_Z_VEL` | 3.5 m/s | Clamp on desired Z velocity |
| `KP_VEL` | 0.14 | Velocity error (m/s) → body rate (rad/s) |
| `KD_VEL` | 0.04 | Filtered velocity error derivative gain |
| `KP_VEL_Z` | 0.08 | Z velocity error → thrust delta |
| `KP_LEVEL` | 2.0 | Attitude angle → levelling rate during TAKEOFF |
| `MIN_FLIGHT_THRUST` | 0.18 | Thrust floor in FLY state |
| `MAX_RATE` | 0.4 rad/s | Pitch and roll rate saturation limit |
| `WAYPOINT_RADIUS` | 1.5 m | Lead-in waypoint arrival threshold |
| `GATE_WAIT_TIMEOUT` | 5.0 s | Max wait for gate map in INIT |
| `CONTROL_HZ` | 100 | Control loop frequency |
| `CV_STALE_TIMEOUT` | 0.3 s | Max age for CV gate estimate to be used |
| `KP_YAW` | 1.5 | Yaw error (rad) → yaw rate (rad/s) |
| `MAX_YAW_RATE` | 1.5 rad/s | Yaw rate saturation limit |

---

## 9. What the Controller Does NOT Do

- **No inner attitude loop**: the controller sends body rates directly; there is no loop that reads back the achieved attitude angles and corrects to a desired pitch/roll angle. Attitude angles accumulate from commanded rates with no feedback on achieved angle.
- **No acceleration feedforward**: the desired velocity is computed from path geometry only; there is no term accounting for centripetal acceleration needed to follow path curves, nor for gravity when the spline climbs or descends.
- **No knowledge of actual pitch/roll magnitude for pitch/roll control**: roll and pitch angles are read for `tilt_comp` in thrust only; they are not used in the body-rate pitch/roll computation. The rate command is purely from velocity error.
- **No explicit speed controller**: the drone has no separate speed controller. Speed is regulated indirectly through velocity error → pitch rate. There is no mechanism that directly limits the rate of change of the velocity setpoint beyond the linear ramp on `speed_cap`.
- **No gate orientation awareness**: the gate opening orientation (quaternion `quat` field available in gate data) is ignored; the drone always targets the center coordinates only.
- **No ODOMETRY or HIGHRES_IMU data used**: these MAVLink messages are received but stubbed (`pass`).
- **No TimeSync consumption**: TIMESYNC is sent at 10 Hz but the timestamps are never used for latency compensation.
- **No cross-track integral**: the cross-track feedback is pure proportional (`KP_CROSS` only); accumulated lateral error does not drive any integral term.
- **No collision avoidance**: COLLISION messages are received and logged but never acted upon by the controller.
- **Yaw in CV mode**: when `GUIDANCE = 'CV_LIVE'`, the yaw target is the direction from drone to the detected gate center (using `error` vector), not the path tangent. `tangent_unit` is None in CV mode.
- **`_lookahead_target` is never called**: it is defined on the Controller class but has no callers in the current codebase.
- **`GUIDANCE` is always `'MAVLINK'`**: the `'CV_LIVE'` and `'CV_PLAN'` branches exist in code but are not activated; CV estimates are computed by `GateVerifier` and written to `shared_data` but are never consumed by the controller in the default configuration.
