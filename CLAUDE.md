# AI Grand Prix — Project Brief for Claude

## What This Is

Autonomous drone racing competition software. The goal is to program an AI pilot that navigates a FPV racing drone through a series of sequential gates as fast as possible — no GPS, no absolute position, vision + MAVLink telemetry only. Maximum run duration: 8 minutes. Round 1 (Qualification) just requires completing the course.

Simulator runs on Windows 11 (FlightSim.exe / Unreal Engine). Python client runs alongside it locally. No Linux support.

---

## Architecture Overview

```
Simulator (FlightSim.exe)
    │
    ├── MAVLink UDP :14550 ──► mavlink_rx.py ──► shared_data dict
    │                                                    │
    └── Vision UDP  :5600 ──► vision_rx.py              │
                              └── GateVerifier           │
                                  ├── gate_detector.py   │
                                  └── pose_estimator.py ─┘
                                                         │
                                                    controller.py ──► SET_ATTITUDE_TARGET ──► Simulator
```

**Key fact:** Everything communicates through a single `shared_data` dict. Threads write to it, the controller reads from it at 100 Hz.

---

## File Map

| File | Role |
|------|------|
| [main.py](PyAIPilotExample/main.py) | Entry point — wires everything together, 100 Hz control loop |
| [setup.py](PyAIPilotExample/setup.py) | Factory: creates all threads/connections (not a setuptools file) |
| [mavlink_rx.py](PyAIPilotExample/mavlink_rx.py) | Thread: receives/parses MAVLink telemetry, logs CSV |
| [timesync.py](PyAIPilotExample/timesync.py) | Thread: sends TIMESYNC @ 10 Hz (currently not consumed by controller) |
| [controller.py](PyAIPilotExample/controller.py) | Main state machine: INIT→IDLE→TAKEOFF→FLY→FINISHED |
| [vision_rx.py](PyAIPilotExample/vision_rx.py) | Thread: reassembles JPEG frames from UDP chunks |
| [gate_verifier.py](PyAIPilotExample/gate_verifier.py) | Subclass of VisionRX: runs CV pipeline per frame, logs estimates |
| [gate_detector.py](PyAIPilotExample/gate_detector.py) | HSV orange segmentation → 4 corners of gate |
| [pose_estimator.py](PyAIPilotExample/pose_estimator.py) | solvePnP → camera frame → NED world position |
| [planner.py](PyAIPilotExample/planner.py) | Builds lead-in/gate waypoints + cubic spline path from gate map |
| [visualize.py](PyAIPilotExample/visualize.py) | Post-flight plots: 3-D trajectory, CV accuracy |

---

## Simulator Interface (VADR-TS-002)

### MAVLink (UDP :14550)

| Message | Direction | Content |
|---------|-----------|---------|
| HEARTBEAT | Sim → Client | Armed status (≥2 Hz minimum) |
| ATTITUDE | Sim → Client | roll, pitch, yaw (rad) |
| LOCAL_POSITION_NED | Sim → Client | pos [x,y,z] m, vel [vx,vy,vz] m/s |
| TIMESYNC | Bidirectional | Clock sync |
| ENCAPSULATED_DATA | Sim → Client | Custom: race status (type=1) or track info (type=2) |
| COLLISION | Sim → Client | collision id, threat_level, impulse |
| SET_ATTITUDE_TARGET | Client → Sim | roll_rate, pitch_rate, yaw_rate (rad/s), thrust [0–1] |

**No GPS. No absolute global position. Coordinate system: NED (North-East-Down), origin = arm point.**

Race status and gate map come via `ENCAPSULATED_DATA` (custom MAVLink payload). Track info is multi-packet and must be reassembled — see `mavlink_rx.py:on_encapsulated_data()`.

### Vision Stream (UDP :5600)

30 Hz, 640×360 JPEG, multi-packet chunked UDP.

```
Header (28 bytes): frame_id(u32), chunk_id(u16), total_chunks(u16), jpeg_size(u32), payload_size(u32), sim_time_ns(u64)
Payload: JPEG bytes
```

Reassemble all chunks by frame_id before decoding.

---

## Physical Constants (from spec)

| Item | Value |
|------|-------|
| Drone size | 280×280×160 mm |
| Gate outer | 2700×2700×260 mm |
| Gate inner (opening) | 1500×1500×260 mm |
| Camera resolution | 640×360 px |
| Camera principal point | cx=320, cy=180 |
| Camera focal length | fx=fy=320 |
| Camera tilt | 20° (spec says upward; code comment notes effective behavior is downward — verify empirically) |
| Physics update rate | 120 Hz |
| Vision stream rate | 30 Hz |
| Max command rate | <100 Hz |

### Coordinate Frames

- **MAV_FRAME_LOCAL_NED**: Origin = arm point on ground. X=North, Y=East, Z=Down.
- **MAV_FRAME_BODY_NED**: Origin = vehicle. X=forward, Y=right, Z=down.
- **Camera**: Same origin as body, tilted 20°. Camera x-axis = body Y. Transform chain: `camera → body → NED`.
- **NED quirk**: Altitude is negative (up = decreasing Z). Gates in the track data give z at the bottom edge; add height/2 to reach center.

---

## Controller State Machine

```
INIT   → wait for first position telemetry; build fallback waypoints
IDLE   → hold minimum thrust; wait for race_started signal (0.5s hysteresis)
TAKEOFF → climb to TAKEOFF_ALT_NED = -0.5 m; level roll/pitch
FLY    → main racing loop (see below)
FINISHED → hover until program exit
```

**FLY loop target priority:**
1. Fresh CV estimate from `shared_data['cv_gate_pos']` (age < 0.3 s, must be in front)
2. Fallback to current waypoint from planner
3. Creep forward (constant pitch -0.2 rad/s) if nothing available

**Waypoint pattern:** alternating lead-in points (12 m before gate) and gate centers. Advancement triggers: sim confirms gate passed (active_gate increments), drone crosses gate plane (x < gate_x - 1.0), or arrives within 1.5 m of lead-in.

**Control output:** `SET_ATTITUDE_TARGET` with `type_mask = ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` (body rates + thrust, quaternion ignored). Sent at 100 Hz.

**Yaw control: currently always 0.0.** This is a known gap.

---

## CV Pipeline

```
BGR image → HSV → orange mask → contours → largest blob → approxPolyDP → 4 corners [TL, TR, BR, BL]
         → solvePnP (camera frame) → rotate to body (R_CAM2BODY) → rotate to NED (Euler ZYX)
         → add drone_pos → world gate center (NED)
```

**Gate detector HSV thresholds:**
- Orange: H 0–25, S 120–255, V 60–255
- Red wrap-around: H 160–179, S 120–255, V 60–255
- Morphological close: 7×7 kernel
- Min contour area: 200 px²

**solvePnP object points** (gate outer boundary corners, half-dimension = 1.36 m):
```
TL: [-1.36, -1.36, 0]  TR: [1.36, -1.36, 0]
BR: [ 1.36,  1.36, 0]  BL: [-1.36, 1.36, 0]
```

**Camera-to-body rotation** (R_CAM2BODY): maps camera axes into body FRD frame. Spec says 20° upward tilt; code comment notes the sign had to be flipped vs. spec — verify with fresh data if CV estimates are systematically off vertically.

---

## shared_data Keys

| Key | Type | Written by | Read by |
|-----|------|-----------|---------|
| `pos` | np.array (3,) NED m | mavlink_rx | controller |
| `vel` | np.array (3,) NED m/s | mavlink_rx | controller |
| `attitude` | dict: roll, pitch, yaw (rad) | mavlink_rx | controller, pose_estimator |
| `time_boot_ms` | int | mavlink_rx | controller |
| `race_status` | dict: active_gate, race_started, race_finished | mavlink_rx | controller |
| `gates` | list of dicts | mavlink_rx | planner, controller |
| `cv_gate_pos` | np.array (3,) NED m | gate_verifier | controller |
| `cv_gate_time` | float (time.time()) | gate_verifier | controller |
| `collision` | dict: id, threat_level, impulse | mavlink_rx | (unused) |

---

## Logging & Diagnostics

All runs log to `PyAIPilotExample/logs/run_YYYYMMDD_HHMMSS/`. Four files are produced per run.

### Log file reference

**`flight_log_*.csv`** — one row per LOCAL_POSITION_NED message (~30 Hz)

| Column | Meaning |
|--------|---------|
| `wall_time_s` | Real wall-clock time (Unix epoch) |
| `time_boot_ms` | Simulator boot time in ms |
| `pos_x/y/z` | Drone position NED (m). Z is negative altitude — pos_z = -1.5 means 1.5 m AGL |
| `vel_x/y/z` | Drone velocity NED (m/s) |
| `speed` | ‖vel‖ (m/s) |
| `roll/pitch/yaw` | Body attitude (rad) |
| `active_gate` | Sim-confirmed gate index currently being targeted (-1 = race not started) |

**`flight_log_*_gates.json`** — written once when the simulator sends the track map

Each entry: `id`, `pos` (NED, z already negated to NED convention), `width`, `height`, `quat`.
Gate `pos` is the **bottom edge** center — add `height/2` to get the opening center (planner does this automatically).

**`gate_estimates_*.csv`** — one row per successful CV gate detection (30 Hz max, only when a gate is visible)

| Column | Meaning |
|--------|---------|
| `sim_time_ns` | Simulator timestamp |
| `est_x/y/z` | CV-estimated gate center (NED, m) |
| `drone_x/y/z` | Drone position at time of estimate |
| `matched_gate_id` | Nearest MAVLink gate (-1 if no gate map yet) |
| `mav_x/y/z` | MAVLink ground-truth position of matched gate |
| `error_x/y/z` | est − mav per axis |
| `error_mag` | ‖est − mav‖ |

**`controller_log.csv`** — 100 Hz controller internals (TAKEOFF and FLY states only)

| Column | Meaning |
|--------|---------|
| `time_s` | Wall-clock time |
| `state` | TAKEOFF or FLY |
| `target_mode` | Which branch selected the target: `spline`, `gate`, `CV(age=Xms)`, `wp[N]`, `creep` |
| `waypoint_idx` | `current_idx` — position in the flat waypoint list |
| `path_idx` | Progress index on smooth spline |
| `target_x/y/z` | What the drone is flying toward (NED, m) |
| `pos_x/y/z` | Drone position at time of command (NED, m) |
| `error_x/y/z` | target − pos |
| `horiz_dist` | Horizontal distance to target |
| `gate_dist` | Distance to next gate waypoint |
| `speed_cap` | Active speed cap (CRUISE_SPEED or GATE_APPROACH_SPEED) |
| `desired_vx/vy/vz` | Velocity setpoint computed by position PID (m/s) |
| `actual_vx/vy/vz` | Drone velocity from telemetry at command time |
| `vel_error_fwd/right/z` | Velocity error projected into body frame |
| `cmd_roll_rate/pitch_rate/yaw_rate` | Commands sent (rad/s) |
| `cmd_thrust` | Thrust sent [0–1] |
| `z_integral` | Accumulated altitude integral (clips at ±0.25) |
| `yaw` | Actual drone yaw (rad) |
| `target_yaw` | Desired yaw toward target (rad, nan if horiz_dist ≤ 1 m) |
| `yaw_err` | Yaw error (rad, nan if horiz_dist ≤ 1 m) |
| `saturated` | 1 if pitch_rate or roll_rate hit MAX_RATE, else 0 |

### Visualization tools

```bash
# Run from PyAIPilotExample/
python visualize.py         # trajectory + CV overlay, auto-picks latest run
python gate_verifier.py     # CV accuracy analysis, auto-picks latest run
python planner.py           # print waypoint list + total path length
```

`visualize.py` produces two saved PNGs in the same run folder:
- `*_trajectory.png` — 6 panels: 3D path, top-down, side view, speed over time, altitude over time. Gold vertical lines = gate confirmation moments. Orange dots = CV estimates. Blue dashed = planned path.
- `*_cv_comparison.png` — CV scatter clouds vs MAVLink truth (★) per gate, plus per-gate mean error bar chart (X/Y/Z separately). Only generated if `gate_estimates_*.csv` exists.

`gate_verifier.py` prints: mean, median, max, std of ‖error‖ across all frames.

### Diagnosing common problems

**Drone stops mid-course / misses a gate**
1. Open `flight_log_*.csv`. Find where `active_gate` stops incrementing — that's the gate it failed on.
2. Run `visualize.py`. Look at the 3D/top-down panel: does the trajectory diverge from the blue planned path before that gate? If yes, the controller lost its target.
3. Check `gate_estimates_*.csv` rows near that time. Sparse rows (few detections) = CV wasn't seeing the gate. Many rows but high `error_mag` = CV was detecting something wrong (false positive or bad pose).

**Drone never moves / race never starts**
- Terminal output should show `[race] gate=0 started=True ...`. If absent, the sim didn't send race start — check MAVLink connection.
- `active_gate` in the CSV stays at -1 throughout.

**CV estimates are consistently wrong (systematic bias)**
- Run `gate_verifier.py` and look at the error-components-vs-time panel.
- Persistent positive/negative `error_z` across all gates → camera tilt sign issue (see Known Gaps).
- Persistent `error_x` or `error_y` offset → solvePnP object point mismatch or wrong gate half-dimension constant.
- High `error_mag` that grows with drone-to-gate distance → perspective / focal length mismatch.

**CV detects gates but controller ignores them**
- `gate_estimates_*.csv` has rows, but the trajectory follows the blue planned path exclusively.
- Check the `cv_gate_time` staleness logic in `controller.py` (CV_STALE_TIMEOUT). If CV estimates arrive but are already old by the time the controller reads them, they'll be discarded.
- Also check the "must be in front" filter: if `cv_gate_pos` is behind the drone's forward vector, it's rejected.

**Drone crashes into a gate or obstacle**
- Terminal prints: `[collision] id=1001 impulse=...` (gate) or `id=1002` (environment).
- Cross-reference the wall_time of the collision with `flight_log_*.csv` to find the drone's position and attitude at impact.
- In `visualize.py`, look at the side-view panel for altitude at that moment — hitting the gate frame rather than flying through the opening usually means an altitude or lateral error.

**Altitude problems (drone too high/low through gates)**
- `visualize.py` altitude-over-time panel: compare drone altitude at each gold gate-confirmation line against the gate height from `gates.json`.
- Check `error_z` in `gate_estimates_*.csv`: if CV is consistently estimating the gate too high or low, the controller will fly to the wrong altitude.

**Speed too slow or drone decelerates too early**
- Speed-over-time panel in `visualize.py`. Look at where speed drops before gate-confirmation events (gold lines).
- If the drone slows well before reaching the gate, the approach distance threshold is triggering too early — adjust `GATE_APPROACH_DIST` in `controller.py`.

**Gate map looks wrong**
- `python planner.py` prints each waypoint and gate center. Cross-check gate positions against expected course geometry.
- Look at `visualize.py` top-down panel: gate rectangles are drawn at their MAVLink positions. If they're in nonsensical locations, the `_gates.json` parsing has an issue (likely the z-sign convention).

---

## Tuning Parameters

Controller gains and thresholds live as named constants at the top of [controller.py](PyAIPilotExample/controller.py) and [planner.py](PyAIPilotExample/planner.py). Read those files for current values — don't trust any cached copy here. Key categories:

- **Thrust**: hover equilibrium, altitude P/I gains
- **Speed**: cruise speed, gate approach speed, approach distance, max speed
- **Position/velocity PID**: horizontal and vertical gains separately
- **Rate limits**: max pitch/roll rate, max climb/descend rate
- **Thresholds**: waypoint arrival radius, CV staleness timeout, takeoff altitude, lead-in distance

---

## Known Gaps & Open Problems

1. **Yaw control**: Always sends 0.0 yaw_rate. Drone doesn't actively point at gates. Could hurt performance on sharp turns.
2. **Camera tilt sign**: Spec says 20° upward; code had to negate. Verify on any new sim version — systematic vertical CV error is a symptom.
3. **No outlier rejection in CV**: Any orange blob passing the area/ratio filter triggers a gate estimate. False positives (e.g., environment decorations) will corrupt the controller target.
4. **Single-gate tracking**: No historical state. Each frame is independent. No gate ID association from CV side.
5. **TimeSync not consumed**: Latency compensation is absent. Fine for localhost but relevant if running on separate machines.
6. **ODOMETRY, HIGHRES_IMU, ACTUATOR_OUTPUT_STATUS**: All stubbed (`pass`). IMU data could improve state estimation.
7. **Altitude sign in planner**: `center[2] = -center[2]` marked as experimental — verify correctness for each track.
8. **Bounding-rect fallback removed**: gate_detector won't fall back if approxPolyDP can't reduce to exactly 4 points. May fail on visually complex gates.

---

## Dependencies

```
pymavlink       MAVLink protocol
opencv-python   HSV segmentation, solvePnP, JPEG decode
numpy           All numerical ops
scipy           Cubic spline (planner)
matplotlib      Post-flight plots
```

Run from `PyAIPilotExample/` directory. Python 3.14.2 confirmed working.

---

## Competition Rules (Round 1)

- Objective: complete the course (start gate → sequential gates → finish gate)
- Max run duration: 8 minutes
- No human interaction during the timed run — instant disqualification
- Course geometry and physics are identical for all participants (deterministic)
- No GPS, no absolute world position from simulator
