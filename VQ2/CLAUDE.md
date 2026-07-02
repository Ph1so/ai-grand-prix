# AI Grand Prix — VQ2 Project Brief for Claude

## What VQ2 Is

Round 2 of the virtual qualifier. The core objective is to qualify for the **Physical Qualifier Event in September 2026**. VQ2 introduces critical constraints that break any VQ1 controller directly: position, attitude, and gate data are all **blocked** at the simulator level. You must fly using only IMU data, vision, and inference.

Simulator binary: `VQ2/AIGP_3379/FlightSim.exe` — launch this first, then log in with your simulator account credentials, and select **Qualification (VQ2)** for a scored run or **Training** for a free practice run.

---

## Critical Changes from VQ1

| What | VQ1 | VQ2 |
|------|-----|-----|
| ATTITUDE telemetry | Available | **BLOCKED** |
| LOCAL_POSITION_NED | Available | **BLOCKED** |
| ODOMETRY | Available | **BLOCKED** |
| GATE_INFO / track data positions | Available (real values) | **NULLED** (struct sent but position/orientation/dimensions are zeroed) |
| Control loop Hz in example | 100 Hz | Example set to 250 Hz (exceeds spec — see bugs) |
| Motor RPM control | Not in example | Added as `update_motor_control()` |
| Sim reset command | Not in example | `MAVLINK_CMD_SIM_RESET = 31000` |

**Consequence:** You cannot rely on any direct position or attitude state from MAVLink. Your entire state estimation must come from HIGHRES_IMU integration and/or camera-based pose estimation.

---

## File Map

| File | Role |
|------|------|
| [main.py](PyAIPilotExample-v2/main.py) | Entry point — wires components, runs control loop |
| [setup.py](PyAIPilotExample-v2/setup.py) | Factory: creates MAVLink connection, threads |
| [mavlink_rx.py](PyAIPilotExample-v2/mavlink_rx.py) | Thread: receives/parses all MAVLink messages |
| [timesync.py](PyAIPilotExample-v2/timesync.py) | TIMESYNC sender @ 10 Hz (has startup bug — see below) |
| [controller.py](PyAIPilotExample-v2/controller.py) | Bare example controller (motor + attitude + velocity stubs) |
| [vision_rx.py](PyAIPilotExample-v2/vision_rx.py) | Thread: reassembles JPEG frames from chunked UDP |
| [requirements.txt](PyAIPilotExample-v2/requirements.txt) | pymavlink, opencv-python, numpy, matplotlib, keyboard |

---

## Architecture Overview

```
Simulator (FlightSim.exe)
    │
    ├── MAVLink UDP :14550 ──► mavlink_rx.py ──► shared_data dict
    │                                                    │
    └── Vision UDP  :5600 ──► vision_rx.py              │
                              └── process_frame()       │
                                  (stub — implement CV) ┘
                                                        │
                                                 controller.py ──► SET_ATTITUDE_TARGET / SET_POSITION_TARGET / SET_ACTUATOR_CONTROL_TARGET ──► Simulator
```

Everything communicates through a single `shared_data` dict. Threads write to it; controller reads at CONTROL_HZ.

---

## Bugs in the Example Code

### 1. TimeSync thread never starts
`setup.py` calls `TimeSync(sim_conn, shared_data)` (the plain constructor), but the thread only starts via the classmethod `TimeSync.create_timesync()`. The timesync thread is **dead** in the example as shipped.

Fix: Change `setup.py` line:
```python
ts_loop = TimeSync(sim_conn, shared_data)
# → should be:
ts_loop = TimeSync.create_timesync(sim_conn, shared_data)
```

### 2. Control loop exceeds spec rate
`controller.py` sets `CONTROL_HZ = 250`. The spec mandates command rate `<100 Hz`. Set this to ≤99 Hz.

### 3. Motor constants are all zero (except MOTOR_FRONT_LEFT)
`MOTOR_FRONT_RIGHT = 1` but `MOTOR_BACK_LEFT = 0` and `MOTOR_BACK_RIGHT = 0` — looks like placeholder constants only.

---

## Simulator Interface (VADR-TS-003 §4, verified against code)

### MAVLink (UDP :14550)

| Message | Direction | VQ2 Status | Notes |
|---------|-----------|------------|-------|
| HEARTBEAT | Sim → Client | Active | armed = `base_mode & MAV_MODE_FLAG_SAFETY_ARMED` |
| TIMESYNC | Bidirectional | Active | Client sends tc1=now_ns, ts1=0 |
| HIGHRES_IMU | Sim → Client | Active | xacc/yacc/zacc, xgyro/ygyro/zgyro, time_usec |
| ATTITUDE | Sim → Client | **BLOCKED** | Handler stub present; will not receive data |
| LOCAL_POSITION_NED | Sim → Client | **BLOCKED** | Handler stub present; will not receive data |
| ODOMETRY | Sim → Client | **BLOCKED** | Handler stub present; will not receive data |
| ENCAPSULATED_DATA | Sim → Client | Active | Race status (type=1) and track chunks (type=2) |
| DATA_TRANSMISSION_HANDSHAKE | Sim → Client | Active | Announces incoming track data batch; `width` = transfer_id, `packets` = total chunk count |
| COLLISION | Sim → Client | Active | id 1001=gate, 1002=environment |
| ACTUATOR_OUTPUT_STATUS | Sim → Client | Active | Motor RPMs (actuator[0..3]) |
| SET_ATTITUDE_TARGET | Client → Sim | Active | rates + thrust, use `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` |
| SET_POSITION_TARGET_LOCAL_NED | Client → Sim | Active | Velocity or position setpoint in NED |
| SET_ACTUATOR_CONTROL_TARGET | Client → Sim | Active | Direct motor RPM control (new in VQ2 example) |
| MAV_CMD_COMPONENT_ARM_DISARM | Client → Sim | Active | param1=1 to arm |
| MAVLINK_CMD_SIM_RESET (31000) | Client → Sim | Active | Custom reset command, not in spec |

**Spec note:** The spec table (§4.3) is incomplete — it lists `HIGHRES_IMU` twice and omits `ACTUATOR_OUTPUT_STATUS`, `COLLISION`, `ENCAPSULATED_DATA`, `DATA_TRANSMISSION_HANDSHAKE`, and motor control. Trust the code over the spec table.

### ENCAPSULATED_DATA Payloads

**Race status** (`data_type = 1`), struct `<BQqqIq`:
| Field | Type | Meaning |
|-------|------|---------|
| data_type | B (uint8) | Always 1 |
| sim_boot_time_ms | Q (uint64) | Server elapsed ms since boot |
| race_start_boot_time_ms | q (int64) | Server ms when race started; <0 if not started |
| race_finish_time_ns | q (int64) | Server ns when race finished; <0 if ongoing |
| active_gate_index | I (uint32) | Current gate being targeted |
| last_gate_race_time | q (int64) | Race time (s) when last gate was passed |

**Track data** (`data_type = 2`), multi-packet, announced by `DATA_TRANSMISSION_HANDSHAKE`:
- Packet header: `<BH` → data_type(B), transfer_id(H); first 3 bytes stripped before storing
- Chunks indexed by `msg.seqnr`; assembled once all expected chunks arrive
- Full payload header: `<H` → num_gates
- Per gate: `<Hfffffffff` = gate_id(H), pos_ned_x/y/z(fff), orient_ned_w/x/y/z(ffff), width(f), height(f) → **38 bytes/gate**
- **In VQ2: gate positions, orientations, and dimensions are nulled (zeroed)** — struct is received but values cannot be trusted

### Collision message
- `msg.id`: 1001 = gate hit, 1002 = environment hit
- `msg.threat_level`: 1 or 2 (2 = harder impact)
- `msg.horizontal_minimum_delta`: **actually impulse magnitude in kg·m/s** (field name is misleading)

### Vision Stream (UDP :5600)

Header format: `<IHHIIQ` — **24 bytes total** (spec §4.6 says 24 B ✓; VQ1 CLAUDE.md incorrectly said 28 B)

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| frame_id | uint32 | 4 | Unique sequence ID |
| chunk_id | uint16 | 2 | Packet index within frame |
| total_chunks | uint16 | 2 | Total packets for this frame |
| jpeg_size | uint32 | 4 | Full reconstructed JPEG size |
| payload_size | uint32 | 4 | JPEG bytes in this packet |
| sim_time_ns | uint64 | 8 | Simulator timestamp (ns) |

- 30 Hz, 640×360 JPEG
- Reassemble all chunks by `frame_id` before decoding with `cv2.imdecode`
- Discard frame if any chunk arrives out of order after assembly is triggered

### Control Interfaces

**Attitude + rates** (most useful):
```python
mavlink_conn.mav.set_attitude_target_send(
    time_boot_ms, target_system, target_component,
    ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,  # ignore quaternion
    [1, 0, 0, 0],  # dummy quat
    roll_rate, pitch_rate, yaw_rate,  # rad/s
    thrust  # 0.0–1.0
)
```

**Velocity setpoint** (NED frame):
```python
mavlink_conn.mav.set_position_target_local_ned_send(
    time_boot_ms, target_system, target_component,
    MAV_FRAME_LOCAL_NED,
    VELOCITY_POSITION_MASK,  # ignore pos + accel + yaw
    0, 0, 0,   # ignored position
    vx, vy, vz,  # velocity m/s
    0, 0, 0,   # ignored accel
    0, 0       # ignored yaw
)
```
Velocity mask: ignore X, Y, Z position + AX, AY, AZ accel + YAW + YAW_RATE; pass only velocities.

**Motor RPM direct**:
```python
mavlink_conn.mav.set_actuator_control_target_send(
    int(time.time() * 1e6),  # NOTE: microseconds, not ms
    target_system, target_component,
    0,  # group
    [FL_rpm, FR_rpm, BL_rpm, BR_rpm, 0, 0, 0, 0]
)
```
Motor index: 0=front-left, 1=front-right, 2=back-left, 3=back-right.

---

## Physical Constants (from spec, not independently verifiable from code)

| Item | Value | Spec Verified? |
|------|-------|----------------|
| Drone | 280×280×160 mm | Spec only |
| Gate outer | 2700×2700×260 mm | Spec only |
| Gate inner opening | 1500×1500×260 mm | Spec only |
| Camera resolution | 640×360 px | ✓ vision_rx.py confirms 640×360 decode |
| Camera principal point | cx=320, cy=180 | Spec only |
| Camera focal length | fx=fy=320 | Spec only |
| Camera tilt | 20° upward | Spec only; VQ1 found sign must be flipped empirically |
| Physics Hz | 120 Hz | Spec only |
| Vision Hz | 30 Hz | Spec only |
| Command rate | <100 Hz | Spec; example code violates this at 250 Hz |
| MAVLink port | 14550 | ✓ main.py |
| Vision port | 5600 | ✓ vision_rx.py |

---

## Coordinate Frames (NED convention)

- **MAV_FRAME_LOCAL_NED**: Origin = arm point on ground. X=North, Y=East, Z=Down (altitude = negative Z).
- **MAV_FRAME_BODY_NED**: Origin = vehicle. X=forward, Y=right, Z=down.
- **Camera frame**: Same origin as body, tilted 20° upward. Camera X = body Y. Need explicit rotation to OpenCV camera convention.
- **Body-to-IMU**: Identity (no offset or rotation between body and IMU frame).

---

## What You Need to Build

The VQ2 example is a skeleton. To race, you need to implement:

1. **State estimation** from HIGHRES_IMU (integrate accel/gyro → velocity/attitude) — the only available telemetry.
2. **Gate detection** from camera frames in `vision_rx.process_frame()` — HSV segmentation or ML detector.
3. **Gate pose estimation** — solvePnP or homography to get 3D gate position from 2D detection.
4. **Controller** — replace the stub in `controller.py` with a real state machine (takeoff → fly → finish).
5. **Fix TimeSync** — start the timesync thread correctly (see bug above).
6. **Respect 100 Hz command rate** — change `CONTROL_HZ` to ≤ 99.

The VQ1 codebase (`VQ1/PyAIPilotExample/`) has working versions of gate_detector.py, pose_estimator.py, gate_verifier.py, planner.py, and a full controller state machine. These are the correct starting point — adapt them for the lack of position/attitude telemetry.

---

## Simulator Setup

1. Extract `AIGP_3379/` (already done in VQ2 folder)
2. Launch `VQ2/AIGP_3379/FlightSim.exe`
3. Log in with simulator account credentials
4. Select **Training** for development or **Qualification (VQ2)** for a scored run
5. In a separate terminal, run: `cd VQ2/PyAIPilotExample-v2 && python main.py`

System requirements: Windows 10/11 64-bit, i7 4770k+, 8 GB RAM, GTX 970+ (RTX 3070 tested), 12 GB storage.

---

## Competition Rules (Round 2)

- Objective: complete the course as fast as possible to qualify for Physical Qualifier (September 2026)
- Faster times rank higher; unlimited attempts
- Best team time counts (not individual)
- No human interaction during a timed run — instant disqualification
- DCL may audit your codebase if cheating is suspected
- Finals: November 2026 in Ohio, $500K prize pool

---

## Spec vs Code Discrepancies (VADR-TS-003)

| Spec Claim | Reality |
|------------|---------|
| §4.3 message table lists HIGHRES_IMU twice | Confirmed duplicate in spec; only one handler in code |
| §4.3 table lists only SET_POSITION_TARGET_LOCAL_NED and SET_ATTITUDE_TARGET as control | Code also supports SET_ACTUATOR_CONTROL_TARGET (motor RPM) |
| §4.3 table omits ENCAPSULATED_DATA, COLLISION, ACTUATOR_OUTPUT_STATUS, DATA_TRANSMISSION_HANDSHAKE | All four are handled in mavlink_rx.py |
| §4.6 header size: 24 bytes | ✓ Correct — struct `<IHHIIQ` = 24 bytes |
| §4.4 command rate <100 Hz | Example code uses 250 Hz — must fix |
| §9.3 blocked: ATTITUDE, LOCAL_POSITION_NED, ODOMETRY, GATE_INFO | ✓ Confirmed: all four have "disabled" comments in mavlink_rx.py; gate data is nulled per on_track_data comment |
| §3.8 camera tilted 20° upward | Empirically found in VQ1 that sign must be negated — recheck on VQ2 |
