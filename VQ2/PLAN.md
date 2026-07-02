# VQ2 Racing Strategy — Implementation Plan

**Goal:** Complete the VQ2 course, qualify for Physical Qualifier (September 2026).
**Scoring:** Time only. Faster = better. Unlimited training runs.

---

## How to use this file
- Check off `[x]` as steps are completed.
- Notes on outcomes go under each step inline.
- Do not skip phases — each phase unblocks the next.

---

## Phase 0 — Day 1: Camera Tilt Fix (30 min)

> **Why first:** A wrong camera tilt sign = 6.4m vertical error at 10m range. Every CV estimate is garbage until this is resolved. Nothing else is worth building before this is verified.

- [x] **0.1** Open `PyAIPilotExample/logs/run_20260615_222757/gate_estimates_20260615_222757.csv`
- [x] **0.2** Filter rows where `error_mag < 5`. Compute `mean(error_z)`.
  - If `mean(error_z) < 0` → flip the z-row sign in `R_CAM2BODY` in `pose_estimator.py`
  - If `mean(error_z) > 0` → sign is correct, leave it
  - If `mean(error_z) ≈ 0` → sign is already correct
- [x] **0.3** ~~Run a VQ1 training lap.~~ Skipped — see outcome notes.

**Outcome notes:** Checked all three available VQ1 runs (20260615_222757, 20260615_223116, 20260618_095124). All show `mean(error_z)` consistently negative: −1.13m, −0.87m, −1.28m respectively. However, VQ1's `pose_estimator.py` has already been empirically fitted to `CAM_TILT = −21.6°` via a least-squares minimization over 2237 detections — the "flip z-row" heuristic in this plan assumed a naive ±20° starting point, which is no longer the case. Decision: **copy VQ1 pose_estimator.py to VQ2 as-is** (CAM_TILT = −21.6°). The remaining ~1m systematic z-bias (CV estimates gate ~1m too high) is noted — compensate at the controller level by targeting slightly below the CV estimate, or accept the bias and let the gate's 1.5m opening provide margin. Step 0.3 skipped because the -21.6° fit was already validated across multiple runs; another training lap would not change the matrix.

---

## Phase 1 — Week 1: Get Any Completion

> **Goal:** One clean run through all gates in VQ2. Time does not matter yet.
> Work in this order — Step 1 must be done before any scored run.

### Step 1.1 — CV Outlier Rejection (2–4 hours)
> A single CV false positive without this sends the drone to a phantom gate. One `if` statement.

File: `PyAIPilotExample-v2/gate_verifier.py`

- [x] **1.1a** Before writing to `shared_data['cv_gate_pos']`: check if estimate jumps > 3m within 150ms window
- [x] **1.1b** VQ2 adaptation: no gate map in VQ2, so reject based on temporal proximity to previous estimate rather than known gate position
- [x] **1.1c** If jump > 3.0m and prev estimate age < 0.15s → discard the estimate, do not update `cv_gate_pos`
- [x] **1.1d** Log discarded estimates to gate_estimates CSV with `outlier_rejected=1` column; also logs to detection_diag CSV for every frame

**Outcome notes:** Implemented in `gate_verifier.py` using a 150ms time window. CV velocity estimation also added in the same file: `vel = -(est_new - est_prev) / dt` for consecutive same-gate frames (time gap < 150ms). Cross-gate transitions produce a gap > 150ms, so spurious velocity from gate-to-gate position jumps is naturally suppressed.

---

### Step 1.2 — Fix CONTROL_HZ (10 minutes)
> VQ2 example has CONTROL_HZ = 250, which violates the spec (<100 Hz). Fix now.

File: `VQ2/PyAIPilotExample-v2/controller.py`

- [x] **1.2a** Changed `CONTROL_HZ = 250` → `CONTROL_HZ = 99`
- [x] **1.2b** Added `ctypes.windll.winmm.timeBeginPeriod(1)` (and `timeEndPeriod(1)` on shutdown) in `main.py`

---

### Step 1.3 — Fix TimeSync Bug (5 minutes)
> The example code starts the TimeSync thread incorrectly — it never runs.

File: `VQ2/PyAIPilotExample-v2/setup.py`

- [x] **1.3a** Changed `ts_loop = TimeSync(sim_conn, shared_data)` → `ts_loop = TimeSync.create_timesync(sim_conn, shared_data)`

---

### Step 1.4 — Gyro Yaw Integration (1 day)
> ATTITUDE is blocked in VQ2. Without yaw, the controller cannot decompose NED velocity into body-frame, and yaw_rate is hardcoded 0.0. Integrate HIGHRES_IMU gyro instead.
> Full AHRS not needed — yaw only. Gyro drift over 2–5s inter-gate window is < 2.5° = < 0.4m lateral error at 10m.

File: `mavlink_rx.py`
- [x] **1.4a** In `on_highres_imu()`, extract `zgyro` (yaw rate, rad/s) and `time_usec`
- [x] **1.4b** Write to `shared_data['gyro_yaw_rate']` and `shared_data['highres_imu_time_us']`

File: `controller.py`
- [x] **1.4c** At start of `update()`: compute `dt = (imu_time_us - last_imu_time_us) * 1e-6`. Integrate: `self._integrated_yaw = wrap(self._integrated_yaw + gyro_yaw_rate * dt)`. Write to `shared_data['integrated_yaw']`.
- [x] **1.4d** Initialize `self._integrated_yaw = 0.0` (treat takeoff heading as NED North). ATTITUDE is blocked in VQ2 so no init from telemetry.
- [x] **1.4e** Yaw rate from path tangent direction: `yaw_rate = clip(-KP_YAW * wrap(target_yaw - integrated_yaw), ±MAX_YAW_RATE)`. `KP_YAW = 1.0` (conservative start).
- [x] **1.4f** All reads of `attitude` (blocked) replaced with `yaw = self._integrated_yaw; roll = 0.0; pitch = 0.0` throughout controller and gate_verifier

**Outcome notes:** Also added VQ1 CV pipeline files (gate_detector.py, pose_estimator.py, planner.py) and new gate_verifier.py to VQ2. All of setup.py, main.py rewritten. Controller is a full VQ1-based state machine adapted for no pos/attitude feedback: time-based TAKEOFF transition (2.5s climb phase), CV_PLAN only, active_gate confirmation for waypoint advancement.

---

### Step 1.5 — End-to-End Completion Test
- [ ] **1.5a** Training run: drone completes all gates
- [ ] **1.5b** Check controller_log.csv: `state` column transitions INIT→IDLE→TAKEOFF→FLY→FINISHED, `active_gate` increments through all gates
- [ ] **1.5c** Check `gate_estimates_*.csv`: CV detections present, outlier_rejected counts low

**Target:** Course completion. Any time.

---

## Phase 2 — Week 2: Speed and Position Anchoring

> **Goal:** Improve reliability and push lap time to < 30s.
> Prerequisites: Phase 1 complete, course completes reliably in training.

### Step 2.1 — active_gate Position Reset (1–2 days)
> The most reliable position fix available. Each gate confirmation gives a known world position.

File: `controller.py`
- [ ] **2.1a** When `active_gate` increments: look up `gate_N_pos` from offline gate map
- [ ] **2.1b** Estimate CV-derived velocity: `cv_vel = (last_cv_pos - prev_cv_pos) / (last_cv_time - prev_cv_time)`. Store last two valid CV estimates with timestamps in `shared_data`.
- [ ] **2.1c** Apply backward correction: `anchor_pos = gate_N_pos - cv_vel * 0.05` (0.05s = estimated active_gate report latency)
- [ ] **2.1d** Reset controller's position estimate to `anchor_pos`

---

### Step 2.2 — CV Depth Split (half day)
> solvePnP depth error at 15m = 1.03m (marginal). At 20m = 1.84m (unusable). Lateral bearing is accurate at any range.

File: `controller.py`
- [ ] **2.2a** Compute `range_to_gate = norm(cv_gate_pos - drone_pos)`
- [ ] **2.2b** If `range_to_gate < 10.0`: use full `cv_gate_pos` as target (depth trusted)
- [ ] **2.2c** If `range_to_gate >= 10.0`: use CV only for yaw correction toward the detected gate center pixel; use planner waypoint for depth/distance

---

### Step 2.3 — Dead-Reckoning Bridge (half day)
> Between active_gate anchors and fresh CV frames, extrapolate position using last CV velocity.

File: `controller.py`
- [ ] **2.3a** Track `last_cv_pos`, `last_cv_time`, `cv_vel` in shared_data
- [ ] **2.3b** When CV is stale (> `CV_STALE_TIMEOUT`): compute `pos_est = last_cv_pos + cv_vel * (now - last_cv_time)`
- [ ] **2.3c** Cap extrapolation at 1.0s. Beyond 1.0s: fall back to spline waypoint
- [ ] **2.3d** Use `pos_est` as the position source for target selection in FLY loop

---

### Step 2.4 — Speed Tuning
- [ ] **2.4a** Increase `CRUISE_SPEED` by 10% from current value
- [ ] **2.4b** Run 3 training laps. If all complete cleanly → repeat
- [ ] **2.4c** On first gate miss: reduce by 5% and lock
- [ ] **2.4d** Tune `GATE_APPROACH_DIST` and `GATE_APPROACH_SPEED` using speed-over-time panel in `visualize.py`

**Target:** Sub-30 second lap in training.

---

## Phase 3 — Week 3: Trajectory Optimization

> **Goal:** Push lap time as low as possible for leaderboard ranking.
> Prerequisites: Phase 2 complete, sub-30s system stable in training.

### Step 3.1 — Offline Gate Map (2–3 days)
> Exploit: map is deterministic, drone starts at (0,0,0) every run, gate dimensions known.
> Method: automated CV at close range (< 10m) where depth is reliable. Multiple runs averaged.

- [ ] **3.1a** Write a mapping script: slow forward crawl at constant altitude, stop and hover 8m before each gate, take 10 frames, average CV estimates
- [ ] **3.1b** Use `MAVLINK_CMD_SIM_RESET = 31000` to programmatically restart runs
- [ ] **3.1c** Run mapping script 3 times per gate. Average the estimates. Flag if std > 0.3m for any gate.
- [ ] **3.1d** Save map as JSON: `vq2_gate_map.json` with `gate_id`, `pos_ned`, `estimated_std`
- [ ] **3.1e** Cross-validate: inter-gate distances and headings should be physically plausible

---

### Step 3.2 — Minimum-Snap Trajectory
- [ ] **3.2a** Replace cubic spline in planner with minimum-snap trajectory through gate centers
- [ ] **3.2b** Safety margin: 0.5m inside the 1.5m inner opening (610mm actual clearance per side — lateral fine, vertical needs camera tilt residual check)
- [ ] **3.2c** Validate using `visualize.py` top-down panel: trajectory should arc smoothly through all gate centers

---

### Step 3.3 — Precision Timing Loop
> Only matters for determinism and high-speed precision. Week 3, not Week 1.

File: `main.py` or `controller.py`
- [ ] **3.3a** Replace `time.sleep(1.0 / CONTROL_HZ)` with:
  ```python
  deadline = time.perf_counter()
  while running:
      controller.update()
      deadline += 1.0 / CONTROL_HZ
      slack = deadline - time.perf_counter()
      if slack > 0.002:
          time.sleep(slack - 0.001)
      while time.perf_counter() < deadline:  # busy-spin last ~1ms
          pass
  ```
- [ ] **3.3b** `timeBeginPeriod(1)` should already be set from Step 1.2b

---

### Step 3.4 — CV Pipeline Profiling
- [ ] **3.4a** Measure actual gate detection frame rate from `gate_estimates_*.csv` timestamps
- [ ] **3.4b** If effective rate < 25 Hz: move gate detection to a background thread with double-buffered frame handoff
- [ ] **3.4c** Profile solvePnP time per frame. If > 15ms: investigate resolution downscaling

---

### Step 3.5 — Final Scored Runs
- [ ] **3.5a** Only switch to VQ2 Qualification mode when training completion rate > 90%
- [ ] **3.5b** Run training lap immediately before each scored run to confirm the system is stable
- [ ] **3.5c** Log all scored run times. Target: sub-25s to be competitive.

---

## Key Constants to Track

> Update these as you tune — don't trust cached values.

| Constant | Location | Current Value | Notes |
|---|---|---|---|
| `CRUISE_SPEED` | controller.py | 8.0 m/s | Increase in Phase 2 |
| `GATE_APPROACH_SPEED` | controller.py | 2.0 m/s | |
| `BRAKE_MARGIN` | controller.py | 5.0 m | replaces GATE_APPROACH_DIST |
| `CV_STALE_TIMEOUT` | controller.py | 0.3s | May need tuning |
| `KP_YAW` | controller.py | 1.0 | Tune in Phase 2 |
| `CONTROL_HZ` | controller.py | 99 | ✓ Fixed |
| `CAM_TILT` | pose_estimator.py | −21.6° | ✓ Verified Phase 0 — copy VQ1 as-is |
| `HOVER_THRUST` | controller.py | 0.28 | Verify on first flight |
| `VQ2_CLIMB_TIME` | controller.py | 2.5 s | Time-based takeoff climb duration |

---

## What Was Deliberately Cut

These were in the original plan and removed by the council:

| Idea | Why Cut |
|---|---|
| Hand-labeling gate corners | Close-range automated CV (< 10m) is sufficient and repeatable |
| Full IMU AHRS for state estimation | Gyro-yaw-only is sufficient; full AHRS is over-engineered for this window length |
| IMU acceleration double-integration | 4.5–22m drift over 30s. Use CV delta for velocity instead |
| Minimum-snap trajectory as Week 1 deliverable | Only adds value after a sub-30s system exists |
| Deterministic command replay | Python timing jitter (~15ms on Windows) makes exact replay unreliable |

---

## Architecture Reference (VQ2)

```
Simulator
  ├── MAVLink UDP :14550
  │     ├── HEARTBEAT          → armed status
  │     ├── HIGHRES_IMU        → xgyro/ygyro/zgyro (integrate for yaw)
  │     ├── ENCAPSULATED_DATA  → race_status (active_gate_index)
  │     ├── COLLISION          → 1001=gate hit, 1002=environment
  │     ├── ACTUATOR_OUTPUT    → motor RPMs
  │     └── [ATTITUDE / LOCAL_POSITION_NED / ODOMETRY → BLOCKED IN VQ2]
  │
  └── Vision UDP :5600
        → 30Hz, 640×360 JPEG, 24-byte header
        → gate_detector (HSV) → solvePnP → cv_gate_pos (NED)

shared_data keys to add/modify for VQ2:
  gyro_yaw_rate      (float, rad/s)     — from HIGHRES_IMU
  integrated_yaw     (float, rad)       — controller integration
  last_cv_pos        (np.array 3,)      — previous valid CV estimate
  last_cv_time       (float)            — timestamp of last_cv_pos
  cv_vel             (np.array 3,)      — (last_cv_pos - prev_cv_pos) / dt
  vq2_gate_map       (list of dicts)    — loaded from vq2_gate_map.json
```

---

*Strategy developed via 3-round council review (Devil's Advocate × State Estimation Expert × Competition Strategist). All three members approved.*
