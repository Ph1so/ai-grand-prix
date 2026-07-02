# Task Brief: Gate-Position Estimation + "Look Toward Target" Control for VQ2

## Problem context

The current controller (`controller.py`) navigates using a **ground-truth gate map**
(`shared_data['gates']`, populated from MAVLink `ENCAPSULATED_DATA` track-info packets)
plus live CV detections (`shared_data['cv_gate_pos']`) as a secondary/correction signal.
This works for the current qualification round, but **VQ2 likely won't provide that
ground-truth gate map** — the drone will need to navigate primarily off noisy,
vision-only gate detections.

Separately, the drone currently doesn't actively orient itself toward gates: yaw is
hard-coded to `0.0` (a documented known gap), and vertical motion happens via
thrust/altitude PID rather than by pointing the nose at the target.

Both problems point at the same fix: **make the drone actively point toward the gate
it's flying to**, using a smoothed estimate of that gate's position built up from CV
detections over time — since a more square-on view also produces better CV detections
(a virtuous cycle).

This breaks into two sub-tasks.

---

## Part A — Smoothed per-gate position estimate from noisy CV detections

Build a running estimate of each gate's NED position from `shared_data['cv_gate_pos']`
samples, robust enough to use as a navigation target when no ground-truth map exists.

### Key empirical fact to seed this with

Measured this session on `logs/run_20260607_192059_with_frames/` (post the just-applied
A11 camera-tilt fix in `pose_estimator.py`): **CV position error grows roughly linearly
with drone-to-gate range**:

| range      | mean ‖error‖ |
|------------|--------------|
| < 5 m      | 1.7 m        |
| 5–10 m     | 0.9 m        |
| 10–15 m    | 4.7 m        |
| 15–20 m    | 9.5 m        |
| 20–30 m    | 13.2 m       |
| 30+ m      | 18.2 m       |

Because the gate is a static point in world (NED) frame, and `camera_to_ned()` already
removes drone ego-motion, the right estimator here is a **range/variance-weighted
running average** (`weight ∝ 1/range²`, or fit directly to the table above). This is
mathematically equivalent to what a Kalman filter converges to for a static-state
process, without the `Q`/`R`/`P` tuning overhead.

A full Kalman/EKF would only be worth the added complexity later if you need to "coast"
the estimate through CV dropouts using drone ego-motion (dead reckoning) — that's a
natural escalation path, not a starting point.

### Open question to resolve early

**How to associate detections with gate IDs without the position map** (currently each
CV frame is independent — known gap #4 in `CLAUDE.md`). Check whether
`shared_data['race_status']['active_gate']` survives into VQ2 even if
`shared_data['gates']` doesn't — it's populated from a *separate* MAVLink signal (race
progress, not the track map). If it does, use increments of `active_gate` as a reset
trigger: zero the running accumulator and start fresh whenever the target gate changes.

---

## Part B — "Point toward target" attitude control

Replace the always-zero yaw and the current "ascend/descend without aiming" vertical
behavior with a controller that orients the drone toward the (estimated or mapped) gate
position — both horizontally (yaw) and vertically (coordinated pitch) — so the gate
stays centered in the camera frame as the drone approaches.

### Critical gotcha

The camera is rigidly mounted at a fixed tilt relative to the body (`R_CAM2BODY` in
`pose_estimator.py`, ~21.6° per the fit just applied). **Pointing the body's nose at
the gate is not the same as centering the gate in the camera frame.** The target
computation needs to invert `R_CAM2BODY` — i.e. solve for the body attitude that puts
the gate on the camera's optical axis, not the body's forward axis.

Also note: on a quadrotor, pitch is coupled to translation (pitching down accelerates
you forward/down) — so this isn't a free "look here" gimbal command, it's a redesign of
how the FLY-state attitude targets are derived, intertwined with the existing
velocity-tracking PID. Scope it as that, not as a small bolt-on.

---

## Files to read first, in this order

1. **`CLAUDE.md`** (project root) — orientation: controller state machine,
   `shared_data` keys table, and the "Known Gaps" section (items #1 yaw-always-zero
   and #4 no-historical-CV-state are exactly what this task closes).
2. **`PyAIPilotExample/controller.py`** — the FLY-state loop where `cmd_roll_rate` /
   `cmd_pitch_rate` / `cmd_yaw_rate` / `cmd_thrust` are computed, and where
   `cv_gate_pos` / `cv_gate_time` are consumed. This is where both new pieces integrate.
3. **`PyAIPilotExample/pose_estimator.py`** — `R_CAM2BODY` and `camera_to_ned()`;
   needed to understand (and invert) the camera-to-body mounting geometry for Part B.
4. **`PyAIPilotExample/gate_verifier.py`** — where `cv_gate_pos` / `cv_gate_time` get
   published into `shared_data`; a natural home for the Part-A running-estimate
   accumulator (or it could live in `controller.py` — worth deciding deliberately).
5. **`PyAIPilotExample/mavlink_rx.py`** — confirm `race_status['active_gate']` is
   populated independently of the `gates` track-info packet (`on_encapsulated_data`),
   to validate the Part-A reset-trigger plan.
6. **`PyAIPilotExample/planner.py`** — shows how the gate map currently drives
   waypoint construction; this becomes the fallback path that Part A/B need to
   substitute for when no map is present.

---

## Suggested sequencing

Start with **Part A** — it's self-contained, lower-risk, immediately testable against
the existing ground-truth gate map (compare smoothed-estimate error vs. raw
single-frame CV error), and useful regardless of whether VQ2 removes the map.

Then build **Part B** on top, since it depends on having a stable target to point at
and is the larger, more coupled behavioral change.
