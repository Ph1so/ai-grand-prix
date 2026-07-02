# Strategy that passed Virtual Qualifier 1

This documents the actual approach the AI pilot used to complete the VQ1 course
(6 gates, ~160 m course length). Only what the final, passing build actually
does is described here — code paths that exist but weren't exercised in the
winning runs (e.g. CV-based navigation) are intentionally omitted.

Confirmed from the winning runs' logs (`run_20260607_011732`, `run_20260607_012243`):
every single FLY-state control tick used `target_mode = 'spline'`. The CV pipeline
ran and logged estimates, but never became the active guidance source — navigation
was 100% MAVLink-telemetry-driven.

---

## 1. Navigate by the MAVLink gate map, not vision

`GUIDANCE = 'MAVLINK'`. On `INIT`, the controller waits for the sim to broadcast
its 6-gate track map over `ENCAPSULATED_DATA` (type=2), then builds the entire
flight plan from that map up front — no per-frame perception in the loop.

This was the foundational decision: the gate map is deterministic and broadcast
before the race starts, so the fastest path to "complete the course" was to trust
that data and fly a pre-computed trajectory through it, rather than depend on a
noisy, frame-by-frame CV pipeline to find each gate live.

## 2. Build a smooth path through the gates, not a sequence of point-to-point hops

`planner.py` converts the 6 raw gate poses into a flight plan in two stages:

1. **Waypoints** (`build_waypoints`): for each gate, compute the opening center
   (`gate_center` — undoes the NED z-sign flip, shifts up by half the gate height,
   applies a small tunable `GATE_Z_BIAS`), and place a "lead-in" waypoint
   `LEAD_IN_DIST = 12 m` before it along the gate's approach direction
   (`gate_approach_dir`, derived from the gate's quaternion normal). Result:
   `start → lead-in-0 → gate-0 → lead-in-1 → gate-1 → ...`
2. **Spline** (`smooth_path`): fit a C2-continuous cubic spline through all of
   those waypoints, parameterized by cumulative chord length. This is the actual
   path the drone follows — it removes the sharp direction changes that a
   waypoint-to-waypoint chase would produce at each corner.

## 3. Follow the spline with feed-forward velocity + cross-track correction

Rather than chasing a single target point (which produces stop-and-go,
proportional-to-distance behavior), `_path_tracking_desired_vel` computes a
velocity setpoint as:

```
desired_vel = cruise_speed * tangent_unit_at_nearest_path_point + cross_track_correction
```

- The **tangent** is taken from a lookahead point ~20 samples ahead of the
  nearest path point (`TANGENT_SAMPLES`), which smooths out spline kinks right
  at waypoints.
- The **cross-track term** (`KP_CROSS = 0.8`, capped at 1.0 m/s independently of
  the forward term) pulls the drone back onto the path without eating into
  forward progress — capping the *combined* vector (an earlier approach) caused
  the drone to visibly slow down on every curve.

This feed-forward-plus-correction structure is what makes the flight look like
continuous racing-line flying rather than a sequence of braking maneuvers at
each gate.

## 4. Speed profile: ramp down on gate approach, ramp back up after

`speed_cap` is a linear ramp between two named speeds:

```
speed_cap = GATE_APPROACH_SPEED + (CRUISE_SPEED - GATE_APPROACH_SPEED)
            * min(1, gate_dist / GATE_APPROACH_DIST)
```

`CRUISE_SPEED = 3.0 m/s` on open stretches, ramping down to
`GATE_APPROACH_SPEED = 1.2 m/s` within `GATE_APPROACH_DIST = 12 m` of the next
gate center. A linear ramp (rather than a hard step at a threshold distance)
avoids the sudden setpoint jump that previously caused the drone to overshoot
into backward-correcting flight right at the gate.

Two more layers keep the velocity setpoint itself well-behaved:
- **Speed brake**: if actual horizontal speed exceeds `speed_cap`, the excess is
  blended into `desired_vel` (scaled by `K_BRAKE = 0.8`) *before* slew limiting,
  so braking can't itself cause a one-frame setpoint discontinuity.
- **Slew limiter** (`MAX_VEL_SLEW = 2.5 m/s²`): caps how fast `desired_vel` itself
  is allowed to change per tick, so any remaining setpoint changes (e.g.
  `speed_cap` shifting near a gate) arrive as a ramp rather than a step —
  preventing the underdamped pitch/roll oscillations a step input would excite.

## 5. Cascaded velocity → attitude control (the structural fix that made flight smooth)

The core control fix of the project: instead of mapping velocity error directly
to an attitude **rate** command (an undamped integrator chain that produced a
persistent ~2.4 s limit-cycle "sway"), the controller now runs two nested loops:

**Outer loop — velocity error → desired tilt angle:**
```python
desired_pitch = clip( KP_VEL_ANGLE * fwd,   ±MAX_TILT_ANGLE)   # KP_VEL_ANGLE = 0.12, MAX_TILT_ANGLE = 0.30 rad
desired_roll  = clip( KP_VEL_ANGLE * right, ±MAX_TILT_ANGLE)
```
where `fwd`/`right` are the (derivative-damped) velocity error projected into
the body frame using the current yaw.

**Inner loop — tilt-angle error → attitude rate (the missing damping term):**
```python
pitch_rate = clip( KP_ANGLE * (pitch - desired_pitch), ±MAX_RATE)   # KP_ANGLE = 2.0
roll_rate  = clip(-KP_ANGLE * (roll  - desired_roll),  ±MAX_RATE)
```
This explicitly commands "stop rotating, you're at the angle that produces the
acceleration you want" instead of letting velocity error drive the rate (and
therefore the angle) indefinitely. The sign convention mirrors the
`_handle_takeoff` leveling formula, generalized from a target of zero to a
non-zero `desired_pitch`/`desired_roll`.

**Sign-convention note that mattered in practice:** the outer-loop pitch sign is
*not* the same as the old single-stage formula's sign. The sim applies
`pitch_rate` with an *inverted* effect on `pitch` but applies `roll_rate` with
the *same*-sign effect on `roll`. The old direct mapping's sign worked only
because that inversion canceled it out in the pitch chain; once the cascade
introduced an explicit angle stage, the inversion was no longer in the loop, so
`desired_pitch` had to carry the net `+fwd → +pitch` relationship directly
(flight-tested: the naively-copied sign sent the drone 85 m the wrong way down
the course). Roll required no such flip.

A velocity-error derivative term (`KD_VEL = 0.04`, exponentially filtered:
`0.3 * raw + 0.7 * previous`) is folded into the outer loop's input to damp
overshoot without amplifying telemetry noise.

Two supporting adjustments round out the attitude/thrust loop:
- **Thrust tilt compensation**: `thrust = HOVER_THRUST / max(0.5, cos(roll)*cos(pitch)) + ...` —
  restores the vertical thrust component lost to tilt, instead of letting
  altitude sag whenever the drone banks into a turn.
- **`MAX_SPEED` rate scaling**: if actual speed exceeds `MAX_SPEED = 10 m/s`,
  `pitch_rate`/`roll_rate` are scaled down proportionally as a hard ceiling.

## 6. Yaw: always point along the spline tangent

```python
target_yaw = atan2(tangent_unit[1], tangent_unit[0])
yaw_rate   = clip(-KP_YAW * yaw_err, ±MAX_YAW_RATE)   # KP_YAW = 1.5, MAX_YAW_RATE = 1.5 rad/s
```

In spline mode, yaw is corrected continuously regardless of cross-track
distance — skipping correction when cross-track error is small lets yaw drift
uncorrected, which misaligns the body frame and sends every subsequent velocity
correction in the wrong NED direction (the body-frame `fwd`/`right` projections
depend on yaw being accurate). The sim applies `yaw_rate` counterclockwise from
above (opposite of NED convention), so the command is negated to produce the
intended clockwise (North→East) rotation.

## 7. Gate-passage detection: geometric plane crossing, not axis-aligned distance

A waypoint advances when either the sim confirms passage (`active_gate`
increments) or the drone's position, projected onto the lead-in→gate approach
direction, is more than 1 m past the gate center:

```python
approach_dir    = (gate_wp - leadin_wp) / |gate_wp - leadin_wp|
drone_past_gate = dot(pos - gate_wp, approach_dir) > 1.0
```

This replaced an earlier axis-aligned check (`pos.x < gate.x - 1.0`) that only
worked for gates facing north — the projection-based check works for gates at
any orientation, which this course's gates are (see `gate_approach_dir`).

## 8. Takeoff: level off, then hold 3 seconds before climbing

On entering `TAKEOFF`, the controller first actively levels any residual
roll/pitch left over from the sim's reset attitude (`KP_LEVEL = 2.0` proportional
leveling — sending zero rates would lock in the reset angle and cause
uncontrolled drift during the climb).

Critically, it then **holds on the ground for `TAKEOFF_DELAY = 3.0 s`** —
sending zero attitude rates and zero/hover thrust without climbing — before
starting the ascent to `TAKEOFF_ALT_NED = -0.5 m`. This was added because the
race-start signal the sim broadcasts fires slightly *before* the official start;
climbing immediately on that signal was triggering an early-start
disqualification. The 3 s hold absorbs that gap.

## 9. Race-start / reset detection

`IDLE` waits for `race_started` to flip from `False` to `True` (requiring the
hysteresis of having explicitly observed `False` first, plus `time_boot_ms`
advancing to confirm the sim clock is actually running) and then a further 0.5 s
settle before transitioning to `TAKEOFF`. Mid-flight, if the sim resets the
drone back near the origin while `race_started` goes false again, the controller
detects this and cleanly re-enters `IDLE` (resetting all per-run state — gate
index, integrators, slew/derivative filters, takeoff-hold timer) so repeated
attempts within a session work without a process restart.
