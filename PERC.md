# Perception Pipeline — CV vs. MAVLink Analysis & Assumptions

Notes from analyzing `run_20260607_012243` (`gate_estimates_*.csv` vs.
`flight_log_*_gates.json`, 78,480 CV detections across 6 gates). Purpose:
catalogue what the CV→pose pipeline currently assumes about the course and
camera, so each assumption can be verified independently before relying on
vision-only navigation in a round without MAVLink gate data.

This is an analysis/notes document only — no fixes proposed here.

---

## 1. What the error data shows

> **⚠ Superseded by Phase 0 — see §6 Phase 0 findings below.** The table and
> commentary in this section are preserved as-is because they're what
> *triggered* the Phase 0 investigation — but two of the three "patterns"
> called out below (gate-0 dominance, and the gate-0/gate-1 correlation
> anomaly) turned out to be measurement-instrument artifacts, not properties
> of the CV pipeline. Read this section as "what raised the alarm," and
> §6 Phase 0 as "what the alarm actually was." The corrected per-gate table
> (computed on deduplicated, genuinely-distinct measurements) lives there.

Per-gate stats (`error = CV estimate − MAVLink truth`, NED metres):

| Gate | n | mean err_x (N) | mean err_y (E) | mean err_z (D) | mean ‖err‖ | mean range | corr(range, ‖err‖) |
|---|---|---|---|---|---|---|---|
| 0 | 43,826 | −0.04 | −0.04 | **+2.77** | 3.27 | 22.1 m | −0.40 |
| 1 | 7,100  | +2.72 | −1.61 | **+5.61** | 7.30 | 19.8 m | +0.05 |
| 2 | 4,559  | +1.95 | +3.03 | **+6.77** | 8.52 | 12.3 m | **+0.99** |
| 3 | 7,191  | +4.91 | −5.41 | **+9.62** | 12.90 | 19.7 m | **+0.95** |
| 4 | 5,605  | +5.19 | +4.98 | **+6.92** | 10.58 | 12.8 m | **+0.98** |
| 5 | 10,199 | +1.28 | −1.84 | **+3.86** | 6.32  | 7.8 m  | **+0.99** |

Overall: mean ‖err‖ = 5.74 m, median = 2.56 m, max = 41.4 m.

Three patterns stand out:

1. **`err_z` (Down) is positive for every single gate** (+2.8 m to +9.6 m,
   never flips sign). This is the one truly systematic, sign-consistent
   bias in the dataset — the CV estimate consistently places the gate
   *lower* (larger Down value) than MAVLink truth.
2. **`err_x` and `err_y` flip sign gate-to-gate** (e.g. err_y is −1.61 at
   G1, +3.03 at G2, −5.41 at G3, +4.98 at G4). This is *not* a fixed
   camera-frame offset — it behaves like something that depends on
   heading/geometry rather than a constant calibration term.
3. **Gates 2–5 show near-perfect correlation between range and error
   magnitude** (0.95–0.99): error grows almost linearly with
   drone-to-gate distance. Gates 0–1 do *not* show this (−0.40, +0.05).
   Also notable: **Gate 0 alone accounts for 56% of all detections**
   (43,826 of 78,480) — visible (or successfully detected) for far
   longer than any other gate.

---

## 2. Pipeline walkthrough (the moving parts, in order)

```
frame → detect_gate() → corners[TL,TR,BR,BL] → estimate_gate_camera_frame() → tvec (camera frame)
      → camera_to_ned(tvec, drone_pos, roll, pitch, yaw) → world gate centre (NED)
```

**Stage 1 — `gate_detector.detect_gate`**
- HSV-threshold for orange (two bands: H 0–25 and H 160–179 red wrap-around),
  morphological close (7×7 kernel).
- Find external contours; keep those with area ≥ 200 px² **and**
  bounding-box aspect ratio in (0.3, 3.0) (`_roughly_square`).
- Take the **largest** surviving contour ("largest = closest").
- Reduce its convex hull to exactly 4 points via `approxPolyDP`, sweeping
  epsilon 0.04 → 0.15 of perimeter until exactly 4 points emerge. **No
  fallback** if it never converges to 4 points — returns `None`.
- Order the 4 points into `[TL, TR, BR, BL]` using coordinate sum (`x+y`)
  for TL/BR and difference (`x−y`) for TR/BL.

**Stage 2 — `pose_estimator.estimate_gate_camera_frame`**
- `cv2.solvePnP` (ITERATIVE) against a fixed object-point set: a flat
  2.72 m × 2.72 m square (half = 1.36 m), centred at the gate-opening
  origin in its own z = 0 plane, ordered to match `[TL, TR, BR, BL]`.
- Camera intrinsics fixed: fx = fy = 320, cx = 320, cy = 180,
  **zero distortion**.
- Returns `tvec` — gate centre in camera frame (x-right, y-down, z-forward).

**Stage 3 — `pose_estimator.camera_to_ned`**
- Two chained rotations: `R_CAM2BODY` (a fixed, hand-derived 20°-tilt
  matrix — code comment notes the spec's "upward" sign had to be flipped
  to match observed data), then `R_b2ned` built fresh each call from the
  *current* MAVLink `roll, pitch, yaw` (ZYX Euler).
- Final position = `drone_pos + R_b2ned @ R_CAM2BODY @ tvec`, where
  `drone_pos = shared_data['pos']`.

**Stage 4 — `gate_verifier` ground-truth matching** (this is the
*analysis* pipeline, not the live CV pipeline — but it shapes how the
chart above should be read)
- `matched_gate_id` = whichever MAVLink gate centre is **nearest in 3-D
  to the CV estimate itself** (`np.argmin(dists)`). It is *not* derived
  from `active_gate` or from which gate is actually in frame.

---

## 3. Assumptions to fact-check (one at a time)

Check these off as each is independently verified against fresh data —
not assumed from this run alone.

### Geometry / object model
- [ ] **A1 — Outer boundary, not inner opening.** The detector always
      traces the gate's *outer* boundary (2.72 m square), never the inner
      opening (1.5 m) or a blend of the two. `OBJ_PTS` is hard-coded to
      the outer half-dimension (1.36 m); if the segmented blob sometimes
      follows the inner edge (or blends, depending on viewing angle/HSV
      bleed), PnP is fed the wrong scale. *(Priority: high — see §4.3)*
- [x] **A2 — Gate is a flat planar square.** ✅ **Confirmed — visual audit,
      `phase1_2_audit.py` (PHASE 1.2 PASS, see write-up).** Six annotated
      `reproject_corners()` overlays spanning the full gate sequence/range
      show the detector consistently tracing the *outer* boundary at every
      sampled range and angle — no front/back-face ambiguity observed. The
      dominant visible-error source is range-scaled silhouette inflation
      (A1's mechanism, confirmed in the same audit), not a face-selection
      problem.
- [x] **A3 — Uniform gate dimensions across the course.** ✅ **Confirmed
      by spec** (§3.1: gates are "consistent throughout the Virtual
      Qualifier 1 track"; §3.7 gives a single gate-dimension spec, not
      per-gate variants — see §5.1). The JSON carries per-gate
      `width`/`height`, but `OBJ_PTS` is a single global constant that
      ignores them — fine for VQ1, but re-check for VQ2 (only *positions*
      are documented as changing between qualifiers, not necessarily
      dimensions).

### Detection logic
- [ ] **A4 — "Largest contour = the relevant (next) gate."** Breaks if a
      second gate becomes visible in the same frame (e.g. looking through
      the current gate toward the next one, or a previous gate still in
      frame) — the detector could lock onto the wrong blob, or a merged
      one.
- [x] **A5 — Aspect-ratio / min-area filters match real gate silhouettes.**
      ✅ **Mostly confirmed, thin marginal tail — `phase1_2_audit.py`
      (PHASE 1.2 PASS).** 98.6% of 38,908 logged frame-attempts pass
      cleanly; of the 1.4% (537) the aspect-ratio band rejects, ~58% sit
      just outside `0.3 < w/h < 3.0` ("marginal," not wild outliers) —
      consistent with steep oblique views compressing the silhouette past
      the band's edge. A small loosening (e.g. 0.25–3.5) would likely
      recover some of these at low risk, but the volume is small enough
      that this is a low-priority polish item, not a structural problem.
- [ ] **A6 — Corner ordering holds at all viewing angles.** `_order_corners`
      (sum/diff heuristic) assumes the gate is viewed close to
      fronto-parallel with only "moderate perspective warp" (per its own
      docstring). At sharp oblique angles it could mis-assign which
      detected corner is TL vs TR vs BL, silently feeding PnP a corner
      permutation that doesn't match `OBJ_PTS`. *(Priority: high — see §4.2)*
- [x] **A7 — `approxPolyDP` reliably converges to 4 points.** ✅ **Mostly
      confirmed, fallback would help only at the margin —
      `phase1_2_audit.py` (PHASE 1.2 PASS).** 98.6% of candidates reduce to
      exactly 4 points (87.9% at the *finest* epsilon — minimal
      simplification needed, i.e. real silhouettes already look like clean
      quads). Of the 1.4% that never converge, 97% are within ±1 point of
      4 — exactly the band a bounding-rect/vertex-drop fallback (known gap
      #8) would likely recover. Polygon-reduction failure is a real but
      *small* contributor; it does not appear to be the main reason gates
      other than #0 have fewer detections (shorter visibility windows /
      range-scaled detection dropoff are bigger factors per Phases 1.1/3).
- [ ] **A8 — HSV thresholds are universal across the course.** Same
      orange/red bands and same S/V floors (120/60) assumed to hold
      regardless of lighting, distance-induced desaturation, motion blur,
      or background contamination anywhere on the track.

### Camera model
- [x] **A9 — Zero lens distortion.** ✅ **Confirmed by spec** (§3.8:
      "We use a standard pinhole camera model without lens distortion" —
      see §5.1). `DIST = zeros(4)` is correct as written, *given the spec
      is accurate* — the published camera model has no radial/tangential
      distortion at the stated focal length.
- [x] **A10 — Fixed intrinsics are exact.** ✅ **Confirmed by spec**
      (§3.8: `[cx,cy]=[320,180]`, `[fx,fy]=[320,320]` — byte-for-byte
      match with `K` in `pose_estimator.py`; see §5.1). One wrinkle: the
      same spec paragraph also states "VFoV = 90°," which is *not*
      consistent with these same numbers (works out to ≈58.7° vertical /
      90° **horizontal** — see §5.2.b). That's an internal inconsistency
      in the spec's own numbers, not a problem with the cx/cy/fx/fy
      values themselves.
- [ ] **A11 — `R_CAM2BODY` is fixed and time-invariant.** Assumes the
      camera is rigidly mounted at exactly 20° (with the
      empirically-flipped sign) and never deviates — no vibration, sag,
      or drift in flight. *(Priority: highest — see §4.1)*
      ⚠️ **Spec partially confirms, partially contradicts this** (§3.8:
      camera shares the body's origin and is tilted 20° — supports "rigid
      mount, fixed magnitude" — but states the tilt is **"upwards,"**
      directly conflicting with the code's empirically-flipped
      (effectively-downward) sign). This is the *newest* spec revision
      (Issue 00.02, changelog literally says **"camera"**), so it isn't
      stale guidance — see §5.2.a. Can't be resolved by reading docs;
      needs targeted empirical testing, and is the prime suspect for the
      persistent positive `err_z` bias in §1.

### Fusion with telemetry
- [x] **A12 — `pos`/`attitude` are synchronous with the camera frame.** ✅
      **Confirmed negligible — `a12_sync_audit.py`.** No timestamp
      alignment exists between the vision frame (its own `sim_time_ns`)
      and whatever `shared_data['pos']`/`attitude` hold at the moment
      `process_frame` runs (the *latest* MAVLink sample, not necessarily
      the sample at the camera's capture instant — TimeSync exists but
      isn't consumed, known gap #5). This is fully testable from
      *existing* Phase-2-instrumented logs with no new flight: every
      `gate_estimates_*.csv` row already records both the camera frame's
      true capture instant (`sim_time_ns`) and the exact telemetry
      `process_frame` used; interpolating the dense (~90 Hz)
      `flight_log_*.csv` series at that instant gives "what *should* have
      been used," and the difference *is* the sync error, by definition.
      Propagated through the same rotation chain `pose_estimator` uses,
      it comes out to **~3–4 cm mean / ~1.3 m worst-case** positional
      effect — **~0.2% of the metres-scale total error** Phases 1.1/3
      already explained — cross-validated on two independent calibration
      runs (`run_20260607_120608`: 0.23%, n=3,940; `run_20260607_124556`:
      0.21%, n=958). Yes, `sync_err` correlates with total error (+0.54 to
      +0.69) — but that's a red herring at this scale: both simply rise
      together when the drone maneuvers harder (more rotation → both more
      telemetry staleness *and* more rotation-chain/oblique-angle error
      per Phase 3), not evidence that sync error *drives* the total. **A12
      is a confirmed non-issue at current speeds and ~90 Hz telemetry
      rate** — revisit only if a future course flies substantially faster
      or the telemetry rate drops.
- [ ] **A13 — Attitude (roll/pitch/yaw) is accurate enough to drive the
      rotation chain.** Any small attitude error is amplified by the lever
      arm from camera to gate — more so at long range, which lines up with
      the range-correlated error growth seen at gates 2–5. *(Priority:
      high — see §4.1, §4.2)*

### Analysis / matching layer (shapes how we *read* the error, not the live pipeline)
- [ ] **A14 — `matched_gate_id` reflects the gate actually in frame.** It's
      actually computed as nearest-MAVLink-centre-to-CV-estimate, *not*
      ground truth about what was visible. A badly-off estimate near a
      gate-boundary region could get silently attributed to the *wrong*
      gate, distorting the per-gate breakdown in §1 — and could be partly
      responsible for the apparent range↔error correlation if far-away,
      large-error estimates get grabbed by a nearer-but-wrong gate's
      centroid.

---

## 4. Suggested verification order (based on what the data points at)

1. **§4.1 → A11, A13** — the *uniform positive `err_z`* across every gate
   (never flips sign, +2.8 m to +9.6 m) points first at the camera
   mount-angle/sign assumption and the attitude-driven rotation chain.
2. **§4.2 → A6, A13** — the *sign-flipping `err_x`/`err_y`* (not a
   constant offset — e.g. err_y goes −1.61 → +3.03 → −5.41 → +4.98 across
   G1–G4) points at corner-ordering or yaw-driven rotation rather than a
   fixed camera-frame bias.
3. **§4.3 → A1, A9, A10** — the *range-correlated blowup at gates 2–5 but
   not 0–1* (corr 0.95–0.99 vs. −0.40/+0.05) points at object-point scale
   or intrinsics/distortion mismatch — though A14 means part of that
   correlation could be a measurement artifact of the nearest-centre
   matching rather than a property of the CV pipeline itself.

---

## 5. Cross-referencing official docs (`ref.md`, `updates.md`,
   `260508_Technical_Spec_0002.md` = VADR-TS-002 Issue 00.02)

### 5.1 Findings that resolve PERC.md assumptions

- **A9 (zero lens distortion) — ✅ confirmed.** §3.8: "We use a standard
  pinhole camera model without lens distortion." Matches `DIST = zeros(4)`
  exactly.
- **A10 (fixed intrinsics) — ✅ confirmed.** §3.8: `[cx,cy]=[320,180]`,
  `[fx,fy]=[320,320]` — byte-for-byte match with `K` in
  `pose_estimator.py`. (Caveat: see 5.2.b — the spec's *VFoV* figure
  quoted in the same breath doesn't itself add up against these numbers.)
- **A3 (uniform gate dimensions) — ✅ confirmed for VQ1.** §3.1: "Gates
  will be visually distinctive to the environment, **but consistent
  throughout the Virtual Qualifier 1 track**." §3.7 gives one
  gate-dimension spec (outer 2700×2700×260 mm, inner 1500×1500×260 mm) —
  no per-gate variants. (Re-verify for VQ2 — `updates.md` only documents
  *positions* changing between qualifiers, not necessarily dimensions.)
- **A11 (camera tilt, fixed mount) — ⚠️ half-confirmed, half-contradicted.**
  §3.8 confirms the camera shares the body's origin and is tilted by 20°
  (supports "rigidly mounted, fixed magnitude"), but states the tilt is
  **"upwards"** — directly conflicting with the code's empirically-flipped,
  effectively-downward sign. See 5.2.a — this is the standout contradiction.

None of the spec material gives grounds to confirm or refute A1, A2, A4–A7,
A12–A14 (detection-algorithm internals, time-sync usage, and the analysis
script's matching logic are simply outside the spec's scope — they remain
open, code/empirical-only checks). **A8 is addressed separately below (5.2.e)
— the docs don't "confirm" it so much as predict its failure.**

### 5.2 Confusing / contradictory points worth keeping in mind

**a. Camera tilt sign — the spec's *newest* clarification still disagrees
with the empirically-tuned code.**
§3.8: "The camera is tilted upwards by 20°." But `pose_estimator.py`'s
`R_CAM2BODY` flips that sign because "data shows the correction must go
in the opposite direction." This isn't stale documentation — Issue
00.02's revision-history row says its *only* change from 00.01 was
**"camera"** (NT, 2026-05-04) — i.e. this is the freshest, most deliberate
official statement about tilt direction available, and it still disagrees
with what flight data shows. Three readings seem possible: (i) the sim
doesn't match its own latest spec, (ii) "upwards" is expressed in a
convention/frame different from the one assumed when building
`R_CAM2BODY`, or (iii) the empirical sign-flip is actually compensating
for a *different*, unrelated sign error elsewhere in the rotation chain
(e.g. in `_euler_zyx`, or in how MAVLink `roll/pitch/yaw` map onto FRD
axes) that happens to cancel out. Any of these would be consistent with
the **persistent, sign-never-flips positive `err_z`** seen across every
gate in §1 — this is the single most important thing to resolve
empirically (ties directly to A11/A13, see §4.1).

**b. The spec's own numbers don't agree with each other: stated VFoV vs.
stated intrinsics.**
§3.8 states `[fx,fy]=[320,320]` at 640×360 resolution *and* "VFoV = 90°"
in the same paragraph. Working the numbers: with `fy=320` and a 360-px
image height, VFoV = 2·atan(180/320) ≈ **58.7°**, not 90°. What *does*
come out to 90° is the **horizontal** FoV: 2·atan(320/320) =
2·atan(1) = 90°. So either the spec mislabelled HFoV as VFoV, or one of
`fx/fy` / resolution / "90°" is simply wrong. Since `fx=fy=320` already
matches the working `K` matrix (and produces plausible PnP ranges — mean
drone-to-gate distances in the 8–22 m band, which line up with course
scale), the likeliest read is a HFoV/VFoV mislabel in the document — but
it's an internal inconsistency worth knowing about, especially if anyone
ever re-derives intrinsics from the stated FoV instead of using
`[fx,fy]` directly.

**c. The MAVLink message table (§4.3) doesn't match the interface this
codebase actually runs against.**
The spec table lists: HEARTBEAT, ATTITUDE, HIGHRES_IMU (listed *twice* —
once as "Vehicle status," once as "Measurements," itself a likely
copy-paste artifact in the doc), SET_POSITION_TARGET_LOCAL_NED,
SET_ATTITUDE_TARGET, TIMESYNC. It does **not** list `LOCAL_POSITION_NED`,
`ENCAPSULATED_DATA`, or `COLLISION` — yet these three are exactly what the
running sim sends and what `mavlink_rx.py` depends on (`pos`/`vel` come
from `LOCAL_POSITION_NED`; **the entire gate map and race-status come from
`ENCAPSULATED_DATA`**). Conversely, `HIGHRES_IMU` and
`SET_POSITION_TARGET_LOCAL_NED` *are* in the spec table, yet `HIGHRES_IMU`
is stubbed (`pass`, known gap #6) and `SET_POSITION_TARGET_LOCAL_NED`
isn't used anywhere — `SET_ATTITUDE_TARGET` is the controller's sole
outbound message. **This is the most strategically significant
contradiction found**: the one channel that currently supplies the gate
map (`ENCAPSULATED_DATA` — the foundation of the entire VQ1-winning
`GUIDANCE = 'MAVLINK'` strategy documented in `STRATEGY.md`) is *absent*
from the officially documented interface. That's consistent with it being
a VQ1 convenience/training-wheel channel rather than a guaranteed,
permanent feature — i.e. one more concrete reason the perception-hardening
work this session is about matters, and probably matters sooner than VQ2.

**d. FAQ's "limited starting-position coordinates only" claim vs. the
continuous `LOCAL_POSITION_NED` stream the whole controller is built on.**
`ref.md`: "Teams may receive limited coordinate information for the
starting position in the virtual qualifier. Beyond that, you should
expect to fly without coordinate/position data." In reality VQ1 streams
`LOCAL_POSITION_NED` (`pos`, `vel`) continuously at ~30 Hz, and *the
entire flight controller* — not just the gate-map navigation strategy —
is built on `shared_data['pos']`/`['vel']` (cascaded velocity→attitude
control, waypoint-arrival checks, slew limiting, everything). If a future
round actually matches the FAQ's stated expectation (no continuous local
position telemetry), it wouldn't just be "swap CV for the MAVLink gate
map" — the *base flight-control loop* as currently written would lose its
primary state-estimation input entirely. Companion FAQ — "Does the drone
know the track/map? Is SLAM required?": "The drone will not 'know' the
track... navigate using onboard sensing (primarily vision)" — reinforces
that the long-term expectation is vision-primary, telemetry-secondary,
which is the inverse of the current VQ1-winning architecture.

**e. HSV-orange detection isn't a "might it generalize?" question — the
docs already predict it won't.**
`updates.md` (26/02): VQ1 is "desaturated... gates highlighted... high
signal-to-noise ratio... Visual guidance aids may be active," whereas VQ2
"will be more complex, with lighting and other distractions to reduce
signal-to-noise ratio... Visual guidance aids will be off," in "a real
3D-scanned environment." This effectively pre-answers assumption **A8**:
the orange/red HSV bands tuned for VQ1's highlighted gates aren't really
a thing left to "verify" — they're a thing **already known to need
replacing**. Worth re-scoping A8 from "is this universal?" to "what
replaces color-keyed segmentation once the color-coding goes away?"

**f. Two more FAQ-vs-spec mismatches on sensor availability** (lower
relevance to the perception pipeline itself, but they establish that the
FAQ and the binding tech spec disagree more than once — useful to know
which one to trust):
  - *Motor RPM*: `ref.md` says onboard sensors will "likely" include
    "motor RPM readouts"; `updates.md`/spec state explicitly "Direct
    access to Engine RPMs is also not given."
  - *Battery status*: `ref.md` says the physical qualifier will "likely"
    provide "battery status"; `updates.md` states "The state of charge of
    the (virtual and physical) battery is not provided. Battery
    performance will not be a limiting factor."
  Likely explanation: the FAQ reflects an earlier/looser expectation that
  the later, binding tech spec tightened up — i.e. **treat the tech spec
  as authoritative whenever the two disagree.**

**g. Camera resolution: FAQ "~12 MP wide-angle" vs. the 640×360 stream
this whole pipeline is tuned around.**
Probably just "physical camera hardware spec" vs. "virtual simulator
stream resolution" rather than a true contradiction — but it's a useful
reminder that transferring this pipeline to the physical drone later
would mean re-tuning *every* pixel-space constant (HSV areas, polygon
epsilons, intrinsics) for a very different sensor, not just re-pointing
it at a new feed.

**h. "Classical Throttle/Roll/Pitch/Yaw" framing (FAQ/updates) vs. the
actual rate-based interface.**
The FAQ describes "classical drone control commands (Throttle, Roll,
Pitch, Yaw)," but the actual outbound message is `SET_ATTITUDE_TARGET`
with `type_mask = ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` — i.e. **body
rate** commands (roll_rate, pitch_rate, yaw_rate) plus thrust, not
stick-style angle/throttle inputs. Not a defect anywhere, just a reminder
that the plain-language FAQ description isn't the operative interface —
the spec's MAVLink message table (despite 5.2.c's gaps) is.

---

## 6. Recommended Course of Action — How to Verify, In Order

**Guiding principle.** The end-to-end ‖error‖ number we have today is the
*sum* of three independent failure sources that must be pulled apart
before any single assumption can be meaningfully "checked off":

  (a) **detection/PnP-stage error** — image corners → camera-frame `tvec`
  (b) **rotation-chain error** — `tvec` → NED, via `R_CAM2BODY` and the
      attitude-driven `R_b2ned`
  (c) **measurement/attribution artifacts** — how we score error in the
      first place (A14's nearest-centroid `matched_gate_id`)

Trying to "verify assumptions one at a time" while (a)/(b)/(c) are still
tangled into one ‖error‖ figure means a fix to the wrong layer could look
like progress by coincidence (or a real fix could look like it did
nothing). So: calibrate the ruler → split the error into its sources using
data we *already have* → design tests that isolate one variable at a time
→ then characterize the robustness items that matter for "rock solid" but
aren't the dominant *current* error source.

Two things make this very tractable:
- The course is **deterministic** (spec §3.5: "course geometry is
  identical... environmental conditions are deterministic") — any test
  maneuver is exactly repeatable run after run.
- `pose_estimator.py` already contains **`reproject_corners()`** — a
  debug-overlay helper that projects the solved PnP pose back onto the
  image — and it does not appear to be called anywhere in the live
  pipeline *or* the analysis tools. It's a free, ready-made visual-audit
  instrument.

### Phase 0 — Calibrate the measuring instrument first (resolves A14)

> ## ✅ PHASE 0 — PASS (verified against `run_20260607_012243`, methodology in `phase0_audit.py`)
>
> Two independent problems were hiding in the §1 numbers. Both are now
> understood, quantified, and (for the first one) trivially correctable —
> here's the writeup.
>
> #### Finding 1 — the dataset is ~17x inflated by repeat-processing of the same camera frame
>
> `gate_estimates_*.csv` logs **78,480 rows**, but grouping by exact-integer
> `sim_time_ns` (the camera's own per-frame timestamp) shows there are only
> **4,651 genuinely distinct camera frames** in the run — each one gets
> independently re-run through `detect_gate → estimate_gate_camera_frame →
> camera_to_ned → log` an average of **~17 times** (max 29), every time
> recombining the *same* image-derived `tvec`/attitude with whatever
> `drone_pos` happens to be live in `shared_data` at that instant. Proof:
> within a repeat-group, `est − drone_pos` (≈ `R_b2ned·R_CAM2BODY·tvec`)
> stays constant to within noise while `drone_pos` itself visibly drifts —
> i.e. the *visual* measurement never changes, only the telemetry snapshot
> it gets stamped with does. ~80% of repeat-rows are exact full-row
> duplicates; the rest are "telemetry-jitter smears" of one underlying
> sample (e.g. one observed group: 19 rows, same `tvec`, `error_mag` constant
> at 18.87 m, `drone_pos` drifting smoothly across the group).
>
> The mechanism is almost certainly in `vision_rx._vision_loop` /
> `gate_verifier.process_frame`: the *same rendered frame's* UDP chunk-set
> appears to be received and fully reassembled multiple times (retransmission,
> or the simulator broadcasting its "current" frame faster than its internal
> render clock advances `sim_time_ns`), and each reassembly re-triggers a full
> CV pass. **This is not just an analysis-log nuisance — `GateVerifier` is the
> same code path that publishes `shared_data['cv_gate_pos']`/`cv_gate_time`
> for the live controller**, so the live pipeline is plausibly burning ~17x
> the CPU it needs to on redundant detect+PnP passes per frame. **Recommended
> fix (not yet applied — flagging for a follow-up session):** have
> `_vision_loop` track the last-processed `frame_id` and skip reassembly/
> reprocessing if it repeats.
>
> The good news: this inflation turns out to be **statistically mild** for
> point estimates — deduplicating barely moves the headline numbers (overall
> mean ‖err‖ 5.74 m raw → 5.47 m deduped, a ~5% shift), because each smeared
> group averages out close to its true value. Its real damage is making `n`
> lie by ~17x (any future claim like "thousands of independent samples per
> gate" would be off by an order of magnitude) and artificially tightening
> per-gate spans in the A14 check below. **Always dedupe by `sim_time_ns`
> (keep first-seen row per group) before computing statistics from this CSV.**
>
> #### Finding 2 — "Gate 0 = 56% of detections" (§1.3) is the countdown/reset hover, not flight
>
> This is the real explanation for the §1.3 "Gate 0 dominance" mystery (and
> closes out Phase 1 item 3 as a side effect). Bucketing the 4,651 deduped
> frames by `‖drone_pos‖` (self-consistent — uses only the `drone_x/y/z`
> columns already logged on each row, no cross-file timestamp alignment
> needed, sidestepping A12 entirely):
>
> - **Exactly 50% of all genuinely-distinct frames in the run (2,339 / 4,651)
>   were logged while the drone sat at the origin** (‖drone_pos‖ < 2 m) —
>   i.e. during the pre-race countdown and/or the post-finish hold the user
>   confirmed exists ("the race has a 3-second countdown... during that
>   period the drone is completely motionless and staring at the first gate").
> - **96% of those at-origin frames (2,243 / 2,339) matched gate 0**; the
>   estimates cluster tightly at `(−23.9, −0.4, 0.9)` ≈ gate 0's true
>   position `(−23.3, −0.4, 0.0)`, mean range ≈ 24 m, mean ‖err‖ ≈ 2.7 m.
>   These are *correct* detections — just of a single fixed, easy, dead-on
>   view, repeated for the entire duration the drone sits still.
> - That single mechanism accounts for **89% of every CV sample
>   `matched_gate_id` ever assigned to gate 0** (2,243 of 2,530 deduped
>   gate-0 rows). Gate 0 isn't "visible for longer during racing" — the
>   controller just stares at it from a fixed spot before the clock starts
>   (and apparently again after the course resets).
>
> **Consequence for §1's gate-0 row specifically:** its reported mean ‖err‖
> of 3.27 m (lowest of all six gates) is an artifact of sample composition,
> not evidence the pipeline tracks gate 0 best. Genuine in-flight gate-0
> measurements look very different:
>
> | Gate-0 subset | n (deduped) | mean range | mean ‖err‖ |
> |---|---|---|---|
> | At origin (‖drone_pos‖ < 2 m) — countdown/reset hover | 2,243 | 24.0 m | 2.70 m |
> | Genuine in-flight approach (‖drone_pos‖ ∈ [10, 30) m) | 186 | 6.5 m | 4.16 m |
> | All in-flight (speed-filtered, ≥ 0.3 m/s) | 353 | 11.0 m | 7.05 m |
>
> i.e. real flight conditions produce **1.5×–2.6× worse** gate-0 error than
> the headline number suggests. When re-deriving any "per-gate accuracy"
> claim, filter to in-flight samples first — the corrected, deduplicated,
> in-flight-only table is:
>
> | Gate | n | mean ‖err‖ | mean range |
> |---|---|---|---|
> | 0 | 353 | 7.05 | 11.0 |
> | 1 | 301 | 7.55 | 10.7 |
> | 2 | 266 | 8.02 | 11.6 |
> | 3 | 414 | 12.34 | 18.9 |
> | 4 | 335 | 10.23 | 12.4 |
> | 5 | 332 | 9.68 | 12.5 |
>
> Overall in-flight: mean ‖err‖ = 9.32 m, median = 7.87 m — noticeably worse
> than the raw-data overall of 5.74 m, because the raw figure is ~50%
> diluted by easy static-viewing samples. **This in-flight table is the
> right baseline to compare future pipeline changes against**, not the §1
> raw table.
>
> #### A14 verdict — nearest-centroid `matched_gate_id` is TRUSTWORTHY (on this course's geometry)
>
> The original A14 worry: a badly-off CV estimate could get silently grabbed
> by the wrong gate's centroid (e.g. inflating the "error grows with range"
> correlation in §1.3 with manufactured cross-gate noise). The most likely
> place for that to happen is exactly the scenario above — multiple gates
> simultaneously visible/detectable from one vantage point (the origin sees
> both gate 0 and gate 1, since the course runs roughly along a line). So
> that's exactly where I stress-tested it: of the 2,339 at-origin frames, 96
> were matched to **gate 1, not gate 0**. For every single one of those 96,
> I checked which gate's true position the `est` actually landed closer to —
> **gate 1's truth was closer in all 96 cases** (e.g. `est=(−49,−2.7,8.0)` is
> 13 m from gate 1's truth vs. 27 m from gate 0's — correctly matched). The
> "overlapping `matched_gate_id` time-spans" the original A14 spot-check
> flagged are fully explained by *genuine simultaneous visibility* from the
> start line, not misattribution.
>
> **A14 passes**: the nearest-centroid heuristic correctly discriminates
> between candidate gates even when multiple are detectable from the same
> spot, at least on this course's roughly-linear geometry. Residual caveat
> (carried forward into Phase 4 item "A4"): tighter-turn course geometries
> where gates sit much closer together angularly could still confuse it —
> that's exactly what Phase 4's multi-gate-visible robustness check is for.
>
> #### Net effect on §1's "three patterns"
> 1. `err_z` always positive — **unaffected**, still the standout
>    sign-consistent bias worth chasing first in Phase 3 (stare test).
> 2. `err_x`/`err_y` sign-flipping gate-to-gate — **unaffected** by either
>    finding (not range- or attribution-related); still an open
>    rotation-chain question for Phase 1.1/Phase 3.
> 3. "Gates 2-5 show range-correlation, gates 0-1 don't" + "gate 0 = 56%" —
>    **fully explained**: gate 0/1's anomalous range/error behavior was the
>    static-hover contamination (constant ~24 m range, low constant error,
>    spanning the whole run) diluting what would otherwise likely be the
>    same range-correlated pattern seen at every other gate. Re-run on the
>    in-flight-only table above and gate 0 (corr ≈ 0.55) and gate 1
>    (corr ≈ 0.98) both look much more like gates 2-5 — the "doesn't fit
>    the pattern" anomaly mostly evaporates once the static samples are
>    removed.
>
> **Bottom line: Phase 0 is closed. The ruler is calibrated.** Use
> `phase0_audit.py` (dedupe by `sim_time_ns`, then optionally filter to
> in-flight) as the standard preprocessing step before computing *any*
> per-gate statistic from `gate_estimates_*.csv` going forward — raw rows
> from this CSV should never be fed directly into an analysis again.

Don't trust another per-gate breakdown until ground-truth attribution is
trustworthy. `matched_gate_id` is currently nearest-MAVLink-centroid-to-
the-CV-estimate — which can silently hand a badly-off estimate to the
*wrong* gate and contaminate every downstream statistic, including the
"error grows with range" pattern in §1.3 (a large, far-away error could
get grabbed by a nearer-but-wrong gate's centroid, manufacturing part of
that correlation). Cross-check it against attribution by `active_gate` /
race-status at the estimate's timestamp — if the two disagree
substantially, treat every number in §1 as provisional until this is
sorted out.

*(Original framing of the Phase-0 task, kept for context — see the ✅
PASS block above for the resolution.)*

### Phase 1 — Mine the data we already have (cheap: zero new flights)

> ## ✅ PHASE 1.1 — PASS (mathematical error decomposition; verified against
> `run_20260607_120608` [stare/yaw_sweep/range_sweep, n=1601 settled CV
> detections with full Phase-2 attitude + raw-`tvec` instrumentation],
> cross-checked against `run_20260607_124556` [oblique_sweep, n=636])
>
> The original framing below has an inherent circularity: "solve for what
> `R_CAM2BODY · tvec` *should* have been" requires *assuming* the rest of
> the rotation chain (`R_b2ned`) is exact — but Phase 3 already proved
> `R_CAM2BODY` is biased, so any such "solve" would just fit one wrong
> assumption to compensate for the other. The two checks below sidestep
> that entirely: (a) is **rotation-invariant** (doesn't touch the rotation
> chain at all), and (b) is a **global least-squares fit** — not a
> "solve" — across 1601 independent samples, so no single sample's
> assumptions dominate.
>
> **(a) Scale check — A1 (detection/PnP-stage), rotation-invariant.**
> Rotations preserve vector length, so `|tvec|` and `|mav − drone_pos|`
> have *exactly* the norms that `R_CAM2BODY·tvec` and
> `R_b2ned·R_CAM2BODY·tvec` would — making `|tvec| / |mav − drone_pos|` a
> **completely assumption-free** probe of the detection+PnP stage alone:
>
> | range bucket | true range | `\|tvec\|/\|true\|` ratio |
> |---|---|---|
> | near | 11.0 m | 0.999 ± 0.017 |
> | mid  | 12.3 m | 0.984 ± 0.022 |
> | far  | 18.4 m | 0.783 ± 0.070 |
>
> `corr(true_range, ratio) = −0.957`. **PnP's range estimate is essentially
> exact at ≤12 m but increasingly *underestimates* true range beyond
> ~15 m** — at 18 m it believes the gate is ~22% closer than it is. This
> *rules out* A1 as originally framed: a fixed wrong `OBJ_PTS`
> half-dimension (or a focal-length/intrinsics error) would produce a
> **constant** ratio offset, not a range-dependent one. The signature
> instead fits a **range-scaled segmentation bias**: the HSV-mask "halo"
> (orange bleed at the boundary) and the 7×7 morphological-close kernel add
> a roughly *fixed pixel-width* inflation to the detected blob — a
> proportionally larger fraction of a small (far) blob than a large (near)
> one, so PnP — trusting `OBJ_PTS`'s size — concludes the camera is closer
> than it truly is, increasingly so as range grows. **Not a wrong
> constant — a range-dependent detection-stage effect** that a tighter HSV
> band / smaller morphological kernel would likely shrink specifically at
> long range (worth re-testing once Phase 1.2 visual audits confirm the
> mechanism directly).
>
> **(b) Angle fit — A11 (rotation-chain-stage), global least-squares.**
> `R_CAM2BODY(t)` is a genuine 1-DOF family (rotation about the fixed
> body-Y axis), so fit its single free parameter `t` by minimizing
> `Σ‖R_CAM2BODY(t)·tvec − R_b2nedᵀ·(mav − drone_pos)‖²` — once across all
> 1601 samples, and again across only the 1075 "scale-clean" samples from
> (a) (range < 14 m, ratio ≈ 1.0) to rule out contamination from (a)'s bias:
>
> | | tilt `t` | RMS body-frame residual |
> |---|---|---|
> | currently coded | **+20.00°** | 10.50 m |
> | global fit (n=1601) | **−21.60°** | 4.67 m |
> | scale-clean-only fit (n=1075) | **−21.85°** | 4.73 m |
> | cross-check fit on independent `oblique_sweep` data (n=636) | **−20.90°** | — |
>
> All three independent fits — two different datasets, two different
> geometries (dead-on/yaw-varying vs. continuously-oblique), two different
> sample subsets — converge to within **~1°** of each other (≈ −21°), and
> the scale-clean subset confirms the fit isn't contaminated by (a)'s bias.
> All land far from the coded +20°, with the **opposite sign**.
>
> *Reconciling with Phase 3's "≈33°" figure*: `stare` found a rock-steady
> `error_z/range ≈ 0.66` and converted it via `arctan(0.66) ≈ 33°` —
> assuming `error_z ≈ range·tan(Δt)`, a relationship that doesn't actually
> match `R_CAM2BODY`'s structure. Worked through properly for `stare`'s
> dead-on geometry (`tvec ≈ (0,0,R)`, level hover ⇒ yaw doesn't touch the
> Z-component): `error_z ≈ (sin(t_coded) − sin(t_true))·range`, so
> `sin(t_true) = sin(20°) − 0.66 = −0.318` ⇒ **`t_true ≈ −18.5°`** — within
> ~3° of all three fits above. The "≈33°" *ratio* was exactly right and
> reproducible; `arctan` was simply the wrong **geometric transform** to
> apply to it. *(The same lesson the oblique_sweep bug taught in a
> different guise: when the matrix's own structure gives an exact
> closed-form relationship, use it — a plausible shortcut (`arctan` of a
> ratio; a circular-mean bearing) can look clean while quietly encoding the
> wrong geometry.)*
>
> **(c) Validation.** Re-running
> `est = drone_pos + R_b2ned · R_CAM2BODY(t) · tvec` with `t = −21.6°`
> instead of the coded `+20°` — *same matrix, one constant changes*:
>
> | | mean `\|err\|` | mean `error_z` |
> |---|---|---|
> | current code (`t=+20°`) | 10.24 m | **+8.16 m** |
> | fitted (`t=−21.6°`) | 3.41 m | **+0.016 m** |
>
> `error_z` collapses from +8.16 m to essentially **zero**; overall mean
> position error drops **66.7%** (10.24 m → 3.41 m) — from changing one
> sign-and-magnitude constant.
>
> **A11 — closed, with a concrete fix.** Set `_t = np.radians(-21.6)` at
> `pose_estimator.py:51` (matrix structure unchanged — only the constant's
> sign and magnitude change). This single edit removes ~⅔ of the
> systematic CV error and effectively zeroes the persistent altitude bias
> visible since §1. The remaining ~3.4 m mean residual is attributable to
> (i) the range-scaled detection bias from (a) — worst at long range,
> exactly where `range_sweep`'s `error_x` super-linear growth lives — and
> (ii) the oblique-angle detectability cliff (Phase 3 §4, A6/A5/A7) once
> the gate is significantly off-centre.
>
> > ✅ **ACTION ITEM — applied.** The fix above (`_t = np.radians(20.0)` →
> > `np.radians(-21.6)` at `pose_estimator.py:51`) was deliberately **held
> > back** until every phase had been scored against one consistent,
> > known-biased baseline — Phase 1.2 (the last one needing fresh
> > instrumentation) closed cleanly with the old `+20°` still in place, so
> > nothing downstream needed re-baselining. With the full investigation
> > now done, **the fix has been applied** (matrix structure unchanged —
> > only `_t`'s sign and magnitude changed). One more wrinkle worth noting
> > while applying it: in `R_CAM2BODY`'s actual coded sign convention,
> > `t ≈ −21.6°` corresponds to an **upward** ~21.6° optical-axis tilt —
> > i.e. it lands close to spec §3.8's original "20° upward" claim (just
> > ~1.6° larger), not in the "opposite, effectively-downward" direction the
> > old `+20°` assumed. That old assumption traced back to an angle
> > recovered via `arctan` of an error/range ratio — exactly the shortcut
> > this write-up's "Reconciling with Phase 3's ≈33° figure" section above
> > already showed uses the wrong geometric transform for this matrix.
> > **Net effect, expected on the next flight**: `error_z` should collapse
> > to ~0 and mean CV position error should drop ~66.7% (10.24 m → 3.41 m,
> > per the validation table above) — re-run and re-score against the
> > corrected pipeline to confirm empirically.
>
> **A1 — re-scoped, not closed.** The *fixed wrong-constant* framing is
> ruled out; the *range-scaled segmentation bias* in (a) is the live
> hypothesis. A Phase 1.2 visual audit at long range would confirm it
> directly: the detected quad should visibly "fatten" relative to the
> gate's true outer edge as the gate shrinks in frame.
>
> **Bottom line: Phase 1.1 is closed.** Both halves of the original
> decomposition — detection/PnP-stage and rotation-chain-stage — now have
> clean, mutually-corroborating, *actionable* numbers, obtained without
> ever "solving for the unknown" (the circularity the original framing had
> never needed solving in the first place).

> ## ✅ PHASE 1.2 — PASS (visual + quantitative audit of A2/A5/A7, run
> `run_20260607_184410` — full 6-gate race finish under `GUIDANCE =
> 'MAVLINK'`, 87.3 s, 2337 deduped CV detections with rvec/corner
> instrumentation; see `phase1_2_audit.py`)
>
> This was the first full-course completion with the extended Phase-2
> logging active end to end (`active_gate` progressed cleanly 0→1→2→3→4→5→6
> at t = 16.0 / 27.3 / 40.6 / 57.4 / 68.6 / 79.7 s) — exactly the spread of
> ranges, angles, and gates Phase 1.2 needed and the single-gate, scripted
> calibration flights couldn't provide.
>
> ### A5 — filter calibration: mostly well-tuned, with a thin marginal tail
> 98.6% of all 38,908 logged frame-attempts pass the aspect/area filter
> cleanly. Of the 1.4% (537) rejected by the `0.3 < w/h < 3.0` aspect-ratio
> band, **~58% sit just outside it** ("marginal" near-misses, not wild
> outliers) — consistent with steep oblique views genuinely compressing the
> silhouette past the band's edge, exactly the failure mode A5 flagged as a
> possibility. **Verdict: the filter is basically right; a small loosening
> (e.g. 0.25–3.5) would likely recover a fraction of these without admitting
> non-gate blobs** — low priority given the volume (1.4% of all attempts).
>
> ### A7 — polygon convergence: clean; a fallback would help only at the margin
> 98.6% of candidate contours reduce to exactly 4 points (87.9% at the
> *finest* epsilon tried — minimal simplification needed, i.e. the true
> silhouette already looks like a clean quad most of the time, not that
> brute-force simplification is forcing a fit). Of the 1.4% that never
> converge, **97% are within ±1 point of 4** (mostly 3- or 5-point hulls) —
> exactly the band a bounding-rect / vertex-drop fallback (known gap #8)
> would likely recover. **Verdict: convergence is not the bottleneck it
> could have been; a fallback is worthwhile but low-yield.**
>
> ### A1 cross-confirmation — range-scaled "fattening," now seen directly [target (b)]
> Phase 1.1 *inferred* a range-scaled segmentation bias from a rotation-
> invariant scale check; this audit confirms the *mechanism* visually and
> independently. Across 768 long-range (≥15 m) samples: `corr(range,
> apparent px-size) = −0.72`, `corr(range, |error_z|) = +0.60` — a clean
> monotonic relationship. Bucketed, mean `|error_z|` runs **~2.0 m at <8 m**
> vs. **~14.2 m at ≥25 m** — a sevenfold growth that tracks range, not a
> fixed offset. The overlay frame `b_long_range_1780883100274728300.jpg`
> (gate 3 @ 27.7 m, `err_mag = 18.87 m`) shows it directly: the small green
> *detected* quad and the larger magenta `reproject_corners()` outline
> visibly diverge — the fitted pose is "fatter" than what was actually
> segmented, exactly the silhouette-inflation-at-range signature A1
> (re-scoped in Phase 1.1) predicted. **Target (b) — closed, with a picture
> to match the numbers.**
>
> ### A8 cross-check — this flight's "impossible" detections are A1's tail, not Phase 4's lock-on [target (c)]
> Target (c) asked to look inside an orphan-detection burst and see what
> `gate_detector` mistakes for a gate. This course's altitude profile is the
> **opposite** of Phase 4's (gate-centre z spans `[-1.4, +24.6]` —
> *descending* — vs. Phase 4's `[+0.03, -26]`, climbing), so Phase 4's
> literal `est_z > +10 ⇒ impossible` bound does **not** transfer (naively
> applying it flagged a misleading 74%). Recomputing the envelope from
> *this* run's own gate map (`[min(centre_z) − 5, max(centre_z) + 5]` =
> `[-6.5, +29.5]`) gives a defensible **707 / 2337 (30.3%)** flagged — still
> a large fraction, but a fundamentally different *kind* of large fraction
> once you look inside it. Discriminating "fixed feature" (Phase 4's
> lock-on signature; `|est_drift|/|drone_drift| ≈ 0`) from "range-bias tail"
> (A1's signature; ratio ≈ 1, `est` tracks the drone / converges as range
> closes) across this flight's three sustained "impossible" bursts gives
> ratios **0.52 / 0.93 / 0.98** — all "tracking," none "fixed." All three
> are real, correctly-`matched_gate_id`'d detections of gates **3, 4, and 5
> at long range**, simply mis-ranged badly enough by PnP to land outside
> the (correct, tight) envelope — A1's mechanism's tail, not a non-gate
> lock-on. **Target (c) — answered, but the answer *revises* rather than
> reproduces the premise**: this run never shows Phase 4's "fixed feature"
> signature at all (good news for A8 on *this* course/version), and it
> independently reinforces Phase 4's "trivial, zero-risk z-bound sanity
> filter" recommendation — that one filter would catch *both* mechanisms'
> worst output, from two unrelated root causes.
>
> ### A2 — visual audit: both predicted regimes observed directly
> Six annotated overlay frames (`<run_dir>/phase1_2_frames/` — green =
> detected corners, magenta = `reproject_corners(tvec, rvec)`, text overlay
> = gate id / range / oblique angle / `err_mag` / `err_z`) span the gate
> sequence and the full range spread. Inspection shows exactly the
> qualitative split Phase 1.2 was designed to surface: close-range frames
> show clean green/magenta overlap and low `err_mag` (the "PnP is
> essentially exact" regime Phase 1.1 quantified at ≤12 m), while long-range
> frames show the dramatic fattening described above. **No frame showed a
> front/back-face ambiguity** (A2's original "which face does the contour
> trace" framing) — the *outer* boundary is what's consistently traced, at
> every range and angle sampled; the dominant visible-error source is
> *range-scaled inflation* of that boundary, not a face-selection problem.
> **A2 — confirmed.**
>
> **Caveat — target (a), the +15°→+25° oblique transition, remains visually
> unsampled.** This race's `matched_gate_id`-relative oblique-angle
> distribution turned out **bimodal** (clusters at 0–15° and ≥30°, a gap
> across 15–25°, `std = 82.3°` — far wider than Phase 3's clean ±40°
> controlled sweep), so *zero* frames landed in the target window. This is
> much more likely an artifact of how a single race pass naturally samples
> approach geometry (you fly *through* a transition, you don't loiter in
> it) than a rediscovery of Phase 3's detectability cliff — but it does mean
> target (a) stays **formally unconfirmed by direct visual evidence**, even
> though Phase 3 already quantified the cliff numerically. A short scripted
> oblique approach (the very `CALIBRATE` / `oblique` leg this run's
> `controller.py` had been left in, before being switched back to
> `MAVLINK`) would close this specific gap if it's ever worth a dedicated
> flight.
>
> **Bottom line: Phase 1.2 is closed — and with it, every phase on the
> list.** Targets (b) and (c) are answered with both numbers and pictures;
> (c)'s answer *revises* Phase 4's working hypothesis for this course rather
> than reproducing it — more interesting and more precise than a simple
> confirmation would have been. Target (a) is the one loose thread (a
> single race pass doesn't sample the transition window), but it's a
> *visual-evidence* gap on an already-numerically-closed question, not an
> open unknown. A2/A5/A7 all score clean-to-good.
>

*(Original framing of this sub-task, kept for context — see the ✅ PASS
block above for the resolution.)*

1. **Mathematical error decomposition.** Since
   `est = drone_pos + R_b2ned · R_CAM2BODY · tvec`, and we know `mav`
   (ground truth) and `drone_pos`, we can solve for what
   `R_CAM2BODY · tvec` *should* have been
   (`R_b2ned⁻¹ · (mav − drone_pos)`) and compare it against what the
   pipeline actually computed. That cleanly splits error into
   "detection/PnP-stage" vs. "rotation-chain-stage" — but it requires
   knowing the attitude at each estimate's instant, which isn't currently
   logged (see Phase 2).
2. **Visual frame audits with `reproject_corners()`.** Sample frames
   spanning the full range of distances/angles seen in a run; overlay (i)
   the detected corners, (ii) the PnP pose reprojected back onto the image,
   and (iii) the gate's visible edges. A single glance then answers: is
   the detector tracing the *outer* or *inner* boundary (A1)? Is corner
   ordering right at oblique angles (A6)? Is `approxPolyDP` converging on
   a sane quad (A7)? Is the HSV mask clean or contaminated (A8)? — This is
   the single highest value-to-effort item on this whole list: the tool
   to do it is already written and sitting unused in the codebase.
3. **Detection-rate audit.** Work out *why* gate 0 alone accounts for 56%
   of all detections (§1.3). Genuinely longer visibility window (course
   geometry / approach geometry), or a higher detection-*failure* rate
   near the other gates? The answer points either at A1/A11 (geometry/
   pose) or at A5/A7/A8 (detection robustness).

### Phase 2 — Minimal added instrumentation (unblocks Phase 1.1 and Phase 3)
Log `roll, pitch, yaw` (and ideally the raw, pre-rotation `tvec`)
alongside each row in `gate_estimates_*.csv`. The CSV currently has
`est`/`drone_pos`/`mav` but not the attitude that produced the rotation —
without it, the decomposition in Phase 1.1 can't be done from existing
logs, and the controlled tests in Phase 3 can't be scored precisely either.

### Phase 3 — Controlled empirical tests (one variable at a time)

> ## ✅ PHASE 3 — PASS (verified against `run_20260607_120608` [stare/yaw_sweep/
> range_sweep] and `run_20260607_124556` [oblique_sweep]; methodology in
> `calibration_plan.py` + `phase3_audit.py`)
>
> All four maneuvers below were flown autonomously — `GUIDANCE = 'CALIBRATE'`,
> `controller._handle_calibrate`, scripted by `calibration_plan.build_calibration_plan` —
> and scored entirely offline from `calibration_log.csv` + `gate_estimates_*.csv`
> (alignment-free: each CV row is matched to its leg by nearest-neighbour
> *state-space* (position, yaw), sidestepping the A12 cross-clock problem
> entirely — see `phase3_audit.py`'s docstring). All four isolated exactly the
> variable they were designed to isolate and produced clean, low-noise,
> mutually-corroborating results.
>
> #### stare — A11 (camera-tilt sign): bias CONFIRMED and QUANTIFIED
> Hovering dead-on collapses `R_b2ned ≈ I`, leaving `est ≈ drone_pos +
> R_CAM2BODY · tvec` — i.e. isolates `R_CAM2BODY` from every attitude effect.
> Result (n=240, ~12 m range): `error_z` mean **+8.14 m**, std **0.153** —
> a |mean|/std ratio of **53**, about as deterministic a signal as this
> pipeline produces (`error_x` and `error_y` are similarly tight: ratios 31
> and 11). This alone confirms the persistent positive `err_z` from §1 is a
> **rotation-chain bias, not detection noise**.
>
> The magnitude closes the loop on §5.2.a's tilt-sign mystery: `error_z /
> range ≈ 0.66` here — and that *exact* ratio reappears, independently,
> across every `range_sweep` bucket (10.1 m → 0.659; 13.1 m → 0.668; 15.2 m
> → 0.658/0.66 net). A **constant error-to-range ratio is the unmistakable
> signature of a fixed angular bias** (a translational/PnP error would not
> scale linearly with range). `arctan(0.66) ≈ 33°` — meaning the camera's
> *true* effective tilt differs from the rotation chain's encoded **20°**
> (`pose_estimator.py` `_t = np.radians(20.0)`, §3.8) by roughly that amount.
> **A11 verdict: the spec's 20° figure (in either sign) does not match this
> build's flight data — the empirically-correct tilt-chain correction is
> closer to ≈33°.** This is now a precise, reproducible re-calibration target,
> not an open question.
>
> #### yaw_sweep — A13 (rotation-chain / yaw coupling): CONFIRMED
> Same hover spot, sweeping yaw ±30° through boresight — position constant,
> so any error-vs-yaw correlation is pure rotation-chain mistracking. Result
> (n=481): `corr(yaw, error_y) = +0.86` (others near zero: `error_x` −0.05,
> `error_z` +0.10). **A13 verdict: confirmed** — `err_y` tracks where the
> camera *points*, not just where the gate *is*, exactly the
> rotation-chain-driven mechanism §1.2 hypothesized for the gate-to-gate
> sign-flipping `err_x`/`err_y` pattern (settles that question in favour of
> A13 over A6 — corner ordering is not the driver).
>
> #### range_sweep — A1/A9/A10 (range-dependent error growth): CONFIRMED & decomposed
> Controlled far→near approach (22 m → 5 m) on the boresight line. Result
> (n=880, range 7.6–16.3 m): `corr(range, error_mag) = +0.96`, with a clean
> monotonic bucket progression — near (10.1 m) → 7.16 m mean error, mid
> (13.1 m) → 10.81 m, far (15.2 m) → 13.21 m. Splitting by component shows
> **two distinct mechanisms layered together**: `error_z` grows
> near-linearly with range (the same fixed-angular-bias signature as
> `stare`, ratio ≈0.66 throughout — i.e. this *is* the A11 bias, just
> observed at varying range), while `error_x` grows **super-linearly**
> (+2.57 → +6.20 → +8.58 m across the same buckets — far steeper than the Z
> trend), pointing at a *second*, distinct effect — most likely a
> focal-length/perspective mismatch (A9/A10 empirical re-check) layered on
> top of the angular bias, not a single root cause. **A1/A9/A10 verdict:
> range-dependent growth is real, reproducible, and decomposes into (a) the
> already-quantified A11 angular bias (Z-axis) plus (b) a separate,
> steeper-than-linear X-axis effect that the spec's "confirmed" intrinsics
> don't explain — worth a focused empirical re-derivation in a follow-up.**
>
> #### oblique_sweep — A6/A5/A7 (oblique-angle robustness): CONFIRMED + a new discovery
> *(Design note: the first attempt at this leg used four discrete
> "fly-to-and-hold" stations at ±20°/±40°. Every one of them entered a
> stable orbit instead of converging — textbook "central-force controller
> can't kill tangential momentum." Rather than patch the controller, the
> leg was redesigned as a single continuous **−40°→+40° traverse**: a linear
> interpolation between the two endpoint stations that, by construction,
> passes exactly through the dead-on line at its midpoint, sweeping the full
> angle range in one motion the controller already flies flawlessly — this
> is what finally produced clean data, on the first live re-run.)*
>
> Result (n=636 settled detections, scored by `score_oblique_sweep` against
> an **exact geometric** dead-on reference bearing — see implementation
> note below): `corr(|oblique angle|, error_mag) = +0.62`, with a clean
> monotonic progression — axis (5.6°) → 7.42 m mean error, mid (16.5°) →
> 8.18 m, oblique (29.3°) → 16.12 m (>2× growth from near-axis to oblique).
> **That alone confirms A6/A5/A7's core worry: PnP/detection accuracy
> degrades measurably as the gate's apparent shape skews at oblique angles.**
>
> But the standout finding is sharper than "accuracy degrades": it's a
> **hard, asymmetric CV-detectability cliff**. Across *every one* of the
> 958 genuinely-distinct CV detections logged during the entire flight (not
> just the scored subset), **not one exceeds +19.7° of oblique angle** —
> while detections continue cleanly out to **−39.5°** on the mirror side.
> This is not a flight-tracking artefact: `calibration_log.csv` shows the
> drone smoothly tracked the planned path out to **+31.2°** the whole time
> (`pos_err` ≈ 2 m, `yaw_err` ≈ 0.8°, no instability) — the CV pipeline
> simply *stops producing detections* past ≈ +20° on one side while
> continuing to work fine to nearly −40° on the other. **A6/A7 verdict:
> something in `gate_detector` (the `_order_corners` sum/diff heuristic, or
> `approxPolyDP`'s no-fallback convergence — known gap #8) behaves
> *asymmetrically* between left- and right-oblique views** — almost
> certainly something direction-dependent in the gate's geometry, markings,
> or lighting as seen from the camera. This now hands Phase 1.2
> (`reproject_corners` visual audit) an exact, high-value target: pull
> frames from the +15°→+25° transition and watch exactly what the detector
> sees as it loses the gate.
>
> *(Implementation note, also a small methodology lesson: scoring this leg
> requires recovering each detection's oblique angle from logged
> drone/gate positions relative to a "dead-on" reference bearing.
> `phase3_audit.py` originally derived that reference as a circular mean of
> the `stare` leg's bearings — reasonable, but when a leg-index-hardcoding
> bug (below) fed it the `oblique_sweep` leg's own bearings instead, the
> self-referential mean was close enough to *look* plausible (a smooth,
> roughly-symmetric ±30° range) without being exact — it was off by enough
> to materially shift the recovered angle distribution and mask the
> detectability cliff above. The fix replaces it with an **exact geometric**
> reference — the same `bearing_from_gate` calculation
> `calibration_plan.build_calibration_plan` uses to lay the leg out,
> computed purely from the drone's pre-CALIBRATE position and the gate's
> MAVLink centre (both already logged, neither carrying CV noise). Verified
> to reproduce the leg's planned ±40° endpoints to within 0.3°. Lesson:
> a result that "looks clean" from an approximate reference can still be
> silently wrong in a way that hides the most interesting finding —
> prefer exact geometric ground truth over statistical approximations
> whenever the inputs to compute it exactly are already on hand.)*
>
> #### A bug this phase exposed (now fixed)
> `phase3_audit.py main()` originally hard-coded `by_leg.get(0/1/2, [])` →
> `score_stare`/`score_yaw_sweep`/`score_range_sweep`, assuming leg-index 0
> is always `stare`. True for the full 7-leg plan; false for any trimmed
> plan (e.g. `CAL_LEG_FILTER = ('oblique',)`, used to re-fly just the failing
> leg without re-flying the ones that already passed) — there, `oblique_sweep`
> occupies index 0, and the hard-coded call silently scored its data as if
> it were `stare`, printing a "LEG: stare" block whose numbers (`error_y`
> std=7.3, `error_z` std=4.9 — 16–50× larger than genuine stare's 0.03/0.15)
> were complete fiction. **Fixed**: legs are now matched **by name**
> (mirroring the pattern `score_oblique_sweep`'s call site already used),
> so the scorer is correct regardless of which legs a future trimmed plan
> flies, in whatever order.
>
> #### Net effect — what Phase 3 resolved
> 1. **A11**: the persistent positive `err_z` from §1 is conclusively a
>    rotation-chain bias (not noise, not detection-stage), and is now
>    *quantified*: a fixed angular discrepancy of ≈33° between the encoded
>    20° tilt and the camera's true effective mounting angle.
> 2. **A13**: the gate-to-gate sign-flipping `err_x`/`err_y` pattern from §1
>    is rotation-chain/yaw-coupling-driven (`corr(yaw, err_y)=+0.86`), not
>    corner-ordering-driven — settles the §4.2 open question in A13's favour.
> 3. **A1/A9/A10**: the range-correlated blowup at gates 2–5 is real and
>    reproducible (`corr=+0.96`), and decomposes into the already-quantified
>    A11 angular bias (Z-axis) plus a separate, steeper-than-linear X-axis
>    effect (likely focal-length/perspective) that the "confirmed by spec"
>    intrinsics don't fully explain.
> 4. **A6/A5/A7**: oblique-angle accuracy genuinely degrades (`corr=+0.62`,
>    >2× growth axis→oblique) — and, more importantly, there's a sharp,
>    *asymmetric* CV-detectability cliff (clean detections to −39.5°, none
>    past +19.7°) that nothing in §1's data could have surfaced, since no
>    logged flight ever held a controlled, continuous oblique sweep before.
>
> **Bottom line: Phase 3 is closed.** All four controlled tests isolated
> precisely the variables they targeted and returned clean, low-noise,
> mutually-consistent, *actionable* numbers — several of which (the ≈33°
> tilt quantification, the asymmetric detectability cliff) are genuinely new
> findings that §1's uncontrolled in-race data could never have produced.
> Phase 1.1 (error decomposition) and Phase 1.2 (`reproject_corners` visual
> audit) now have precise, quantified targets to chase rather than open
> questions — see them re-scoped accordingly below.

*(Original framing of the Phase-3 task, kept for context — see the ✅ PASS
block above for the resolution.)*

Because the course is deterministic, these are exactly repeatable:
- **Stare test** — hover level (roll≈pitch≈yaw≈0) at a fixed, known range
  directly in front of a gate. This collapses `R_b2ned ≈ I`, leaving
  `est ≈ drone_pos + R_CAM2BODY · tvec` — i.e. isolates `R_CAM2BODY`
  from all attitude effects. **Directly resolves the A11 tilt-sign
  contradiction (§5.2.a)**: solve for the rotation that actually maps
  `tvec` onto `(mav − drone_pos)` and read off whether reality says +20°,
  −20°, or something else.
- **Yaw-sweep test** — from that same hover spot, sweep yaw through a
  range while still viewing the gate, and watch how `err_x`/`err_y`
  respond. If they track yaw predictably, that confirms the
  sign-flipping-x/y pattern (§1.2) is rotation-chain-driven (A13) rather
  than corner-ordering-driven (A6) — or the reverse.
- **Range-sweep test** — controlled straight-line approach (far → near)
  on a single, cleanly-tracked gate, logging error vs. range. Confirms or
  refutes whether the range-correlated blowup at gates 2–5 (§1.3) is a
  genuine geometric/optical effect (A1 scale, or A9/A10 — note §5.2.a
  already proved the *sim doesn't necessarily match its own latest spec*,
  so "confirmed by spec" ≠ "confirmed empirically in this build") versus
  an artifact that Phase 0 should have already mostly removed.
- **Oblique-angle test** — approach a gate off-axis at angles representative
  of the course's actual turns. Stress-tests corner-ordering (A6) and
  detection robustness (A5, A7) at the geometric extremes the live course
  will actually present, rather than whatever angles happened to occur in
  one logged run.

### Phase 4 — Robustness characterization (matters for "rock solid"; lower priority for explaining *today's* error numbers)

> ## ✅ PHASE 4 — PASS (mined entirely from the existing full-course race log
> `run_20260607_012243` — 6 gates, 2550 deduped CV detections, no new flight
> needed; see `phase4_audit.py`)
>
> Both A4 and A8 turn out to be answerable with a single insight: the
> simulator already tells us, every tick, which gate the controller is
> *currently chasing* (`flight_log_*.csv: active_gate`) — a sharper
> ground-truth for "what should the camera reasonably be seeing right now"
> than raw nearest-gate-by-distance (which doesn't know which way the
> camera points). Time-aligning that onto each CV detection turns both
> "probe for a failure mode" questions into simple counting exercises.
>
> ### A4 — largest-contour / multi-gate ambiguity: a clean PASS
> Of 2172 valid in-race detections, only **34 (1.6%)** report a
> `matched_gate_id` different from the simulator-confirmed `active_gate` —
> and **none of those 34 are confident** (mean `error_mag` for mismatches
> is **20.9 m**, vs. 8.6 m for matches; every one is a noisy, low-quality
> detection of the *correct* gate that happened to nearest-match a
> different gate's MAVLink position by coincidence). Even restricting to
> the 243 frames (11.2%) where ≥2 gates were geometrically within Phase 3's
> empirical ~22 m detection envelope, the **confident-mismatch rate is
> exactly 0.0%**. CV never once confidently locked onto the wrong gate.
> **Verdict: A4 is a non-issue for this course's geometry** — a single-file
> corridor where gates essentially never visually overlap in frame. (That's
> itself the finding, and a caveat worth keeping: this characterizes *this
> course*, not the algorithm in the abstract — a tighter, twistier course
> with gates visible through one another could still expose it. The
> instrument now exists to check that the moment such a course shows up.)
>
> ### A8 — HSV universality / false positives: found one, and it's bigger than expected
> 23.7% of detections (515/2172) claim a `matched_gate_id` whose true
> position is **beyond the ~22 m detection envelope** — geometrically
> implausible on its face. Of those, **323 (≈15% of every CV detection
> logged this entire flight)** have an `est` that sits more than 5 m from
> *every* real gate centre — true orphans, corresponding to nothing in the
> gate map at all.
>
> These orphans are not noise — **97.5% sit immediately adjacent to another
> orphan in frame sequence**, forming **9 distinct, sustained bursts** (the
> longest: **132 consecutive frames, 6.4 seconds**, at ~25 Hz). And each
> burst is *tightly clustered in 3-D* — e.g. burst 1's `est` centroid sits
> at `(-103.6, -14.6, 38.8)` with a standard deviation of just **(5.6, 4.8,
> 6.5) m**, while the *drone* moves **17 m** through that same 6.4-second
> window. **A fixed `est` while the observing platform moves substantially
> is the unmistakable signature of the pipeline locking onto and
> consistently re-estimating the pose of one single fixed real-world
> feature** — exactly the "non-gate orange object" failure mode A8 was
> written to anticipate, demonstrated concretely, in VQ1 — supposedly the
> *friendliest* environment for colour-keyed segmentation (§5.2.e).
>
> The estimated poses aren't just *wrong* — they're **physically
> impossible**: every real gate and the entire logged flight envelope sit
> between `z = +0.03` and `z = -26` m (NED-down: at-or-above the arm
> point), yet **88% of orphan detections report `est_z > +10 m`** — i.e.
> *more than 10 m underground* — with a median of **+34.7 m** (over 30 m
> below the ground the drone never gets near). There is no real-world
> object that could produce this; it's the rotation chain confidently
> placing a "gate" tens of metres into solid earth.
>
> **A8 — confirmed, characterized, and quantified**: roughly **1 in 7 of
> every CV detection this pipeline logs is physically-impossible garbage**,
> arriving in multi-second bursts triggered by at least one real,
> repeatable environmental feature along the course — not edge-case noise,
> a structural property of the current detector + no-outlier-rejection
> combination (known gap #3). `gate_verifier.py`'s headline accuracy
> numbers (mean/median/max/std of `‖error‖`) **do not exclude these** —
> they're averaged in alongside genuine detections, meaning the "CV
> accuracy" figure this whole investigation has been chasing is itself
> diluted by a ~15% contamination rate from detections of *nothing at all*.
>
> **Did it matter operationally, this run?** No — by luck, not design: the
> controller **never once entered CV-targeting mode this entire flight**
> (`target_mode` shows `CV(age=...)` on **0 of 7327** controller ticks; the
> whole course was flown on `spline` + waypoint fallback). So none of these
> bursts ever reached `cv_gate_pos` as a live target. But the FLY loop's
> documented target priority puts "fresh CV estimate" *first* — the
> architecture's intent is for CV to be the *primary* guidance source, with
> the spline as fallback. A future tuning pass, or a course/lighting
> condition that makes CV estimates "fresh" and "in front" more often,
> would hand a 6-second burst of `est` positions 50–90 m from any real gate
> straight to the controller as its #1-priority target. That's not a
> hypothetical — it's exactly the scenario the architecture is designed to
> reach for.
>
> **A concrete, low-risk fix**: a trivial sanity filter — reject any
> `cv_gate_pos` whose `z` falls outside the flight envelope (e.g.
> `[-30, +5]` m, derivable directly from the gate map + arm-point origin,
> no tuning required) — would eliminate **all 323** orphan detections
> outright, with **zero risk of rejecting a real gate** (no gate or flight
> position in this course's data — or plausibly any course's, given the
> physics — comes anywhere near `z = +30` m). This is the single
> highest-leverage, lowest-risk addition `gate_verifier.py` could get: it
> doesn't require fixing *why* the false trigger happens, just refusing to
> propagate its physically-impossible output downstream — directly closing
> known gap #3 for the most egregious failure class.
>
> **Bottom line: Phase 4 is closed.** A4 is verified clean (for this
> course's geometry, with the caveat noted). A8 went from "predicted to be
> a problem in VQ2" to "demonstrated, quantified, and traced to a specific,
> fixable structural gap *already in VQ1*" — arguably the most consequential
> finding of the whole investigation, since it affects every single number
> `gate_verifier.py` and Phases 1.1/3 have reported (all computed over a
> detection stream that is ~15% physically-impossible noise).
>

*(Original framing of this phase, kept for context — see the ✅ PASS block
above for the resolution.)*

- **A8 (HSV universality)** — the docs already predict this fails outside
  VQ1 (§5.2.e: "visual guidance aids will be off" + reduced
  signal-to-noise in VQ2/the 3D-scanned environment). The goal here isn't
  "verify" so much as **characterize exactly where the current bands
  break** (lighting extremes, distance-induced desaturation, non-gate
  orange objects/"non-gated objects" per `updates.md`) — so the failure
  envelope, and the requirements for whatever replaces color-keyed
  segmentation, are known in advance rather than discovered live.
- **A4 (largest-contour heuristic)** — specifically probe scenarios where
  more than one gate could plausibly be visible at once (tight turns,
  looking through the active gate toward the next one).

### Suggested execution order

> **Status update — final order was 0 → 2 → 3 → 1.1 → 4 → 1.2 — every
> phase now ✅ PASS:** Phase 2's logging turned out to be a hard prerequisite for Phase
> 3's scoring (you can't score a controlled maneuver against attitude/raw-
> `tvec` data that isn't being logged), so it was pulled forward and done
> alongside the calibration-flight build-out — and that same logging is what
> then made Phase 1.1's decomposition possible straight from the
> calibration-flight logs, no new flight needed. Phase 4 turned out to be
> *equally* mineable from existing logs (the race run already encodes
> "what gate should the camera be seeing right now" via `active_gate`, and
> Phase 3 had already established the detection-range envelope needed to
> spot geometrically-implausible detections) — so it was pulled forward too,
> ahead of Phase 1.2, which turned out to be the *only* phase that genuinely
> required new instrumentation and a fresh flight — and, once that flight
> landed (`run_20260607_184410`, full 6-gate finish), closed cleanly too.
> That cascading "mine what's already loggable first" choice
> paid off repeatedly — Phase 3 produced quantified answers no amount of
> in-race mining could have (`corr(yaw,err_y)=+0.86`, `corr(range,err)=+0.96`,
> an asymmetric detectability cliff), Phase 1.1 turned those into one
> concrete code fix (`_t = np.radians(-21.6)`, ~⅔ error collapse), and Phase
> 4 — working purely off the *original, pre-calibration* race log — surfaced
> what may be the single most consequential finding of the whole
> investigation: **~15% of every CV detection this pipeline has ever logged
> is physically-impossible noise**, silently baked into every accuracy number
> Phases 1.1 and 3 reported. Phase 1.2 then closed the loop on all three
> targets those upstream phases handed it — confirming (b) the range-scaled
> "fattening" mechanism with both numbers and pictures, *revising* (c) Phase
> 4's orphan-burst hypothesis for this course (range-bias tails on real
> gates, not a fixed-feature lock-on — though the same proposed filter
> catches both), and leaving only (a) the oblique-transition visual check as
> a thin, non-blocking loose end. **Every phase on the list is now ✅ PASS.**

1. ~~**Phase 0** (A14)~~ — ✅ **PASS**, see write-up above.
2. ~~**Phase 2** (logging) → **Phase 3** (stare/yaw/range/oblique)~~ — ✅
   **PASS** (both), see write-up above. All four A11/A13/A1/A9/A10/A6/A5/A7
   threads are now resolved with quantified, reproducible numbers.
3. ~~**Phase 1.1** (mathematical error decomposition)~~ — ✅ **PASS**, see
   write-up above. Three independent fits across two datasets converged on
   `t ≈ −21°` (not Phase 3's "≈33°" — that figure's *ratio* was right but its
   `arctan` conversion used the wrong geometric transform); substituting it
   collapses `error_z` to ~0 and mean position error by 66.7%. A1 was
   re-scoped from "wrong constant" to "range-scaled segmentation bias," with
   a Phase 1.2 visual audit as the natural next confirmation.
4. ~~**Phase 4** (A8, A4)~~ — ✅ **PASS**, see write-up above. A4: clean (0%
   confident multi-gate confusion, even when ≥2 gates are geometrically in
   range — a property of this course's corridor geometry). A8: confirmed and
   quantified far beyond the original "characterize where it breaks" framing
   — **~15% of every CV detection logged this entire flight is physically-
   impossible** (`est_z` tens of metres underground), arriving in long,
   tightly-clustered bursts (up to 132 consecutive frames) that are the
   signature of the pipeline locking onto one fixed real-world feature and
   mis-estimating its pose as a gate — *in VQ1*, the friendliest possible
   environment. A trivial z-bound sanity filter on `cv_gate_pos` would
   eliminate all of it at zero risk to genuine detections.
5. ~~**Phase 1.2** (`reproject_corners` visual audit)~~ — ✅ **PASS**, see
   write-up above (`run_20260607_184410`, full 6-gate finish, 2337 deduped
   detections). Of its three inherited targets: **(b)** long-range
   "fattening" is directly confirmed in both numbers (`corr(range,
   |error_z|) = +0.60`, mean `|error_z|` ~2.0 m → ~14.2 m from <8 m to
   ≥25 m) and a picture (overlay `b_long_range_1780883100274728300.jpg`,
   gate 3 @ 27.7 m, `err_mag = 18.87 m`); **(c)** the orphan-burst mechanism
   is answered but *revised* — this course's three "impossible" bursts are
   all A1 range-bias tails on real, correctly-`matched_gate_id`'d gates
   (drift ratios 0.52/0.93/0.98 — "tracking," not Phase 4's "fixed feature"
   lock-on), independently reinforcing the case for one shared z-bound
   sanity filter from a second root cause; **(a)** the +15°→+25° oblique
   transition stayed visually unsampled (this race's angle distribution is
   bimodal with a gap there) — a loose end on an already-closed numerical
   question, not an open unknown. A2 confirmed (outer boundary traced
   consistently, no front/back-face ambiguity); A5 (98.6% filter pass rate)
   and A7 (98.6% polygon convergence) both score clean-to-good with only
   thin marginal tails that a small loosening / fallback would mop up.
