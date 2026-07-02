# Task Brief: Co-linear Gate Merging Corrupts CV Pose Estimates

## Problem statement

A new CV failure mode was identified that **is not covered by existing audits A4 or
A8** (see `PERC.md`), and is dangerous precisely because it produces detections that
look superficially valid and pass every existing filter.

**The mechanism:**

When the drone is approaching gate *N* and the next gate, *N+1*, is visible *through*
gate *N*'s opening (because the two gates are roughly co-linear from the drone's
viewpoint — exactly the geometry of this course's gate-to-gate transitions), the HSV
orange mask (`_orange_mask_from_hsv` in `gate_detector.py`) picks up orange pixels from
**both** gates in the same frame. The 7×7 morphological close operation can then bridge
the gap between the two separate blobs into a single merged contour.

`detect_gate` selects the **largest** contour and runs `approxPolyDP` on its convex
hull (`_four_corners`). The convex hull of the merged blob is *not* the convex hull of
either individual gate — it's an enlarged, distorted quadrilateral that partially spans
both gates. If `approxPolyDP` happens to converge to 4 points on this merged shape
(plausible, since two roughly-aligned rectangles can still approximate a quad), the
resulting "corners" are handed to `solvePnP` as if they were a single gate's 2.72 m ×
2.72 m outer boundary — but they don't correspond to *any* real gate's geometry.

**The result:**

- `tvec` magnitude is wrong — the merged blob looks "bigger" than a single gate would
  at that range, so PnP infers the drone is closer than it actually is.
- `rvec` is wrong — the quad is skewed toward the background gate's position, encoding
  a false rotation.
- **Critically, the estimate still gets attributed to the correct gate** by
  `matched_gate_id` (nearest-neighbor match against the gate map in `gate_verifier.py`)
  — the corrupted position is *closer to gate N's true center than to any other gate's*,
  so it passes that sanity check. This is exactly why **A4 (matched-gate-ID sanity) and
  A8 (physically-impossible-value filtering) never catch it** — the ID is "correct" and
  the magnitude isn't wild enough to look "impossible," it's just systematically wrong.

This means the corruption hits **silently**, polluting both the live controller's
targeting and the offline `gate_estimates_*.csv` accuracy analysis with detections that
look like clean, successful, correctly-matched hits.

## Why this matters especially

This failure mode is most likely to occur during **boresight/co-linear approach** —
i.e., exactly the geometric regime where CV would otherwise be most reliable (square-on
view, closing range, larger apparent gate size). It silently injects bias into the
*best-case* data, which is also the data a future "look toward target" / running-estimate
system (see `VQ2_PREP_BRIEF.md`) would lean on most heavily.

## Diagnostic signature to look for

The user's hypothesis for how to spot this in existing logs:

> `est_z` and `est_x` errors that are correlated with **drone-to-gate-(N+1) distance**
> (not just drone-to-gate-N distance), appearing specifically when the drone is on the
> boresight approach line.

Concretely, this means: pull rows from `gate_estimates_*.csv` where `matched_gate_id ==
N`, and check whether the *residual* error correlates better with `‖drone_pos −
gate_center(N+1)‖` than with `‖drone_pos − gate_center(N)‖` (the latter is what the
existing `phase0_audit.py`/`phase4_audit.py` analyses condition on — which is part of
why this slipped through).

Other angles worth checking:
- `detection_diag_*.csv` rows where `detect_result == 'ok'` but `best_area` is
  anomalously large for the drone's known range to gate *N* (a merged blob would be
  bigger than a clean single-gate silhouette predicts).
- Whether `best_aspect` for these frames differs systematically from the aspect-ratio
  distribution of confirmed-clean detections.
- Visual confirmation: step through frames from the approach segments to gates
  0→1, 1→2, etc. using `audits/inspect_corners.py --play` (note: frame snapshots are
  now saved at 1/sec of sim time by `gate_verifier.py`, specifically to support this
  kind of offline visual audit) and look for detected quads that visibly span two gates.

## Files to read first

1. **`PyAIPilotExample/gate_detector.py`** — `detect_gate`, `_orange_mask_from_hsv`,
   `_four_corners`, `_roughly_square`. This is ground zero: where segmentation,
   contour selection, and corner-reduction happen, and where the merge first corrupts
   the pipeline.
2. **`PyAIPilotExample/pose_estimator.py`** — `estimate_gate_camera_frame` (the
   solvePnP call that blindly trusts the 4 corners) and `reproject_corners` (already
   exists for A2's visual ground-truth checks — likely reusable as a runtime
   self-consistency filter, see "Possible fix directions" below).
3. **`PyAIPilotExample/gate_verifier.py`** — `process_frame`, where `matched_gate_id`
   is computed via nearest-center matching (showing exactly why this corruption sails
   through unflagged), and where the new 1 fps frame snapshots are written.
4. **`PyAIPilotExample/audits/inspect_corners.py`** — the visual debugging tool,
   recently extended with FPV-style `--play` playback — the fastest way to *see*
   merged-blob detections directly.
5. **`PERC.md`** — read the A4 and A8 write-ups specifically, to understand exactly
   what those checks *do* test for and confirm why this failure mode falls through the
   gap between them. This new failure mode should get documented there once understood
   (it doesn't fit neatly under any existing assumption — may warrant a new entry).
6. **`PyAIPilotExample/planner.py`** (`gate_center`, course/gate ordering) and
   `visualize.py`'s top-down trajectory panel — to identify *which* gate-to-gate
   transitions on this course are roughly co-linear (i.e., at risk) vs. which involve
   sharp turns (not at risk).

## Suggested order of work

1. **Confirm the hypothesis before fixing anything.**
   - Use the gate map (`planner.py` / `visualize.py` top-down view) to identify which
     consecutive-gate pairs are roughly co-linear from the approach direction.
   - Pull `gate_estimates_*.csv` rows for those gates and test the
     drone-to-gate-(N+1)-distance correlation described above.
   - Visually confirm via `inspect_corners.py --play` on frames from those approach
     windows — look for quads that visibly straddle two gates.
2. **Quantify the impact** — how many frames/what fraction of "successful" detections
   are actually merged-blob corruptions, and how much they skew the per-gate error
   stats in `gate_estimates_*.csv` (this affects how urgent a fix is, and gives a
   before/after baseline).
3. **Design and implement a fix.** One principled candidate worth strongly considering:
   **PnP reprojection-residual filtering** — after `solvePnP` succeeds, use the
   already-existing `reproject_corners(rvec, tvec)` to project `OBJ_PTS` back into image
   space and compare against the detected corners. A merged/distorted quad from two
   co-linear gates is *not* well-explained by a single planar 2.72 m × 2.72 m square at
   any pose, so its reprojection residual should be large — making this a purely
   geometric self-consistency check that requires no gate-map knowledge (so it
   generalizes to VQ2, unlike a track-aware exclusion approach). Other directions
   (shape/concavity analysis on the merged contour to detect and split two overlapping
   rectangles, tuning the morphological-close kernel size, etc.) are also worth
   weighing — but should be compared against this one on cost/generality grounds.
4. **Validate the fix** — rerun, recompute CV-vs-ground-truth stats (mirroring the
   per-gate / per-range breakdown done for the A11 tilt-fix validation), and confirm
   (a) the merged-blob detections are now rejected or corrected, and (b) legitimate
   clean detections aren't being thrown out as collateral damage.
5. **Document the finding in `PERC.md`** — this failure mode doesn't fit cleanly under
   any existing lettered assumption (A1–A14); it may warrant being written up as a new,
   explicitly-named failure mode (the user's own description above is a strong starting
   draft for that write-up) so future audits know to check for it.
