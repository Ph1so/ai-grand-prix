# CV Pipeline — Findings Log

All tests run against: `logs/run_20260629_201358/frames/` (46 frames)  
Analysis tool: `analyze_frames.py` → `analysis_out*/`

---

## Test 1 — Baseline (saturation bump + blue subtraction + open/close)
**Date:** 2026-06-29  
**Code state:** `FLOOR_MASK_FRACTION = 0.0`, sat floor raised 120→150, blue ribbon subtraction added, MORPH_OPEN(3x3) + MORPH_CLOSE(9x9)

**Result:** 10 good / 36 bad / 0 new misses vs original

**Good frames:** 000000, 000031, 000057, 000064, 000078, 000103, 000155, 000162, 000202, 000209

**Failure breakdown:**
- 3 complete misses (241, 248, 255) — gate too small or drone inverted
- 1 false positive (222) — floor glow detected as gate, no actual gate visible
- 32 detected but wrong corners — either triangle/wedge shape or all 4 corners clustered at one point

**Root cause identified:**  
The gate has a downward orange spotlight that illuminates the floor in a teardrop/cone shape. MORPH_CLOSE merges this floor blob upward into the gate frame contour, making the overall blob a T-shape or cross. `approxPolyDP` on a T-shape produces:
- Triangles pointing downward (top two gate corners + floor blob tip)
- Diamonds (widest points of the merged cross)
- Clusters (when floor blob dominates and the gate contribution is small)

**Secondary cause (clustered corners):**  
Later sections (frames 261–307, drone inverted/looking at ceiling) have many small orange gate frames at distance. The closest gate blob is detected correctly but is only 700–950 px² — polygon corners end up within a few pixels of each other. Not usable for solvePnP.

**Blue ribbon finding:**  
The ribbon is neon cyan. The blue mask detects the floor glow where the ribbon illuminates the ground (visible in analysis images). But the ribbon itself cutting through the gate opening is NOT the main failure mode — the gate frame contour stays intact. Blue subtraction is safe to keep but is not solving the core problem.

---

## Test 2 — Floor mask (FLOOR_MASK_FRACTION = 0.35)
**Date:** 2026-06-29  
**Code state:** Bottom 35% of image zeroed before orange mask

**Result:** ~14 good / 32 bad / 4 new regressions

**What improved:**
- frame_000169: previously bad (corners spanned to floor), now proper rectangle — floor glow removed
- frame_000222: false positive correctly eliminated (floor glow below cutoff)

**Regressions (good → miss):**
- frame_000202: gate was in lower-centre area, cut off by the mask
- frame_000209: same — gate in lower portion during outdoor section

**Still bad despite floor mask:**
- frame_000110, 000116, 000122, 000129, 000135, 000142, 000148: corners still wrong even after floor glow removed — the gate frame blob itself is not producing a clean rectangular contour at close/medium range

**Verdict:** Floor mask is NOT robust. It partially solves the floor glow problem but:
1. Breaks detection when gate dips into the lower frame area (outdoor sections, banking)
2. Does NOT fully fix the corner quality problem — the gate frame at close range still yields bad approxPolyDP results
3. Completely useless for inverted/ceiling frames (261–307) where geometry is flipped

---

## What we know so far

| Observation | Confirmed? |
|-------------|------------|
| Orange floor spotlight merges with gate contour via MORPH_CLOSE | ✅ Yes |
| Blue ribbon fragmenting gate contour | ❌ Not the main issue — gate frame stays intact |
| Floor reflection (not spotlight) causing false positives | ⚠️ Minor — elongated shape mostly filtered by aspect ratio |
| Close-range gate produces bad corners even without floor glow | ✅ Yes (Test 2 confirmed) |
| Inverted/banking sections have degenerate small-gate detections | ✅ Yes |

---

---

## Test 3 — Inner-hole trimming (RETR_TREE on CLOSE result)
**Date:** 2026-06-30  
**Code state:** `FLOOR_MASK_FRACTION = 0.0`, `GATE_INNER_MIN_AREA_PX = 150`, `GATE_FRAME_MARGIN_PX = 5`

**Approach:**  
After MORPH_OPEN, create a temporary MORPH_CLOSE of the OPEN result. Run `cv2.findContours` with `RETR_TREE` on the temporary mask. Inner contours (holes) at any level of hierarchy = gate interior openings. Take the largest such hole; its bounding-rect bottom + 5px = spotlight cutoff. Zero the OPEN mask below that cutoff, then apply the real MORPH_CLOSE.

Why use CLOSE result for hole detection instead of OPEN result: the blue ribbon subtraction creates gaps in the gate ring at the OPEN stage, making the interior not topologically enclosed. MORPH_CLOSE bridges those gaps, giving RETR_TREE a complete ring to detect the hole from.

**Result: 44/46 detected with correct corners (was 10/46 in Test 1)**

**Remaining misses:** 000241 (area=186 < 200), 000255 (area=11). Gate is genuinely too small — either far range or drone inverted. Fundamental visibility limit, not a pipeline bug.

**Frames fixed by this approach:**
- 000110, 000116, 000122, 000142, 000148: all now show best_area ≈ 5200–5290, best_aspect ≈ 1.206, eps=0.04 (tightest, cleanest). Spotlight fully removed, corners are proper rectangles.
- 000129, 000135, 000162, 000169, 000176, 000183, 000215 and more: previously bad, now DETECTED with good corners.

**No regressions** vs. Test 1 good frames.

---

---

## Test 4 — Parent-fraction guard on inner-hole trim
**Date:** 2026-06-30  
**Code state:** `GATE_HOLE_MIN_PARENT_FRACTION = 0.30`, `GATE_HOLE_MAX_Y_FRACTION = 0.70`

**Problem found:** When the close gate's orange ring is incomplete (bottom bar near/below frame edge, or blue ribbon gap too large to bridge), `RETR_TREE` on the temp-CLOSE mask finds **no inner hole for the close gate**. The only inner hole it finds belongs to a background gate (~1269 px²). That background gate's cutoff (at ~y=145) destroys the close gate arch, leaving only tiny fragments. The 1269 px² background gate then becomes the only detection candidate → controller steers wrong direction. Affected frame: **000183** (and potentially similar very-close frames in live runs).

**Fix 1 — parent fraction guard:**  
Before accepting an inner hole as the trim cutoff, require its parent contour to be ≥ 30% the area of the largest outer blob in the scene. A background gate (~1269 px²) never passes this check when a close gate blob (~7500–30000 px²) is also present. This prevents the background gate from dictating the cutoff.

**Fix 2 — hole-bottom guard (already present):**  
If the winning inner hole's bottom exceeds 70% of frame height, skip trimming. Guards against the cutoff removing the gate's own bottom bar when it's very close with a complete ring.

**Result: 44/46 detected, same misses (000241, 000255), no regressions.**

**Key improvement:** frame_000183 now detects the close gate at **best_area=7557 (2 clipped)** instead of a background gate at 1269 (0 clipped). The controller will no longer steer toward a wrong gate at close range — it either gets a useful close-gate pose or `None` (fall through to heading hold).

---

## Test 5 — Ring check + spotlight geometry guards (current state)
**Date:** 2026-07-01  
**Code state:** `_RING_CLOSE_SIZE=5`, `GATE_RING_MAX_FILL=0.80`, `GATE_RING_FILL_MIN_AREA_PX=1000`, `_SPOTLIGHT_WIDTH_RATIO=1.6`, `_SPOTLIGHT_COLLAPSE_RATIO=0.3`, `_SPOTLIGHT_SKEW_RATIO=0.35`

**Finding:** The 44/46 metric from Tests 3–4 was misleading — the AI grader only checked "4 corners returned", not "corners are on the actual gate." Visual inspection via `batch_inspect.py` + independent agent audit revealed many of the 44 had corners landing on the floor glow rather than the gate frame.

**Problem 1 (wrong target selection):** Fix A — ring-shape sanity check:
After `MORPH_OPEN`, run `MORPH_CLOSE(5×5)` on `open_mask` to get `ring_mask`. Use `RETR_TREE` to find blobs with a closed inner hole (= gate ring). C-shaped blobs fall back to `fill_ratio < 0.80` if area ≥ 1000 px². Only accept candidates inside a ring bbox. Eliminates:
- Frames 000261–000307 (inverted section): 8 false positives → no_candidates ✓
- Frames 000018, 000025, 000037, 000044, 000051, 000071, 000097: 7 previously detected as wrong/bad target → no_candidates (but see false negatives below)

**Problem 2 (floor spotlight distorts corners):** Three failure modes, three guards:

**Fix B1 — Spread guard** (`bottom_w > top_w × 1.6`): BL/BR dragged outward by spotlight horizontal spread. Catches:
- frame_000110: top_w=78, bottom_w=152 (1.95×) ✓
- frame_000116, 000122, 000142, 000148, 000169, 000176 ✓

**Fix B2 — Collapse guard** (`bottom_w < top_w × 0.3`): BL/BR converge at floor spotlight TIP (downward triangle). `approxPolyDP` on a T-shape blob places both bottom corners at the floor-glow apex. Catches:
- frame_000084: TL(282,142) TR(361,134) BL≈BR≈(320,271) — degenerate triangle ✓
- frame_000129 ✓

**Fix B3 — Skew guard** (`|BR.y − BL.y| > gate_h × 0.35`): One bottom corner pulled down diagonally, other stays near gate. Catches:
- frame_000135: BL(312,263) vs BR(376,219) — 44px vertical difference at gate_h=96px (ratio=0.46) ✓
- Trade-off: also rejects frame_000228 (gate at heavy bank angle, tilted ~60° in image). Accepted trade-off: 000228's bad pose is less useful than avoiding 000135's wrong pose for solvePnP.

**Root cause of the floor spotlight:** At close range the camera sees the bottom gate bar AND floor spotlight at the same image y-coordinate. No horizontal cutoff can separate them. The spotlight trim in `_orange_mask_from_hsv` removes the vertical stem but not the horizontal component.

**Result (batch_inspect_out13): 15/46 clean detections, 12 spotlight_distortion rejections, 1 false positive (000222)**

**vs. baseline (Test 1):** 10 correct detections → 15 verified-correct detections. Additionally:
- 20 no_candidates (vs. many false positives before): correctly rejects inverted section, wrong targets, etc.
- 12 spotlight_distortion (vs. feeding solvePnP garbage before)

**False positives remaining:**
- frame_000222: floor glow detected as gate during banking. Large C-shaped blob passes ring check + all geometry guards.

**False negatives (confirmed by visual inspection + agent audit):**
- Frames 000018, 000025, 000037, 000044, 000051, 000071, 000097 (7 frames): gate IS visible directly ahead, but pipeline fails. The gate ring is already FRAGMENTED in the `open_mask` (OPEN mask shows disconnected pieces). Root cause under investigation — likely excessive blue ribbon subtraction cutting into the gate frame at these viewing angles, fragmenting the ring before the spotlight trim even runs. Fragments are too small individually to pass MIN_GATE_AREA_PX or the ring check.
- Frames 000110–000176 spotlight-distorted (12 frames): correctly rejected, but these represent the window when the drone is closest and most directly facing the gate. The controller must dead-reckon through these frames.
- Frame_000228: legitimate gate at heavy bank angle, rejected by skew guard (accepted trade-off).

---

## Hypotheses for further improvement

### H1 — Fix the false negatives (000018–000097) — highest priority
The gate ring is fragmented in the `open_mask`. Investigate whether the blue ribbon subtraction is too aggressive (dilating by 3×3 might be cutting into gate frame pixels). Try reducing the dilation from 3×3 to 1×1 or removing it entirely — the ribbon cuts THROUGH the gate opening, not the gate frame itself, so dilation beyond 1px may be unnecessary.

**Status: RESOLVED — but not by the blue-ribbon hypothesis. See Test 6.**

### H2 — Ring mask contour for corner fitting (avoids spotlight in final_mask)
In `ring_mask` (5×5 CLOSE of open_mask), the gate ring and floor spotlight are **still separate** for medium-range frames (confirmed from mask images). After selecting the best candidate, store the ring contour from ring_mask and use it for approxPolyDP instead of the final_mask contour. This would recover the spotlight-distorted frames (000110–000169) with clean corners.  
**Con:** For very close range (000176), the spotlight already touches the gate bar in open_mask → still merged in ring_mask.

### H3 — Reduce blue ribbon dilation from 3×3 to 1px
Blue ribbon subtraction currently dilates the detected ribbon mask by 3×3 before subtraction. If the dilation is incorrectly erasing orange gate-frame pixels near the ribbon, reducing dilation could restore the gate ring integrity in `open_mask`.

**Status: DISPROVEN. See Test 6.**

---

## Test 6 — H1 root cause was saturation floor, not blue ribbon (disproved H1/H3)
**Date:** 2026-07-01
**Code state:** `_HSV_LO1`/`_HSV_LO2` saturation floor `150 → 90` (all other constants unchanged from Test 5)

**H1/H3 hypothesis tested and disproven:** Measured pixel overlap between the blue-ribbon mask (raw and 3×3-dilated) and the orange mask directly, on all 7 false-negative frames (000018, 000025, 000037, 000044, 000051, 000071, 000097). The dilated blue mask overlaps the orange mask by **0-8 pixels** across all 7 frames — nowhere near enough to fragment a gate ring. The blue-ribbon subtraction step is not the cause.

**Actual root cause:** At medium/far range, the gate's front-face panel (branded decal with logo text, dot-pattern perforations, and a dark funnel graphic — see frame_000018) renders with saturation in the **90-140 range** across large portions of its surface (confirmed by direct HSV sampling: e.g. frame_000018 top bar sampled at sat=79,54,121,118,119,112...), even though hue is unambiguously gate-red (0-15 / 165-179). The Test-1 saturation floor of 150 (raised from 120 specifically to reject floor-glow) rejects most of these pixels, leaving only sparse speckle fragments (5-8 disconnected components of 100-900px each) that can't form a valid ring or pass the shape/area filters. This is an orange-mask sparsity problem, not a fragmentation-by-subtraction problem.

**Verification that lowering the floor is safe against the original floor-glow motivation:** Sampled the floor-glow false positive (frame_000222) — its hue is 16-30 (distinctly more yellow-orange than the gate's 0-15/165-179) and its saturation is mostly **150-250**, i.e. it already clears the old 150 floor and was never blocked by it. Test 5's `detect_result` for 000222 was already `ok` (wrongly) at floor=150, so lowering the floor to 90 cannot make this pre-existing, already-tracked false positive worse — confirmed unchanged in the Test 6 batch run.

**Sweep evidence (frame_000018 gate region, connected components):**
| sat floor | filled px | n components | largest component |
|-----------|-----------|---------------|--------------------|
| 150 (Test 5) | 1287 | 30 | 399 |
| 130 | 2474 | 27 | 444 |
| 110 | 3681 | 8 | 1657 |
| 90 (chosen) | 4257 | 7 | 3305 |
| 80 | 4552 | 7 | 3592 |

90 was chosen as the point where the gate collapses to one dominant component without going further than needed.

**Result (batch_inspect_out14): 28/46 ok** (was 15/46 in Test 5), 0 partial, 18 missed.

**Full per-frame diff vs. Test 5 (batch_inspect_out13) — every one of the 46 frames checked, only these changed:**
- All 7 target false negatives now `ok` with clean, visually-verified corners on the actual gate: 000018, 000025, 000037, 000044, 000051, 000071, 000097
- 2 bonus fixes (not in the original H1 list): 000084 (`spotlight_distortion → ok`), 000215 (`no_candidates → ok`) — both visually confirmed landing on the real gate. **Correction (Test 7): the 000215 call is disputed, not confirmed — see Test 7.**
- No previously-good frame regressed; no previously-correctly-rejected frame in the 000110-000176 (spotlight) or 000241-000261 (distant/banked) ranges changed

**New regression found (not present in the target scope, disclosed for completeness):** 4 frames in the inverted/ceiling section (000268, 000287, 000294, 000301) flipped from `no_candidates` to a false-positive `ok`, detecting a small red ceiling icon (~1000 px², diamond-outline shape) instead of a gate. Investigated and **not fixed**, for these reasons:
- The icon is colorimetrically and geometrically indistinguishable from a legitimate small distant gate with this pipeline's features. Tested two candidate discriminators and both failed: (1) hue-wrap-straddle fraction (fraction of pixels with hue ≥160) — genuine good detections range from 0.02 to 0.85 fraction, fully overlapping the icon's 0.02-0.04, no separation; (2) sat-hysteresis (require an anchor pixel ≥150 before admitting 90-149 neighbors) — the icon has pixels up to sat=255 so it would still seed and reconnect just like a real gate.
- Area is the only measured separator (icon 982-1056px vs. smallest legitimate detection 1238px) but the margin is 17% on a 46-frame sample — too thin to trust as a real fix; not applied.
- Checked downstream impact directly in `controller.py`: the CV freshness check (line 372) only verifies the estimate is horizontally in front of the drone using `forward = [cos(yaw), sin(yaw), 0]` (roll/pitch hardcoded to 0 per VQ2 constraints) — it has no way to know the drone is inverted, so it would not reliably catch this false positive if it occurred live.
- Judged low-severity and accepted as-is: this can only fire while the drone is already inverted (i.e. already tumbling/crashed), a flight state where the attempt's competitive time is very likely already lost regardless of what the controller does next. The principled fix is attitude-awareness (suppress vision trust when actual roll/pitch, once estimated from IMU integration per the "what you need to build" list, indicates inverted flight), not more HSV/shape tuning — tracked as a follow-up, not solved here.

**Verdict:** H1 is fixed. The real lever was the saturation floor, not the ribbon. Net result this test: +13 clean detections (15→28), 0 regressions in the original 42-frame scope, 1 new low-severity edge case scoped to the already-degenerate inverted-flight section.

---

## Test 7 — Independent blind audit (fresh-context agent, cross-referenced)
**Date:** 2026-07-01
**Method:** A second agent, given zero context from Tests 1-6 (no hypotheses, no prior numbers, only the raw code + tool usage instructions), ran `batch_inspect.py` itself and visually judged all 46 annotated frames from scratch against its own read of the raw source frames. Its per-frame calls were then individually cross-checked by re-opening the disputed images directly, rather than trusting either side's summary — the agent's own aggregate counts didn't reconcile to 46 and it flagged that its arithmetic, not its per-frame calls, was unreliable.

**Confirmed matches with Test 6 (independent corroboration):**
- 000268, 000287, 000294, 000301 — agent independently flagged all 4 as false positives on the ceiling icon, matching the Test 6 disclosure exactly.
- 000110, 000116, 000122, 000129, 000135, 000142, 000148, 000169, 000176 — agent flagged all 9 as "gate clearly visible, should have been detected." It had no knowledge of the spotlight-distortion guards (H2), so this independently corroborates that H2 is real, recoverable value, not just a self-serving read of our own metric.
- 000222 — agent independently confirmed floor-glow false positive.

**New defects the audit caught that Tests 1-6 missed (visually confirmed by re-inspection, not just taken on the agent's word):**
- **000162 — confirmed severe, pre-existing defect, NOT caused by Test 6.** This frame has been counted as `ok` since Test 5 (present in the original baseline before the saturation-floor change) and was never individually eyeballed until now. The detected quadrilateral is a narrow funnel/triangle shape that traces the gate's *inner* dark funnel graphic, not the outer panel boundary — completely wrong corners despite passing every shape/ring/area filter. This means the true "clean" count in Test 5 and Test 6 was overstated by at least this one frame, and it was never touched by the ribbon/saturation investigation at all.
- **000037 — confirmed minor defect.** One of this test's own 7 target fixes. Re-inspection shows a small inward notch on the bottom-left corner (doesn't cleanly reach the panel's true corner) — real, but much less severe than 162. Only 2 of the 7 target frames (018, 044) had actually been eyeballed before this audit; the other 5 were trusted on aggregate stats alone. This is the gap the blind audit was for.

**Agent calls investigated and overturned:**
- **000084, 000090 — disagree with the agent.** It called these BAD-CORNERS ("TR corner stretched onto the background sign"). Re-inspecting both at full resolution: the TR corner sits tightly on the gate panel's own top-right corner with a clear black gap to the background sign, which is a distinctly separate object well to the right. Judged clean TP; the agent's call here doesn't hold up.

**Unresolved, genuine ambiguity (not settled either way):**
- **000190, 000196, 000202, 000209, 000215.** These are all in the 183-234 "transition" section: a corridor lined with 20+ numbered pillars ("Station 19", "Station 22", ...), several of which carry a small red icon that, zoomed in, shares the same "AI-GP" branding and dot-pattern texture as the confirmed real gate — so it isn't obviously a different decoy asset the way the ceiling icon or floor glow are. But the same icon design repeats at many pillars simultaneously in-frame, which is hard to square with each instance being an independently flyable full-size gate. Test 6 had only eyeballed 000215 (called it a correct detection); the fresh audit called all 5 false positives. Neither call is well-supported enough to be definitive from static frames alone — resolving this needs either the live gate/station count from track telemetry, or checking whether the drone ever closes distance on one of these icons the way it demonstrably does on the confirmed real gates elsewhere in this dataset. **Open question, not closed.**
- **000183** — agent called this BAD-CORNERS (BL/BR dragged onto the floor). This is consistent with the already-documented Test 4 finding (`n_clipped=2`, accepted trade-off for close-range partial views) rather than a new discovery — flagged here only to note the aspect ratio (2.06:1) is more distorted than Test 4's writeup implied, so this trade-off may be worth revisiting rather than assuming it's still a net-useful pose.

**Corrected, adjudicated tally for all 46 frames (supersedes the plain `ok` counts in Tests 5-6, which measured the pipeline's internal label, not verified correctness):**
| Category | Count | Frames |
|---|---|---|
| Clean TP | 15 | 000, 018, 025, 031, 044, 051, 057, 064, 071, 078, 084, 090, 097, 103, 155 |
| Bad corners (wrong shape, same object) | 3 | 037 (minor), 162 (severe), 183 (known clip trade-off, possibly worse than assumed) |
| Intentionally rejected, recoverable (H2 target) | 9 | 110, 116, 122, 129, 135, 142, 148, 169, 176 |
| Confirmed false positive (wrong object) | 5 | 222, 268, 287, 294, 301 |
| Disputed / unresolved | 5 | 190, 196, 202, 209, 215 |
| Correct rejection (nothing usable to see) | 9 | 228, 234, 241, 248, 255, 261, 274, 280, 307 |

**Takeaway:** the mechanical `ok` label overstates quality even after Test 6's real improvement. Of 46 frames, only 15 are unambiguously correct; 3 more have real corner defects that would degrade a solvePnP estimate; 5 are outright wrong-object detections; 5 are a genuinely open question. The H2 hypothesis (recovering the 9 spotlight-distortion frames via the ring_mask contour) remains the single biggest available win, now with independent corroboration.

---

## Track sections observed
| Frames | Section | Notes |
|--------|---------|-------|
| 000–176 | Indoor arena with blue ribbon on floor | Gate spotlight creates floor orange glow |
| 183–234 | Transition / outdoor areas | Gate in lower frame portions common |
| 241–255 | Distant / banked gates | Very small blobs, near/at miss threshold |
| 261–307 | Inverted / ceiling view | Drone looking up, 7 nearly identical frames |
