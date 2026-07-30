# Report Outline — Markerless Vision-Based Joint-Angle Tracking of a da Vinci EndoWrist

Working title; sharpen once results exist (e.g. "...for Characterising
Cable-Drive Hysteresis in a Surgical Robotics Demonstrator").

Status tags per section:
- [NOW]    writable today — content is settled, won't change with results
- [DESIGN] writable today as design/methods; numbers slot in later
- [DATA]   blocked on captured/experimental data

FIRST ACTION: confirm from the module handbook (or Aban) the word limit,
required structure, referencing style, and marking criteria — this outline
follows the conventional MSc structure (Aban's own report is the template
precedent) but the handbook wins. Note figure-heavy methods sections eat
word count fast; the GT chapter especially will need pruning discipline.

---

## Abstract  [last]
One paragraph: problem (cable-drive hysteresis breaks the commanded→actual
mapping), approach (markerless segmentation + geometry tracker, graded
against a purpose-built GT instrument), headline numbers (GT uncertainty;
tracker per-joint MAE; fps), and what it enables (hysteresis
characterisation beyond the linear model; closed-loop feedback as future
work).

## 1. Introduction  [NOW — draft in full]
- 1.1 Context: RViM surgical-robotics public-engagement demonstrator; why
  joint-level feedback matters for it (Habeeb et al. HSMR25 names
  closed-loop feedback as future work — this project fills that gap).
- 1.2 The problem: cable-driven EndoWrist transmission; hysteresis/backlash;
  Aban's linear kinematic model baseline (~2.1° mean / ~4.2° max error,
  worse in coupled configurations) — the tracker exists to measure what the
  model cannot predict.
- 1.3 Aims & objectives (numbered, testable — mirror these in Discussion):
  O1 GT methodology with quantified uncertainty; O2 markerless segmentation
  tracker (θ_W, θ_G1, θ_G2; roll constrained in v1); O3 evaluation vs GT
  (per-class mIoU, per-joint MAE, fps); O4 hysteresis characterisation vs
  the linear model on an improved validation grid; (O5 optional: live demo
  integration / scoring).
- 1.4 Report structure paragraph.

## 2. Background  [NOW — draft in full]
- 2.1 da Vinci Si EndoWrist mechanics: wheels 1–4, joint nomenclature
  (θ_R, θ_W, θ_G1, θ_G2), sign conventions (reproduce/adapt Aban Fig 2.1.2
  with credit), cable/pulley coupling.
- 2.2 Kinematic modelling of tendon-driven instruments: Focacci et al. 2007
  linear transmission matrix (Eq.3 as used by Aban); Kim et al. 2014
  dynamic tendon models; where linear models fail (coupling, hysteresis).
- 2.3 Vision-based surgical-tool tracking: fiducial/marker-based methods vs
  markerless; segmentation-based part tracking; brief SAM/foundation-model
  labelling context; why a light runtime model (real-time constraint).
- 2.4 Gap statement: no existing joint-level measurement on the lab's
  demonstrator; predecessor work is open-loop model-only.

## 3. Ground-Truth Methodology  [NOW — draft in full; final numbers pending
   the real trial session]
- 3.1 Principle: GT as a removable measuring instrument to grade the
  tracker; requirements (sub-degree target, no interference with training
  images at deployment).
- 3.2 Camera calibration: ChArUco board, working-distance protocol, locked
  focus/exposure capture rules; reprojection RMS.
- 3.3 θ_W via fiducial flags: two-marker rigid flags (DICT_5X5_100), long
  lever arm, session zero-referencing cancelling mounting offsets,
  undistorted-coordinate angle convention, --flip-sign one-time check.
- 3.4 θ_G via silhouette line-fit: why jaws can't be flagged (~2 mm), PCA
  band fit, diagnostics (perp_rms, straightness, SUSPECT flag), zero-wrist
  protocol (jaws referenced to shaft axis; coupled configs deferred to rig).
- 3.5 Validation experiments: repeatability (1σ), known-rotation (slope,
  RMS residual), cross-method agreement; failure-mode analysis
  (out-of-plane tilt → ~1.7°, detectable and rejected via perp_rms).
- 3.6 Resulting uncertainty budget → the yardstick the tracker is graded
  against. [final numbers: real trial session]
- Figures: flag/board photo; annotated QC frame; line-fit overlay;
  repeatability + known-rotation plots.

## 4. Markerless Tracker  [DESIGN — draft structure + design rationale now]
- 4.1 System overview figure: camera → undistort → segmentation →
  per-part PCA → joint angles + confidence → ROS2/consumers.
- 4.2 Dataset: capture spec rationale (pose grid, background/lighting
  variants, viewpoint band, distractors, flag-robustness set), session-based
  splits and why (near-duplicate leakage), jaw-identity convention,
  SAM-assisted labelling workflow, class boundary rules. [numbers: DATA]
- 4.3 Model: choice rationale (YOLO11s-seg vs alternatives; real-time
  requirement; mask-boundary-fidelity caveat), training configuration —
  including why flip augmentation must be disabled (jaw identity), imgsz
  choice; HPC setup. [curves/metrics: DATA]
- 4.4 Geometry layer: port of the GT PCA machinery to masks; distal
  direction-pinning via the kinematic chain; confidence redesign
  (elongation replaces edge perp_rms — explain why the semantics change on
  filled masks); zero-offset + sign conventions shared with GT; synthetic
  validation (100-pose grid, max err 0.76°, mean 0.25°). [NOW]
- 4.5 Runtime + interface: frozen output (angles + timestamp + per-angle
  confidence; θ_R present, conf 0 in v1); ROS2 JointState topics; fps
  budget. [integration details: on-site]

## 5. Evaluation  [DATA — define metrics + table skeletons now]
- 5.1 Segmentation: per-class mIoU on the held-out SESSION (state the
  anti-leakage protocol explicitly — examiners reward this); qualitative
  failure gallery (shadow/clutter/flagged frames).
- 5.2 Geometry in isolation: compute_frame on ground-truth masks vs GT
  angles — separates geometry error from segmentation error.
- 5.3 Full pipeline: per-joint MAE vs GT on paired (flagged) frames;
  confidence-vs-error relationship (does low conf predict high error? —
  strong figure if yes); fps on the demo hardware.
- 5.4 Hysteresis characterisation: commanded vs tracker-measured sweeps;
  hysteresis loops per joint; comparison vs linear-model prediction on the
  improved validation grid (mirror Aban's 22 configs + denser coupled
  sampling + repeats); like-for-like table against his Table 3.3-C.

## 6. Discussion  [after results]
- Answer each objective O1–O4 explicitly with numbers.
- Error-budget narrative: GT uncertainty vs geometry-layer error vs
  segmentation error — where does the total come from?
- Limitations (be unflinching; this earns marks): roll constrained → jaw
  identity assumption; clevis as weakest PCA target; iPhone↔demo-camera
  domain gap; single instrument/background scope; GT available only in
  flagged configurations for θ_W.
- Fallback narrative if segmentation underperformed: fiducial tracking as
  the working demo, segmentation as characterised future work (frame as an
  engineering decision, not a failure).
- Future work: full-roll handling, hysteresis compensation in the control
  loop, closed-loop demo, dynamic tendon models (Kim et al.).

## 7. Conclusion  [last]
Three short paragraphs: what was built, what was measured, what it enables.

## References
Core set already identified: Aban Merali MSc 2024; Habeeb et al. HSMR25;
Focacci et al. 2007; Kim et al. 2014; + segmentation/model papers (YOLO,
SAM, and 2–3 surgical tool-tracking works from 2.3).

## Appendices
A: GT tooling reference (scripts, flag_config.json, checklist).
B: Capture spec + training config (link/reproduce from repo docs).
C: Validation-grid raw tables.
D: Code repository pointer (JangB-R — confirm pushed + public/shareable).

---

## Writing order (fit around lab work; ~500 words/day beats a final-week crisis)
1. §3 GT methodology — fully settled, longest, do first.
2. §1 Introduction + §2 Background — settled, needs only reference reading.
3. §4.2–4.4 design rationale — settled as of this week's commits.
4. §5 skeleton tables/figure placeholders — so experiments fill slots, and
   missing-data gaps become visible EARLY.
5. Results → Discussion → Conclusion → Abstract, in that order, as data lands.
