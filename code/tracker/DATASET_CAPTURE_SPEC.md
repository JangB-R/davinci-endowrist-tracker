# Segmentation Dataset — Capture Spec (v1)

Purpose: capture ~150–250 marker-free images of the Cadiere EndoWrist for
fine-tuning the runtime segmentation model (classes: shaft, clevis, jaw1,
jaw2). Rig-mounted, commanded poses. Roll held near zero (v1 constraint).

Camera: iPhone 15 main lens, same setup as GT sessions. Demo camera is
undecided — reserve a 30–50 image top-up session on the demo camera once
chosen (fine-tune / test top-up). Do NOT treat the iPhone-only model as
demo-ready until it is checked on the demo camera.

---

## 0. Standing capture rules (same as GT sessions)

- Focus + exposure LOCKED at working distance before the session starts.
- No digital zoom, no video stabilisation, highest still resolution.
- Landscape orientation, consistent throughout.
- Same instrument face toward camera in ALL images (clevis "S" stamp facing
  camera) so jaw1/jaw2 are always on the same image side — this is what makes
  the jaw identity convention labelable. If the face flips, jaw identity
  flips: discard or re-shoot.

## 1. Jaw identity convention (decide once, never violate)

- jaw1 = Gripper1 (Wheel2), jaw2 = Gripper2 (Wheel3), per Aban Fig 2.1.2.
- At zero roll with the stamp facing the camera, note WHICH image side jaw1
  is on (top/bottom) on day 1, write it in `capture_log.csv` header AND in
  this file, and keep it fixed for every session and every label.
- Roll tolerance: keep commanded roll within ±10° of zero. Beyond that the
  jaws approach the flexion plane edge-on and identity/visibility degrade.

jaw1 image-side at zero roll: ____________  (fill in on day 1)

## 2. Pose grid (rig-commanded)

Command in joint-angle terms via the rig's mapping; log commanded values per
frame. Commanded ≠ actual (hysteresis) — that is fine, the log is for
coverage auditing, not labels.

Core grid — deployment background, nominal viewpoint (35 poses ≈ 35 images):

- theta_W: {−60, −40, −20, 0, +20, +40, +60}° (7 values; clip to rig's safe
  range if narrower — record actual range used).
- Gripper states (5): both closed / both half-open / both fully open /
  jaw1 open + jaw2 closed / jaw1 closed + jaw2 open.
- All 7 × 5 = 35 combinations, one still each.

Background/lighting variants (≈ 60 images):

- Repeat a 15-pose subset (every other theta_W × alternating grip states,
  cover the extremes) under each of 4 background × lighting combos (§3).

Viewpoint band (≈ 25 images):

- Demo geometry is "somewhat variable": move the camera ±15–20° in azimuth
  and elevation and ±20% in distance around nominal. Mixed poses (vary theta_W
  and grip each shot, don't repeat the grid). Keep focus/exposure re-locked
  after each camera move.

Distractor/realism set (≈ 20 images):

- Gloved and bare hands in frame near/behind the instrument (demo has people).
- Other tools/cables in background, partial occlusion of the SHAFT only
  (never occlude clevis/jaws — those must stay measurable in v1).

Flag-robustness set (≈ 15 images):

- Shaft + clevis fiducial flags MOUNTED, mixed poses, deployment background.
- Labelled like all others; flags themselves are background (unlabelled).
- Purpose: lets the tracker run on flagged frames so tracker-vs-GT theta_W
  comparison can use paired frames without the flags being out-of-distribution.

Running total ≈ 155 images for train/val.

## 3. Backgrounds × lighting (4 combos minimum)

The demo runs in a cluttered lab — a fabric-only model will fail there.

1. Deployment background as-is (real bench clutter), ceiling lights.
2. Deployment background, rearranged clutter + side lamp (harsh shadows).
3. Different lab surface/wall, dimmer lighting (lights half off / blinds).
4. Matt fabric (GT-style) control, ceiling lights.

Roughly 60–70% of all images on backgrounds 1–3, ≤ 25% on fabric.

## 4. Held-out TEST session (separate day — do not skip)

- ~40–50 images, captured on a DIFFERENT day/session from training data:
  fresh rig mount, fresh camera placement, deployment background.
- Include ~10 flagged frames (for paired pipeline-vs-GT evaluation) and, once
  the demo camera exists, ~20 frames from it.
- NEVER let test-session images into train/val. Split by SESSION, not by
  random shuffle — consecutive frames are near-duplicates and random splits
  leak, inflating mIoU. This is the go/no-go metric; keep it honest.

## 5. Folder structure + logging

```
data/
  raw/
    s01_2026-07-XX_bg1_lightA_vpN/      # one folder per session/condition
      IMG_0001.jpg ...
      capture_log.csv
    s02_...
  dataset/                               # produced later by selection script
    train/  val/  test/                  # split BY SESSION
```

`capture_log.csv` columns:
`filename, theta_W_cmd, theta_G1_cmd, theta_G2_cmd, theta_R_cmd, background_id,
lighting_id, viewpoint_id, flags_mounted, notes`

Fill it live during capture (30 s/frame). A frame without a log row is a
frame you can't audit for coverage gaps.

## 6. Labelling workflow (Roboflow, SAM-assisted)

1. Project type: **Instance Segmentation**; classes exactly: `shaft`,
   `clevis`, `jaw1`, `jaw2` (background implicit — never label it).
2. Upload per-session batches so provenance survives into splits.
3. Use **Smart Polygon** (SAM-backed): click the part, adjust with +/−
   clicks, assign class. Expected pace after warm-up: 1–2 min/image.
4. Class boundaries (write these in the project description too):
   - shaft: dark tube up to the shaft/clevis joint line.
   - clevis: the stamped "S" yoke, from that joint line to the jaw pivot.
   - jaw1/jaw2: each blade from pivot to tip, per §1 side convention.
   - Boundary pixels: when in doubt at joints, favour consistency over
     precision — pick a rule (e.g. split at the visible pivot pin) and reuse it.
   - Flags in the flag-robustness set: unlabelled (background).
5. QC pass: second run-through of every image at 1× zoom checking (a) jaw
   identity side, (b) no missing part, (c) no class bleeding across joints.
6. Export: COCO JSON (archival) + model-native format at training time
   (e.g. YOLO-seg TXT). Turn OFF Roboflow's built-in augmentations on export
   — augmentation happens in the training pipeline where we control it.

## 7. Known traps (read before training, not after)

- **Horizontal/vertical flip augmentation swaps jaw1/jaw2 identity.** Default
  YOLO training uses fliplr=0.5. It MUST be disabled (or handled with a
  label-swap-aware transform). A model trained with naive flips will be
  confidently wrong about which jaw is which.
- Domain gap to the demo camera: plan the top-up session; don't discover it
  during the demo.
- Session leakage (§4): report mIoU only on the held-out session.
- Coverage audit before labelling everything: plot the capture_log commanded
  angles; fill gaps with a short second capture rather than labelling a
  skewed set.
