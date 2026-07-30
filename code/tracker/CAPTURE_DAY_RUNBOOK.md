# Capture-Day Runbook — full data-collection system setup

Companion to DATASET_CAPTURE_SPEC.md (what to shoot) and
../groundtruth/TRIAL_SESSION_CHECKLIST.md (how to run a GT session).
This file: how to set the whole system up and run the day.

Priority if time runs short (work top-down, cut bottom-up):
  1. GT trial session (iPhone)  — least-proven link, highest value
  2. ELP mount + calibration    — freezes the deployment viewpoint
  3. ELP core pose grid         — the training set's backbone
  4. Background/lighting variants + iPhone minority + flag set
  5. Viewpoint band + distractors

## A. Kit list (pack the night before)

- [ ] iPhone: >= 10 GB free, Settings->Camera->Formats = Most Compatible,
      Settings->Photos->Transfer = Keep Originals, lens wiped
- [ ] iPhone tripod/mount + USB cable (for transfer AND as backup)
- [ ] ELP camera + varifocal lens; SmallRig arm + super clamp
- [ ] Tape (lens rings + witness marks), USB extension lead
- [ ] ChArUco board, rigid and flat
- [ ] Flag sheet: printed flags mounted on stalks/clips, spare prints
- [ ] Matt background fabric + means of hanging it
- [ ] Desk lamp (harsh side-light variant), tape measure, marker pen
- [ ] Laptop: repo synced, anaconda env working, >= 20 GB free
- [ ] Printed copy of this runbook + DATASET_CAPTURE_SPEC.md +
      TRIAL_SESSION_CHECKLIST.md + blank capture_log rows

Laptop pre-flight (run at home the night before):
- [ ] `python code/tracker/capture_frames.py --source 0 --out scratch`
      works with any webcam / the ELP if already delivered
- [ ] `python code/tracker/test_mask_geometry_synthetic.py` passes
- [ ] `python code/groundtruth/inpaint_flags.py --selftest` passes
- [ ] check_flags.py + measure_flag_angles.py run on an old test image

## B. Physical layout (order matters)

1. Rig + instrument first. Confirm the instrument's working envelope with
   Aban (where the wrist actually moves during sweeps/demo).
2. MEASURE AND RECORD the available camera working distance: ______ cm.
   (< ~50 cm: ELP 12 mm end is fine. Longer: order fixed 16/25 mm lens,
   still capture today on iPhone — do not block the day on this.)
3. Clamp the ELP arm to the RIG'S OWN structure (same table/frame), short
   arm extension, at the demo viewpoint. Aim so the working envelope fills
   the frame with margin.
4. Set ELP lens: zoom to frame the envelope, focus on the wrist at working
   distance, then TAPE both rings. Witness-mark the arm joints.
5. Strain-relieve the USB cable to the arm, not the camera.
6. iPhone tripod: GT position per TRIAL_SESSION_CHECKLIST (measurement
   plane square to sensor). This is a SEPARATE station from the ELP; they
   don't interfere and can both stay up all day.
7. Background fabric behind the instrument for GT blocks; lab clutter
   stays for deployment-background training blocks (swap per block).
8. Photograph the whole setup from two angles (methods figure + re-rig
   reference). Record camera positions with the tape measure.

## C. System bring-up checks (before any real data)

ELP:
- [ ] `capture_frames.py --source <N> --try-lock` streams 1920x1080
- [ ] wave-hand test: preview does NOT adapt -> exposure locked. If it
      adapts, try vendor tool / different index; note outcome either way
- [ ] test frame: jaw blade spans >= ~15 px when instrument at working
      distance (zoom the saved JPEG). Record px width: ______
- [ ] ChArUco calibration of the ELP at working distance ->
      `elp_calibration.npz`; reproj RMS recorded: ______ px
- [ ] unplug/replug: settings survived? ______

iPhone:
- [ ] focus/exposure locked at working distance (long-press AE/AF lock)
- [ ] 3-4 board verification shots -> residuals consistent with existing
      calibration; if moved distance/refocused: full recalibration now
- [ ] `check_flags.py` pass on a flagged test shot at working distance

Folders (laptop, inside the OneDrive-synced project):
```
data/raw/s01_<date>_gt-trial_iphone/
data/raw/s02_<date>_bg1_lightA_vpN_elp/
...one folder per session block, capture_log.csv in each
```

## D. Run order

1. GT TRIAL SESSION (iPhone) — follow TRIAL_SESSION_CHECKLIST end-to-end:
   zero-ref image, known-rotation sweep on the rig, repeatability set,
   gripper silhouette set at zero wrist. Process ON the laptop before
   tearing anything down: measure_flag_angles.py + analyze_session.py must
   produce sane numbers WHILE the setup still exists to re-shoot.
2. ELP CORE GRID — rig steps the pose grid (spec §2), SPACE per pose in
   capture_frames.py, log row per frame (commanded angles from the rig).
3. VARIANTS — background/lighting combos (spec §3), 15-pose subset each.
4. IPHONE MINORITY SET — ~30-40 mixed poses + the flag-robustness set
   (flags mounted, flags unlabelled later).
5. VIEWPOINT BAND + DISTRACTORS — move ELP arm through +/-15-20 deg and
   +/-20% distance (re-aim only, rings stay taped); hands/tools in frame.
   RE-AIM THE ARM BACK to witness marks afterwards; verify with one
   comparison frame against a step-2 image.

## E. End of day (do not skip; 20 min)

- [ ] Offload iPhone via USB into session folders; file count == log rows
      for EVERY folder before deleting anything from the phone
- [ ] Quick QC skim: 1 in 10 frames at 1:1 zoom — sharpness, exposure,
      jaw visibility; note bad blocks for re-shoot
- [ ] Confirm OneDrive has synced (icon = done, not syncing)
- [ ] capture_log gaps filled while memory is fresh; note anything odd
- [ ] git add/commit: calibration .npz files, logs, this runbook with the
      blanks filled in
- [ ] Coverage audit at home: plot commanded angles from all logs; list
      gaps -> next visit's shot list
