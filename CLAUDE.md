# CLAUDE.md — da Vinci EndoWrist Vision Tracker (KCL MSc/MRes)

Standing context for Claude Code sessions. Keep this current — edit when
decisions change. Sections marked **OPEN** are unresolved; do not treat them
as settled.

---

## 1. Project in one paragraph

Markerless vision-based end-effector tracker for a da Vinci Si EndoWrist
(Cadiere Forceps), for the RViM lab's surgical-robotics public-engagement
demonstrator. Pipeline: external camera → deep-learning segmentation of wrist
parts → geometry layer converts masks to joint angles (theta_W wrist pitch,
theta_G1/theta_G2 grippers, theta_R roll). Purpose: characterise and ultimately
compensate the cable-driven hysteresis/backlash between commanded motor angles
and actual wrist angles, versus the lab's existing linear kinematic model
(Aban Merali MSc, ~2.1 deg mean / ~4.2 deg max error, worse in coupled configs).

Supervisors: Aban Merali, Prof. Christos Bergeles. Student GitHub: JangB-R.

## 2. Priority order (READ FIRST)

1. **The tracker is THE deliverable.** Tracker-first. This is the intellectual
   headline and what earns the marks.
2. **Scoring system / web UI is OPTIONAL** — build only if time permits after
   the tracker works. Not a necessity.
3. **Fiducial-based tracking is the documented FALLBACK** if segmentation
   underperforms — the flag/silhouette GT tooling already does this, so a
   marker-based tracker is a working demo, with segmentation positioned as
   future work.
4. **Go/no-go checkpoint ~week 3–4:** "is a fine-tuned model producing usable
   masks on held-out images?" If no → fall back to fiducial tracking or revive
   scoring as the safe deliverable. Do not discover the crunch late.

Risk note: tracker-first with a never-before-trained model on a ~6-week clock
is the ambitious, higher-risk path with no guaranteed-deliverable safety net
except the fiducial fallback. Timeline is the biggest risk in the project.

## 3. Timeline

- Today: 16 July 2026. **Final report due 29 August 2026 (~6 weeks).**
- The original "6 weeks remote + 6 weeks on-site" split no longer applies.
  Student can be **on-site now**. Get access to the motorised rig ASAP rather
  than deferring rig work to a later phase.
- Implication: GT methodology work is essentially DONE — stop polishing it.
  Centre of gravity must shift to the segmentation model immediately.

## 4. Nomenclature (use consistently — from Aban MSc Fig 2.1.2)

- Control wheels 1–4: Wheel1→Wrist(pitch), Wheel2→Gripper1, Wheel3→Gripper2,
  Wheel4→Roll. Motor angles theta_M1–theta_M4 map to wheels 1–4.
- Joint angles: theta_R (roll), theta_W (wrist pitch), theta_G1, theta_G2
  (grippers; combine for yaw).
- Positive = counter-clockwise, top view, at zero roll. Roll axis defined tip→base.
- Segments (shaft outward): **shaft** (dark tube) → **clevis** = wrist-link,
  the stamped "S" yoke the jaws pivot on → two **jaws** (thin serrated blades).

## 5. Joint-angle definitions (CONFIRMED with Aban)

- **theta_W** = clevis axis vs shaft axis (wrist pitch).
- **theta_G1, theta_G2** = each jaw relative to the CLEVIS (wrist-link) axis,
  PER-JAW (not jaw-to-jaw), matching Eq.3's separate gripper rows.
- Gripper zero/sign: **confirmed as assumed** (per-jaw vs clevis axis, signs
  per Fig 2.1.2). Aligns with his Table 3.3-C for like-for-like comparison.

## 6. Ground-truth methodology (BUILT + VALIDATED — do not re-litigate)

GT is a measuring instrument to grade the tracker; removed for training/deploy.

- **theta_W:** fiducial flags on shaft + clevis (ArUco marker pairs,
  DICT_5X5_100, referenced to the ChArUco board), long lever arm.
- **theta_G1/G2:** **silhouette line-fit** on the jaw blade edges (jaws too thin
  ~2mm to flag reliably). Repeatability sub-degree (<=0.6 deg) on matt
  background with flexion plane square to camera; degrades to ~1.7 deg under
  out-of-plane tilt — DETECTABLE via perp_rms / SUSPECT flag, so reject those.
  Clip-flags (jaw1c/jaw2c) remain in config as documented fallback only.
- **Remote gripper measurement done at ZERO WRIST**, jaws referenced to the
  shaft axis (shaft/collar/clevis axes coincide at zero wrist; shaft flag never
  occludes jaws). Coupled (non-zero-wrist) gripper + roll GT deferred to the
  rig (rigid pose-holding + commanded-angle data).
- GT uncertainty quantified via three experiments (analyze_session.py):
  repeatability (1-sigma), known-rotation (slope~1, RMS residual), cross-method.
- Capture rule: matt non-textured background, focus/exposure LOCKED at working
  distance, no digital zoom, no video stabilisation, flexion plane square to
  sensor.

## 7. Tooling (code/groundtruth/) — all synthetically validated

- `generate_flag_sheet.py` — printable flag sheet (PDF w/ true physical size).
- `check_flags.py --pass wrist|gripper1|gripper2` — pre-capture detection +
  foreshortening check.
- `measure_flag_angles.py` — main GT: flags → theta_W; ingests `--jaw-csv` to
  compute theta_G = jaw − shaft-axis. Reads `flag_config.json`.
- `blade_linefit.py --jaw-csv ...` — interactive click-seeded jaw line-fit;
  exports jaw angles keyed by image for the pipeline. Diagnostics: perp_rms,
  straightness, SUSPECT flag.
- `analyze_session.py {repeatability|known-rotation|cross-method}` — GT stats.
- `flag_config.json` — flag IDs/sizes + joint definitions (source of truth).
- `TRIAL_SESSION_CHECKLIST.md` — end-to-end run procedure.

Note: `blade_linefit.py`'s PCA machinery is a PROTOTYPE of the tracker's
geometry layer — at deployment the segmentation mask replaces click-seeds and
the same line-fit runs on mask pixels; perp_rms/straightness become per-frame
confidence.

## 8. Segmentation model plan

- **SAM/SAM2 for LABELLING assistance only** (CVAT / Roboflow / Label Studio),
  NOT as the runtime model — vanilla SAM is heavy and prompt-driven, not
  real-time. Fine-tune a LIGHT automatic model (e.g. YOLO-seg / compact
  Mask R-CNN / SegFormer) for live-video inference. Confirm any RViM-preferred
  checkpoint before committing.
- **Classes (4):** shaft, clevis, jaw1, jaw2 (background implicit).
- **Roll constrained in v1:** keep roll near zero for the tracked demo to avoid
  jaw1/jaw2 swap ambiguity; full-roll handling = future work.
- Dataset target ~100–300 marker-free images. MUST span: gripper openings,
  wrist pitch, AND realistic backgrounds/lighting (not just matt fabric — the
  demo runs in a cluttered lab; a fabric-only model will fail there).
- Metrics defined up front: per-class mIoU (segmentation); per-joint MAE vs GT
  (pipeline); fps (demo viability).

## 9. Interface (tracker → consumers)

- Tracker publishes joint angles + timestamp + per-angle confidence.
- Integrates with the **existing xArm ROS2 interface** from Aban's GitHub repo
  (`abanmerali/daVinci-EndoWrist-Instrument-Control-System` and related) — pull
  the existing message/topic conventions from there rather than inventing new.
- Freeze the interface early so the (optional) scoring system can develop
  against a simulated/replayed publisher.
- **OPEN: ROS2 distro** — check repo README / package.xml (Humble? Iron?
  Jazzy?). Install the matching version in the on-site environment.

## 10. Environment

- Dev machine: Windows 11, PowerShell. Python: `C:/Users/RAYAAN/anaconda3/python.exe`.
- Project root: `C:\Users\RAYAAN\OneDrive\Desktop\Uni\Healthcare Tech\Solo project`
- Layout: `code/calibration/`, `code/groundtruth/`, `code/tracker/` (to build),
  `code/scoring/` (optional).
- Camera: **iPhone 15 main lens** — sufficient (calibration 1.35 px RMS). Do NOT
  buy a dedicated camera; if the live demo needs a fixed USB feed, source a
  cheap webcam/machine-vision cam ON-SITE with Aban from lab stock.
- Calibration: ChArUco (calib.io, 8×11, 15mm checker, 11mm marker, DICT_4X4_50,
  requires `setLegacyPattern(True)` — even row count). Output `.npz`.
- **CREATE HPC:** user `k22010593`. Queue `bmeis_teach_gpu`, 1 GPU, 8 CPU,
  48 GB RAM, interactive Jupyter, **5-hour session cap** → checkpoint often so
  timeouts don't lose training progress. Setup cmd:
  `source /users/k22010593/jvenv/bin/activate`. Ask whether a non-interactive
  batch (sbatch) queue exists for longer unattended runs.

## 11. Key references

- Aban Merali, MSc Report, KCL 2024 — predecessor device, kinematic model
  (Eq.3 transmission matrix), baseline perf, limitations. Figs 2.1.2 (joint
  nomenclature), 1.1.7/1.1.8 (pulleys). Table 3.3-C (22-config error data —
  raw data provided by Aban; mirror + extend for validation grid).
- Habeeb et al., HSMR25 — current xArm 7 demonstrator; names closed-loop
  feedback as future work (the gap this project fills).
- Focacci et al. 2007 — source of the linear transmission matrix.
- Kim et al. 2014 — dynamic tendon model (relevant for stretch goals).

## 12. Working conventions for Claude Code

- Commit to git before significant sessions; allowlist SPECIFIC commands, not
  `Bash(*)`. Repo is local — **OPEN: confirm pushed to GitHub (user JangB-R).**
- Validate new measurement/geometry code SYNTHETICALLY (known inputs → known
  outputs) before trusting it on real data — this is the established pattern.
- Prefer prose/matplotlib outputs saved to file; keep formatting minimal.
- Student is a medical student, competent in Python but NOT an expert engineer
  and has NOT trained a segmentation model before — explain the DL/fine-tuning
  workflow, flag risks early, push back when something is unwise, don't
  over-reassure.

## 13. OPEN items (resolve + update this file)

- [ ] ROS2 distro name (repo README/package.xml).
- [ ] GitHub push confirmed (local commit exists; remote unknown).
- [ ] Working-distance recalibration not yet done → current `.npz` provisional.
- [ ] Trial session not yet run → GT chain unproven on real end-to-end data.
- [ ] Improved validation grid design (Aban wants better than his 22 configs —
      plan: mirror 22 + denser coupled-region sampling + repeats for variance).
- [ ] Runtime segmentation model choice (SAM-assisted labelling agreed; light
      runtime model TBD; check RViM preference).
