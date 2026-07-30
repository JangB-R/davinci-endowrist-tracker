# Related Work Notes — feeds report §2.2/2.3

Compiled 27 Jul 2026 from web search. Verify details against the actual PDFs
before citing; page/venue details to be completed in the reference manager.

## Group A — Vision/sensing for cable-drive hysteresis (same PURPOSE as us)

1. Hwang et al., IEEE RA-L 2020 — "Efficiently Calibrating Cable-Driven
   Surgical Robots with RGBD Fiducial Sensing and Recurrent Neural Networks"
   (arXiv:2003.08520; Berkeley AUTOLAB, dVRK).
   Camera: EXTERNAL RGBD camera tracking a 3D-printed fiducial frame on the
   arm/end-effector. 1800 samples in 31 min; LSTM + linear compensation
   models; end-effector tracking error 2.96 mm -> 0.65 mm.
   Relevance: closest philosophical match to our project (external sensing,
   removable fiducial GT, learned compensation vs cable effects). They
   measure TIP POSITION, not joint angles; we measure per-joint angles,
   which is finer-grained for characterisation.

2. "Efficient data-driven joint-level calibration of cable-driven surgical
   robots", npj Robotics 2024 (RAVEN-II).
   Sensing: temporarily mounted external joint ENCODERS (80,000 PPR), not
   vision. DNN vs linear vs polynomial compensation; joint errors ~2-8 deg
   before -> ~0.1 deg after (76-89% reduction).
   Relevance: the non-vision GT alternative; good benchmark numbers for
   "how well can compensation do with perfect joint sensing". Their linear-
   vs-learned comparison is the same experimental shape as ours vs Aban's
   linear matrix.

3. Baek et al., "Image-based hysteresis compensator for a flexible
   endoscopic surgery robot" (IEEE/RSJ 2019) + follow-ups: LBHJAE (hybrid
   image+kinematic Kalman estimation under partial occlusion), wire-tension
   hysteresis classifier, and ViO-Com (vision-optimised feed-forward,
   CycleGAN segmentation + Siamese CNN joint-angle estimation + Bouc-Wen
   model, ~67% accuracy improvement; K-FLEX platform, KAIST).
   Camera: endoscopic view of the tool; image-based joint-angle estimation
   in 2D — the closest match to our mono in-plane measurement approach.
   Relevance: validates segmentation->angle->compensation as a pipeline;
   their Bouc-Wen/learned models are candidate hysteresis models beyond the
   linear matrix (alongside Kim et al. 2014, already in refs).

## Group B — Articulated instrument pose estimation (same TRACKER TECH)

4. SurgRIPE challenge, MICCAI 2023 (arXiv:2501.02990).
   Camera: da Vinci Si endoscope, MONO (left channel), 960x540 @ 25 Hz.
   GT: "keydot" marker on the instrument -> PnP -> marker removed from the
   images by deep INPAINTING so training/eval images are marker-free
   (GT validation: 0.253 mm / 0.302 deg). Target: 6DoF wrist pose.
   Winner: YOLOv5 + SurfEmb; occlusion degrades all methods.
   Relevance: state-of-the-art benchmark for markerless instrument pose;
   the marker-inpainting trick is the literature's answer to our
   "flags visible in paired GT frames" problem (our flag-robustness
   training subset is the lightweight alternative — cite this to justify).

5. Hao, Özgüner et al. — "Vision-Based Surgical Tool Pose Estimation for
   the da Vinci Robotic Surgical System" (CWRU; PMC6706092).
   Camera: da Vinci stereo endoscope. Method: CAD silhouette rendering +
   particle filter + Chamfer matching; joint angles in the state vector.
   ~0.6 mm / 2.4 deg (sim), but ~0.3 fps — far from real time.
   Relevance: model-based alternative to learning; shows the accuracy/speed
   trade-off that motivates our light segmentation + closed-form geometry.

6. UCSD ARClab line (Yip group): SuPer Deep (arXiv:2003.03472) — DL keypoint
   detection on the tool + kinematics for surgical perception; "Robotic Tool
   Tracking under Partially Visible Kinematic Chain" (arXiv:2102.06235) —
   keypoints + lumped-error estimation, stereo endoscope; later work with
   probabilistic geometric primitives (arXiv:2403.04971) and differentiable
   rendering (arXiv:2503.05953).
   Relevance: the KEYPOINT alternative to part segmentation — sparse joints
   instead of dense masks; typically fuses kinematic priors with vision.
   We deliberately avoid kinematic priors: our tracker must be INDEPENDENT
   of commanded angles because measuring the command->actual gap IS the
   experiment. Worth one Discussion sentence.

7. SurgiPose (arXiv:2512.18068, 2025): differentiable rendering on MONO
   surgical video -> tool trajectories + joint angles (for imitation
   learning data). Relevance: recent proof that mono video -> joint angles
   is viable; optimisation-based, not real-time-first.

8. Xu et al., "Occlusion-robust markerless surgical instrument pose
   estimation", Healthcare Technology Letters 2024 — brief cite for the
   markerless trend.

## Group C — Instrument PART segmentation (same MODEL TASK)

9. EndoVis 2017 Robotic Instrument Segmentation challenge (da Vinci Xi
   endoscope, 1280x1024, 8 sequences): part classes SHAFT / WRIST /
   CLASPERS — near-identical to our shaft/clevis/jaw scheme, except we
   split the two jaws into separate classes for per-jaw angles (jaw1/jaw2),
   which the literature generally does NOT do (identity comes from
   left/right consistency at constrained roll — our convention).
   Methods: TernausNet, LinkNet, OR-UNet, DeepLabV3+.
   Relevance: pretraining/comparison baseline; per-class IoU conventions.

10. "Lightweight DNN for Articulated Joint Detection of Surgical
    Instrument..." 2022 (PMC9485358): BiSeNet-V2-based keypoint detector,
    720x576 mono laparoscope, 96.3% PCK@15px, 53-85 fps.
    Relevance: real-time lightweight networks are proven at 50+ fps —
    supports the YOLO11s-seg runtime choice; PCK is a metric option if we
    ever add keypoints.

## Camera-setup synthesis (what this means for OUR rig)

- Literature default is the ENDOSCOPE (mono 960x540-1280x1024 @ 25 Hz, or
  stereo for depth) because they target in-vivo scenes. Our demonstrator is
  benchtop — an external camera is the right call and matches Group A
  (Berkeley: external RGBD; RAVEN: external encoders).
- Our iPhone (4K, locked optics, calibrated, fixed working distance) has
  HIGHER resolution than anything surveyed; resolution is not our risk.
- The structural gap vs literature: MONO 2D in-plane measurement makes
  out-of-plane angles unobservable. Literature answers: stereo, RGBD, or
  model-based 3D (CAD + rendering). This is exactly why our v1 constrains
  roll ~0 and squares the flexion plane to the sensor, and why coupled/roll
  GT is deferred. Frame as a scoped design decision with the literature
  escape routes named (second camera / stereo / differentiable rendering).
- Nobody surveyed measures PER-JAW angles vs GT with quantified GT
  uncertainty at sub-degree level on a benchtop EndoWrist — the niche is
  real: joint-level, per-jaw, uncertainty-quantified benchtop
  characterisation for a public demonstrator.

## Improvement ideas harvested (candidate future work / stretch)

- Temporal hysteresis models on tracker output: linear + LSTM (Hwang),
  Bouc-Wen (ViO-Com), vs Aban's static linear matrix — directly the
  comparison our validation grid enables.
- Marker inpainting (SurgRIPE) instead of / alongside the flag-robustness
  training subset for paired GT frames.
- Keypoint head or pivot-point detection to stabilise the clevis axis (our
  weakest PCA target) — clevis axis from shaft-end->jaw-pivot line.
- Kalman/temporal smoothing of per-frame angles using confidence as
  measurement noise (cheap, keeps independence from commanded angles).
- Second viewpoint (cheap webcam) for out-of-plane detection only — even
  unmeasured, it can flag SUSPECT frames like the GT's perp_rms does.
