# Markerless Vision-Based Joint-Angle Tracking of a da Vinci EndoWrist

Markerless, vision-based measurement of the joint angles of a cable-driven
da Vinci Si EndoWrist (Cadiere forceps) from a single external camera, as a
step toward closed-loop control of cable-drive hysteresis.

Pipeline: **external camera → deep-learning part segmentation → geometry
layer → joint angles + per-angle confidence → ROS 2.**

MSc/MRes Healthcare Technologies project, Robotics and Vision in Medicine
(RViM) Lab, King's College London. Supervisors: Aban Merali,
Prof. Christos Bergeles.

## Results (independent held-out session)

- **Gripper angles: 0.71° / 0.68° mean absolute error** — sub-degree, and
  below the lab's linear kinematic model (~2.1° mean).
- **Wrist pitch: ~3° MAE** — limited by the near-round clevis mask giving the
  principal-axis fit no well-defined direction; the per-angle confidence
  flags this automatically (near-zero confidence for wrist pitch, high for
  the grippers).
- **Real-time:** ~114 fps on an RTX 5060 laptop GPU. Segmentation model:
  YOLO11s-seg (fine-tuned).

## How it works

The instrument is segmented into four parts (shaft, clevis, jaw1, jaw2). A
geometry layer fits a principal axis to each mask and forms joint angles as
differences of part-axis angles: wrist pitch = clevis − shaft, per-jaw
grippers = jaw − clevis (or − shaft at zero wrist). Each angle carries a
confidence derived from the mask's shape (elongation), so the tracker
reports which measurements it can be trusted on. An independent ground-truth
instrument (fiducial flags + jaw-silhouette line-fitting) grades the tracker.

## Repository layout

```
code/
  calibration/     ChArUco camera calibration
    calibrate_camera.py        intrinsics + distortion from board images
    elp_calibration.npz        deployment-camera calibration
  groundtruth/     independent ground-truth instrument (grades the tracker)
    measure_flag_angles.py     wrist pitch from fiducial flags
    blade_linefit.py           gripper angles from jaw silhouette (PCA)
    inpaint_flags.py           flag removal for paired evaluation
    analyze_session.py         GT validation statistics
    check_flags.py             pre-capture flag-detection check
    flag_config*.json          flag IDs and joint definitions
  tracker/         the tracker itself
    mask_geometry.py           geometry layer (masks -> angles + confidence)
    run_tracker.py             runtime tracker (video / webcam / image dir)
    evaluate_tracker.py        held-out tracker-vs-GT evaluation
    ros2_tracker_node.py       ROS 2 (Jazzy) publishing node
    capture_frames.py          dataset / evaluation capture
    test_mask_geometry_synthetic.py   synthetic validation of the geometry
```

## Setup

Python 3.11+, with a CUDA build of PyTorch for GPU inference:

```bash
pip install ultralytics opencv-python numpy matplotlib
# for a Blackwell (RTX 50-series) GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
# 1. Calibrate (ChArUco board images in code/calibration/calib_images/)
python code/calibration/calibrate_camera.py

# 2. Run the tracker on a live camera
python code/tracker/run_tracker.py --model best.pt --source 0 \
    --calib code/calibration/elp_calibration.npz --imgsz 960

# 3. Evaluate against ground truth
python code/tracker/evaluate_tracker.py --model best.pt \
    --frames <eval_frames_dir> --gt <gt.csv> \
    --calib code/calibration/elp_calibration.npz --imgsz 960
```

Segmentation training uses Ultralytics YOLO (`yolo segment train`,
`model=yolo11s-seg.pt`, `fliplr=0` to preserve jaw identity).

## Data and models

Datasets, trained weights (`best.pt`), calibration image sets and training
runs are **not** tracked here (see `.gitignore`) — they are large and/or
contain lab imagery. The code reproduces them from a captured dataset.
