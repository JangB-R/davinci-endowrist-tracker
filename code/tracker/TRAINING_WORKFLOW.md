# Segmentation Model — Choice + CREATE HPC Training Workflow (v1)

Decision: fine-tune **Ultralytics YOLO11-seg, small variant (`yolo11s-seg.pt`,
COCO-pretrained)**. Confirm with Aban that RViM has no preferred checkpoint
before committing (CLAUDE.md §13).

## 1. Why this model

- Real-time: the demo needs live video. YOLO11s-seg runs 30+ fps on any
  recent GPU and usable rates on CPU; nano (`yolo11n-seg`) is the fallback if
  the demo machine is weak.
- Small-dataset friendly: COCO-pretrained weights + ~150 images is a routine
  fine-tune, not research.
- Lowest workflow risk for a first-time trainer: one pip install, one train
  command, automatic checkpointing/resume, direct Roboflow export format.

Rejected: Mask R-CNN / detectron2 (heavier, slower, more setup, no benefit at
this scale); SegFormer (semantic-only is workable for our 4 distinct classes
but slower and a clunkier pipeline); SAM/SAM2 at runtime (prompt-driven, not
real-time — labelling assistance only, as planned).

KNOWN LIMITATION — mask boundary fidelity: YOLO-seg builds masks from
low-resolution prototypes upsampled to image size, so mask EDGES are soft.
Our angles come from PCA line-fits over all mask pixels, which averages out
edge noise, but this is exactly what the known-rotation and GT-comparison
validation must confirm. Mitigations if angle noise is too high: train/infer
at larger imgsz (960→1280), or keep the model's mask as a region prior and
refine edges classically (threshold/Canny within the mask) before the
line-fit. Do NOT assume mask mAP ≈ angle accuracy — measure per-joint MAE.

Licence note: Ultralytics is AGPL-3.0 — fine for this academic project;
flag to the lab if the demonstrator ever becomes a distributed product.

## 2. Reality check on compute

~150–200 images, yolo11s-seg, 200 epochs, imgsz 960 → typically **well under
1 hour** on a single modern GPU. The 5-hour Jupyter cap is not a real
constraint for this project; checkpointing (automatic, every epoch to
`last.pt`) covers the rare timeout. Still ask HPC support whether a
non-interactive sbatch queue exists (CLAUDE.md open item) — nice for sweeps,
not required.

## 3. One-time HPC setup

```bash
source /users/k22010593/jvenv/bin/activate
pip install ultralytics            # pulls torch if missing; if the venv has
                                   # a CUDA torch already, keep it
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

If `cuda.is_available()` is False inside the GPU Jupyter session, stop and
fix (module load / torch reinstall per CREATE docs) before training.

## 4. Data onto the HPC

1. Roboflow → Export → format **YOLOv11 (segmentation)** → download zip.
   Export with augmentations OFF (capture spec §6).
2. Upload the zip via the Jupyter file browser (or scp/rsync per CREATE
   docs), unzip to `/users/k22010593/endowrist/dataset/`.
3. Check `data.yaml` inside the export: 4 names in the exact order
   `shaft, clevis, jaw1, jaw2`; fix `path:` to the absolute dataset dir.
4. Verify the split matches the SESSION split (capture spec §4) — Roboflow's
   default random split must be overridden at export or re-done by hand.

## 5. Train

```bash
yolo segment train \
  model=yolo11s-seg.pt \
  data=/users/k22010593/endowrist/dataset/data.yaml \
  epochs=200 patience=50 \
  imgsz=960 batch=-1 \
  fliplr=0.0 flipud=0.0 \
  degrees=15 translate=0.1 scale=0.3 \
  hsv_h=0.015 hsv_s=0.5 hsv_v=0.5 \
  project=/users/k22010593/endowrist/runs name=v1
```

Parameter rationale:
- **fliplr=0.0 flipud=0.0 — NON-NEGOTIABLE.** Default fliplr=0.5 mirror-swaps
  jaw1/jaw2 identity and silently poisons training (capture spec §7).
- degrees/translate/scale: geometric variety that does NOT break jaw
  identity; rotation ±15° matches the "somewhat variable" viewpoint band.
- hsv_*: lighting/colour robustness — cheap insurance for the iPhone→demo
  camera domain gap.
- imgsz=960: thin jaws (~2 mm) need resolution; 640 default is too coarse.
  If GPU memory allows, try a 1280 run and compare angle error, not just mAP.
- batch=-1: auto-fit to GPU memory.
- patience=50: early-stops a converged run; epochs=200 is a ceiling.

Resume after any interruption:
```bash
yolo segment train resume model=/users/k22010593/endowrist/runs/v1/weights/last.pt
```

## 6. Evaluate (go/no-go evidence)

```bash
yolo segment val model=.../runs/v1/weights/best.pt \
  data=.../data.yaml split=test imgsz=960
```

- Ultralytics reports per-class mask mAP50/mAP50-95. Our declared metric is
  per-class mIoU — compute it with a small script over the test predictions
  (to be written with the geometry layer; same mask-loading code).
- Look at pictures, not just numbers: `yolo segment predict` on the held-out
  test session + the flag-robustness frames. Failure modes that matter:
  jaw1/jaw2 swaps, clevis/shaft boundary wander, dropped masks under the
  harsh-shadow lighting combo.
- Go/no-go (~week 3–4): usable masks on the held-out SESSION (not a random
  split). If no → fiducial fallback per CLAUDE.md §2.

## 7. Get the model back + runtime

- Download `best.pt` (~20 MB) via the Jupyter file browser.
- Runtime inference on the demo machine:
  `model = YOLO("best.pt"); r = model(frame, imgsz=960)` — masks in
  `r[0].masks`, feed to the geometry layer.
- Measure fps on the ACTUAL demo machine early (open question: what hardware
  runs the demo?). If too slow: `yolo11n-seg`, imgsz 640, or ONNX export
  (`yolo export format=onnx`) for CPU.

## 8. Smoke test BEFORE the real dataset exists (do this first)

De-risk the whole loop with ~10 throwaway images of the instrument (existing
bench photos are fine, flags and all):
label in Roboflow → export → upload → 20-epoch train → predict → download.
Every tooling problem (account, export format, venv, CUDA, resume, transfer)
surfaces in this cheap rehearsal instead of during the real week-3 window.
The resulting model will be bad — that is irrelevant.
