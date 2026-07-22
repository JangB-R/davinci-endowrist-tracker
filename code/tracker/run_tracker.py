r"""
run_tracker.py -- markerless EndoWrist tracker runtime (ROS-free).

Pipeline per frame:
  source frame -> (optional) undistort -> YOLO-seg inference -> best mask per
  class -> mask_geometry.compute_frame() -> joint angles + confidence ->
  display overlay + CSV log.

This is the standalone/dev runtime; ros2_tracker_node.py wraps the same
functions for publishing on-site.

Inputs you need:
  --model   trained weights (best.pt from TRAINING_WORKFLOW.md)
  --source  webcam index ("0"), a video file, or a directory of images
  --calib   camera_calibration.npz (optional but recommended: matches the GT
            convention of measuring on undistorted coordinates)
  --zero-offsets  JSON from a --calibrate-zero run (optional)

One-time per camera setup (mirrors the GT tools):
  1. Sign check: command a known small positive rotation per Fig 2.1.2; if
     the reported sign is inverted, always pass --flip-sign.
  2. Zero capture: put the instrument at commanded zero, run with
     --calibrate-zero 20 -> writes zero_offsets.json, then pass it via
     --zero-offsets on every later run.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe run_tracker.py `
      --model best.pt --source 0 `
      --calib "..\calibration\camera_calibration.npz" `
      --zero-offsets zero_offsets.json --log session.csv

Keys: q/Esc quit.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mask_geometry import (JOINT_NAMES, PART_NAMES, calibrate_zero,
                           compute_frame)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Frame sources
# ----------------------------------------------------------------------------

def open_source(source):
    """Yields (frame_bgr, is_live). Accepts webcam index, video file, or
    directory of images."""
    p = Path(source)
    if p.is_dir():
        paths = sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS)
        if not paths:
            raise SystemExit(f"no images in {p}")
        def gen():
            for x in paths:
                img = cv2.imread(str(x))
                if img is not None:
                    yield img, False
        return gen()
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {source}")
    def gen():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, source.isdigit()
        cap.release()
    return gen()


# ----------------------------------------------------------------------------
# Undistortion (same convention as measure_flag_angles.py: angles are read
# off undistorted coordinates; here the whole frame is undistorted once)
# ----------------------------------------------------------------------------

def load_undistort(npz_path, shape):
    data = np.load(npz_path)
    keys = {k.lower(): k for k in data.files}
    K = next(data[keys[c]] for c in ("camera_matrix", "mtx", "k", "intrinsics")
             if c in keys)
    dist = next(data[keys[c]] for c in ("dist_coeffs", "dist", "distortion",
                                        "dist_coefs", "d") if c in keys)
    h, w = shape[:2]
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
    return m1, m2


# ----------------------------------------------------------------------------
# YOLO -> masks dict
# ----------------------------------------------------------------------------

def masks_from_result(result, shape, min_conf):
    """Highest-confidence detection per class, rasterised from the polygon
    outline (masks.xy is already in original-image coordinates)."""
    masks, confs = {}, {}
    if result.masks is None or result.boxes is None:
        return masks, confs
    names = result.names
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    for i in np.argsort(-conf):
        name = names[cls[i]]
        if name not in PART_NAMES or name in masks or conf[i] < min_conf:
            continue
        poly = result.masks.xy[i]
        if poly is None or len(poly) < 3:
            continue
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        masks[name] = m.astype(bool)
        confs[name] = float(conf[i])
    return masks, confs


# ----------------------------------------------------------------------------
# Overlay
# ----------------------------------------------------------------------------

PART_COLORS = {"shaft": (200, 200, 0), "clevis": (0, 200, 0),
               "jaw1": (0, 0, 255), "jaw2": (255, 0, 0)}


def draw_overlay(frame, fr, fps):
    out = frame
    for name, p in fr.parts.items():
        col = PART_COLORS.get(name, (255, 255, 255))
        cv2.putText(out, f"{name} e={p['elongation']} c={p['conf']}",
                    (10, 30 + 26 * list(PART_NAMES).index(name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    y = out.shape[0] - 20
    for j in reversed(JOINT_NAMES):
        v = fr.angles.get(j)
        txt = f"{j}: {'---' if v is None else f'{v:7.2f}'} deg  " \
              f"(conf {fr.confidence.get(j, 0.0):.2f})"
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        y -= 32
    cv2.putText(out, f"{fps:.1f} fps", (out.shape[1] - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if fr.warnings:
        cv2.putText(out, ";".join(fr.warnings)[:80], (10, out.shape[0] - 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def draw_axes(frame, fr, masks):
    """Tint masks and draw fitted principal axes through part centroids."""
    for name, m in masks.items():
        col = np.array(PART_COLORS.get(name, (255, 255, 255)), np.uint8)
        frame[m] = (0.6 * frame[m] + 0.4 * col).astype(np.uint8)
    for name, p in fr.parts.items():
        ys, xs = np.nonzero(masks.get(name, np.zeros((1, 1), bool)))
        if xs.size == 0:
            continue
        c = np.array([xs.mean(), ys.mean()])
        a = np.radians(p["angle_deg"])
        d = np.array([np.cos(a), -np.sin(a)])
        half = 1.6 * np.sqrt(xs.size)
        p0 = tuple(np.round(c - d * half).astype(int))
        p1 = tuple(np.round(c + d * half).astype(int))
        cv2.line(frame, p0, p1, PART_COLORS.get(name, (255, 255, 255)), 2)
    return frame


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def csv_row(t, idx, fr, fps):
    row = {"t_unix": round(t, 3), "frame": idx, "fps": round(fps, 1),
           "warnings": ";".join(fr.warnings)}
    for j in JOINT_NAMES:
        v = fr.angles.get(j)
        row[f"{j}_deg"] = None if v is None else round(v, 3)
        row[f"{j}_conf"] = round(fr.confidence.get(j, 0.0), 3)
    for name in PART_NAMES:
        p = fr.parts.get(name, {})
        for k in ("n_px", "elongation", "perp_rms_px", "straightness"):
            row[f"{name}_{k}"] = p.get(k)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="trained YOLO-seg .pt")
    ap.add_argument("--source", required=True,
                    help="webcam index, video file, or image directory")
    ap.add_argument("--calib", default=None, help="camera_calibration.npz")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="detector confidence floor per part")
    ap.add_argument("--flip-sign", action="store_true")
    ap.add_argument("--zero-offsets", default=None,
                    help="zero_offsets.json from a --calibrate-zero run")
    ap.add_argument("--calibrate-zero", type=int, default=0, metavar="N",
                    help="capture N frames at commanded zero, write "
                         "zero_offsets.json, exit")
    ap.add_argument("--log", default=None, help="CSV output path")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO   # lazy: mask_geometry stays import-light
    model = YOLO(args.model)

    zero_offsets = {}
    if args.zero_offsets:
        zero_offsets = json.loads(Path(args.zero_offsets).read_text())

    maps = None
    writer = fh = None
    zero_frames = []
    t_prev, fps = time.time(), 0.0

    for idx, (frame, is_live) in enumerate(open_source(args.source)):
        if args.calib:
            if maps is None:
                maps = load_undistort(args.calib, frame.shape)
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

        result = model(frame, imgsz=args.imgsz, verbose=False)[0]
        masks, det_confs = masks_from_result(result, frame.shape, args.min_conf)
        fr = compute_frame(masks, det_confs=det_confs,
                           flip_sign=args.flip_sign,
                           zero_offsets=None if args.calibrate_zero else zero_offsets)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)) if idx else 0.0
        t_prev = now

        if args.calibrate_zero:
            if all(fr.angles[j] is not None for j in ("theta_W", "theta_G1",
                                                      "theta_G2")):
                zero_frames.append(fr)
                print(f"zero frame {len(zero_frames)}/{args.calibrate_zero}")
            if len(zero_frames) >= args.calibrate_zero:
                offs = calibrate_zero(zero_frames)
                out = Path(__file__).with_name("zero_offsets.json")
                out.write_text(json.dumps(offs, indent=2))
                print(f"zero offsets {offs} -> {out}")
                break

        if args.log:
            row = csv_row(now, idx, fr, fps)
            if writer is None:
                fh = open(args.log, "w", newline="")
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)

        if not args.no_display:
            vis = draw_axes(frame.copy(), fr, masks)
            cv2.imshow("tracker", draw_overlay(vis, fr, fps))
            wait = 1 if is_live else 0        # image dirs: key-step frames
            if cv2.waitKey(wait) & 0xFF in (ord("q"), 27):
                break

    if fh:
        fh.close()
        print(f"log -> {Path(args.log).resolve()}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
