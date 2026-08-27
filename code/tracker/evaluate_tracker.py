r"""
evaluate_tracker.py -- grade the tracker against ground truth on a held-out
session. Runs the segmentation model + geometry layer on marker-free (inpainted)
frames, joins the result to the GT CSV by filename, and reports per-joint MAE
plus the figures the report needs.

Run ONCE PER BLOCK (basenames like eval_0001.jpg collide across blocks, so
keep them separate):

  # Block A -- theta_W
  python evaluate_tracker.py --model runs/v2/weights/best.pt \
      --frames "data/raw/eval_2026-08-25_heldout/clean" \
      --gt "blockA_gt.csv" --calib "../calibration/elp_calibration.npz" \
      --commanded "data/raw/eval_2026-08-25_heldout/capture_log.csv" \
      --zero-offsets zero_offsets.json --imgsz 1280 --out-prefix evalA

  # Block B -- theta_G1 / theta_G2
  python evaluate_tracker.py --model runs/v2/weights/best.pt \
      --frames "data/raw/eval_2026-08-25_gripper_matt/clean" \
      --gt "blockB_gt.csv" --calib "../calibration/elp_calibration.npz" \
      --commanded "data/raw/eval_2026-08-25_gripper_matt/capture_log.csv" \
      --zero-offsets zero_offsets.json --imgsz 1280 --out-prefix evalB

Outputs (per --out-prefix):
  <prefix>_merged.csv     one row per frame: GT, tracker, confidence, commanded
  <prefix>_mae.csv        per-joint MAE / RMSE / N / max
  <prefix>_scatter.png    tracker vs GT, per joint, with unit-slope line
  <prefix>_error_vs_cmd.png   |error| vs commanded angle (if --commanded given)
  <prefix>_conf_vs_error.png  confidence vs |error|, per joint

Only joints present (non-blank) in the GT CSV are evaluated, so Block A
yields theta_W and Block B yields theta_G1/theta_G2 automatically.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from mask_geometry import JOINT_NAMES, PART_NAMES, compute_frame
from run_tracker import load_undistort, masks_from_result

GT_COL = {"theta_W": "theta_W_deg", "theta_G1": "theta_G1_deg",
          "theta_G2": "theta_G2_deg", "theta_R": "theta_R_deg"}
CMD_COL = {"theta_W": "theta_W_cmd", "theta_G1": "theta_G1_cmd",
           "theta_G2": "theta_G2_cmd", "theta_R": "theta_R_cmd"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_keyed(csv_path, colmap):
    """Return {basename: {joint: float}} for the joints whose column exists
    and has a numeric value, keyed by the CSV's 'image' column basename."""
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = Path(row.get("image", "")).name
            if not name:
                continue
            vals = {}
            for joint, col in colmap.items():
                v = _fnum(row.get(col))
                if v is not None:
                    vals[joint] = v
            if vals:
                out[name] = vals
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True, help="inpainted/marker-free dir")
    ap.add_argument("--gt", required=True, help="GT CSV (measure_flag_angles)")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--commanded", default=None, help="capture_log.csv (optional)")
    ap.add_argument("--zero-offsets", default=None)
    ap.add_argument("--flip-sign", action="store_true")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--gripper-ref", choices=["clevis", "shaft"], default="clevis",
                    help="reference axis for theta_G1/G2. Use 'shaft' for the "
                         "zero-wrist gripper eval -- it matches the GT and "
                         "avoids the unreliable clevis axis.")
    ap.add_argument("--auto-zero", action="store_true",
                    help="zero-reference each tracker joint to the frame whose "
                         "GT is nearest zero, matching the GT's zero-referencing "
                         "(removes constant offset from an uncalibrated zero).")
    ap.add_argument("--out-prefix", default="eval")
    args = ap.parse_args()

    gt = load_keyed(args.gt, GT_COL)
    if not gt:
        raise SystemExit(f"no usable GT rows in {args.gt}")
    joints = sorted({j for v in gt.values() for j in v},
                    key=lambda j: JOINT_NAMES.index(j))
    print(f"GT covers {len(gt)} frames; evaluating joints: {joints}")
    cmd = load_keyed(args.commanded, CMD_COL) if args.commanded else {}

    zero_offsets = {}
    if args.zero_offsets and Path(args.zero_offsets).exists():
        zero_offsets = json.loads(Path(args.zero_offsets).read_text())

    from ultralytics import YOLO
    model = YOLO(args.model)
    maps = None

    rows = []
    frames_dir = Path(args.frames)
    for name, gvals in sorted(gt.items()):
        fp = frames_dir / name
        if not fp.exists() or fp.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if args.calib:
            if maps is None:
                maps = load_undistort(args.calib, img.shape)
            img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
        res = model(img, imgsz=args.imgsz, verbose=False)[0]
        masks, det_confs = masks_from_result(res, img.shape, args.min_conf)
        fr = compute_frame(masks, det_confs=det_confs, flip_sign=args.flip_sign,
                           zero_offsets=zero_offsets)
        if args.gripper_ref == "shaft":
            sgn = -1.0 if args.flip_sign else 1.0
            sh = fr.parts.get("shaft", {}).get("angle_deg")
            for jj, jw in (("theta_G1", "jaw1"), ("theta_G2", "jaw2")):
                jw_a = fr.parts.get(jw, {}).get("angle_deg")
                fr.angles[jj] = (None if (sh is None or jw_a is None)
                                 else wrap180(sgn * (jw_a - sh)))
        row = {"image": name}
        for j in joints:
            g = gvals.get(j)
            t = fr.angles.get(j)
            row[f"{j}_gt"] = None if g is None else round(g, 3)
            row[f"{j}_track"] = None if t is None else round(t, 3)
            row[f"{j}_conf"] = round(fr.confidence.get(j, 0.0), 3)
            row[f"{j}_err"] = (None if (g is None or t is None)
                               else round(wrap180(t - g), 3))
            row[f"{j}_cmd"] = cmd.get(name, {}).get(j)
        row["warnings"] = ";".join(fr.warnings)
        rows.append(row)

    if not rows:
        raise SystemExit("no frames matched GT -- check --frames points at the "
                         "inpainted dir and filenames match the GT 'image' column")

    # zero-reference the tracker to the near-zero-GT frame, per joint, so it
    # matches the GT's own zero-referencing (removes a constant offset).
    if args.auto_zero:
        for j in joints:
            paired = [r for r in rows if r[f"{j}_gt"] is not None
                      and r[f"{j}_track"] is not None]
            if not paired:
                continue
            zrow = min(paired, key=lambda r: abs(r[f"{j}_gt"]))
            t0 = zrow[f"{j}_track"]
            for r in rows:
                if r[f"{j}_track"] is not None:
                    r[f"{j}_track"] = round(wrap180(r[f"{j}_track"] - t0), 3)
                    if r[f"{j}_gt"] is not None:
                        r[f"{j}_err"] = round(wrap180(r[f"{j}_track"]
                                                      - r[f"{j}_gt"]), 3)
        print(f"auto-zeroed tracker to near-zero-GT frame per joint")

    # per-joint stats
    stats = []
    for j in joints:
        errs = [r[f"{j}_err"] for r in rows if r[f"{j}_err"] is not None]
        n_missing = sum(1 for r in rows if r[f"{j}_track"] is None)
        if errs:
            a = np.abs(errs)
            stats.append({"joint": j, "n": len(errs),
                          "MAE_deg": round(float(a.mean()), 3),
                          "RMSE_deg": round(float(np.sqrt((a ** 2).mean())), 3),
                          "max_abs_deg": round(float(a.max()), 3),
                          "bias_deg": round(float(np.mean(errs)), 3),
                          "n_missing": n_missing})
        else:
            stats.append({"joint": j, "n": 0, "MAE_deg": None, "RMSE_deg": None,
                          "max_abs_deg": None, "bias_deg": None,
                          "n_missing": n_missing})

    # write CSVs
    p = args.out_prefix
    with open(f"{p}_merged.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{p}_mae.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)

    print("\n=== per-joint accuracy (tracker vs GT) ===")
    for s in stats:
        if s["MAE_deg"] is None:
            print(f"  {s['joint']}: no paired frames "
                  f"({s['n_missing']} missing tracker output)")
        else:
            print(f"  {s['joint']}: MAE={s['MAE_deg']:.2f}  RMSE={s['RMSE_deg']:.2f}"
                  f"  max={s['max_abs_deg']:.2f}  bias={s['bias_deg']:+.2f}  "
                  f"N={s['n']}  (missing={s['n_missing']})")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not available -- CSVs written, skipping plots.")
        return

    def scatter():
        fig, axs = plt.subplots(1, len(joints), figsize=(5 * len(joints), 4.6),
                                squeeze=False)
        for ax, j in zip(axs[0], joints):
            g = [r[f"{j}_gt"] for r in rows if r[f"{j}_err"] is not None]
            t = [r[f"{j}_track"] for r in rows if r[f"{j}_err"] is not None]
            ax.scatter(g, t, s=28, alpha=0.8)
            if g:
                lo, hi = min(g + t), max(g + t)
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel(f"{j} GT (deg)")
            ax.set_ylabel(f"{j} tracker (deg)")
            ax.set_title(j)
            ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(f"{p}_scatter.png", dpi=140)
        plt.close(fig)

    def conf_vs_error():
        fig, axs = plt.subplots(1, len(joints), figsize=(5 * len(joints), 4.2),
                                squeeze=False)
        for ax, j in zip(axs[0], joints):
            c = [r[f"{j}_conf"] for r in rows if r[f"{j}_err"] is not None]
            e = [abs(r[f"{j}_err"]) for r in rows if r[f"{j}_err"] is not None]
            ax.scatter(c, e, s=28, alpha=0.8)
            ax.set_xlabel(f"{j} confidence")
            ax.set_ylabel(f"{j} |error| (deg)")
            ax.set_title(j)
        fig.tight_layout()
        fig.savefig(f"{p}_conf_vs_error.png", dpi=140)
        plt.close(fig)

    def error_vs_cmd():
        have = any(r.get(f"{j}_cmd") is not None
                   for r in rows for j in joints)
        if not have:
            return
        fig, axs = plt.subplots(1, len(joints), figsize=(5 * len(joints), 4.2),
                                squeeze=False)
        for ax, j in zip(axs[0], joints):
            xs = [_fnum(r.get(f"{j}_cmd")) for r in rows if r[f"{j}_err"] is not None]
            ys = [abs(r[f"{j}_err"]) for r in rows if r[f"{j}_err"] is not None]
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None]
            if pts:
                ax.scatter([x for x, _ in pts], [y for _, y in pts], s=28, alpha=0.8)
            ax.set_xlabel(f"{j} commanded (deg)")
            ax.set_ylabel(f"{j} |error| (deg)")
            ax.set_title(j)
        fig.tight_layout()
        fig.savefig(f"{p}_error_vs_cmd.png", dpi=140)
        plt.close(fig)

    scatter()
    conf_vs_error()
    error_vs_cmd()
    print(f"\nwrote {p}_merged.csv, {p}_mae.csv, and figures ({p}_*.png)")


if __name__ == "__main__":
    main()
