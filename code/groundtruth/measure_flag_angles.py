"""
measure_flag_angles.py
Ground-truth angle measurement for EndoWrist validation captures.

For each image in a session:
  1. Detect the ChArUco board (DICT_4X4_50, legacy pattern) -> world reference,
     plane-tilt check (board plane vs image plane).
  2. Detect flag markers (DICT_5X5_100). Each flag = two markers (base, tip);
     the flag's angle is the in-plane angle of the base->tip line, measured on
     UNDISTORTED image coordinates. Positive = counter-clockwise as displayed.
  3. Subtract the session zero-reference image's angles (cancels mounting
     offsets), and form joint angles per the config's joint_definitions:
         theta_W  = (clevis - shaft) relative to zero pose
         theta_G1 = (jaw1 - clevis)  relative to zero pose
         theta_G2 = (jaw2 - clevis)  relative to zero pose
  4. Write one CSV row per image, with warnings for low span, high board
     tilt, or missing detections. Optionally writes annotated QC images.

Sign convention: establish once per setup with a known small positive
rotation per Figure 2.1.2 (positive CCW, top view, zero roll); if the sign
comes out inverted for your camera placement, pass --flip-sign.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe measure_flag_angles.py `
      --images "path/to/session_01" `
      --calib  "path/to/camera_calibration.npz" `
      --zero   "zero_ref.jpg" `
      --out    "session_01_gt.csv" `
      --annotate "path/to/session_01_qc"
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------

def load_calibration(npz_path):
    data = np.load(npz_path)
    keys = {k.lower(): k for k in data.files}
    K = dist = None
    for cand in ("camera_matrix", "mtx", "k", "intrinsics"):
        if cand in keys:
            K = data[keys[cand]]
            break
    for cand in ("dist_coeffs", "dist", "distortion", "dist_coefs", "d"):
        if cand in keys:
            dist = data[keys[cand]]
            break
    if K is None or dist is None:
        raise KeyError(
            f"Could not find camera matrix / distortion in {npz_path}. "
            f"Available keys: {data.files}")
    return np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64)


def build_board(board_cfg):
    adict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, board_cfg["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (board_cfg["squares_x"], board_cfg["squares_y"]),
        board_cfg["square_length_m"], board_cfg["marker_length_m"], adict)
    if board_cfg.get("legacy_pattern", False):
        board.setLegacyPattern(True)
    return board


def build_flag_detector(cfg):
    adict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, cfg["flag_dictionary"]))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Flags are small in-frame: relax the minimum perimeter so 6 mm markers
    # at working distance are not discarded before decoding.
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMax = 35
    return cv2.aruco.ArucoDetector(adict, params)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def wrap_deg(a):
    """Wrap angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def undistort_px(pts, K, dist):
    """Undistort pixel points, returning pixel coordinates (P=K)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2)


def flag_angle(base_center, tip_center, K, dist):
    """In-plane CCW-positive angle (deg) of base->tip on undistorted coords,
    plus span in undistorted pixels."""
    und = undistort_px([base_center, tip_center], K, dist)
    d = und[1] - und[0]
    # Image y axis points down; negate dy so positive = CCW as displayed.
    return math.degrees(math.atan2(-d[1], d[0])), float(np.hypot(*d))


def board_tilt_deg(charuco_corners, charuco_ids, board, K, dist):
    """Angle between the board plane normal and the camera optical axis."""
    if charuco_ids is None or len(charuco_ids) < 6:
        return None, 0 if charuco_ids is None else len(charuco_ids)
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, _ = cv2.solvePnP(obj_pts, img_pts, K, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, len(charuco_ids)
    R, _ = cv2.Rodrigues(rvec)
    normal_cam = R[:, 2]  # board z-axis in camera frame
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal_cam[2])))))
    return tilt, len(charuco_ids)


# ----------------------------------------------------------------------------
# Per-image measurement
# ----------------------------------------------------------------------------

def measure_image(gray, cfg, flag_detector, charuco_detector, board, K, dist):
    """Returns dict: flags -> {angle, span, base_px, tip_px}, board tilt, warnings."""
    result = {"flags": {}, "board_tilt": None, "board_corners": 0,
              "warnings": []}

    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
    tilt, ncorn = board_tilt_deg(ch_corners, ch_ids, board, K, dist)
    result["board_tilt"], result["board_corners"] = tilt, ncorn
    if tilt is None:
        result["warnings"].append("board_not_found")
    elif tilt > 5.0:
        result["warnings"].append(f"board_tilt_{tilt:.1f}deg")

    corners, ids, _ = flag_detector.detectMarkers(gray)
    centers = {}
    if ids is not None:
        for c, mid in zip(corners, ids.flatten()):
            centers[int(mid)] = c.reshape(4, 2).mean(axis=0)

    for name, f in cfg["flags"].items():
        b, t = f["id_base"], f["id_tip"]
        if b in centers and t in centers:
            ang, span = flag_angle(centers[b], centers[t], K, dist)
            result["flags"][name] = {"angle": ang, "span": span,
                                     "base_px": centers[b], "tip_px": centers[t]}
            if span < 150:
                result["warnings"].append(f"{name}_span_{span:.0f}px")
        elif b in centers or t in centers:
            result["warnings"].append(f"{name}_partial")
    return result


def annotate(img, meas, joint_vals):
    out = img.copy()
    for name, fdata in meas["flags"].items():
        p0 = tuple(np.round(fdata["base_px"]).astype(int))
        p1 = tuple(np.round(fdata["tip_px"]).astype(int))
        cv2.line(out, p0, p1, (0, 0, 255), 2)
        cv2.circle(out, p0, 6, (0, 255, 0), -1)   # base = green
        cv2.circle(out, p1, 6, (255, 0, 0), -1)   # tip  = blue
        cv2.putText(out, f"{name} {fdata['angle']:.2f}", (p1[0] + 8, p1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    y = 30
    for jname, val in joint_vals.items():
        txt = f"{jname}: {val:.2f} deg" if val is not None else f"{jname}: ---"
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2, cv2.LINE_AA)
        y += 32
    if meas["board_tilt"] is not None:
        cv2.putText(out, f"board tilt: {meas['board_tilt']:.1f} deg", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True,
                    help="Directory of session images")
    ap.add_argument("--calib", required=True, help="camera_calibration.npz")
    ap.add_argument("--config", default=str(script_dir / "flag_config.json"))
    ap.add_argument("--zero", default=None,
                    help="Filename of the zero-reference image "
                         "(default: alphabetically first)")
    ap.add_argument("--out", default="gt_angles.csv")
    ap.add_argument("--jaw-csv", default=None,
                    help="Optional jaw-angle CSV from blade_linefit.py "
                         "(columns image,jaw,jaw_line_deg,...). Supplies the "
                         "moving angle for theta_G1/theta_G2 when the config's "
                         "moving_source includes silhouette.")
    ap.add_argument("--annotate", default=None,
                    help="Directory for annotated QC images (optional)")
    ap.add_argument("--flip-sign", action="store_true",
                    help="Negate all joint angles (set after one-time sign "
                         "check against a known positive rotation)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    K, dist = load_calibration(args.calib)
    board = build_board(cfg["board"])
    charuco_detector = cv2.aruco.CharucoDetector(board)
    flag_detector = build_flag_detector(cfg)

    img_dir = Path(args.images)
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {img_dir}")

    zero_path = (img_dir / args.zero) if args.zero else paths[0]
    if zero_path not in paths:
        raise SystemExit(f"Zero-reference image {zero_path} not in {img_dir}")
    paths.remove(zero_path)
    paths.insert(0, zero_path)
    print(f"Zero-reference image: {zero_path.name}")

    if args.annotate:
        Path(args.annotate).mkdir(parents=True, exist_ok=True)

    joints = cfg["joint_definitions"]
    joint_names = [j for j in joints if not j.startswith("comment")]
    sign = -1.0 if args.flip_sign else 1.0

    zero_flag_angles = {}   # flag name -> absolute angle at zero pose
    rows = []
    flag_cols = list(cfg["flags"].keys())

    # Load jaw-line angles (silhouette) keyed by image -> {jaw: (angle, quality)}.
    jaw_lookup = {}
    if args.jaw_csv and Path(args.jaw_csv).exists():
        import csv as _csv
        with open(args.jaw_csv, newline="") as fh:
            for r in _csv.DictReader(fh):
                jaw_lookup.setdefault(r["image"], {})[r["jaw"]] = (
                    float(r["jaw_line_deg"]), r.get("quality", ""))
        print(f"Loaded jaw angles for {len(jaw_lookup)} image(s) "
              f"from {Path(args.jaw_csv).name}")
    zero_jaw_angles = {}   # jaw name -> absolute jaw-line angle at zero pose

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            rows.append({"image": p.name, "warnings": "unreadable"})
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meas = measure_image(gray, cfg, flag_detector, charuco_detector,
                             board, K, dist)

        if i == 0:
            for name, fdata in meas["flags"].items():
                zero_flag_angles[name] = fdata["angle"]
            for jaw, (ang, _q) in jaw_lookup.get(p.name, {}).items():
                zero_jaw_angles[jaw] = ang
            missing = [f for f in flag_cols if f not in zero_flag_angles]
            if missing:
                print(f"WARNING: zero image missing flags {missing}; "
                      f"dependent joints will be blank all session.")

        row = {"image": p.name, "is_zero": int(i == 0),
               "board_tilt_deg": (None if meas["board_tilt"] is None
                                  else round(meas["board_tilt"], 2)),
               "board_corners": meas["board_corners"]}

        rel = {}
        for name in flag_cols:
            fdata = meas["flags"].get(name)
            if fdata is not None and name in zero_flag_angles:
                rel[name] = wrap_deg(fdata["angle"] - zero_flag_angles[name])
            row[f"{name}_abs_deg"] = (None if fdata is None
                                      else round(fdata["angle"], 3))
            row[f"{name}_rel_deg"] = (None if name not in rel
                                      else round(rel[name], 3))
            row[f"{name}_span_px"] = (None if fdata is None
                                      else round(fdata["span"], 1))

        # Fold silhouette jaw angles into rel{} so they zero-reference exactly
        # like flags. Track quality so SUSPECT fits surface as a warning.
        jaw_quality = {}
        for jaw, (ang, q) in jaw_lookup.get(p.name, {}).items():
            if jaw in zero_jaw_angles:
                rel[jaw] = wrap_deg(ang - zero_jaw_angles[jaw])
            row[f"{jaw}_line_deg"] = round(ang, 3)
            jaw_quality[jaw] = q
            if q == "SUSPECT":
                meas["warnings"].append(f"{jaw}_fit_suspect")

        joint_vals = {}
        # Aliases: descriptive reference tokens -> actual rel{} key.
        alias = {"shaft_at_zero_wrist": "shaft"}
        for jname in joint_names:
            mv = joints[jname]["moving"]
            rf = joints[jname]["reference"]
            mv = alias.get(mv, mv)
            rf = alias.get(rf, rf)
            if mv in rel and rf in rel:
                joint_vals[jname] = sign * wrap_deg(rel[mv] - rel[rf])
            else:
                joint_vals[jname] = None
            row[f"{jname}_deg"] = (None if joint_vals[jname] is None
                                   else round(joint_vals[jname], 3))

        row["warnings"] = ";".join(meas["warnings"])
        rows.append(row)

        if args.annotate:
            qc = annotate(img, meas, joint_vals)
            cv2.imwrite(str(Path(args.annotate) / f"qc_{p.name}"), qc)

        jtxt = "  ".join(f"{j}={joint_vals[j]:.2f}" if joint_vals[j] is not None
                         else f"{j}=---" for j in joint_names)
        wtxt = f"  [{row['warnings']}]" if row["warnings"] else ""
        print(f"{p.name}: {jtxt}{wtxt}")

    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, None)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
