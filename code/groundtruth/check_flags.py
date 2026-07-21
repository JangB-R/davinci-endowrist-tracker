"""
check_flags.py  --  quick sanity check on a single photo of the mounted flags.

Reports which flag markers (DICT_5X5_100, ids 10/11/20/21/30/31/40/41) are
detected, the white-quiet-zone / contrast health, and writes an annotated
copy so you can SEE what the detector saw. Use this before any real session.

Usage:
  C:/Users/RAYAAN/anaconda3/python.exe check_flags.py path\to\photo.jpeg
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

EXPECTED = {10: "shaft.base", 11: "shaft.tip", 20: "clevis.base",
            21: "clevis.tip", 30: "jaw1.base", 31: "jaw1.tip",
            40: "jaw2.base", 41: "jaw2.tip"}

# Which flag-marker ids each capture pass requires (moving + reference flag).
PASSES = {
    "wrist":    {10, 11, 20, 21},   # theta_W : shaft + clevis
    "gripper1": {20, 21, 30, 31},   # theta_G1: clevis + jaw1
    "gripper2": {20, 21, 40, 41},   # theta_G2: clevis + jaw2
    "all":      set(EXPECTED),
}

def main():
    ap = argparse.ArgumentParser(description="Sanity-check mounted flags in one photo.")
    ap.add_argument("image")
    ap.add_argument("--pass", dest="cap_pass", default="all",
                    choices=sorted(PASSES),
                    help="Capture pass: check only the flags this pass needs "
                         "(default: all). wrist=shaft+clevis, "
                         "gripper1=clevis+jaw1, gripper2=clevis+jaw2.")
    args = ap.parse_args()
    needed = PASSES[args.cap_pass]
    p = Path(args.image)
    img = cv2.imread(str(p))
    if img is None:
        sys.exit(f"could not read {p}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMax = 35
    det = cv2.aruco.ArucoDetector(d, params)

    corners, ids, rejected = det.detectMarkers(gray)
    found = set(int(i) for i in ids.flatten()) if ids is not None else set()

    print(f"\nImage: {p.name}  ({gray.shape[1]}x{gray.shape[0]} px)")
    print(f"Pass: '{args.cap_pass}'  (needs ids {sorted(needed)})")
    need_found = needed & found
    print(f"Required flags detected: {len(need_found)}/{len(needed)}.\n")
    for mid in sorted(EXPECTED):
        req = " *" if mid in needed else "  "
        mark = "OK " if mid in found else "MISSING"
        print(f"  [{mark}]{req} id {mid:>2}  {EXPECTED[mid]}")
    print("  (* = required for this pass)")

    # Per-marker size and foreshortening (skew) report.
    if ids is not None:
        print("\n  detected marker geometry:")
        for c, mid in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2)
            sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            side = float(np.mean(sides))
            # Aspect ratio of opposite-side pairs: ~1 = square-on, <0.6 = edge-on.
            pair1 = (sides[0] + sides[2]) / 2
            pair2 = (sides[1] + sides[3]) / 2
            aspect = min(pair1, pair2) / max(pair1, pair2)
            notes = []
            if side < 20:
                notes.append("small: move closer")
            if aspect < 0.6:
                notes.append(f"FORESHORTENED (aspect {aspect:.2f}): "
                             "flag not facing camera")
            note = "  <-- " + "; ".join(notes) if notes else ""
            print(f"    id {int(mid):>2}: {side:5.1f} px, aspect {aspect:.2f}{note}")

    out = img.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
    if len(rejected):
        cv2.aruco.drawDetectedMarkers(out, rejected, borderColor=(0, 0, 255))
        print(f"\n  {len(rejected)} rejected candidate(s) drawn in RED "
              f"(often = quiet-zone/contrast problems).")
    outp = p.with_name(p.stem + "_check.png")
    cv2.imwrite(str(outp), out)
    print(f"\nAnnotated image written to: {outp}")
    if need_found == needed:
        print(f"\nPASS '{args.cap_pass}' READY: all required flags detected.")
    else:
        missing = sorted(needed - found)
        print(f"\nNot ready for pass '{args.cap_pass}'. Missing required "
              f"ids {missing}. If a flag shows FORESHORTENED above, rotate "
              f"the instrument so it faces the camera; otherwise check "
              f"glare / quiet-zone / framing.")

if __name__ == "__main__":
    main()
