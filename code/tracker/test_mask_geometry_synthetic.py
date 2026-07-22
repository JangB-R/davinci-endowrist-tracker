r"""
test_mask_geometry_synthetic.py -- synthetic validation of mask_geometry.py.

Renders a fake EndoWrist as filled rotated rectangles (one binary mask per
part, chained tip-to-tail: shaft -> clevis -> jaw1/jaw2) at KNOWN commanded
angles across a grid of shaft orientations, wrist pitches and jaw openings,
then checks that compute_frame() recovers theta_W / theta_G1 / theta_G2.

Also checks: flip_sign symmetry, zero-offset round trip, missing-part
handling (None angle + zero confidence), and jaw-side diagnostic consistency.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe test_mask_geometry_synthetic.py

Pass criterion: max |recovered - commanded| < 1.0 deg on every joint
(rect PCA is exact up to rasterisation, so expect well under that).
"""

import math

import cv2
import numpy as np

from mask_geometry import (FrameResult, calibrate_zero, compute_frame,
                           wrap_deg)

H, W = 900, 1200
TOL_DEG = 1.0


def draw_part(angle_deg, start, length, width):
    """Filled rotated rectangle from `start`, along display-CCW `angle_deg`
    (image y down => dy = -sin). Returns (mask HxW bool, end_point)."""
    a = math.radians(angle_deg)
    v = np.array([math.cos(a), -math.sin(a)])
    n = np.array([-v[1], v[0]])
    p = np.asarray(start, float)
    q = p + v * length
    corners = np.array([p + n * width / 2, q + n * width / 2,
                        q - n * width / 2, p - n * width / 2])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(corners).astype(np.int32)], 1)
    return m.astype(bool), q


def make_masks(shaft_ang, theta_w, theta_g1, theta_g2):
    """Chain the four parts; jaw angles follow the joint definitions:
    clevis = shaft + theta_w ; jaw_k = clevis + theta_gk. The chain starts
    half a shaft-length behind the canvas centre so every shaft orientation
    (including ~180 deg) stays inside the canvas."""
    a = math.radians(shaft_ang)
    v = np.array([math.cos(a), -math.sin(a)])
    start = np.array([W / 2, H / 2]) - v * 250
    masks = {}
    masks["shaft"], p = draw_part(shaft_ang, start, 300, 40)
    clevis_ang = shaft_ang + theta_w
    masks["clevis"], p = draw_part(clevis_ang, p, 80, 44)
    masks["jaw1"], _ = draw_part(clevis_ang + theta_g1, p, 120, 12)
    masks["jaw2"], _ = draw_part(clevis_ang + theta_g2, p, 120, 12)
    return masks


def main():
    grid_shaft = [0.0, 20.0, -35.0, 80.0, 170.0]
    grid_w = [-60.0, -30.0, 0.0, 30.0, 60.0]
    grid_g = [(0.0, 0.0), (15.0, -15.0), (40.0, -10.0), (5.0, -45.0)]

    errs = {"theta_W": [], "theta_G1": [], "theta_G2": []}
    n_cases = 0
    for sa in grid_shaft:
        for tw in grid_w:
            for g1, g2 in grid_g:
                fr = compute_frame(make_masks(sa, tw, g1, g2))
                truth = {"theta_W": tw, "theta_G1": g1, "theta_G2": g2}
                for j, t in truth.items():
                    assert fr.angles[j] is not None, f"{j} missing at {sa,tw,g1,g2}"
                    assert fr.confidence[j] > 0.3, (
                        f"{j} low confidence {fr.confidence[j]} at {sa,tw,g1,g2}")
                    errs[j].append(abs(wrap_deg(fr.angles[j] - t)))
                n_cases += 1

    print(f"grid: {n_cases} poses")
    ok = True
    for j, e in errs.items():
        e = np.array(e)
        status = "PASS" if e.max() < TOL_DEG else "FAIL"
        ok &= status == "PASS"
        print(f"  {j}: max_err={e.max():.3f} deg  mean_err={e.mean():.3f} deg  [{status}]")

    # flip_sign symmetry
    fr_p = compute_frame(make_masks(10, 25, 12, -8))
    fr_n = compute_frame(make_masks(10, 25, 12, -8), flip_sign=True)
    for j in ("theta_W", "theta_G1", "theta_G2"):
        assert abs(fr_p.angles[j] + fr_n.angles[j]) < 1e-6, "flip_sign broken"
    print("  flip_sign symmetry: PASS")

    # zero-offset round trip: calibrate at a slightly-off 'zero' pose, then a
    # known excursion must come back relative to that zero.
    zero_frames = [compute_frame(make_masks(15, 2.0, 1.0, -1.5))
                   for _ in range(3)]
    offs = calibrate_zero(zero_frames)
    fr = compute_frame(make_masks(15, 32.0, 21.0, -11.5), zero_offsets=offs)
    for j, expect in (("theta_W", 30.0), ("theta_G1", 20.0), ("theta_G2", -10.0)):
        err = abs(wrap_deg(fr.angles[j] - expect))
        assert err < TOL_DEG, f"zero-offset round trip {j}: err {err:.3f}"
    print("  zero-offset round trip: PASS")

    # missing part: no clevis -> every joint None with confidence 0
    m = make_masks(10, 20, 10, -10)
    del m["clevis"]
    fr = compute_frame(m)
    assert all(fr.angles[j] is None and fr.confidence[j] == 0.0
               for j in ("theta_W", "theta_G1", "theta_G2")), "missing-part handling"
    assert "clevis_missing" in fr.warnings
    print("  missing-part handling: PASS")

    # theta_R frozen in interface
    fr = compute_frame(make_masks(0, 0, 10, -10))
    assert fr.angles["theta_R"] is None and fr.confidence["theta_R"] == 0.0
    # jaw-side diagnostic: open jaws sit on opposite sides of the clevis axis
    assert fr.parts["jaw1"]["side"] * fr.parts["jaw2"]["side"] == -1, "jaw sides"
    print("  interface fields + jaw-side diagnostic: PASS")

    print("\nALL PASS" if ok else "\nFAILURES above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
