r"""
inpaint_flags.py -- remove fiducial flags from GT frames by inpainting, so
tracker evaluation can run on paired frames (flag-based GT + marker-free-
looking image of the SAME physical instant). Re-capturing without flags is
impossible: commanded angles are hysteretic/path-dependent, so a re-visited
pose is a different pose. (Trick per SurgRIPE challenge, arXiv:2501.02990.)

Pipeline per image:
  1. Detect flag markers (flag_config.json dictionary, same params as
     measure_flag_angles.py).
  2. For each flag (base+tip marker pair): mask = dilated filled convex hull
     of both marker quads -- covers the whole printed strip. --dilate-frac
     scales the margin; --extra-mask adds fixed polygons (e.g. clips/stalks)
     applied to every frame in the session.
  3. cv2.inpaint (Telea). Good enough on the matt GT background; if you see
     smearing on textured scenes, upgrade path is a deep inpainter (LaMa).
  4. Save inpainted image (+ optional side-by-side QC image). Frames where a
     flag is only partially detected are SKIPPED with a warning (no mask ->
     no safe inpaint; those frames lack full GT anyway).

VALIDATION before trusting on real data:
  - `--selftest` (no args needed): synthetic scene -> paste flag strip ->
    detect -> inpaint -> report pixel error vs the pristine scene and check
    markers are no longer detectable. Run after any parameter change.
  - On real data: run the tracker on inpainted frames vs a few genuinely
    flag-free control frames of similar poses; angle deltas attributable to
    inpainting should be well under GT uncertainty. Also ALWAYS eyeball the
    QC images: the mask must never bite into shaft/clevis/jaw silhouettes.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe inpaint_flags.py --selftest
  C:/Users/RAYAAN/anaconda3/python.exe inpaint_flags.py `
      --images "path\to\gt_session" --out "path\to\gt_session_inpainted" `
      --qc "path\to\gt_session_inpaint_qc"
  optional: --dilate-frac 0.6   margin as fraction of mean marker side
            --radius 7          cv2.inpaint radius (px)
            --extra-mask masks.json   {"polygons": [[[x,y],...], ...]}
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_detector(cfg):
    adict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, cfg["flag_dictionary"]))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMax = 35
    return cv2.aruco.ArucoDetector(adict, params)


def flag_masks(gray_shape, corners_by_id, flags_cfg, dilate_frac):
    """One combined uint8 mask for all fully-detected flags. Returns
    (mask, complete_flags, partial_flags)."""
    mask = np.zeros(gray_shape, np.uint8)
    complete, partial = [], []
    for name, f in flags_cfg.items():
        b, t = f["id_base"], f["id_tip"]
        have = [i for i in (b, t) if i in corners_by_id]
        if len(have) == 1:
            partial.append(name)
            continue
        if not have:
            continue
        pts = np.concatenate([corners_by_id[i].reshape(4, 2) for i in (b, t)])
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        # margin: fraction of the mean marker side length
        sides = [np.linalg.norm(corners_by_id[i].reshape(4, 2)[k] -
                                corners_by_id[i].reshape(4, 2)[(k + 1) % 4])
                 for i in (b, t) for k in range(4)]
        m = np.zeros(gray_shape, np.uint8)
        cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 255)
        r = max(3, int(dilate_frac * float(np.mean(sides))))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (2 * r + 1, 2 * r + 1)))
        mask |= m
        complete.append(name)
    return mask, complete, partial


def inpaint_image(img, detector, flags_cfg, dilate_frac, radius,
                  extra_polys=()):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    by_id = {}
    if ids is not None:
        for c, mid in zip(corners, ids.flatten()):
            by_id[int(mid)] = c
    mask, complete, partial = flag_masks(gray.shape, by_id, flags_cfg,
                                         dilate_frac)
    for poly in extra_polys:
        cv2.fillPoly(mask, [np.round(np.asarray(poly)).astype(np.int32)], 255)
    if not mask.any():
        return None, complete, partial
    out = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return out, complete, partial, mask


# ----------------------------------------------------------------------------
# Self-test: synthetic scene, no real data needed
# ----------------------------------------------------------------------------

def selftest(cfg):
    rng = np.random.default_rng(0)
    H, W = 700, 900
    noise = rng.normal(0.0, 2.5, (H, W, 1))            # mild sensor noise
    base = np.clip(178.0 + noise, 0, 255).astype(np.uint8).repeat(3, axis=2)
    cv2.rectangle(base, (100, 320), (700, 370), (60, 55, 50), -1)  # "shaft"
    pristine = base.copy()

    # paste a flag strip (white strip + base/tip markers of the shaft flag)
    adict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, cfg["flag_dictionary"]))
    ms = 64                                            # marker side, px
    strip = np.full((ms + 24, 2 * ms + 3 * 12 + ms // 2, 3), 255, np.uint8)
    for k, mid in enumerate((cfg["flags"]["shaft"]["id_base"],
                             cfg["flags"]["shaft"]["id_tip"])):
        mk = cv2.aruco.generateImageMarker(adict, mid, ms)
        x0 = 12 + k * (ms + 12 + ms // 2)
        strip[12:12 + ms, x0:x0 + ms] = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
    sy, sx = 180, 300                                  # above the "shaft"
    scene = pristine.copy()
    scene[sy:sy + strip.shape[0], sx:sx + strip.shape[1]] = strip

    det = build_detector(cfg)
    res = inpaint_image(scene, det, {"shaft": cfg["flags"]["shaft"]},
                        dilate_frac=0.6, radius=7)
    assert res[0] is not None, "selftest: flag not detected in synthetic scene"
    out, complete, partial, mask = res
    assert complete == ["shaft"] and not partial

    # 1) strip must cover: no markers detectable afterwards
    c2, ids2, _ = det.detectMarkers(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
    assert ids2 is None or len(ids2) == 0, "selftest: markers survived inpaint"

    # 2) restoration error inside the mask vs pristine background
    diff = cv2.absdiff(out, pristine).mean(axis=2)
    in_mask = diff[mask > 0]
    print(f"selftest: masked px={in_mask.size}  mean_err={in_mask.mean():.2f} "
          f"grey-levels  p99={np.percentile(in_mask, 99):.1f}")

    # 3) instrument pixels outside the mask must be untouched
    outside = cv2.absdiff(out, scene).mean(axis=2)[mask == 0]
    assert float(outside.max()) == 0.0, "selftest: pixels outside mask changed"

    ok = in_mask.mean() < 6.0
    print("selftest:", "PASS" if ok else
          "FAIL (mean restoration error too high)")
    cv2.imwrite(str(Path(__file__).with_name("inpaint_selftest_qc.png")),
                np.hstack([scene, out]))
    print(f"QC image -> inpaint_selftest_qc.png (before | after)")
    raise SystemExit(0 if ok else 1)


def main():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", help="directory of GT session images")
    ap.add_argument("--out", help="output directory for inpainted images")
    ap.add_argument("--config", default=str(script_dir / "flag_config.json"))
    ap.add_argument("--dilate-frac", type=float, default=0.6)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--extra-mask", default=None,
                    help='JSON {"polygons": [[[x,y],...], ...]} applied to '
                         "every frame (clips/stalks)")
    ap.add_argument("--qc", default=None,
                    help="directory for before|after QC images")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.selftest:
        selftest(cfg)
    if not args.images or not args.out:
        raise SystemExit("--images and --out required (or --selftest)")

    extra = []
    if args.extra_mask:
        extra = json.loads(Path(args.extra_mask).read_text())["polygons"]

    det = build_detector(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.qc:
        Path(args.qc).mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for p in sorted(Path(args.images).iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name}: unreadable, skipped")
            continue
        res = inpaint_image(img, det, cfg["flags"], args.dilate_frac,
                            args.radius, extra)
        if res[0] is None:
            print(f"{p.name}: no complete flags detected, SKIPPED")
            n_skip += 1
            continue
        out, complete, partial, _mask = res
        if partial:
            print(f"{p.name}: partial flags {partial} NOT masked -- frame "
                  f"skipped (no safe inpaint)")
            n_skip += 1
            continue
        cv2.imwrite(str(out_dir / p.name), out)
        if args.qc:
            cv2.imwrite(str(Path(args.qc) / f"qc_{p.name}"),
                        np.hstack([img, out]))
        n_ok += 1
        print(f"{p.name}: inpainted flags {complete}")
    print(f"\n{n_ok} inpainted -> {out_dir.resolve()}  ({n_skip} skipped)")


if __name__ == "__main__":
    main()
