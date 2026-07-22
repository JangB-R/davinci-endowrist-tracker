r"""
mask_geometry.py -- geometry layer of the markerless tracker.

Converts per-part segmentation masks (shaft, clevis, jaw1, jaw2) into joint
angles (theta_W, theta_G1, theta_G2) with per-angle confidence. This is the
deployment version of the PCA line-fit prototyped in
code/groundtruth/blade_linefit.py: the segmentation mask replaces the
click-seeded bright-pixel band and the same principal-axis fit runs on the
mask pixels; the fit diagnostics become per-frame confidence.

Conventions (identical to the GT tools):
  - Angles are in-plane, on (ideally undistorted) image coordinates,
    positive = counter-clockwise as displayed (image y negated).
  - Joint angles are differences of part-axis angles, per flag_config.json's
    joint_definitions:
        theta_W  = clevis_axis - shaft_axis
        theta_G1 = jaw1_axis   - clevis_axis
        theta_G2 = jaw2_axis   - clevis_axis
    then global sign flip (--flip-sign, one-time check per camera setup, same
    as measure_flag_angles.py) and optional per-joint zero offsets captured
    at a commanded-zero pose (replaces the GT session zero-reference image).
  - theta_R is NOT measured in v1 (roll constrained near zero) but is present
    in the output with confidence 0.0 so the published interface is frozen now.

Direction pinning: a PCA axis is undirected (+d and -d are the same line).
The GT tool pinned direction by click order; here it is pinned by the
kinematic chain: every part axis points DISTALLY (shaft toward clevis,
clevis away from shaft, jaws away from clevis), so angle differences are
unambiguous.

Confidence (0..1 per joint) = min over the joint's two parts of:
  clip(n_px / (2*min_px)) * elongation term * detector confidence
where elongation = sqrt(major/minor PCA std). A filled mask's perp_rms is
dominated by part WIDTH (unlike the GT edge fit), so elongation -- how
line-like the mask is -- is the meaningful shape score. The thresholds in
QUALITY are provisional; tune them on the first real labelled frames.

Pure numpy/cv2; no model, no ROS. Validate with
test_mask_geometry_synthetic.py before trusting on real masks.
"""

from dataclasses import dataclass, field
import math

import numpy as np

PART_NAMES = ("shaft", "clevis", "jaw1", "jaw2")
JOINT_DEFS = {  # joint -> (moving part, reference part); mirrors flag_config
    "theta_W":  ("clevis", "shaft"),
    "theta_G1": ("jaw1", "clevis"),
    "theta_G2": ("jaw2", "clevis"),
}
JOINT_NAMES = ("theta_W", "theta_G1", "theta_G2", "theta_R")

# Per-part quality expectations for confidence scoring. PROVISIONAL defaults
# sized for ~1080-4K frames at the calibrated working distance -- tune on the
# first real labelled frames and keep the tuned values here, in one place.
QUALITY = {
    "shaft":  {"min_px": 800, "elong_zero": 2.0, "elong_full": 6.0},
    "clevis": {"min_px": 300, "elong_zero": 1.05, "elong_full": 1.7},
    "jaw1":   {"min_px": 150, "elong_zero": 1.5, "elong_full": 4.0},
    "jaw2":   {"min_px": 150, "elong_zero": 1.5, "elong_full": 4.0},
}


def wrap_deg(a):
    """Wrap angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def part_axis(mask):
    """PCA principal-axis fit of a binary mask.

    Returns dict with centroid c(2,), undirected unit direction d(2,), n_px,
    elongation, perp_rms_px, straightness -- or None if the mask is empty /
    degenerate. Direction is pinned later by pin_directions()."""
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if n < 20:
        return None
    c = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.stack([xs, ys]).astype(float))
    evals, evecs = np.linalg.eigh(cov)          # ascending eigenvalues
    lam_minor, lam_major = max(float(evals[0]), 0.0), max(float(evals[1]), 0.0)
    if lam_major <= 0:
        return None
    d = evecs[:, 1].astype(float)
    d /= np.linalg.norm(d)
    return {
        "c": c, "d": d, "n_px": n,
        "elongation": math.sqrt(lam_major / max(lam_minor, 1e-9)),
        "perp_rms_px": math.sqrt(lam_minor),
        "straightness": lam_major / (lam_major + lam_minor),
    }


def _pin(ax, from_pt, to_pt):
    """Flip ax['d'] if needed so it points from from_pt toward to_pt."""
    if float(ax["d"] @ (np.asarray(to_pt) - np.asarray(from_pt))) < 0:
        ax["d"] = -ax["d"]


def pin_directions(axes):
    """Pin every axis distal along the kinematic chain. Only pins pairs that
    are jointly present (a joint needs both its parts anyway)."""
    sh, cl = axes.get("shaft"), axes.get("clevis")
    if sh and cl:
        _pin(sh, sh["c"], cl["c"])              # shaft points toward clevis
        _pin(cl, sh["c"], cl["c"])              # clevis points away from shaft
    elif cl:
        jaw_cs = [axes[j]["c"] for j in ("jaw1", "jaw2") if axes.get(j)]
        if jaw_cs:                              # fallback: distal = toward jaws
            _pin(cl, cl["c"], np.mean(jaw_cs, axis=0))
    for j in ("jaw1", "jaw2"):
        if axes.get(j) and cl:
            _pin(axes[j], cl["c"], axes[j]["c"])  # jaw points away from clevis


def axis_angle_deg(ax):
    """Display-CCW-positive angle of a pinned axis (image y negated)."""
    return math.degrees(math.atan2(-ax["d"][1], ax["d"][0]))


def _part_conf(name, ax, det_conf, quality):
    q = quality[name]
    if ax is None or ax["n_px"] < q["min_px"]:
        return 0.0
    c_n = min(1.0, ax["n_px"] / (2.0 * q["min_px"]))
    c_e = (ax["elongation"] - q["elong_zero"]) / (q["elong_full"] - q["elong_zero"])
    c_e = min(1.0, max(0.0, c_e))
    return c_n * c_e * (1.0 if det_conf is None else float(det_conf))


@dataclass
class FrameResult:
    angles: dict = field(default_factory=dict)       # joint -> deg or None
    confidence: dict = field(default_factory=dict)   # joint -> 0..1
    parts: dict = field(default_factory=dict)        # part -> diagnostics
    warnings: list = field(default_factory=list)


def compute_frame(masks, det_confs=None, flip_sign=False, zero_offsets=None,
                  quality=None):
    """masks: {part_name: HxW bool/uint8}; det_confs: {part_name: 0..1} from
    the detector (optional); zero_offsets: {joint: deg} captured at commanded
    zero (optional). Returns FrameResult."""
    det_confs = det_confs or {}
    zero_offsets = zero_offsets or {}
    quality = quality or QUALITY
    sign = -1.0 if flip_sign else 1.0

    res = FrameResult()
    axes = {}
    for name in PART_NAMES:
        m = masks.get(name)
        axes[name] = part_axis(np.asarray(m).astype(bool)) if m is not None else None
        if axes[name] is None and m is not None:
            res.warnings.append(f"{name}_degenerate_mask")
        elif m is None:
            res.warnings.append(f"{name}_missing")
    pin_directions(axes)

    part_conf = {}
    for name in PART_NAMES:
        ax = axes[name]
        part_conf[name] = _part_conf(name, ax, det_confs.get(name), quality)
        if ax is not None:
            res.parts[name] = {
                "angle_deg": round(axis_angle_deg(ax), 3),
                "n_px": ax["n_px"],
                "elongation": round(ax["elongation"], 2),
                "perp_rms_px": round(ax["perp_rms_px"], 2),
                "straightness": round(ax["straightness"], 4),
                "conf": round(part_conf[name], 3),
            }

    # jaw-side diagnostic: sign of the cross product clevis_axis x (jaw
    # centroid - clevis centroid). At near-zero roll jaw1/jaw2 should sit on
    # consistent, opposite-ish sides; a flipped pair suggests an identity swap.
    cl = axes.get("clevis")
    if cl:
        for j in ("jaw1", "jaw2"):
            if axes.get(j) and j in res.parts:
                v = axes[j]["c"] - cl["c"]
                cross = cl["d"][0] * v[1] - cl["d"][1] * v[0]
                res.parts[j]["side"] = int(np.sign(cross)) if cross else 0

    for jname, (mv, rf) in JOINT_DEFS.items():
        if axes.get(mv) is not None and axes.get(rf) is not None:
            raw = sign * wrap_deg(axis_angle_deg(axes[mv]) - axis_angle_deg(axes[rf]))
            res.angles[jname] = wrap_deg(raw - zero_offsets.get(jname, 0.0))
            res.confidence[jname] = min(part_conf[mv], part_conf[rf])
        else:
            res.angles[jname] = None
            res.confidence[jname] = 0.0
    res.angles["theta_R"] = None                 # v1: roll not measured
    res.confidence["theta_R"] = 0.0
    return res


def calibrate_zero(frame_results):
    """Mean raw joint angles over frames captured at the commanded-zero pose
    (run compute_frame with zero_offsets=None). Save the returned dict as
    JSON and pass it back as zero_offsets at runtime."""
    offsets = {}
    for jname in JOINT_DEFS:
        vals = [fr.angles[jname] for fr in frame_results
                if fr.angles.get(jname) is not None]
        if vals:
            offsets[jname] = round(float(np.mean(vals)), 3)
    return offsets
