"""
analyze_session.py  --  quantify ground-truth uncertainty for the EndoWrist
flag-based angle measurements.

Consumes the CSV(s) written by measure_flag_angles.py and runs the three
GT-quantification experiments agreed with the supervisor:

  repeatability   How tightly does the pipeline reproduce the SAME pose?
                  -> reports 1-sigma (this is the GT uncertainty you quote).

  known-rotation  Rotate the instrument by KNOWN amounts; does the pipeline
                  recover them? -> fits measured = slope*nominal + intercept,
                  reports slope (ideal 1.0), RMS residual, max residual.

  cross-method    Do independent methods (flag / manual / jig) agree on the
                  same poses? -> bias, SD of difference, 95% limits of
                  agreement (Bland-Altman), max |difference|.

A MANIFEST (optional) supplies per-image labels without editing the
generated gt CSV. Manifest is a CSV with an 'image' column plus any of:
  pose_id      groups repeats / links the same pose across methods
  nominal_deg  the known/commanded angle (for known-rotation)
  group        split a session into named sub-experiments

Examples (PowerShell):
  python analyze_session.py repeatability --gt rep_session.csv --joint theta_W
  python analyze_session.py known-rotation --gt rot.csv --manifest rot_manifest.csv --joint theta_W
  python analyze_session.py cross-method --csv flag=flag.csv --csv manual=manual.csv --key pose_id --joint theta_G1
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

JOINT_COLS = ["theta_W_deg", "theta_G1_deg", "theta_G2_deg"]
JOINT_CHOICES = ["theta_W", "theta_G1", "theta_G2",
                 "theta_W_deg", "theta_G1_deg", "theta_G2_deg"]


def norm_joint(j):
    """Accept 'theta_W' or 'theta_W_deg'; return the CSV column name."""
    if j is None:
        return None
    return j if j.endswith("_deg") else j + "_deg"

FLAG_REL_COLS = ["shaft_rel_deg", "clevis_rel_deg", "jaw1_rel_deg", "jaw2_rel_deg"]
SPAN_COLS = ["shaft_span_px", "clevis_span_px", "jaw1_span_px", "jaw2_span_px"]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def join_manifest(rows, manifest_path):
    """Left-join manifest columns onto gt rows by 'image'."""
    if not manifest_path:
        return rows
    man = {r["image"]: r for r in read_csv(manifest_path) if r.get("image")}
    extra_cols = [c for c in (next(iter(man.values()), {})) if c != "image"]
    for r in rows:
        m = man.get(r.get("image"), {})
        for c in extra_cols:
            r.setdefault(c, m.get(c))
            if r.get(c) in (None, ""):
                r[c] = m.get(c)
    return rows


def resolve_joint(rows, joint):
    """If joint is None, pick joint columns with >=3 numeric values."""
    joint = norm_joint(joint)
    if joint:
        return [joint]
    out = []
    for c in JOINT_COLS:
        n = sum(1 for r in rows if to_float(r.get(c)) is not None)
        if n >= 3:
            out.append(c)
    return out


def circ_guard(vals, label):
    """Warn if values span enough to risk angle wrap-around (naive stats)."""
    if max(vals) - min(vals) > 180.0:
        print(f"  WARNING: {label} spans >180 deg; naive linear stats may be "
              f"wrong near the +/-180 wrap. Check the data.")


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------

def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "ptp": float(np.ptp(a)),
    }


# ----------------------------------------------------------------------------
# Mode: repeatability
# ----------------------------------------------------------------------------

def mode_repeatability(args):
    rows = join_manifest(read_csv(args.gt), args.manifest)

    groups = {}
    if args.group_col:
        for r in rows:
            groups.setdefault(r.get(args.group_col) or "_", []).append(r)
    else:
        groups["all"] = rows

    joints = resolve_joint(rows, args.joint)
    if not joints:
        raise SystemExit("No joint columns with >=3 values found. "
                         "Use --joint or check the CSV.")

    print(f"\n=== REPEATABILITY  ({Path(args.gt).name}) ===")
    print("1-sigma (std) is the ground-truth repeatability you quote.\n")

    report_rows = []
    for gname, grows in groups.items():
        if args.group_col:
            print(f"-- group: {gname}  (n_images={len(grows)})")
        for jc in joints:
            vals = [to_float(r.get(jc)) for r in grows]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                print(f"  {jc:>14}: <2 values, skipped")
                continue
            circ_guard(vals, jc)
            s = summarize(vals)
            print(f"  {jc:>14}:  n={s['n']:<3}  mean={s['mean']:8.3f}  "
                  f"1sigma={s['std']:6.3f}  range={s['ptp']:6.3f}  "
                  f"[{s['min']:.3f}, {s['max']:.3f}] deg")
            report_rows.append({"group": gname, "quantity": jc, **s})

        # Per-flag relative-angle and span noise (diagnoses the noisy flag).
        for fc in FLAG_REL_COLS:
            vals = [to_float(r.get(fc)) for r in grows]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                s = summarize(vals)
                report_rows.append({"group": gname, "quantity": fc, **s})
        for sc in SPAN_COLS:
            vals = [to_float(r.get(sc)) for r in grows]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 1:
                a = np.asarray(vals)
                report_rows.append({"group": gname, "quantity": sc,
                                    "n": a.size, "mean": float(a.mean()),
                                    "std": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
                                    "min": float(a.min()), "max": float(a.max()),
                                    "ptp": float(np.ptp(a))})

    _write_report_csv(args.out_prefix, "repeatability", report_rows)

    if HAVE_MPL and args.plot:
        _plot_repeatability(groups, joints, args)


def _plot_repeatability(groups, joints, args):
    for jc in joints:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (gname, grows) in enumerate(groups.items()):
            vals = [to_float(r.get(jc)) for r in grows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            x = np.full(len(vals), i) + np.random.uniform(-0.05, 0.05, len(vals))
            ax.scatter(x, vals, alpha=0.7, label=gname)
            ax.errorbar(i, np.mean(vals),
                        yerr=(np.std(vals, ddof=1) if len(vals) > 1 else 0),
                        fmt="_", color="k", capsize=6, markersize=20)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(list(groups.keys()))
        ax.set_ylabel(f"{jc} (deg)")
        ax.set_title(f"Repeatability: {jc}  (bar = mean +/- 1sigma)")
        if args.group_col:
            ax.legend(fontsize=8)
        fig.tight_layout()
        out = f"{args.out_prefix}_repeatability_{jc}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  plot -> {Path(out).resolve()}")


# ----------------------------------------------------------------------------
# Mode: known-rotation
# ----------------------------------------------------------------------------

def mode_known_rotation(args):
    rows = join_manifest(read_csv(args.gt), args.manifest)
    nominal_col = args.nominal_col
    joints = resolve_joint(rows, args.joint)

    print(f"\n=== KNOWN-ROTATION  ({Path(args.gt).name}) ===")
    print("Ideal: slope=1.000, intercept=mounting offset, residuals ~ chain noise.\n")

    report_rows = []
    for jc in joints:
        pairs = [(to_float(r.get(nominal_col)), to_float(r.get(jc)))
                 for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            print(f"  {jc:>14}: <3 paired points, skipped")
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        # Least-squares line y = m x + b
        A = np.vstack([x, np.ones_like(x)]).T
        (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - (m * x + b)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rms = float(np.sqrt(np.mean(resid ** 2)))
        maxr = float(np.max(np.abs(resid)))
        print(f"  {jc:>14}:  n={len(pairs):<3}  slope={m:6.4f}  "
              f"intercept={b:7.3f}  R^2={r2:6.4f}  "
              f"RMS_resid={rms:6.3f}  max_resid={maxr:6.3f} deg")
        report_rows.append({"quantity": jc, "n": len(pairs), "slope": m,
                            "intercept": b, "r2": r2, "rms_resid": rms,
                            "max_resid": maxr})
        if HAVE_MPL and args.plot:
            _plot_known_rotation(x, y, m, b, resid, jc, args)

    _write_report_csv(args.out_prefix, "known_rotation", report_rows)


def _plot_known_rotation(x, y, m, b, resid, jc, args):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.linspace(x.min(), x.max(), 100)
    ax1.scatter(x, y, alpha=0.8, label="measured")
    ax1.plot(xs, m * xs + b, "r-", label=f"fit (slope {m:.3f})")
    ax1.plot(xs, xs + b, "k--", alpha=0.5, label="unity (slope 1)")
    ax1.set_xlabel(f"nominal (deg)")
    ax1.set_ylabel(f"measured {jc} (deg)")
    ax1.set_title(f"Known-rotation: {jc}")
    ax1.legend(fontsize=8)
    ax2.scatter(x, resid, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("nominal (deg)")
    ax2.set_ylabel("residual (deg)")
    ax2.set_title(f"Residuals (RMS {np.sqrt(np.mean(resid**2)):.3f} deg)")
    fig.tight_layout()
    out = f"{args.out_prefix}_known_rotation_{jc}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  plot -> {Path(out).resolve()}")


# ----------------------------------------------------------------------------
# Mode: cross-method
# ----------------------------------------------------------------------------

def mode_cross_method(args):
    methods = {}
    for spec in args.csv:
        if "=" not in spec:
            raise SystemExit(f"--csv expects name=path, got '{spec}'")
        name, path = spec.split("=", 1)
        methods[name] = read_csv(path)
    if len(methods) < 2:
        raise SystemExit("cross-method needs >=2 --csv name=path entries")

    key = args.key
    joints = [norm_joint(args.joint)] if args.joint else JOINT_COLS

    print(f"\n=== CROSS-METHOD AGREEMENT  (key='{key}') ===")
    print("bias = mean(A-B); 95% LoA = bias +/- 1.96*SD(A-B) (Bland-Altman).\n")

    names = list(methods)
    indexed = {}
    for name, rows in methods.items():
        d = {}
        for r in rows:
            k = r.get(key)
            if k:
                d[k] = r
        indexed[name] = d

    report_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            for jc in joints:
                diffs, means = [], []
                common = set(indexed[a_name]) & set(indexed[b_name])
                for k in sorted(common):
                    va = to_float(indexed[a_name][k].get(jc))
                    vb = to_float(indexed[b_name][k].get(jc))
                    if va is not None and vb is not None:
                        diffs.append(va - vb)
                        means.append((va + vb) / 2)
                if len(diffs) < 2:
                    continue
                d = np.array(diffs)
                bias = float(d.mean())
                sd = float(d.std(ddof=1))
                loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
                maxabs = float(np.max(np.abs(d)))
                print(f"  {a_name} vs {b_name}  [{jc}]:  n={len(d):<3}  "
                      f"bias={bias:7.3f}  SD={sd:6.3f}  "
                      f"95%LoA=[{loa_lo:.3f}, {loa_hi:.3f}]  "
                      f"max|diff|={maxabs:6.3f} deg")
                report_rows.append({"pair": f"{a_name}_vs_{b_name}",
                                    "quantity": jc, "n": len(d), "bias": bias,
                                    "sd": sd, "loa_lo": loa_lo, "loa_hi": loa_hi,
                                    "max_abs": maxabs})
                if HAVE_MPL and args.plot:
                    _plot_bland_altman(np.array(means), d, bias, sd,
                                       a_name, b_name, jc, args)

    _write_report_csv(args.out_prefix, "cross_method", report_rows)


def _plot_bland_altman(means, diffs, bias, sd, a_name, b_name, jc, args):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(means, diffs, alpha=0.8)
    ax.axhline(bias, color="b", label=f"bias {bias:.3f}")
    ax.axhline(bias + 1.96 * sd, color="r", ls="--", label="95% LoA")
    ax.axhline(bias - 1.96 * sd, color="r", ls="--")
    ax.set_xlabel(f"mean of methods (deg)")
    ax.set_ylabel(f"{a_name} - {b_name} (deg)")
    ax.set_title(f"Bland-Altman {jc}: {a_name} vs {b_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = f"{args.out_prefix}_bland_{jc}_{a_name}_{b_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  plot -> {Path(out).resolve()}")


# ----------------------------------------------------------------------------

def _write_report_csv(prefix, mode, report_rows):
    if not report_rows:
        print("  (no results to write)")
        return
    out = f"{prefix}_{mode}.csv"
    fields = list({k for r in report_rows for k in r})
    # stable column order
    head = [c for c in ("group", "pair", "quantity", "n") if c in fields]
    rest = [c for c in fields if c not in head]
    fields = head + rest
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in report_rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\n  report -> {Path(out).resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("repeatability", help="1-sigma GT uncertainty")
    pr.add_argument("--gt", required=True)
    pr.add_argument("--manifest", default=None)
    pr.add_argument("--joint", default=None, choices=JOINT_CHOICES)
    pr.add_argument("--group-col", default=None,
                    help="manifest column to split groups by (e.g. 'group')")
    pr.add_argument("--out-prefix", default="analysis")
    pr.add_argument("--plot", action="store_true")
    pr.set_defaults(func=mode_repeatability)

    pk = sub.add_parser("known-rotation", help="linearity vs known angles")
    pk.add_argument("--gt", required=True)
    pk.add_argument("--manifest", default=None)
    pk.add_argument("--nominal-col", default="nominal_deg")
    pk.add_argument("--joint", default=None, choices=JOINT_CHOICES)
    pk.add_argument("--out-prefix", default="analysis")
    pk.add_argument("--plot", action="store_true")
    pk.set_defaults(func=mode_known_rotation)

    pc = sub.add_parser("cross-method", help="agreement between methods")
    pc.add_argument("--csv", action="append", required=True,
                    help="name=path ; repeat for each method")
    pc.add_argument("--key", default="pose_id")
    pc.add_argument("--joint", default=None, choices=JOINT_CHOICES)
    pc.add_argument("--out-prefix", default="analysis")
    pc.add_argument("--plot", action="store_true")
    pc.set_defaults(func=mode_cross_method)

    args = ap.parse_args()
    if not HAVE_MPL and getattr(args, "plot", False):
        print("matplotlib unavailable; skipping plots.")
    args.func(args)


if __name__ == "__main__":
    main()
