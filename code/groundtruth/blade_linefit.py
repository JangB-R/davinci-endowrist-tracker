r"""
blade_linefit.py  --  interactive, click-seeded line fitting for the EndoWrist
jaws (or shaft / any straight metal edge), to test whether silhouette-based
angle measurement is good enough WITHOUT flags.

How it works:
  - Opens the image. For each blade you want to measure, click TWO rough points
    roughly along the blade (tip end, then pivot end). Order doesn't matter.
  - The tool searches a band between/around those two clicks, finds the bright
    metal pixels, and fits a line by PCA (principal axis of the bright cluster).
  - It draws the fitted line and prints the angle PLUS honest fit-quality
    diagnostics so you can tell if it actually locked onto the blade edge or
    wandered onto background:
        n_px            how many bright pixels it fitted (more = better)
        perp_rms_px     scatter perpendicular to the line; SMALL = clean edge,
                        LARGE = it grabbed wood grain / glare too
        straightness    fraction of variance along the principal axis (~1.0
                        = a clean line; <0.98 = blobby, not a line)

Controls:
  left-click      add a seed point (every 2 clicks = one blade fit)
  u               undo last blade
  r               reset all
  s               save annotated image + print summary
  q / Esc         quit

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe blade_linefit.py path\to\photo.jpeg
  optional: --bandwidth 60   (half-width of search band in px; widen for
                              thick blades, narrow if it grabs neighbours)
            --bright-pct 78  (brightness percentile inside band counted as
                              metal; raise if background leaks in)

Sign convention: angle is in image coordinates, positive = counter-clockwise
as displayed (y-axis-down negated). Subtract two blade angles for the opening,
or subtract the shaft angle to reference a jaw to the shaft.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


class Fitter:
    def __init__(self, img, bandwidth, bright_pct):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        self.H, self.W = self.gray.shape
        self.bandwidth = bandwidth
        self.bright_pct = bright_pct
        self.seeds = []          # pending click points
        self.blades = []         # list of dicts with fit results
        # Downscale factor for display so big phone images fit on screen.
        self.scale = min(1.0, 1100 / max(self.W, self.H))
        self.disp_base = cv2.resize(
            img, (int(self.W * self.scale), int(self.H * self.scale)))

    # ---- geometry ----
    def fit_band(self, p0, p1):
        p0 = np.array(p0, float)
        p1 = np.array(p1, float)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 5:
            return None
        vn = v / L
        n = np.array([-vn[1], vn[0]])
        half = self.bandwidth
        xmin = int(max(0, min(p0[0], p1[0]) - half))
        xmax = int(min(self.W, max(p0[0], p1[0]) + half))
        ymin = int(max(0, min(p0[1], p1[1]) - half))
        ymax = int(min(self.H, max(p0[1], p1[1]) + half))
        sub = self.gray[ymin:ymax, xmin:xmax]
        if sub.size == 0:
            return None
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        along = (xx - p0[0]) * vn[0] + (yy - p0[1]) * vn[1]
        perp = (xx - p0[0]) * n[0] + (yy - p0[1]) * n[1]
        band = (along > -10) & (along < L + 10) & (np.abs(perp) < half)
        if band.sum() < 50:
            return None
        thr = np.percentile(sub[band], self.bright_pct)
        bright = band & (sub > thr)
        if bright.sum() < 30:
            return None
        pts = np.stack([xx[bright], yy[bright]], -1).astype(float)
        c = pts.mean(0)
        _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        # PCA axis is undirected (+d and -d describe the same line). Pin the
        # direction to the user's seed order (p0 -> p1) so the reported angle
        # is unambiguous and angle-differences are meaningful.
        if d @ (p1 - p0) < 0:
            d = -d
        # diagnostics
        proj_along = (pts - c) @ d
        proj_perp = (pts - c) @ np.array([-d[1], d[0]])
        perp_rms = float(np.sqrt(np.mean(proj_perp ** 2)))
        var_total = float(np.sum(s ** 2))
        straightness = float(s[0] ** 2 / var_total) if var_total > 0 else 0.0
        ang = float(np.degrees(np.arctan2(-d[1], d[0])))
        return {"c": c, "d": d, "ang": ang, "n": int(bright.sum()),
                "perp_rms": perp_rms, "straightness": straightness,
                "p0": p0, "p1": p1}

    # ---- interaction ----
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = x / self.scale, y / self.scale
            self.seeds.append((fx, fy))
            if len(self.seeds) == 2:
                res = self.fit_band(self.seeds[0], self.seeds[1])
                if res:
                    self.blades.append(res)
                    self._print_blade(len(self.blades) - 1, res)
                else:
                    print("  fit failed (too few bright pixels) - try again, "
                          "click closer along the actual blade")
                self.seeds = []
            self.redraw()

    def _print_blade(self, i, r):
        quality = ("clean" if r["perp_rms"] < 6 and r["straightness"] > 0.985
                   else "SUSPECT - check overlay")
        print(f"  blade {i}: angle={r['ang']:7.2f} deg   n={r['n']:<6} "
              f"perp_rms={r['perp_rms']:5.2f}px  straight={r['straightness']:.4f}  "
              f"[{quality}]")

    def redraw(self):
        disp = self.disp_base.copy()
        colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]
        for i, r in enumerate(self.blades):
            col = colors[i % len(colors)]
            c, d = r["c"], r["d"]
            p = ((c - d * 500) * self.scale).astype(int)
            q = ((c + d * 500) * self.scale).astype(int)
            cv2.line(disp, tuple(p), tuple(q), col, 2)
            lbl = ((c) * self.scale).astype(int)
            cv2.putText(disp, f"{i}:{r['ang']:.1f}", tuple(lbl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        for s in self.seeds:
            sp = (int(s[0] * self.scale), int(s[1] * self.scale))
            cv2.circle(disp, sp, 4, (0, 255, 255), -1)
        # pairwise angles overlay
        y = 24
        for i in range(len(self.blades)):
            for j in range(i + 1, len(self.blades)):
                a = abs(((self.blades[i]["ang"] - self.blades[j]["ang"]) + 180)
                        % 360 - 180)
                cv2.putText(disp, f"angle {i}-{j}: {a:.2f} deg", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
                y += 24
        cv2.imshow("blade_linefit", disp)

    def summary(self, out_path):
        print("\n=== SUMMARY ===")
        for i, r in enumerate(self.blades):
            self._print_blade(i, r)
        for i in range(len(self.blades)):
            for j in range(i + 1, len(self.blades)):
                a = abs(((self.blades[i]["ang"] - self.blades[j]["ang"]) + 180)
                        % 360 - 180)
                print(f"  angle between blade {i} and {j}: {a:.2f} deg")
        vis = self.img.copy()
        colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]
        for i, r in enumerate(self.blades):
            c, d = r["c"], r["d"]
            p = (c - d * 600).astype(int)
            q = (c + d * 600).astype(int)
            cv2.line(vis, tuple(p), tuple(q), colors[i % len(colors)], 4)
        cv2.imwrite(out_path, vis)
        print(f"  annotated image -> {Path(out_path).resolve()}")

    def export_jaw_csv(self, csv_path, image_name, labels):
        """Append one row per labelled blade to a jaw-angle CSV that
        measure_flag_angles.py can join on 'image'. labels maps blade index
        -> 'jaw1'/'jaw2'. Existing rows for the same image+jaw are replaced."""
        import csv as _csv
        fields = ["image", "jaw", "jaw_line_deg", "perp_rms_px",
                  "straightness", "n_px", "quality"]
        rows = []
        p = Path(csv_path)
        if p.exists():
            with open(p, newline="") as fh:
                rows = [r for r in _csv.DictReader(fh)]
        for i, r in enumerate(self.blades):
            jaw = labels.get(i)
            if jaw is None:
                continue
            quality = ("clean" if r["perp_rms"] < 8 and r["straightness"] > 0.985
                       else "SUSPECT")
            rows = [x for x in rows
                    if not (x["image"] == image_name and x["jaw"] == jaw)]
            rows.append({"image": image_name, "jaw": jaw,
                         "jaw_line_deg": round(r["ang"], 3),
                         "perp_rms_px": round(r["perp_rms"], 2),
                         "straightness": round(r["straightness"], 4),
                         "n_px": r["n"], "quality": quality})
        with open(p, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  jaw angles -> {p.resolve()}")
        n_susp = sum(1 for x in rows if x["image"] == image_name
                     and x["quality"] == "SUSPECT")
        if n_susp:
            print(f"  NOTE: {n_susp} SUSPECT fit(s) for this image - "
                  f"re-seed or re-shoot before trusting.")


def main():
    ap = argparse.ArgumentParser(description="Click-seeded blade line fit.")
    ap.add_argument("image")
    ap.add_argument("--bandwidth", type=int, default=60)
    ap.add_argument("--bright-pct", type=float, default=78)
    ap.add_argument("--jaw-csv", default=None,
                    help="If set, pressing 's' also appends jaw-line angles "
                         "to this CSV (keyed by image filename) for "
                         "measure_flag_angles.py to consume.")
    ap.add_argument("--labels", default=None,
                    help="Comma list mapping blade index to jaw, e.g. "
                         "'0=jaw1,1=jaw2'. If omitted with --jaw-csv set, "
                         "you'll be prompted in the terminal on save.")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    f = Fitter(img, args.bandwidth, args.bright_pct)
    cv2.namedWindow("blade_linefit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("blade_linefit", f.disp_base.shape[1], f.disp_base.shape[0])
    cv2.setMouseCallback("blade_linefit", f.on_mouse)
    f.redraw()
    print("Click 2 points along each blade (tip, then pivot). "
          "Keys: u=undo  r=reset  s=save  q=quit")

    out_path = str(Path(args.image).with_name(
        Path(args.image).stem + "_bladefit.png"))
    image_name = Path(args.image).name

    preset_labels = {}
    if args.labels:
        for tok in args.labels.split(","):
            idx, jaw = tok.split("=")
            preset_labels[int(idx)] = jaw.strip()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("u") and f.blades:
            f.blades.pop()
            f.redraw()
            print("  undid last blade")
        if k == ord("r"):
            f.blades, f.seeds = [], []
            f.redraw()
            print("  reset")
        if k == ord("s"):
            f.summary(out_path)
            if args.jaw_csv:
                labels = dict(preset_labels)
                if not labels:
                    print("  label blades for the jaw CSV "
                          "(blade index shown in overlay):")
                    for i in range(len(f.blades)):
                        ans = input(f"    blade {i} -> jaw1 / jaw2 / skip: ").strip()
                        if ans in ("jaw1", "jaw2"):
                            labels[i] = ans
                f.export_jaw_csv(args.jaw_csv, image_name, labels)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
