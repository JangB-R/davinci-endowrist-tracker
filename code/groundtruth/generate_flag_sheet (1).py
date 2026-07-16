"""
generate_flag_sheet.py
Generates a printable sheet of fiducial flags for EndoWrist ground-truth capture.

Each flag = two ArUco markers (DICT_5X5_100) on a strip, base -> tip.
Print at 100% scale (no 'fit to page'), then VERIFY the 50 mm scale bar
with a ruler before cutting anything. Acceptable: 50.0 +/- 0.2 mm.
Note: scale accuracy only affects span-vs-precision estimates, NOT angle
accuracy (angles are scale-invariant), so minor printer error is tolerable.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe generate_flag_sheet.py --dpi 600 --out flag_sheet.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(script_dir / "flag_config.json"),
                    help="Path to flag_config.json (default: next to this script)")
    ap.add_argument("--dpi", type=int, default=600,
                    help="Print resolution. 600 recommended for 6 mm markers.")
    ap.add_argument("--out", default="flag_sheet.png")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    dict_name = cfg["flag_dictionary"]
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    dpi = args.dpi
    page_w, page_h = mm_to_px(190, dpi), mm_to_px(150, dpi)  # fits A4 with margin
    sheet = np.full((page_h, page_w), 255, dtype=np.uint8)

    y_cursor = mm_to_px(10, dpi)
    x_margin = mm_to_px(10, dpi)
    label_scale = dpi / 300.0

    for name, f in cfg["flags"].items():
        marker_px = mm_to_px(f["marker_mm"], dpi)
        # White quiet zone around each marker: >= 1 marker-bit width.
        # 5x5 dict => 7 bits incl. border; quiet zone = marker_px / 7, min 2 mm.
        quiet = max(mm_to_px(2, dpi), marker_px // 7)
        span_px = mm_to_px(f["span_mm"], dpi)
        strip_h = marker_px + 2 * quiet
        strip_w = span_px + marker_px + 2 * quiet

        x0, y0 = x_margin, y_cursor
        # Strip outline (cut line)
        cv2.rectangle(sheet, (x0, y0), (x0 + strip_w, y0 + strip_h), 0, 1)

        for which, mid in (("base", f["id_base"]), ("tip", f["id_tip"])):
            img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)
            cx = x0 + quiet + (0 if which == "base" else span_px)
            cy = y0 + quiet
            sheet[cy:cy + marker_px, cx:cx + marker_px] = img

        label = (f"{name}  ids {f['id_base']}->{f['id_tip']}  "
                 f"span {f['span_mm']}mm  marker {f['marker_mm']}mm  "
                 f"(base=left, tip=right)")
        cv2.putText(sheet, label, (x0 + strip_w + mm_to_px(4, dpi), y0 + strip_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * label_scale, 0,
                    max(1, int(label_scale)), cv2.LINE_AA)
        y_cursor += strip_h + mm_to_px(8, dpi)

    # 50 mm scale-verification bar
    bar_len = mm_to_px(50, dpi)
    bx, by = x_margin, y_cursor + mm_to_px(5, dpi)
    cv2.rectangle(sheet, (bx, by), (bx + bar_len, by + mm_to_px(3, dpi)), 0, -1)
    cv2.putText(sheet, "SCALE BAR: must measure 50.0 mm (+/- 0.2 mm). Print at 100%.",
                (bx, by + mm_to_px(10, dpi)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * label_scale, 0, max(1, int(label_scale)), cv2.LINE_AA)

    # Write outputs with PHYSICAL SIZE embedded. OpenCV's imwrite stores no DPI
    # metadata, so print dialogs guess the scale (and guess wrong). The PDF is
    # the primary artefact: PDFs are defined in physical units, so printing at
    # "Actual size" is unambiguous. A DPI-tagged PNG is written as backup.
    from PIL import Image
    pil_img = Image.fromarray(sheet)
    out_base = Path(args.out).with_suffix("")
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    pil_img.save(pdf_path, "PDF", resolution=float(dpi))
    pil_img.save(png_path, dpi=(dpi, dpi))
    print(f"Wrote {pdf_path.resolve()}  <-- PRINT THIS ONE, at 'Actual size' / 100%")
    print(f"Wrote {png_path.resolve()}  (backup, DPI-tagged)")
    print("After printing: verify scale bar. Cover markers with MATTE tape only;")
    print("glossy lamination can kill detection under lamps.")


if __name__ == "__main__":
    main()
