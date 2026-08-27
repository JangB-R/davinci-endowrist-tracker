r"""
capture_frames.py -- keypress frame grabber for training-set capture from a
UVC camera (ELP demo camera). Saves full-resolution stills with sequential
names into a session folder, ready for capture_log.csv rows.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe capture_frames.py `
      --source 0 --out "..\..\data\raw\s03_2026-08-XX_bg1_lightA_vpN" `
      --width 1920 --height 1080

Keys (default sequential mode):
  SPACE   save current frame  (s01_0001.jpg, s01_0002.jpg, ...)
  q/Esc   quit

Paired mode (--paired), for a held-out eval session where each pose needs a
flagged GT frame AND a marker-free tracker frame of the same held pose:
  f   save FLAG frame   -> poseNNN_flag.jpg   (flags on; for ground truth)
  c   save CLEAN frame  -> poseNNN_clean.jpg  (clevis flag removed; ends pose)
  z   save zero-ref     -> zero_flag.jpg
  q/Esc   quit
The filename encodes the pairing, so flag<->clean stay matched automatically;
pose counter resumes from existing poseNNN_clean.jpg files. Keep this session
in its OWN folder and never upload it to the training/Roboflow project.

Logging: every saved frame gets a row appended to capture_log.csv in --out
(created with a header if missing). background_id / lighting_id /
viewpoint_id are parsed from the session folder name tokens (bg*, light*,
vp*) or overridden with --bg/--light/--vp; --flags-mounted and --notes set
the remaining columns. Commanded angles are left BLANK — fill them in Excel
after each capture block from your sweep plan. On startup, any existing
frames in --out that have no log row are back-filled.

Notes:
  - Lens focus/zoom and exposure must already be set and locked (spec §0).
    --try-lock asks the driver to disable auto-exposure/auto-WB via UVC;
    VERIFY it stuck by watching the preview while waving a hand in front —
    brightness must NOT adapt. Some boards ignore these properties.
  - Numbering continues from existing files in --out, so re-running mid-
    session is safe.
"""

import argparse
import csv
from pathlib import Path

import cv2

LOG_FIELDS = ["filename", "theta_W_cmd", "theta_G1_cmd", "theta_G2_cmd",
              "theta_R_cmd", "background_id", "lighting_id", "viewpoint_id",
              "flags_mounted", "notes"]


def session_tokens(folder_name):
    """bg1 / lightA / vpN tokens from the session folder name, if present."""
    tok = {"background_id": "", "lighting_id": "", "viewpoint_id": ""}
    for part in folder_name.split("_"):
        if part.startswith("bg"):
            tok["background_id"] = part
        elif part.startswith("light"):
            tok["lighting_id"] = part
        elif part.startswith("vp"):
            tok["viewpoint_id"] = part
    return tok


def append_log(out_dir, rows):
    p = out_dir / "capture_log.csv"
    new = not p.exists()
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def logged_filenames(out_dir):
    p = out_dir / "capture_log.csv"
    if not p.exists():
        return set()
    with open(p, newline="") as fh:
        return {r["filename"] for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0", help="webcam index")
    ap.add_argument("--out", required=True, help="session folder")
    ap.add_argument("--prefix", default=None,
                    help="filename prefix (default: folder name up to first _)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--try-lock", action="store_true",
                    help="attempt to disable auto-exposure/auto-white-balance")
    ap.add_argument("--preview-width", type=int, default=960,
                    help="preview window width in px (saved frames stay "
                         "full resolution)")
    ap.add_argument("--bg", default=None, help="override background_id")
    ap.add_argument("--light", default=None, help="override lighting_id")
    ap.add_argument("--vp", default=None, help="override viewpoint_id")
    ap.add_argument("--flags-mounted", action="store_true",
                    help="mark rows as flagged frames (flag-robustness set)")
    ap.add_argument("--notes", default="", help="notes column for all rows")
    ap.add_argument("--paired", action="store_true",
                    help="paired held-out eval mode: keys f/c/z save "
                         "poseNNN_flag.jpg / poseNNN_clean.jpg / zero_flag.jpg")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or out.name.split("_")[0]
    existing = sorted(out.glob(f"{prefix}_*.jpg"))
    n = int(existing[-1].stem.split("_")[-1]) if existing else 0
    if n:
        print(f"resuming numbering after {existing[-1].name}")

    tok = session_tokens(out.name)
    if args.bg:
        tok["background_id"] = args.bg
    if args.light:
        tok["lighting_id"] = args.light
    if args.vp:
        tok["viewpoint_id"] = args.vp

    def make_row(fname, note=None, flags_val=None):
        return {"filename": fname, "theta_W_cmd": "", "theta_G1_cmd": "",
                "theta_G2_cmd": "", "theta_R_cmd": "",
                "flags_mounted": (int(args.flags_mounted) if flags_val is None
                                  else int(flags_val)),
                "notes": args.notes if note is None else note, **tok}

    unlogged = [p.name for p in existing if p.name not in logged_filenames(out)]
    if unlogged:
        append_log(out, [make_row(f, note="backfilled") for f in unlogged])
        print(f"back-filled {len(unlogged)} existing frame(s) into "
              f"capture_log.csv -- fill their commanded angles")

    cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.source}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.try_lock:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual on DSHOW
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        print("requested manual exposure/WB -- VERIFY the preview does not "
              "adapt to a hand waved in front of the lens")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"streaming {w}x{h} -> {out.resolve()}")
    pose = 0
    if args.paired:
        done = sorted(out.glob("pose*_clean.jpg"))
        pose = max((int(p.stem[4:7]) for p in done), default=0)
        if pose:
            print(f"resuming paired capture after pose {pose:03d}")
        print("PAIRED:  f = FLAG frame   c = CLEAN frame (ends pose)   "
              "z = zero-ref   q/Esc = quit")
    else:
        print("SPACE = save    q/Esc = quit")

    frame = None

    def save(name, note, flags_val):
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        append_log(out, [make_row(name, note=note, flags_val=flags_val)])
        print(f"  saved {name} (+log row)")

    cur = pose + 1          # pose currently being captured (paired mode)
    flag_done = False
    while True:
        ok, frame = cap.read()
        if not ok:
            print("stream ended / camera lost")
            break
        scale = min(1.0, args.preview_width / frame.shape[1])
        disp = (frame if scale >= 1.0 else
                cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA))
        status = (f"pose {cur:03d}  flag:{'Y' if flag_done else '-'}"
                  if args.paired else f"saved: {n}")
        cv2.putText(disp, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("capture_frames", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if args.paired:
            if k == ord("z"):
                save("zero_flag.jpg", "zero-ref", 1)
            elif k == ord("f"):
                save(f"pose{cur:03d}_flag.jpg", f"pose{cur:03d} flag", 1)
                flag_done = True
            elif k == ord("c"):
                if not flag_done:
                    print(f"  WARNING: no flag frame saved for pose {cur:03d}")
                save(f"pose{cur:03d}_clean.jpg", f"pose{cur:03d} clean", 0)
                cur += 1
                flag_done = False
        elif k == ord(" "):
            n += 1
            save(f"{prefix}_{n:04d}.jpg", None, None)

    cap.release()
    cv2.destroyAllWindows()
    if args.paired:
        print(f"paired capture done: {cur - 1} pose(s) in {out.resolve()}")
    else:
        print(f"{n} frames in {out.resolve()}")
    print("capture_log.csv updated -- now fill the commanded-angle columns")


if __name__ == "__main__":
    main()
