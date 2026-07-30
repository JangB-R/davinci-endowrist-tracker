r"""
capture_frames.py -- keypress frame grabber for training-set capture from a
UVC camera (ELP demo camera). Saves full-resolution stills with sequential
names into a session folder, ready for capture_log.csv rows.

Usage (PowerShell):
  C:/Users/RAYAAN/anaconda3/python.exe capture_frames.py `
      --source 0 --out "..\..\data\raw\s03_2026-08-XX_bg1_lightA_vpN" `
      --width 1920 --height 1080

Keys:
  SPACE   save current frame  (s01_0001.jpg, s01_0002.jpg, ...)
  q/Esc   quit

Notes:
  - Lens focus/zoom and exposure must already be set and locked (spec §0).
    --try-lock asks the driver to disable auto-exposure/auto-WB via UVC;
    VERIFY it stuck by watching the preview while waving a hand in front —
    brightness must NOT adapt. Some boards ignore these properties.
  - Numbering continues from existing files in --out, so re-running mid-
    session is safe.
"""

import argparse
from pathlib import Path

import cv2


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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or out.name.split("_")[0]
    existing = sorted(out.glob(f"{prefix}_*.jpg"))
    n = int(existing[-1].stem.split("_")[-1]) if existing else 0
    if n:
        print(f"resuming numbering after {existing[-1].name}")

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
    print("SPACE = save    q/Esc = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("stream ended / camera lost")
            break
        scale = min(1.0, args.preview_width / frame.shape[1])
        disp = (frame if scale >= 1.0 else
                cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA))
        cv2.putText(disp, f"saved: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("capture_frames", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            n += 1
            name = f"{prefix}_{n:04d}.jpg"
            cv2.imwrite(str(out / name),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
            print(f"  saved {name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"{n} frames in {out.resolve()} -- now fill capture_log.csv")


if __name__ == "__main__":
    main()
