"""
Camera calibration from ChArUco board images.
Run from code/calibration/ with images in calib_images/.
"""

import cv2
import numpy as np
import glob
import os
from PIL import Image, ImageOps

# --- EDIT THESE TO MATCH YOUR PRINTED BOARD ---
SQUARES_X = 11           # number of checker squares along the long side
SQUARES_Y = 8            # number of checker squares along the short side
SQUARE_LENGTH = 0.015    # checker side length, METRES (measure your print!)
MARKER_LENGTH = 0.011    # ArUco marker side, METRES (measure your print!)
IMAGE_DIR = "calib_images"
DEBUG_DIR = "debug_output"
# ----------------------------------------------

# Dictionaries to try, in order of likelihood for your board
DICTS_TO_TRY = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
]


def load_image(path):
    """Load image respecting EXIF rotation. Returns BGR numpy array."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)  # crucial for iPhone JPEGs
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_image_files(folder):
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def detect_best_dictionary(sample_images):
    """Try every dictionary on a few sample images, return the best."""
    print("\nProbing dictionaries to find the right one...")
    scores = {}
    for name, dict_id in DICTS_TO_TRY:
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(d)
        total = 0
        for path in sample_images:
            img = load_image(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, ids, _ = detector.detectMarkers(gray)
            total += 0 if ids is None else len(ids)
        scores[name] = (total, dict_id)
        print(f"  {name:25s}: {total} markers across {len(sample_images)} sample images")

    best_name = max(scores, key=lambda k: scores[k][0])
    best_count, best_id = scores[best_name]
    if best_count == 0:
        return None, None
    return best_name, best_id


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    images = find_image_files(IMAGE_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGE_DIR}/")
    print(f"Found {len(images)} images in {IMAGE_DIR}/")

    # Sanity check the first image
    first = load_image(images[0])
    print(f"First image size after EXIF correction: "
          f"{first.shape[1]} x {first.shape[0]} pixels")
    cv2.imwrite(os.path.join(DEBUG_DIR, "first_image_loaded.jpg"), first)
    print(f"Saved {DEBUG_DIR}/first_image_loaded.jpg "
          f"-- open this and check the board is right-side up.")
    # Force dictionary explicitly — board is labelled DICT_4X4
    best_name = "DICT_4X4_50"
    best_id = cv2.aruco.DICT_4X4_50
    print(f"\nUsing dictionary: {best_name} (forced)")

    # Build the ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(best_id)
    board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    board.setLegacyPattern(True)   # <-- the fix
    detector = cv2.aruco.CharucoDetector(board)

    all_corners, all_ids = [], []
    image_size = None
    accepted = 0

    for i, fname in enumerate(images):
        img = load_image(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)

        n_corners = 0 if ch_ids is None else len(ch_ids)
        status = "OK" if n_corners > 10 else "SKIP"
        print(f"  [{status}] {os.path.basename(fname)}: {n_corners} corners")

        if n_corners > 10:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            accepted += 1

            # Save annotated debug image for the first few
            if accepted <= 3:
                vis = img.copy()
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
                out = os.path.join(DEBUG_DIR, f"detected_{accepted}.jpg")
                cv2.imwrite(out, vis)

    print(f"\nAccepted {accepted} of {len(images)} images.")
    if accepted < 10:
        raise SystemExit("Need at least 10 good images. Recapture with more "
                         "variety and check debug_output/ to see what was detected.")
# Diagnostic: report what we're calibrating against
    print(f"\nBoard parameters being used:")
    print(f"  squaresX = {SQUARES_X}, squaresY = {SQUARES_Y}")
    print(f"  square length = {SQUARE_LENGTH} m")
    print(f"  marker length = {MARKER_LENGTH} m")
    print(f"  Image size: {image_size}")
    print(f"  Legacy pattern: {board.getLegacyPattern()}")
    print(f"  Total chessboard corners on this board: "
          f"{len(board.getChessboardCorners())}")
    print(f"  Per-image corner counts:")
    counts = [len(ids) for ids in all_ids]
    print(f"    min {min(counts)}, max {max(counts)}, "
          f"median {int(np.median(counts))}")
    
    print("Running calibration...")
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=cv2.CALIB_FIX_K3,
    )

    print(f"\n{'='*50}")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"{'='*50}")
    if ret < 0.5:
        print("Excellent.")
    elif ret < 1.0:
        print("Good, usable.")
    elif ret < 2.0:
        print("Marginal -- consider recapturing for better variety.")
    else:
        print("Poor -- something is off. Recapture or check setup.")

    print(f"\nCamera matrix K:\n{K}")
    print(f"\nDistortion coefficients:\n{dist.ravel()}")

    np.savez("camera_calibration.npz",
             camera_matrix=K,
             dist_coeffs=dist,
             image_size=image_size,
             reprojection_error=ret,
             dictionary=best_name)
    print("\nSaved camera_calibration.npz")

    print("\nPer-image reprojection: mean / worst-corner")
    obj_points_full = board.getChessboardCorners()
    errors, worsts = [], []
    for i, (rv, tv, corners_i, ids_i) in enumerate(zip(rvecs, tvecs, all_corners, all_ids)):
        obj_points = obj_points_full[ids_i.flatten()]
        proj, _ = cv2.projectPoints(obj_points, rv, tv, K, dist)
        diff = np.linalg.norm((corners_i - proj).reshape(-1, 2), axis=1)
        mean_err = diff.mean()
        worst_err = diff.max()
        errors.append(mean_err)
        worsts.append(worst_err)
        flag = " <-- worst is very large" if worst_err > 10 else ""
        print(f"  Image {i+1:3d}: mean {mean_err:6.3f} px, "
              f"worst-corner {worst_err:6.3f} px, "
              f"{len(ids_i)} corners{flag}")

    errors = np.array(errors)
    worsts = np.array(worsts)
    print(f"\nMean of per-image means:   {errors.mean():.3f} px")
    print(f"Max worst-corner across all images: {worsts.max():.3f} px")
# Verification: compute total RMS reprojection error manually
    # This is the standard definition and should match if all is well.
    total_error_sq = 0.0
    total_points = 0
    for rv, tv, corners_i, ids_i in zip(rvecs, tvecs, all_corners, all_ids):
        obj_points = obj_points_full[ids_i.flatten()]
        proj, _ = cv2.projectPoints(obj_points, rv, tv, K, dist)
        diff = (corners_i - proj).reshape(-1, 2)
        total_error_sq += np.sum(diff ** 2)
        total_points += len(diff)
    manual_rms = np.sqrt(total_error_sq / total_points)
    print(f"\nManual overall RMS reprojection error: {manual_rms:.4f} px")
    print(f"OpenCV-reported `ret` value:          {ret:.4f} px")
    print(f"Mean of per-image errors:             {errors.mean():.4f} px")

if __name__ == "__main__":
    main()