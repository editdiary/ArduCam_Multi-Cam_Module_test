import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Intrinsic camera calibration using a checkerboard pattern.")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Path to folder containing calibration images (.jpg / .jpeg / .png)")
    parser.add_argument("--board_w", type=int, default=10,
                        help="Number of inner corners along the width (default: 10)")
    parser.add_argument("--board_h", type=int, default=7,
                        help="Number of inner corners along the height (default: 7)")
    parser.add_argument("--square_size", type=float, default=36.0,
                        help="Size of one square in mm (default: 36.0)")
    parser.add_argument("--model", type=str, default="fisheye", choices=["pinhole", "fisheye"],
                        help="Camera projection model (default: fisheye)")
    parser.add_argument("--outlier_thresh", type=float, default=5.0,
                        help="Per-image RMS threshold (px) for outlier rejection (default: 5.0, 0 = disabled)")
    parser.add_argument("--focal_length_mm", type=float, default=None,
                        help="Lens focal length in mm for initial K estimate (e.g. 1.56)")
    parser.add_argument("--pixel_pitch_um", type=float, default=3.0,
                        help="Sensor pixel pitch in µm (default: 3.0 for AR0234)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path (default: <img_dir>/intrinsic_result.npz)")
    return parser.parse_args()


def build_object_points(board_w, board_h, square_size):
    objp = np.zeros((board_h * board_w, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(img_path, board_size):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None, gray.shape[::-1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners, gray.shape[::-1]


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_pinhole(obj_points, img_points, image_size):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return rms, K, dist, rvecs, tvecs


def calibrate_fisheye(obj_points, img_points, image_size, f_init=None):
    obj_pts_fish = [p.reshape(-1, 1, 3) for p in obj_points]
    img_pts_fish = [p.reshape(-1, 1, 2) for p in img_points]

    if f_init is None:
        f_init = image_size[0] / np.pi  # fallback: 180° HFOV assumption
    K = np.array([[f_init,    0., image_size[0] / 2.0],
                  [   0., f_init, image_size[1] / 2.0],
                  [   0.,    0.,               1.    ]], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    n = len(obj_pts_fish)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n)]

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW            |
             cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_pts_fish, img_pts_fish, image_size,
        K, D, rvecs, tvecs, flags
    )
    return rms, K, D, rvecs, tvecs


# ── Per-image reprojection error ──────────────────────────────────────────────

def per_image_rms_pinhole(obj_pts, img_pts, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    diff = img_pts.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def per_image_rms_fisheye(obj_pts, img_pts, rvec, tvec, K, dist):
    projected, _ = cv2.fisheye.projectPoints(
        obj_pts.reshape(-1, 1, 3), rvec, tvec, K, dist
    )
    diff = img_pts.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_errors(model, obj_points, img_points, rvecs, tvecs, K, dist):
    fn = per_image_rms_fisheye if model == "fisheye" else per_image_rms_pinhole
    return [fn(obj_points[i], img_points[i], rvecs[i], tvecs[i], K, dist)
            for i in range(len(obj_points))]


# ── Iterative outlier rejection ───────────────────────────────────────────────

def iterative_calibrate(model, obj_points, img_points, valid_files, image_size, outlier_thresh, f_init=None):
    """Calibrate, remove outliers above threshold, repeat until stable."""
    if model == "fisheye":
        calibrate = lambda o, i, s: calibrate_fisheye(o, i, s, f_init)
    else:
        calibrate = calibrate_pinhole

    indices = list(range(len(valid_files)))
    removed_files = []
    iteration = 0

    while True:
        iteration += 1
        sub_obj = [obj_points[i] for i in indices]
        sub_img = [img_points[i] for i in indices]

        rms, K, dist, rvecs, tvecs = calibrate(sub_obj, sub_img, image_size)
        errors = compute_errors(model, sub_obj, sub_img, rvecs, tvecs, K, dist)

        if outlier_thresh <= 0:
            break

        bad_local = [j for j, e in enumerate(errors) if e > outlier_thresh]
        if not bad_local:
            break

        # Remove only the single worst outlier per iteration for stability
        worst = max(bad_local, key=lambda j: errors[j])
        removed_files.append(valid_files[indices[worst]])
        print(f"  [iter {iteration}] Removed outlier ({errors[worst]:.2f} px): "
              f"{Path(valid_files[indices[worst]]).name}")
        indices.pop(worst)

        if len(indices) < 10:
            print(f"  [WARNING] Only {len(indices)} images remain — stopping outlier removal.")
            break

    used_files = [valid_files[i] for i in indices]
    sub_obj    = [obj_points[i] for i in indices]
    sub_img    = [img_points[i] for i in indices]
    return rms, K, dist, rvecs, tvecs, errors, used_files, removed_files, sub_obj, sub_img


# ── FOV ───────────────────────────────────────────────────────────────────────

def compute_fov_pinhole(K, image_size):
    w, h = image_size
    fov_h = 2 * np.degrees(np.arctan2(w / 2.0, K[0, 0]))
    fov_v = 2 * np.degrees(np.arctan2(h / 2.0, K[1, 1]))
    return fov_h, fov_v


def compute_fov_fisheye(K, image_size):
    # Equidistant projection: r = f * θ  →  FOV = 2 * (W/2) / f
    w, h = image_size
    fov_h = np.degrees(w / K[0, 0])
    fov_v = np.degrees(h / K[1, 1])
    return fov_h, fov_v


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(args, detect_failed, outlier_removed, used_files,
                  K, dist, rms_global, per_image_errors, image_size, output_path):
    board_str = f"{args.board_w}x{args.board_h}"
    total = len(detect_failed) + len(outlier_removed) + len(used_files)
    sep = "=" * 55

    print(f"\n{sep}")
    print("  Intrinsic Calibration Results")
    print(sep)
    print(f"  Model      : {args.model}")
    print(f"  Board      : {board_str} inner corners  |  Square: {args.square_size} mm")
    print(f"  Image dir  : {args.img_dir}")

    # ── Image selection ──────────────────────────────────────────
    print(f"\n[Image Selection]")
    print(f"  Total          : {total:>3d} images")
    print(f"  Used           : {len(used_files):>3d} images  (calibration OK)")
    print(f"  Outlier removed: {len(outlier_removed):>3d} images  (RMS > {args.outlier_thresh:.1f} px)")
    print(f"  Detection fail : {len(detect_failed):>3d} images  (corner not found)")
    if outlier_removed:
        for f in outlier_removed:
            print(f"      [outlier] {Path(f).name}")
    if detect_failed:
        for f in sorted(detect_failed):
            print(f"      [no corner] {Path(f).name}")

    # ── Camera matrix ────────────────────────────────────────────
    print(f"\n[Camera Matrix]")
    print(f"  fx = {K[0,0]:10.4f} px    fy = {K[1,1]:10.4f} px")
    print(f"  cx = {K[0,2]:10.4f} px    cy = {K[1,2]:10.4f} px")
    print(f"  skew = {K[0,1]:.6f}")

    # ── Distortion coefficients ───────────────────────────────────
    d = dist.ravel()
    if args.model == "fisheye":
        print(f"\n[Distortion Coefficients]  (equidistant: k1, k2, k3, k4)")
        labels = ["k1", "k2", "k3", "k4"]
    else:
        print(f"\n[Distortion Coefficients]  (pinhole: k1, k2, p1, p2, k3, ...)")
        labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    for i, val in enumerate(d):
        label = labels[i] if i < len(labels) else f"d{i}"
        print(f"  {label} = {val:+.8f}")

    # ── FOV ──────────────────────────────────────────────────────
    if args.model == "fisheye":
        fov_h, fov_v = compute_fov_fisheye(K, image_size)
    else:
        fov_h, fov_v = compute_fov_pinhole(K, image_size)
    print(f"\n[Field of View]  ({'equidistant' if args.model == 'fisheye' else 'pinhole'})")
    print(f"  Horizontal FOV = {fov_h:.2f} deg")
    print(f"  Vertical   FOV = {fov_v:.2f} deg")
    print(f"  Image size     = {image_size[0]} x {image_size[1]} px")

    # ── Per-image reprojection error ──────────────────────────────
    print(f"\n[Reprojection Error per Image]")
    sorted_errors = sorted(zip(used_files, per_image_errors), key=lambda x: x[1])
    max_name_len = max(len(Path(f).name) for f, _ in sorted_errors)
    max_name_len = min(max_name_len, 45)
    header = f"  {'Rank':>4}  {'Filename':<{max_name_len}}  Error (px)"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, (fname, err) in enumerate(sorted_errors, start=1):
        name = Path(fname).name
        if len(name) > max_name_len:
            name = name[:max_name_len - 3] + "..."
        print(f"  {rank:>4}  {name:<{max_name_len}}  {err:.6f}")

    mean_rms = float(np.mean(per_image_errors))
    max_rms  = float(np.max(per_image_errors))
    min_rms  = float(np.min(per_image_errors))
    print(f"\n  Mean  RMS reprojection error : {mean_rms:.6f} px")
    print(f"  Min   RMS reprojection error : {min_rms:.6f} px")
    print(f"  Max   RMS reprojection error : {max_rms:.6f} px")
    print(f"  Overall RMS (cv2.calibrate)  : {rms_global:.6f} px")

    # ── Saved ─────────────────────────────────────────────────────
    print(f"\n[Saved]")
    print(f"  Results saved to : {output_path}")
    print(f"  Keys : camera_matrix, dist_coeffs, rvecs, tvecs,")
    print(f"         rms, per_image_errors, image_size, used_files, model")
    print(sep + "\n")


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        print(f"[ERROR] Image directory not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else img_dir / "intrinsic_result.npz"

    board_size = (args.board_w, args.board_h)
    objp = build_object_points(args.board_w, args.board_h, args.square_size)

    image_paths = sorted(set(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.JPG")) +
        list(img_dir.glob("*.JPEG")) +
        list(img_dir.glob("*.PNG"))
    ))

    if not image_paths:
        print(f"[ERROR] No images found in: {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(image_paths)} images for checkerboard ({args.board_w}x{args.board_h})"
          f"  [model: {args.model}]")

    obj_points = []
    img_points = []
    valid_files = []
    detect_failed = []
    image_size = None

    for i, img_path in enumerate(image_paths, start=1):
        corners, size = detect_corners(img_path, board_size)
        status = "OK" if corners is not None else "FAIL"
        print(f"  [{i:>3}/{len(image_paths)}] {img_path.name}  ->  {status}")
        if corners is not None:
            obj_points.append(objp)
            img_points.append(corners)
            valid_files.append(str(img_path))
            if image_size is None:
                image_size = size
        else:
            detect_failed.append(str(img_path))

    if len(valid_files) < 10:
        print(f"\n[ERROR] Not enough valid images ({len(valid_files)} found, need >= 10).", file=sys.stderr)
        sys.exit(1)

    if args.focal_length_mm is not None:
        f_init = args.focal_length_mm / (args.pixel_pitch_um * 1e-3)
        print(f"  Initial fx = {f_init:.1f} px  "
              f"(from {args.focal_length_mm}mm / {args.pixel_pitch_um}µm)")
    else:
        f_init = None
        print(f"  Initial fx = {(image_size[0] / np.pi):.1f} px  (fallback: 180° HFOV assumption)")

    thresh_str = f"{args.outlier_thresh:.1f} px" if args.outlier_thresh > 0 else "disabled"
    print(f"\nRunning {args.model} calibration  [outlier threshold: {thresh_str}]")

    rms_global, K, dist, rvecs, tvecs, per_image_errors, used_files, outlier_removed, \
        used_obj, used_img = iterative_calibrate(
            args.model, obj_points, img_points, valid_files, image_size,
            args.outlier_thresh, f_init=f_init
        )

    np.savez(
        str(output_path),
        camera_matrix=K,
        dist_coeffs=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        rms=np.float64(rms_global),
        per_image_errors=np.array(per_image_errors),
        image_size=np.array(image_size),
        used_files=np.array(used_files),
        model=np.array(args.model)
    )

    print_results(
        args, detect_failed, outlier_removed, used_files,
        K, dist, rms_global, per_image_errors, image_size, output_path
    )


if __name__ == "__main__":
    main()
