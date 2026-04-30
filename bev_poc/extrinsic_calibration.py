import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrinsic camera calibration for IPM using a ground-plane checkerboard."
    )
    parser.add_argument("--left_img",  type=str, required=True,
                        help="Left camera image path")
    parser.add_argument("--right_img", type=str, required=True,
                        help="Right camera image path")
    parser.add_argument("--left_npz",  type=str, required=True,
                        help="Left camera intrinsic .npz")
    parser.add_argument("--right_npz", type=str, required=True,
                        help="Right camera intrinsic .npz")
    parser.add_argument("--board_w",     type=int,   default=10,
                        help="Inner corners along y-axis / width (default: 10)")
    parser.add_argument("--board_h",     type=int,   default=7,
                        help="Inner corners along x-axis / depth (default: 7)")
    parser.add_argument("--square_size", type=float, default=36.0,
                        help="Square size in mm (default: 36.0)")
    parser.add_argument("--origin_x",   type=float, default=0.3,
                        help="x-coordinate of nearest inner corner in meters (default: 0.3)")
    parser.add_argument("--origin_y",   type=float, default=0.0,
                        help="y-coordinate of nearest inner corner in meters (default: 0.0)")
    parser.add_argument("--origin_z",   type=float, default=0.0,
                        help="z-coordinate of board surface, i.e. board thickness (default: 0.0)")
    parser.add_argument("--output_dir", type=str,   default="calib_params",
                        help="Output directory for result .npz files (default: calib_params)")
    return parser.parse_args()


# ── World coordinates ─────────────────────────────────────────────────────────
# World frame: x=forward, y=left(+), z=up
# Board is flat on the ground (z=0).
# board_h corners along x (forward), board_w corners along y (left).
# Nearest corner = front-right = (origin_x, origin_y, origin_z).
#
# OpenCV findChessboardCorners returns corners row-major (top→bottom, left→right
# in the image).  For a forward-facing camera:
#   row 0   → top of image → farthest from camera → largest x
#   col 0   → left of image → leftmost             → largest y
#
def build_world_points(board_w, board_h, square_size_mm,
                       origin_x_m, origin_y_m=0.0, origin_z_m=0.0):
    s = square_size_mm / 1000.0
    pts = []
    for row in range(board_h):
        for col in range(board_w):
            x = origin_x_m + (board_h - 1 - row) * s
            y = origin_y_m + (board_w - 1 - col) * s
            pts.append([x, y, origin_z_m])
    return np.array(pts, dtype=np.float32)


# ── Corner detection ──────────────────────────────────────────────────────────

def detect_corners(img_path, board_size):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
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


# ── Fisheye solvePnP ──────────────────────────────────────────────────────────

def solve_pnp_fisheye(obj_pts, img_corners, K, D, min_height=0.05):
    """
    Fisheye-aware solvePnP with planar-ambiguity resolution.

    SOLVEPNP_IPPE returns both conjugate solutions for a planar target.
    Physical validity filters:
      - cam_z > min_height  : camera is above the board
      - R[2,0] > 0          : camera faces forward (+x direction)

    Returns (rvec, tvec, rms, is_valid).
    is_valid=True  → solution passed both physical filters.
    is_valid=False → fallback: no physically valid solution found for this
                     corner ordering (caller should prefer other orderings).
    """
    undist = cv2.fisheye.undistortPoints(
        img_corners.reshape(-1, 1, 2), K, D, R=None, P=K
    )
    zeros4 = np.zeros((4, 1), dtype=np.float64)
    try:
        n_sol, rvecs_out, tvecs_out, _ = cv2.solvePnPGeneric(
            obj_pts.reshape(-1, 1, 3).astype(np.float64),
            undist.reshape(-1, 1, 2).astype(np.float64),
            K, zeros4,
            flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        return None, None, float("inf"), False

    best_valid = (None, None, float("inf"))    # passes both physical filters
    best_any   = (None, None, float("inf"))    # unconditional best (fallback)

    for i in range(n_sol):
        rv, tv = rvecs_out[i], tvecs_out[i]
        R, _ = cv2.Rodrigues(rv)
        cam_z     = float((-R.T @ tv).ravel()[2])
        cam_fwd_x = float(R[2, 0])
        rms = _reprojection_rms(obj_pts, img_corners, rv, tv, K, D)

        if rms < best_any[2]:
            best_any = (rv, tv, rms)
        if cam_z > min_height and cam_fwd_x > 0.0 and rms < best_valid[2]:
            best_valid = (rv, tv, rms)

    if best_valid[0] is not None:
        return (*best_valid, True)
    if best_any[0] is not None:
        return (*best_any, False)
    return None, None, float("inf"), False


def _reprojection_rms(obj_pts, img_corners, rvec, tvec, K, D):
    proj, _ = cv2.fisheye.projectPoints(
        obj_pts.reshape(-1, 1, 3), rvec, tvec, K, D
    )
    diff = img_corners.reshape(-1, 2) - proj.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def solve_with_auto_flip(obj_pts, corners, board_h, board_w, K, D, min_height=0.05):
    """
    Try 4 possible corner orderings and return the best result.

    Cameras tilted sideways (e.g. ±45° yaw) see the board with
    left-right mirrored relative to a forward-facing camera, so a
    simple full-reverse is not always enough.  The 4 orderings cover:
      normal   – top-left first, left-to-right within each row
      flip-y   – mirror columns (left↔right within each row)
      flip-x   – mirror rows   (top↔bottom row order)
      flip-xy  – full reverse  (= flip-x + flip-y)
    """
    c = corners.reshape(board_h, board_w, 1, 2)
    candidates = [
        ("normal",  c.reshape(-1, 1, 2)),
        ("flip-y",  c[:, ::-1].reshape(-1, 1, 2)),
        ("flip-x",  c[::-1, :].reshape(-1, 1, 2)),
        ("flip-xy", c[::-1, ::-1].reshape(-1, 1, 2)),
    ]
    # Physically valid solutions (cam_z > 0, facing forward) are always
    # preferred over fallback solutions, regardless of RMS difference.
    best_valid   = (None, None, float("inf"), "none")
    best_fallback = (None, None, float("inf"), "none")

    for label, pts in candidates:
        rvec, tvec, rms, is_valid = solve_pnp_fisheye(obj_pts, pts, K, D, min_height)
        if rvec is None:
            continue
        if is_valid and rms < best_valid[2]:
            best_valid = (rvec, tvec, rms, label)
        if rms < best_fallback[2]:
            best_fallback = (rvec, tvec, rms, label)

    if best_valid[0] is not None:
        return best_valid
    return best_fallback


# ── Camera pose extraction ────────────────────────────────────────────────────

def extract_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).ravel()          # camera position in world frame

    # ZYX Euler angles (world frame → camera frame)
    yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    pitch = float(np.degrees(np.arctan2(-R[2, 0],
                  np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
    roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return R, cam_pos, roll, pitch, yaw, T


# ── Printing ──────────────────────────────────────────────────────────────────

def print_result(label, img_path, rvec, tvec, rms, flip_label, K, D, is_valid=True):
    R, cam_pos, roll, pitch, yaw, T = extract_pose(rvec, tvec)
    sep = "=" * 55

    print(f"\n{sep}")
    print(f"  Extrinsic Calibration — {label}")
    print(sep)
    print(f"  Image : {Path(img_path).name}")

    print(f"\n[Corner Detection]")
    print(f"  Status          : OK")
    valid_str = "OK (physically valid)" if is_valid else "FALLBACK (no valid solution found)"
    print(f"  Corner ordering : {flip_label}")
    print(f"  Solution        : {valid_str}")
    print(f"  Reprojection RMS: {rms:.6f} px")

    print(f"\n[Rotation Vector]")
    rv = rvec.ravel()
    print(f"  rvec = [{rv[0]:+.6f},  {rv[1]:+.6f},  {rv[2]:+.6f}]  (rad)")

    print(f"\n[Rotation Matrix  R  (world → camera)]")
    for i in range(3):
        print(f"  [{R[i,0]:+.6f}  {R[i,1]:+.6f}  {R[i,2]:+.6f}]")

    print(f"\n[Translation Vector  tvec  (world → camera, camera frame)]")
    tv = tvec.ravel()
    print(f"  [{tv[0]:+.6f},  {tv[1]:+.6f},  {tv[2]:+.6f}]  (m)")

    print(f"\n[Camera Pose in World]")
    print(f"  Position : x={cam_pos[0]:+.4f} m  "
          f"y={cam_pos[1]:+.4f} m  "
          f"z={cam_pos[2]:+.4f} m  (height above ground)")
    print(f"  Roll     = {roll:+.3f} deg  (x-axis tilt)")
    print(f"  Pitch    = {pitch:+.3f} deg  (y-axis tilt, negative = looking down)")
    print(f"  Yaw      = {yaw:+.3f} deg  (z-axis rotation)")

    print(f"\n[4×4 Transform  T_cam_world  (world → camera)]")
    for i in range(4):
        row_str = "  [" + "  ".join(f"{T[i,j]:+.6f}" for j in range(4)) + "]"
        print(row_str)

    print(sep)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_result(out_path, rvec, tvec, rms, K, D, image_size):
    R, cam_pos, roll, pitch, yaw, T = extract_pose(rvec, tvec)
    np.savez(
        str(out_path),
        rvec=rvec,
        tvec=tvec,
        R_mat=R,
        T_cam_world=T,
        cam_pos=cam_pos,
        roll=np.float64(roll),
        pitch=np.float64(pitch),
        yaw=np.float64(yaw),
        rms=np.float64(rms),
        camera_matrix=K,
        dist_coeffs=D,
        image_size=np.array(image_size),
        model=np.array("fisheye"),
    )
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_intrinsics(npz_path):
    data = np.load(npz_path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    square_size_m = args.square_size / 1000.0
    board_size    = (args.board_w, args.board_h)
    obj_pts       = build_world_points(args.board_w, args.board_h,
                                       args.square_size, args.origin_x,
                                       args.origin_y, args.origin_z)

    cameras = [
        ("LEFT",  args.left_img,  args.left_npz,  "left_extrinsic_result.npz"),
        ("RIGHT", args.right_img, args.right_npz, "right_extrinsic_result.npz"),
    ]

    for label, img_path, npz_path, out_name in cameras:
        print(f"\nProcessing {label} camera …")

        K, D = load_intrinsics(npz_path)
        corners, image_size = detect_corners(img_path, board_size)

        if corners is None:
            print(f"  [ERROR] Checkerboard not found in {img_path}", file=sys.stderr)
            continue

        rvec, tvec, rms, flip_label = solve_with_auto_flip(
            obj_pts, corners, args.board_h, args.board_w, K, D,
            min_height=args.origin_z + 0.05
        )

        R_check, _ = cv2.Rodrigues(rvec)
        cam_z_check = float((-R_check.T @ tvec).ravel()[2])
        fwd_x_check = float(R_check[2, 0])
        is_valid = cam_z_check > (args.origin_z + 0.05) and fwd_x_check > 0.0

        print_result(label, img_path, rvec, tvec, rms, flip_label, K, D, is_valid)

        out_path = output_dir / out_name
        save_result(out_path, rvec, tvec, rms, K, D, image_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
