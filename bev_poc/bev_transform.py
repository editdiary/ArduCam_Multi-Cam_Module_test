import argparse
from pathlib import Path

import cv2
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Bird's Eye View (BEV/IPM) images using fisheye calibration data."
    )
    parser.add_argument("--left_img",   type=str, required=True)
    parser.add_argument("--right_img",  type=str, required=True)
    parser.add_argument("--cal_dir",    type=str, default="calib_params")
    parser.add_argument("--x_max",      type=float, default=3.0,
                        help="Forward range in meters (default: 3.0)")
    parser.add_argument("--x_min",      type=float, default=0.0,
                        help="Minimum forward distance in meters (default: 0.0)")
    parser.add_argument("--y_range",    type=float, default=1.5,
                        help="Half lateral range; BEV covers [-y_range, +y_range] (default: 1.5)")
    parser.add_argument("--res",        type=float, default=0.02,
                        help="Resolution in meters/pixel (default: 0.02 → 2 cm/px)")
    parser.add_argument("--output_dir", type=str,   default="bev_output")
    parser.add_argument("--save_lut",   action="store_true",
                        help="Save LUT + quality map to cal_dir for reuse")
    parser.add_argument("--load_lut",   action="store_true",
                        help="Load pre-computed LUT from cal_dir (skip recomputation)")
    parser.add_argument("--no_grid",    action="store_true",
                        help="Disable metric grid overlay")
    return parser.parse_args()


# ── Calibration loading ───────────────────────────────────────────────────────

def load_cal(cal_dir, side):
    base = Path(cal_dir)
    intr = np.load(base / f"{side}_intrinsic_result.npz")
    extr = np.load(base / f"{side}_extrinsic_result.npz")
    K    = intr["camera_matrix"].astype(np.float64)
    D    = intr["dist_coeffs"].astype(np.float64)
    rvec = extr["rvec"].astype(np.float64)
    tvec = extr["tvec"].astype(np.float64)
    return K, D, rvec, tvec


# ── BEV config ────────────────────────────────────────────────────────────────

class BevConfig:
    def __init__(self, x_min, x_max, y_range, res):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = -y_range
        self.y_max = +y_range
        self.res   = res
        self.H = round((x_max - x_min) / res)
        self.W = round((y_range * 2)   / res)

    def pixel_to_world(self, rows, cols):
        X = self.x_max - rows * self.res
        Y = self.y_max - cols * self.res
        return X, Y

    def matches(self, npz):
        """Check whether a loaded LUT npz matches this BEV config."""
        return (
            abs(float(npz["x_min"]) - self.x_min) < 1e-6 and
            abs(float(npz["x_max"]) - self.x_max) < 1e-6 and
            abs(float(npz["y_min"]) - self.y_min) < 1e-6 and
            abs(float(npz["y_max"]) - self.y_max) < 1e-6 and
            abs(float(npz["res"])   - self.res)   < 1e-9
        )


# ── LUT building ──────────────────────────────────────────────────────────────

def build_lut(K, D, rvec, tvec, bev: BevConfig, img_h, img_w):
    """
    Build remap tables and a quality map for one camera.

    Quality = cos(θ) where θ is the angle between the BEV world point and the
    camera's optical axis.  Pixels near the optical axis get high quality (≈1);
    extreme-angle pixels get low quality (≈0), which suppresses blurry/stretched
    regions during blending.

    Returns:
        map_x, map_y   float32 H×W  — remap tables (-1 = out of view)
        quality        float32 H×W  — per-pixel quality weight in [0, 1]
    """
    H, W = bev.H, bev.W
    rows, cols = np.meshgrid(np.arange(H, dtype=np.float32),
                              np.arange(W, dtype=np.float32), indexing='ij')
    X, Y = bev.pixel_to_world(rows, cols)
    Z    = np.zeros_like(X, dtype=np.float32)

    world_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 1, 3).astype(np.float32)

    # ── remap tables ────────────────────────────────────────────────────────
    img_pts, _ = cv2.fisheye.projectPoints(world_pts, rvec, tvec, K, D)
    map_x = img_pts[:, 0, 0].reshape(H, W).astype(np.float32)
    map_y = img_pts[:, 0, 1].reshape(H, W).astype(np.float32)

    oob = (map_x < 0) | (map_x >= img_w) | (map_y < 0) | (map_y >= img_h)
    map_x[oob] = -1.0
    map_y[oob] = -1.0

    # ── quality map: cos(θ) from optical axis ────────────────────────────────
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t    = np.asarray(tvec, dtype=np.float64).ravel()
    P_cam = (R @ world_pts.reshape(-1, 3).T + t[:, None]).T  # (N, 3)

    cos_theta = P_cam[:, 2] / (np.linalg.norm(P_cam, axis=1) + 1e-9)
    quality   = np.clip(cos_theta, 0.0, 1.0).reshape(H, W).astype(np.float32)
    quality[oob] = 0.0

    return map_x, map_y, quality


# ── LUT save / load ───────────────────────────────────────────────────────────

def save_lut(cal_dir, side, map_x, map_y, quality, bev: BevConfig):
    path = Path(cal_dir) / f"{side}_bev_lut.npz"
    np.savez(str(path),
             map_x=map_x, map_y=map_y, quality=quality,
             x_min=bev.x_min, x_max=bev.x_max,
             y_min=bev.y_min, y_max=bev.y_max, res=bev.res)
    print(f"  LUT saved → {path}")


def load_lut(cal_dir, side, bev: BevConfig):
    path = Path(cal_dir) / f"{side}_bev_lut.npz"
    if not path.exists():
        return None, None, None
    data = np.load(str(path))
    if not bev.matches(data):
        print(f"  [WARN] Cached LUT params differ from current BEV config — recomputing.")
        return None, None, None
    return data["map_x"], data["map_y"], data["quality"]


# ── BEV rendering ─────────────────────────────────────────────────────────────

def apply_lut(img, map_x, map_y):
    return cv2.remap(img, map_x, map_y,
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(0, 0, 0))


def blend_bev(left_bev, right_bev, left_quality, right_quality):
    """
    Quality-weighted blend of two BEV images.

    Weight for each camera = cos(θ) from its optical axis.
    This ensures that extreme-angle (blurry/stretched) pixels from one camera
    do not contaminate the sharp pixels of the other camera in the same region.
    """
    l_valid = np.any(left_bev  > 0, axis=2)
    r_valid = np.any(right_bev > 0, axis=2)

    wl = (left_quality  * l_valid).astype(np.float32)[:, :, None]
    wr = (right_quality * r_valid).astype(np.float32)[:, :, None]
    w_total = wl + wr

    combined = np.where(
        w_total > 0,
        (left_bev.astype(np.float32) * wl +
         right_bev.astype(np.float32) * wr) / (w_total + 1e-9),
        0.0
    )
    return combined.clip(0, 255).astype(np.uint8)


def draw_grid(bev_img, bev: BevConfig, interval_m=1.0):
    img = bev_img.copy()
    color_grid   = (80,  80,  80)
    color_center = (0,  200,   0)
    color_text   = (255, 0, 0)

    for x in np.arange(np.ceil(bev.x_min / interval_m) * interval_m,
                        bev.x_max + 1e-9, interval_m):
        row = int(round((bev.x_max - x) / bev.res))
        if 0 <= row < bev.H:
            cv2.line(img, (0, row), (bev.W - 1, row), color_grid, 1)
            cv2.putText(img, f"{x:.0f}m", (4, row - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_text, 1)

    for y in np.arange(np.ceil(bev.y_min / interval_m) * interval_m,
                        bev.y_max + 1e-9, interval_m):
        col = int(round((bev.y_max - y) / bev.res))
        if 0 <= col < bev.W:
            lc = color_center if abs(y) < 1e-6 else color_grid
            cv2.line(img, (col, 0), (col, bev.H - 1), lc, 2 if lc == color_center else 1)

    orig_row = bev.H - 1
    orig_col = int(round(bev.W / 2))
    cv2.drawMarker(img, (orig_col, orig_row), color_center,
                   cv2.MARKER_CROSS, 12, 2)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bev = BevConfig(args.x_min, args.x_max, args.y_range, args.res)
    print(f"BEV : x=[{bev.x_min}, {bev.x_max}]m  "
          f"y=[{bev.y_min}, {bev.y_max}]m  "
          f"res={bev.res}m/px  size={bev.W}×{bev.H}px")

    results = {}
    qualities = {}

    for side, img_path in [("left", args.left_img), ("right", args.right_img)]:
        print(f"\nProcessing {side} camera …")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] Cannot read {img_path}")
            continue
        img_h, img_w = img.shape[:2]

        map_x, map_y, quality = None, None, None

        if args.load_lut:
            map_x, map_y, quality = load_lut(args.cal_dir, side, bev)
            if map_x is not None:
                print(f"  LUT loaded from {args.cal_dir}/{side}_bev_lut.npz")

        if map_x is None:
            K, D, rvec, tvec = load_cal(args.cal_dir, side)
            print(f"  Building LUT ({bev.W}×{bev.H}) …")
            map_x, map_y, quality = build_lut(K, D, rvec, tvec, bev, img_h, img_w)
            valid_pct = 100.0 * np.sum(map_x >= 0) / (bev.H * bev.W)
            print(f"  Valid BEV pixels: {valid_pct:.1f}%")

        if args.save_lut:
            save_lut(args.cal_dir, side, map_x, map_y, quality, bev)

        bev_img = apply_lut(img, map_x, map_y)
        if not args.no_grid:
            bev_img = draw_grid(bev_img, bev)

        out_path = out_dir / f"{side}_bev.jpg"
        cv2.imwrite(str(out_path), bev_img)
        print(f"  Saved → {out_path}")

        results[side]   = bev_img
        qualities[side] = quality

    if "left" in results and "right" in results:
        combined = blend_bev(results["left"], results["right"],
                             qualities["left"], qualities["right"])
        if not args.no_grid:
            combined = draw_grid(combined, bev)
        out_path = out_dir / "combined_bev.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"\nCombined BEV saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
