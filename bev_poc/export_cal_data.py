"""
Export calib_params/*.npz to human-readable YAML files and print a verification summary.
Usage: python export_cal_data.py [--cal_dir calib_params]
"""
import argparse
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cal_dir", default="calib_params")
    return p.parse_args()


# ── YAML helpers ─────────────────────────────────────────────────────────────

def _mat_yaml(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.ravel()
    rows, cols = (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (1, arr.size)
    data_str = ", ".join(f"{v:.10g}" for v in flat)
    return (f"{name}: !!opencv-matrix\n"
            f"   rows: {rows}\n"
            f"   cols: {cols}\n"
            f"   dt: d\n"
            f"   data: [ {data_str} ]\n")


def _scalar_yaml(name, val):
    return f"{name}: {val}\n"


# ── Per-type export ───────────────────────────────────────────────────────────

def export_intrinsic(data, out_path):
    K   = data["camera_matrix"]
    D   = data["dist_coeffs"].ravel()
    sz  = data["image_size"] if "image_size" in data else [0, 0]
    rms = float(data["rms"]) if "rms" in data else float("nan")
    mdl = str(data["model"]) if "model" in data else "fisheye"

    lines = ["%YAML:1.0\n---\n",
             _mat_yaml("camera_matrix", K),
             _mat_yaml("dist_coeffs", D.reshape(1, -1)),
             f"image_size: [ {int(sz[0])}, {int(sz[1])} ]\n",
             _scalar_yaml("model", mdl),
             _scalar_yaml("rms", f"{rms:.8g}"),
             ]
    out_path.write_text("".join(lines), encoding="utf-8")


def export_extrinsic(data, out_path):
    rvec = data["rvec"].ravel()
    tvec = data["tvec"].ravel()
    R    = data["R_mat"]
    T    = data["T_cam_world"]
    pos  = data["cam_pos"].ravel()
    rms  = float(data["rms"]) if "rms" in data else float("nan")

    lines = ["%YAML:1.0\n---\n",
             _mat_yaml("rvec", rvec),
             _mat_yaml("tvec", tvec),
             _mat_yaml("R_mat", R),
             _mat_yaml("T_cam_world", T),
             _mat_yaml("cam_pos", pos),
             _scalar_yaml("rms", f"{rms:.8g}"),
             ]
    out_path.write_text("".join(lines), encoding="utf-8")


def export_bev_lut(data, out_path):
    lines = ["%YAML:1.0\n---\n",
             _scalar_yaml("x_min", float(data["x_min"])),
             _scalar_yaml("x_max", float(data["x_max"])),
             _scalar_yaml("y_min", float(data["y_min"])),
             _scalar_yaml("y_max", float(data["y_max"])),
             _scalar_yaml("res",   float(data["res"])),
             f"map_shape: [ {data['map_x'].shape[0]}, {data['map_x'].shape[1]} ]\n",
             "# map_x / map_y arrays omitted (large float32 arrays)\n",
             ]
    out_path.write_text("".join(lines), encoding="utf-8")


# ── Console summary ───────────────────────────────────────────────────────────

def summarise_intrinsic(name, data):
    K   = data["camera_matrix"]
    D   = data["dist_coeffs"].ravel()
    rms = float(data["rms"]) if "rms" in data else float("nan")
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  fx = {K[0,0]:.4f}   fy = {K[1,1]:.4f}")
    print(f"  cx = {K[0,2]:.4f}   cy = {K[1,2]:.4f}")
    dist_labels = ["k1", "k2", "k3", "k4", "k5", "k6"]
    for i, v in enumerate(D):
        lbl = dist_labels[i] if i < len(dist_labels) else f"d{i}"
        print(f"  {lbl} = {v:+.8f}")
    print(f"  RMS = {rms:.6f} px")
    # sanity
    fx_ok = 500 < K[0,0] < 530
    print(f"  fx check (500~530) : {'✅' if fx_ok else '❌  UNEXPECTED'}")


def summarise_extrinsic(name, data):
    pos  = data["cam_pos"].ravel()
    R    = data["R_mat"]
    rms  = float(data["rms"]) if "rms" in data else float("nan")
    roll  = float(data["roll"])  if "roll"  in data else float("nan")
    pitch = float(data["pitch"]) if "pitch" in data else float("nan")
    yaw   = float(data["yaw"])   if "yaw"   in data else float("nan")
    fwd   = R[2, :]

    z_ok   = pos[2] > 0.05
    fwd_ok = fwd[0] > 0.0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  cam_pos : x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f} m")
    print(f"  R[2,:]  : {fwd[0]:+.4f}  {fwd[1]:+.4f}  {fwd[2]:+.4f}  ← camera forward in world")
    print(f"  Roll={roll:+.2f}°  Pitch={pitch:+.2f}°  Yaw={yaw:+.2f}°")
    print(f"  RMS = {rms:.6f} px")
    print(f"  z > 0  (above ground) : {'✅' if z_ok   else '❌  CAMERA BELOW GROUND'}")
    print(f"  fwd_x > 0 (faces fwd) : {'✅' if fwd_ok else '❌  CAMERA FACING BACKWARD'}")


def summarise_bev_lut(name, data):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  x: [{float(data['x_min']):.2f}, {float(data['x_max']):.2f}] m")
    print(f"  y: [{float(data['y_min']):.2f}, {float(data['y_max']):.2f}] m")
    print(f"  res : {float(data['res'])*100:.1f} cm/px")
    h, w = data["map_x"].shape
    valid = int(np.sum(data["map_x"] >= 0))
    print(f"  map : {w}×{h} px  |  valid = {valid}/{w*h} ({100*valid/(w*h):.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

HANDLERS = {
    "intrinsic": (export_intrinsic, summarise_intrinsic),
    "extrinsic": (export_extrinsic, summarise_extrinsic),
    "bev_lut":   (export_bev_lut,   summarise_bev_lut),
}


def detect_type(stem):
    if "intrinsic" in stem:
        return "intrinsic"
    if "extrinsic" in stem:
        return "extrinsic"
    if "bev_lut" in stem or "lut" in stem:
        return "bev_lut"
    return None


def main():
    args = parse_args()
    cal_dir = Path(args.cal_dir)

    npz_files = sorted(cal_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {cal_dir}")
        return

    print(f"Found {len(npz_files)} npz file(s) in '{cal_dir}'")

    for npz_path in npz_files:
        kind = detect_type(npz_path.stem)
        if kind is None:
            print(f"\n[SKIP] {npz_path.name} — unrecognised type")
            continue

        data = np.load(str(npz_path), allow_pickle=True)
        export_fn, summary_fn = HANDLERS[kind]

        # Console summary
        summary_fn(npz_path.stem, data)

        # YAML export
        yaml_path = npz_path.with_suffix(".yaml")
        export_fn(data, yaml_path)
        print(f"  → exported: {yaml_path.name}")

    print(f"\nAll YAML files saved to '{cal_dir}/'")


if __name__ == "__main__":
    main()
