#!/usr/bin/env python3
"""
trigger_mode 설정/해제 유틸리티

사용법:
  python3 trigger_mode_ctrl.py on        # 모든 카메라 trigger_mode=1
  python3 trigger_mode_ctrl.py off       # 모든 카메라 trigger_mode=0
  python3 trigger_mode_ctrl.py status    # 현재 상태 확인
  python3 trigger_mode_ctrl.py on -d 0   # 특정 카메라만
"""

import argparse
import sys

from ar0234_cam.v4l2_utils import fourcc, V4L2_BA10, detect_cameras, v4l2_get, v4l2_set


def main():
    parser = argparse.ArgumentParser(description="trigger_mode 설정/해제")
    parser.add_argument("action", choices=["on", "off", "status"])
    parser.add_argument("-d", "--device", type=int, action="append", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = detect_cameras()
        if not args.device:
            print("[ERROR] BA10 카메라를 찾을 수 없습니다.")
            sys.exit(1)

    cam_ids = sorted(args.device)

    if args.action == "on":
        for dev in cam_ids:
            v4l2_set(dev, "trigger_mode", 1)
            v4l2_set(dev, "frame_timeout", 2000)
            print(f"  cam{dev}: trigger_mode=1, frame_timeout=2000")
        print("[INFO] 완료")

    elif args.action == "off":
        for dev in cam_ids:
            v4l2_set(dev, "trigger_mode", 0)
            v4l2_set(dev, "frame_timeout", 2000)
            print(f"  cam{dev}: trigger_mode=0, frame_timeout=2000")
        print("[INFO] 완료")

    elif args.action == "status":
        for dev in cam_ids:
            tm = v4l2_get(dev, "trigger_mode")
            ft = v4l2_get(dev, "frame_timeout")
            print(f"  cam{dev}: trigger_mode={tm}, frame_timeout={ft}")


if __name__ == "__main__":
    main()
