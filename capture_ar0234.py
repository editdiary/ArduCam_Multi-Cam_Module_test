#!/usr/bin/env python3
"""
AR0234 BA10 Raw 캡처 → SW Demosaic → GStreamer HW JPEG 인코딩 테스트 스크립트

카메라가 출력하는 BA10(10-bit Bayer GRGR/BGBG) raw 데이터를:
  1. V4L2를 통해 raw 그대로 수신
  2. OpenCV로 SW demosaic (Bayer → BGR)
  3. GStreamer nvjpegenc로 HW 가속 JPEG 인코딩
  4. 파일로 저장

사용법:
  python3 capture_ar0234.py -d 0 -n 3
  python3 capture_ar0234.py -d 1 --width 1280 --height 720
  python3 capture_ar0234.py --fallback   # OpenCV imwrite 방식으로 저장
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 하위 호환 re-export
# test_codes/ 등에서 `from capture_ar0234 import fourcc, demosaic, ...` 유지
# ---------------------------------------------------------------------------
from ar0234_cam.v4l2_utils import fourcc
from ar0234_cam.isp import (
    demosaic, _USE_GPU, _GAMMA_POST_LUT,
)
from ar0234_cam.auto_exposure import AutoExposure
from ar0234_cam.encoder import GstJpegEncoder

if _USE_GPU:
    from ar0234_cam.isp import demosaic_gpu


def capture(args):
    """카메라에서 BA10 raw 프레임을 캡처하고 JPEG로 저장한다."""

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] /dev/video{args.device} 열기 실패", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, fourcc('B', 'A', '1', '0'))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 디바이스: /dev/video{args.device}")
    print(f"[INFO] 요청 해상도: {args.width}x{args.height}")
    print(f"[INFO] 실제 해상도: {actual_w}x{actual_h}")

    os.makedirs(args.output_dir, exist_ok=True)

    use_gpu = _USE_GPU and not args.cpu
    if use_gpu:
        print("[INFO] GPU 가속 demosaic 모드 (PyCUDA)")
    else:
        print("[INFO] CPU demosaic 모드" + (" (--cpu 지정)" if args.cpu else " (PyCUDA 없음)"))

    # 수동 노출/게인 설정
    if args.exposure is not None or args.gain is not None:
        import fcntl
        import v4l2
        try:
            vd = open(f'/dev/video{args.device}', 'w')
            if args.exposure is not None:
                ctrl = v4l2.v4l2_control()
                ctrl.id = v4l2.V4L2_CID_EXPOSURE
                ctrl.value = max(2, min(65535, args.exposure))
                fcntl.ioctl(vd, v4l2.VIDIOC_S_CTRL, ctrl)
                print(f"[INFO] 수동 노출: {ctrl.value}")
            if args.gain is not None:
                ctrl = v4l2.v4l2_control()
                ctrl.id = 0x009e0903
                ctrl.value = max(100, min(1200, args.gain))
                fcntl.ioctl(vd, v4l2.VIDIOC_S_CTRL, ctrl)
                print(f"[INFO] 수동 게인: {ctrl.value}")
            vd.close()
        except Exception as e:
            print(f"[WARN] 수동 노출/게인 설정 실패: {e}", file=sys.stderr)

    # AE (자동 노출) 초기화
    ae = None
    if args.ae:
        try:
            ae = AutoExposure(args.device, target_luma=args.ae_target, speed=args.ae_speed)
            print(f"[INFO] 자동 노출(AE) 활성화 (target={args.ae_target}, speed={args.ae_speed})")
        except Exception as e:
            print(f"[WARN] AE 초기화 실패: {e}", file=sys.stderr)

    # JPEG 인코더 초기화
    encoder = None
    if not args.fallback:
        try:
            encoder = GstJpegEncoder(actual_w, actual_h, quality=args.quality)
            print(f"[INFO] GStreamer nvjpegenc HW 인코더 초기화 완료 (quality={args.quality})")
        except Exception as e:
            print(f"[WARN] GStreamer 인코더 초기화 실패: {e}", file=sys.stderr)
            print("[WARN] OpenCV fallback으로 전환합니다.", file=sys.stderr)
            args.fallback = True

    if args.fallback:
        print(f"[INFO] OpenCV imwrite 모드 (quality={args.quality})")

    # 첫 프레임 읽기 테스트
    print("[INFO] 카메라 응답 확인 중...")
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.", file=sys.stderr)
        print("[ERROR] 카메라 연결 상태와 드라이버를 확인하세요.", file=sys.stderr)
        cap.release()
        sys.exit(1)
    print(f"[INFO] 프레임 수신 확인 (shape={test_frame.shape}, dtype={test_frame.dtype})")

    print("[INFO] 센서 안정화 중 (4프레임 추가 스킵)...")
    for _ in range(4):
        cap.read()

    # 캡처 시작
    print(f"[INFO] {args.num_captures}장 캡처 시작")
    captured = 0
    for i in range(args.num_captures):
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] 프레임 {i+1} 읽기 실패", file=sys.stderr)
            continue

        t_read = time.time()

        if use_gpu:
            bgr, ch_avgs = demosaic_gpu(frame, actual_h, actual_w)
            if ae:
                ae.update(ch_avgs[0], ch_avgs[1], ch_avgs[2])
        else:
            bgr = demosaic(frame, actual_h, actual_w)
            if ae:
                avgs = cv2.mean(bgr)
                ae.update(avgs[0], avgs[1], avgs[2])
        t_demosaic = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"ar0234_dev{args.device}_{timestamp}.jpg"
        filepath = os.path.join(args.output_dir, filename)

        if args.fallback:
            cv2.imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        else:
            jpeg_data = encoder.encode(bgr)
            if jpeg_data is None:
                print(f"[ERROR] 프레임 {i+1} 인코딩 실패", file=sys.stderr)
                continue
            with open(filepath, "wb") as f:
                f.write(jpeg_data)

        t_end = time.time()

        file_size = os.path.getsize(filepath)
        print(
            f"  [{i+1}/{args.num_captures}] {filename} "
            f"({actual_w}x{actual_h}, {file_size/1024:.1f}KB) "
            f"read={1000*(t_read-t_start):.0f}ms "
            f"demosaic={1000*(t_demosaic-t_read):.0f}ms "
            f"encode={1000*(t_end-t_demosaic):.0f}ms "
            f"total={1000*(t_end-t_start):.0f}ms"
        )
        captured += 1

    if ae:
        ae.close()
    if encoder:
        encoder.close()
    cap.release()

    print(f"\n[DONE] {captured}장 저장 완료 → {os.path.abspath(args.output_dir)}/")


def main():
    parser = argparse.ArgumentParser(
        description="AR0234 BA10 캡처 → SW Demosaic → HW JPEG 인코딩"
    )
    parser.add_argument(
        "-d", "--device", type=int, default=0,
        help="비디오 디바이스 번호 (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="프레임 너비 (default: 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="프레임 높이 (default: 1080)"
    )
    parser.add_argument(
        "-n", "--num-captures", type=int, default=1,
        help="캡처 매수 (default: 1)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./output",
        help="출력 디렉토리 (default: ./output)"
    )
    parser.add_argument(
        "--quality", type=int, default=85,
        help="JPEG 품질 1-100 (default: 85)"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="GStreamer 대신 OpenCV imwrite로 저장 (디버깅/비교용)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="GPU 대신 CPU demosaic 사용 (비교/디버깅용)"
    )
    parser.add_argument(
        "--ae", action="store_true",
        help="자동 노출(AE) 활성화 (실험적)"
    )
    parser.add_argument(
        "--ae-target", type=int, default=120,
        help="AE 목표 밝기 0-255 (default: 120)"
    )
    parser.add_argument(
        "--ae-speed", type=float, default=0.3,
        help="AE 수렴 속도 0.0-1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--exposure", type=int, default=None,
        help="수동 노출 설정 (2~65535, 미지정시 센서 기본값 사용)"
    )
    parser.add_argument(
        "--gain", type=int, default=None,
        help="수동 아날로그 게인 설정 (100~1200, 미지정시 센서 기본값 사용)"
    )

    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
