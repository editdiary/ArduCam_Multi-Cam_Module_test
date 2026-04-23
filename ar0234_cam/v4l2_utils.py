"""V4L2 공통 유틸리티.

FourCC 인코딩, 카메라 감지, v4l2-ctl 래퍼 등
모든 모듈에서 공유하는 V4L2 관련 헬퍼를 제공한다.
"""

import os
import subprocess

import cv2


def fourcc(a, b, c, d):
    """4개의 ASCII 문자를 V4L2 FourCC 정수로 인코딩한다."""
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


#: AR0234 센서의 10-bit Bayer 출력 포맷 (V4L2_PIX_FMT_SGRBG10)
V4L2_BA10 = fourcc('B', 'A', '1', '0')


def detect_cameras(fourcc_code=V4L2_BA10):
    """BA10 포맷을 지원하는 V4L2 카메라 디바이스 번호 목록을 반환한다.

    Args:
        fourcc_code: 검색할 FourCC 코드 (기본: BA10)

    Returns:
        정렬된 디바이스 번호 리스트 (예: [0, 1])
    """
    found = []
    for i in range(16):
        if not os.path.exists(f'/dev/video{i}'):
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if int(cap.get(cv2.CAP_PROP_FOURCC)) == fourcc_code:
            found.append(i)
        cap.release()
    return found


def v4l2_set(device, name, value):
    """v4l2-ctl로 컨트롤 값을 설정한다.

    내부적으로 ``v4l2-ctl -d /dev/videoN -c name=value`` 를 실행한다.
    고빈도 호출에는 fcntl ioctl을 직접 사용하는 것이 효율적이다.

    Args:
        device: 디바이스 번호 (int)
        name: 컨트롤 이름 (예: "trigger_mode", "exposure")
        value: 설정할 값 (int 또는 문자열로 변환 가능한 값)
    """
    subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{device}", "-c", f"{name}={value}"],
        capture_output=True, timeout=5
    )


def v4l2_get(device, name):
    """v4l2-ctl로 컨트롤의 현재 값을 읽는다.

    ``v4l2-ctl --list-ctrls`` 출력에서 ``value=`` 항목을 파싱한다.

    Args:
        device: 디바이스 번호 (int)
        name: 컨트롤 이름 (예: "trigger_mode", "exposure")

    Returns:
        값 문자열 (예: "1", "5000"), 컨트롤을 찾지 못하면 "?"
    """
    result = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{device}", "--list-ctrls"],
        capture_output=True, text=True, timeout=5
    )
    for line in result.stdout.splitlines():
        if name in line:
            for part in line.split():
                if part.startswith("value="):
                    return part.split("=")[1]
    return "?"


def check_trigger_mode(device):
    """카메라의 trigger_mode가 1(활성)인지 확인한다.

    Args:
        device: 디바이스 번호 (int)

    Returns:
        trigger_mode=1이면 True
    """
    result = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{device}", "--list-ctrls"],
        capture_output=True, text=True, timeout=2
    )
    for line in result.stdout.splitlines():
        if "trigger_mode" in line and "value=1" in line:
            return True
    return False
