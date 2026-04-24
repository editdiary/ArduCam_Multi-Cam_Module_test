"""V4L2 공통 유틸리티.

FourCC 인코딩, 카메라 감지, v4l2-ctl 래퍼 등
모든 모듈에서 공유하는 V4L2 관련 헬퍼를 제공한다.
"""

import os
import re
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


def has_video_capture_cap(device):
    """해당 디바이스가 V4L2_CAP_VIDEO_CAPTURE (영상 캡처) 능력을 가졌는지.

    UVC USB 카메라는 보통 capture 노드(/dev/videoN)와 metadata-capture 전용
    노드(/dev/videoN+1)를 함께 노출한다. 메타 노드를 cv2.VideoCapture 로 열려
    하면 OpenCV가 "can't open camera by index" 경고를 찍는다. `v4l2-ctl --info`
    출력의 Device Caps 비트를 검사해 실제 캡처 가능한 노드만 통과시킨다.

    Args:
        device: 디바이스 번호 (int)

    Returns:
        True  — Video Capture 비트(0x1) 있음
        False — 비트 없음, 또는 조회 실패
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{device}", "--info"],
            capture_output=True, text=True, timeout=2,
        )
    except Exception:
        return False
    m = re.search(r"Device Caps\s*:\s*(0x[0-9a-fA-F]+)", result.stdout)
    if not m:
        return False
    V4L2_CAP_VIDEO_CAPTURE = 0x1
    return bool(int(m.group(1), 16) & V4L2_CAP_VIDEO_CAPTURE)


def list_resolutions(device, fourcc_filter=None):
    """v4l2-ctl --list-formats-ext 를 파싱해 지원 해상도 리스트를 반환한다.

    Args:
        device: 디바이스 번호 (int)
        fourcc_filter: 특정 FourCC(예: "BA10", "YUYV")만 필터링. None이면 모든 포맷 병합.

    Returns:
        [(w, h), ...] — 중복 제거, 해상도(픽셀 수) 내림차순. 실패 시 빈 리스트.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{device}", "--list-formats-ext"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []

    fmt_re = re.compile(r"'(.{4})'")
    size_re = re.compile(r"Size:\s*Discrete\s*(\d+)x(\d+)")
    current_fourcc = None
    found = set()
    for line in result.stdout.splitlines():
        m = fmt_re.search(line)
        if m and "Size" not in line:
            # 'Y16 ' 처럼 공백 포함 FourCC 대응 (trailing space strip).
            current_fourcc = m.group(1).strip()
            continue
        m = size_re.search(line)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if fourcc_filter is None or current_fourcc == fourcc_filter:
                found.add((w, h))

    return sorted(found, key=lambda wh: -wh[0] * wh[1])


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
