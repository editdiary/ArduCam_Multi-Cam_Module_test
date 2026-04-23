"""GPIO 트리거 제어.

ArduCam FRSYNC(Frame Sync) 동기화를 위한 GPIO 펄스 출력을 제공한다.
GPIO 펄스를 보내면 trigger_mode=1인 모든 카메라가 동시에 1프레임을 캡처한다.

gpiod 라이브러리가 있으면 직접 제어하고(빠름),
없으면 gpioset subprocess로 폴백한다(느리지만 설치 불필요).

모듈 변수:
    TRIG_CHIP (str): GPIO 칩 이름 (Jetson AGX Orin 기준 "gpiochip0")
    TRIG_LINE (int): FRSYNC 신호에 연결된 GPIO 라인 번호 (146)
    _USE_GPIOD (bool): gpiod 라이브러리 사용 가능 여부
"""

import subprocess
import time

# FRSYNC#1 GPIO 설정
TRIG_CHIP = "gpiochip0"
TRIG_LINE = 146

_USE_GPIOD = False
try:
    import gpiod
    from gpiod.line import Direction, Value
    _USE_GPIOD = True
except ImportError:
    pass

_trig_request = None


def _get_trig_line():
    """gpiod 라인 핸들을 캐싱하여 반환한다."""
    global _trig_request
    if _trig_request is None and _USE_GPIOD:
        chip = gpiod.Chip(f"/dev/{TRIG_CHIP}")
        _trig_request = chip.request_lines(
            config={TRIG_LINE: gpiod.LineSettings(direction=Direction.OUTPUT)},
            consumer="ar0234_cam",
        )
    return _trig_request


def gpio_pulse(duration_ms=1):
    """GPIO 트리거 펄스를 출력한다.

    Args:
        duration_ms: 펄스 지속 시간 (밀리초, 기본 1)
    """
    req = _get_trig_line()
    if req:
        req.set_value(TRIG_LINE, Value.ACTIVE)
        time.sleep(duration_ms / 1000.0)
        req.set_value(TRIG_LINE, Value.INACTIVE)
    else:
        subprocess.run(
            ["gpioset", "--mode=time", f"--usec={duration_ms * 1000}",
             TRIG_CHIP, f"{TRIG_LINE}=1"],
            capture_output=True, timeout=2
        )
