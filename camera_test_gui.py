#!/usr/bin/env python3
"""카메라 테스트 GUI (Flask 웹 기반).

브라우저에서 카메라 선택 → 촬영 → 미리보기 → 저장을 수행할 수 있는
웹 기반 카메라 테스트 도구.

사용법:
  python3 camera_test_gui.py
  python3 camera_test_gui.py --port 9999 --cpu
  python3 camera_test_gui.py --width 1280 --height 720

브라우저 접속:
  http://<jetson-ip>:8888/
"""

import argparse
import base64
import collections
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from ar0234_cam.v4l2_utils import (
    V4L2_BA10, check_trigger_mode, has_video_capture_cap, list_resolutions,
    v4l2_get, v4l2_set,
)
from ar0234_cam.isp import _USE_GPU, demosaic

if _USE_GPU:
    from ar0234_cam.isp import demosaic_gpu
    import pycuda.driver as cuda
    _cuda_ctx = cuda.Context.get_current()
else:
    _cuda_ctx = None

from ar0234_cam.gpio import gpio_pulse
from ar0234_cam.sync import parallel_grab, parallel_retrieve

# ---------------------------------------------------------------------------
# 앱 상태
# ---------------------------------------------------------------------------

app = Flask(__name__)


class AppState:
    """서버 전역 상태."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cameras = []           # 감지된 카메라 목록
        self.captured = {}          # {dev: {"bgr": ndarray, "jpeg_b64": str, "timestamp": str}}
        self.use_gpu = _USE_GPU
        # CLI --width/--height 의 기본값. 개별 카메라 해상도가 지정되지 않았을 때
        # fallback으로 사용된다.
        self.default_width = 1920
        self.default_height = 1080
        self._cam_res = {}          # {dev: (w, h)} — per-camera 해상도
        self._open_caps = {}        # {dev: cv2.VideoCapture} 열린 카메라 세션
        self._cap_info = {}         # {dev: {"w": int, "h": int, "trigger": bool, "type": str}}
        # OpenCV VideoCapture 는 thread-safe 하지 않으므로, 스트리머 백그라운드
        # 루프와 /api/capture 요청이 같은 cap 의 grab/retrieve 를 동시에 건드리지
        # 않도록 직렬화한다. demosaic/encode 는 락 밖에서.
        self.cap_lock = threading.Lock()
        # 단일 카메라 스트리밍 스코프: 활성 Streamer 인스턴스 또는 None.
        self.streamer = None


state = AppState()

# ---------------------------------------------------------------------------
# 카메라 감지
# ---------------------------------------------------------------------------


def _get_v4l2_card_name(dev_num):
    """v4l2-ctl --info로 카메라 카드 이름을 가져온다."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{dev_num}", "--info"],
            capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.splitlines():
            if "Card type" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown"


def detect_all_cameras():
    """AR0234(BA10) 및 일반 USB 카메라를 모두 감지한다.

    Returns:
        [{"dev": int, "type": "ar0234"|"usb", "name": str, "label": str}, ...]
    """
    found = []
    for i in range(16):
        if not os.path.exists(f"/dev/video{i}"):
            continue
        # UVC 카메라의 metadata-only 노드 등 Video Capture 능력이 없는 디바이스는
        # cv2.VideoCapture 로 열면 OpenCV 경고가 뜨므로 사전에 필터링.
        if not has_video_capture_cap(i):
            continue

        # AR0234 (BA10 포맷) 확인
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, V4L2_BA10)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if int(cap.get(cv2.CAP_PROP_FOURCC)) == V4L2_BA10:
            cap.release()
            card = _get_v4l2_card_name(i)
            found.append({
                "dev": i,
                "type": "ar0234",
                "name": f"AR0234 ({card})",
                "label": f"/dev/video{i} - AR0234 ({card})",
            })
            continue

        # 일반 USB 카메라 확인
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        ret, _ = cap.read()
        cap.release()
        if ret:
            card = _get_v4l2_card_name(i)
            found.append({
                "dev": i,
                "type": "usb",
                "name": f"USB ({card})",
                "label": f"/dev/video{i} - USB ({card})",
            })

    return found


# ---------------------------------------------------------------------------
# 촬영 로직
# ---------------------------------------------------------------------------


def _bgr_to_jpeg_b64(bgr, quality=85):
    """BGR 이미지를 base64 JPEG 문자열로 변환한다."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _round_timing(t):
    """타이밍 딕셔너리를 표시용으로 반올림."""
    out = {}
    for k, v in t.items():
        if v is None:
            out[k] = None
        elif isinstance(v, int):
            out[k] = v
        elif "delta" in k:
            out[k] = round(v, 3)
        else:
            out[k] = round(v, 2)
    return out


def _drain_queue(cap, max_drops=10, quick_ms=30):
    """V4L2 내부 큐에 쌓인 오래된 프레임들을 비운다.

    grab()이 빠르게 반환되면 큐에 남아있던 버퍼가 회수된 것으로 판단해
    계속 drain, 길어지면 센서가 다음 프레임을 만드는 중(=큐가 비었음)으로
    판단해 중단한다. 트리거 모드/continuous 모드 모두 안전.

    Returns:
        drain된 프레임 수.
    """
    drops = 0
    for _ in range(max_drops):
        t0 = time.monotonic()
        ok = cap.grab()
        dt = (time.monotonic() - t0) * 1000
        if not ok:
            break
        drops += 1
        if dt > quick_ms:
            break
    return drops


def _parallel_drain(caps, cam_ids, max_drops=10, quick_ms=30):
    """여러 카메라의 큐를 병렬로 drain."""
    results = {}

    def _d(dev):
        results[dev] = _drain_queue(caps[dev], max_drops, quick_ms)

    threads = [threading.Thread(target=_d, args=(d,)) for d in cam_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    return results


def _all_open_ar0234_devs():
    """트리거가 활성화된 열린 AR0234 카메라 dev 번호 리스트."""
    return [d for d, info in state._cap_info.items()
            if info["type"] == "ar0234" and info.get("trigger")]


def _sync_pulse_grab_ar0234():
    """GPIO 펄스 + 열린 모든 AR0234 카메라 병렬 grab.

    GPIO는 하드웨어 공유 라인이라 한 번의 펄스가 트리거 활성화된 모든
    AR0234 카메라의 V4L2 큐에 프레임을 추가한다. 타겟 외 카메라 큐가
    누적되는 것을 막기 위해, 매 펄스마다 모든 AR0234 카메라를 동시에
    grab하여 큐 상태를 동기화 유지한다. retrieve는 호출자가 타겟에
    대해서만 수행한다.

    Returns:
        (grabs, pulse_ms, grab_ms) — grabs는 parallel_grab과 동일한 형식
        {dev: {"ok": bool, "grab_ms": float}}.
    """
    all_ar0234 = _all_open_ar0234_devs()
    all_caps = {d: state._open_caps[d] for d in all_ar0234}

    t_pulse = time.monotonic()
    gpio_pulse()
    pulse_ms = (time.monotonic() - t_pulse) * 1000

    if not all_caps:
        return {}, pulse_ms, 0.0

    t_grab = time.monotonic()
    grabs = parallel_grab(all_caps, all_ar0234)
    grab_ms = (time.monotonic() - t_grab) * 1000
    return grabs, pulse_ms, grab_ms


def _release_caps():
    """열려 있는 모든 카메라 세션을 해제한다.

    Flask threaded=True 환경에서 /api/cameras/refresh 와 /api/params,
    /api/capture 등이 동시에 실행되면 iteration 중 dict가 수정되어
    RuntimeError 가 난다. 스냅샷을 뜨고, 락으로 다른 경로와 상호배제.
    """
    # 스트리머가 돌고 있으면 cap release 전에 먼저 중지 (루프가 released cap
    # 을 건드리지 않도록).
    if state.streamer is not None:
        state.streamer.stop()
        state.streamer = None
    with state.lock:
        caps = list(state._open_caps.items())
        state._open_caps.clear()
        state._cap_info.clear()
    for dev, cap in caps:
        try:
            cap.release()
        except Exception:
            pass


# Default 프리셋 (Indoor 기준) — UI 슬라이더 초기값과 일치시켜 페이지 로드 시
# v4l2_get read-back이 700 같은 센서 부팅값이 아니라 900/100으로 보이도록 한다.
_AR0234_DEFAULT_EXPOSURE = 900
_AR0234_DEFAULT_GAIN = 100


def _init_ar0234_defaults(dev):
    """AR0234 카메라에 override_enable과 Default 프리셋 값을 써둔다.

    startup / camera refresh / resolution 변경 / ensure_cap_open 등 세션 리셋
    지점에서 호출하여 `/api/params` GET이 항상 일관된 값(900/100)을 반환하도록
    만든다. 멱등(idempotent)이므로 반복 호출해도 안전.
    """
    v4l2_set(dev, "override_enable", 1)
    v4l2_set(dev, "exposure", _AR0234_DEFAULT_EXPOSURE)
    v4l2_set(dev, "analogue_gain", _AR0234_DEFAULT_GAIN)


def _ensure_cap_open(dev_num, cam_type):
    """카메라 세션을 열어 유지한다. 이미 열려있으면 기존 세션을 반환.

    sync_monitor와 동일하게 카메라를 한 번 열면 release하지 않아서
    trigger_mode 리셋을 방지한다.

    Returns:
        (cap, actual_w, actual_h, is_trigger) 튜플
    """
    if dev_num in state._open_caps:
        info = state._cap_info[dev_num]
        return state._open_caps[dev_num], info["w"], info["h"], info["trigger"]

    # trigger_mode 상태를 카메라 열기 전에 확인 (열고 나면 리셋됨)
    is_trigger = (cam_type == "ar0234") and check_trigger_mode(dev_num)

    # Tegra 카메라 드라이버: override_enable=1 이어야 사용자가 v4l2로 쓴
    # exposure / analogue_gain 값이 sensor_mode 프리셋을 덮어쓰며 실제 센서
    # 레지스터에 적용된다. STREAMON 이후엔 반영이 보장되지 않으므로
    # VideoCapture 열기 전에 설정.
    if cam_type == "ar0234":
        _init_ar0234_defaults(dev_num)

    w, h = state._cam_res.get(
        dev_num, (state.default_width, state.default_height))

    cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
    # V4L2 DMA 버퍼 링을 최소화 — stale frame lag 방지.
    # REQBUFS 이전에 호출되어야 반영되므로 fourcc/size 설정보다 먼저 둔다.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cam_type == "ar0234":
        cap.set(cv2.CAP_PROP_FOURCC, V4L2_BA10)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    elif cam_type == "usb":
        # UVC 카메라는 포맷마다 지원 해상도 범위가 다르다. YUYV 는 대역폭 제약으로
        # 고해상도만 지원하는 경우가 많아(예: 1280x720 이상만), 저해상도(800x600,
        # 640x480, 320x240) 선택이 실제로 반영되지 않고 V4L2 가 가장 가까운
        # YUYV 해상도로 롤백하는 문제가 있다. MJPG 는 UVC 표준이고 저해상도까지
        # 넓게 커버하므로 기본 포맷으로 강제. OpenCV 가 자동 디코드해서 BGR 반환.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 워밍업: sync_monitor 패턴(초기 펄스+grab 1회 + 5회 루프).
    # 트리거 펄스는 하드웨어 공유 라인이라 기존에 열린 AR0234 카메라의
    # 큐에도 프레임을 추가하므로, 해당 카메라들도 같이 grab해서 누적을
    # 방지한다 (큐 동기화 불변).
    other_caps = [state._open_caps[d] for d in _all_open_ar0234_devs()]

    if cam_type == "ar0234" and is_trigger:
        gpio_pulse(duration_ms=1)
        cap.grab()
        for c in other_caps:
            try:
                c.grab()
            except Exception:
                pass

        for _ in range(5):
            gpio_pulse()
            cap.grab()
            cap.retrieve()
            for c in other_caps:
                try:
                    c.grab()
                except Exception:
                    pass
            time.sleep(0.02)
    else:
        # 비트리거 AR0234 / USB: 단순 워밍업
        for _ in range(3):
            cap.grab()
            cap.retrieve()

    # STREAMON(첫 grab) 시점에 Tegra 드라이버가 exposure/analogue_gain을
    # V4L2 default(700/100) 로 한 번 리셋한다(override_enable=1 이어도 그렇다).
    # 워밍업 이후 재주입하면 이후 캡처까지 sticky하므로 여기서 다시 써준다.
    if cam_type == "ar0234":
        v4l2_set(dev_num, "exposure", _AR0234_DEFAULT_EXPOSURE)
        v4l2_set(dev_num, "analogue_gain", _AR0234_DEFAULT_GAIN)

    state._open_caps[dev_num] = cap
    state._cap_info[dev_num] = {
        "w": actual_w, "h": actual_h,
        "trigger": is_trigger, "type": cam_type,
    }

    return cap, actual_w, actual_h, is_trigger


def capture_single(dev_num, cam_type):
    """단일 카메라에서 1프레임을 촬영한다.

    카메라 세션을 유지한 채 grab/retrieve만 호출하여
    trigger_mode 리셋을 방지한다.

    Returns:
        (bgr, timing) 튜플. 실패 시 (None, timing).
        timing dict: pulse_ms, grab_ms, retrieve_ms, isp_ms.
    """
    cap, actual_w, actual_h, is_trigger = _ensure_cap_open(dev_num, cam_type)
    timing = {"pulse_ms": 0.0, "grab_ms": 0.0,
              "retrieve_ms": 0.0, "isp_ms": 0.0,
              "drain_ms": 0.0, "drained": 0}

    # grab/retrieve 는 cap_lock 으로 직렬화 — 백그라운드 스트리머가 동일 cap
    # 을 동시에 건드릴 때 OpenCV 비-thread-safe 상태에 빠지는 것을 방지.
    # demosaic 은 락 밖에서.
    frame = None
    with state.cap_lock:
        # 큐에 쌓인 stale 프레임 제거 — 항상 최신 프레임이 나오도록
        t0 = time.monotonic()
        timing["drained"] = _drain_queue(cap)
        timing["drain_ms"] = (time.monotonic() - t0) * 1000

        if is_trigger:
            # 펄스 + 열린 모든 AR0234 병렬 grab (큐 동기화)
            grabs, pulse_ms, grab_ms = _sync_pulse_grab_ar0234()
            timing["pulse_ms"] = pulse_ms
            timing["grab_ms"] = grab_ms
            if not grabs.get(dev_num, {}).get("ok", False):
                return None, timing
        else:
            t0 = time.monotonic()
            ret = cap.grab()
            timing["grab_ms"] = (time.monotonic() - t0) * 1000
            if not ret:
                return None, timing

        t0 = time.monotonic()
        ret, frame = cap.retrieve()
        timing["retrieve_ms"] = (time.monotonic() - t0) * 1000
        if not ret:
            return None, timing

    t0 = time.monotonic()
    if cam_type == "ar0234":
        if state.use_gpu and _cuda_ctx is not None:
            _cuda_ctx.push()
            try:
                bgr, _ = demosaic_gpu(frame, actual_h, actual_w, buf_id=dev_num)
            finally:
                _cuda_ctx.pop()
        else:
            bgr = demosaic(frame, actual_h, actual_w)
    else:
        bgr = frame
    timing["isp_ms"] = (time.monotonic() - t0) * 1000

    return bgr, timing


def capture_multi(dev_nums, cam_infos):
    """다중 카메라 동시 촬영.

    AR0234 카메라는 GPIO 트리거로 동기 촬영하고,
    USB 카메라는 일반 grab/retrieve로 촬영한다.
    카메라 세션을 유지하여 trigger_mode 리셋을 방지한다.

    Returns:
        (results, error, timing) 튜플. timing dict는 pulse_ms, grab_ms,
        retrieve_ms, isp_ms, delta_grab_ms(AR0234 2대 이상일 때만 값 있음).
    """
    ar0234_devs = [d for d in dev_nums if cam_infos[d]["type"] == "ar0234"]
    usb_devs = [d for d in dev_nums if cam_infos[d]["type"] == "usb"]

    results = {}
    error = None
    timing = {"pulse_ms": 0.0, "grab_ms": 0.0, "retrieve_ms": 0.0,
              "isp_ms": 0.0, "delta_grab_ms": None,
              "drain_ms": 0.0, "drained": 0}

    # AR0234 다중 촬영 (GPIO 트리거)
    if ar0234_devs:
        caps = {}
        for dev in ar0234_devs:
            cap, _, _, is_trigger = _ensure_cap_open(dev, "ar0234")
            if not is_trigger:
                return {}, (f"cam{dev} trigger_mode 비활성. "
                           "먼저: python3 trigger_mode_ctrl.py on"), timing
            caps[dev] = cap

        # 펄스 + 열린 모든 AR0234 병렬 grab (타겟 외 cam이 있어도 큐 sync 유지).
        # Multi에서도 drain은 하지 않음 — 카메라별 독립 drain은 큐 비대칭을
        # 만들어 sync를 깬다. 대신 매 pulse마다 모든 AR0234를 동시에 grab해
        # 큐 상태를 항상 동일하게 맞춘다.
        # grab/retrieve 는 cap_lock 으로 직렬화 (스트리머와 공존 보장).
        raws = {}
        errors = []
        with state.cap_lock:
            grabs, pulse_ms, grab_ms = _sync_pulse_grab_ar0234()
            timing["pulse_ms"] = pulse_ms
            timing["grab_ms"] = grab_ms

            # sync delta는 "타겟" 카메라들의 grab 시간으로만 계산
            target_grab_times = [grabs[d]["grab_ms"] for d in ar0234_devs
                                 if grabs.get(d, {}).get("ok")]
            if len(target_grab_times) >= 2:
                timing["delta_grab_ms"] = (max(target_grab_times) -
                                            min(target_grab_times))

            t0 = time.monotonic()
            raws, errors = parallel_retrieve(caps, ar0234_devs, grabs)
            timing["retrieve_ms"] = (time.monotonic() - t0) * 1000
            # release 하지 않음 — 세션 유지

        if errors:
            error = f"촬영 실패 카메라: {errors}"

        t0 = time.monotonic()
        if state.use_gpu and _cuda_ctx is not None:
            _cuda_ctx.push()
        try:
            for dev in ar0234_devs:
                if dev in raws:
                    # 카메라마다 해상도가 다를 수 있으므로 per-cam 으로 조회.
                    info = state._cap_info[dev]
                    h, w = info["h"], info["w"]
                    if state.use_gpu:
                        bgr, _ = demosaic_gpu(raws[dev], h, w, buf_id=dev)
                    else:
                        bgr = demosaic(raws[dev], h, w)
                    results[dev] = bgr
        finally:
            if state.use_gpu and _cuda_ctx is not None:
                _cuda_ctx.pop()
        timing["isp_ms"] = (time.monotonic() - t0) * 1000

    # USB 카메라 촬영 (순차 실행 — 타이밍은 grab/retrieve 버킷에 누적)
    # Multi 모드에서는 drain을 하지 않는다 (AR0234 쪽 주석 참조, 일관성 유지).
    for dev in usb_devs:
        cap, _, _, _ = _ensure_cap_open(dev, "usb")
        with state.cap_lock:
            t0 = time.monotonic()
            ret = cap.grab()
            grab_dt = (time.monotonic() - t0) * 1000
            if ret:
                t0 = time.monotonic()
                ret, bgr = cap.retrieve()
                retrieve_dt = (time.monotonic() - t0) * 1000
                if ret:
                    results[dev] = bgr
                    timing["grab_ms"] += grab_dt
                    timing["retrieve_ms"] += retrieve_dt
                    continue
        error = (error or "") + f" cam{dev} USB 촬영 실패."

    return results, error, timing


# ---------------------------------------------------------------------------
# 라이브 스트리머 (단일 카메라)
# ---------------------------------------------------------------------------


class Streamer:
    """선택된 카메라 1대에 대한 백그라운드 캡처 + JPEG 인코딩 루프.

    state._open_caps 의 영속 cap 을 빌려 쓰며, 절대 release 하지 않는다.
    grab/retrieve 는 state.cap_lock 로 직렬화해 /api/capture 와 안전하게 공존.
    demosaic/encode 는 락 밖에서 실행해 다른 요청을 블로킹하지 않는다.

    MJPEG 라우트는 wait_frame(last_seq) 으로 새 프레임이 올 때까지 블로킹하며
    클라이언트 disconnect 와 무관하게 루프는 계속 돈다(명시적 stop() 만 종료).
    """

    def __init__(self, dev, cam_type, fps=15):
        self._dev = dev
        self._type = cam_type
        self._fps = max(1, int(fps))

        self._cond = threading.Condition()
        self._running = False
        self._thread = None
        self._latest_jpeg = None
        self._seq = 0
        self._stats = collections.deque(maxlen=30)

    @property
    def dev(self):
        return self._dev

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"streamer-{self._dev}", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def wait_frame(self, last_seq, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout=timeout)
            return self._seq, self._latest_jpeg

    def get_stats(self):
        with self._cond:
            entries = list(self._stats)
        if not entries:
            return {"fps": 0, "grab_ms": 0, "isp_ms": 0, "encode_ms": 0}
        n = len(entries)
        return {
            "fps": round(sum(e["fps"] for e in entries) / n, 1),
            "grab_ms": round(sum(e["grab"] for e in entries) / n, 2),
            "isp_ms": round(sum(e["isp"] for e in entries) / n, 2),
            "encode_ms": round(sum(e["encode"] for e in entries) / n, 2),
        }

    def _loop(self):
        # GPU demosaic 사용 시 CUDA 컨텍스트를 이 스레드에 바인딩
        # (sync.py::_capture_loop 의 push/pop 패턴과 동일).
        ctx = _cuda_ctx if (state.use_gpu and _cuda_ctx is not None) else None
        if ctx is not None:
            ctx.push()

        interval = 1.0 / self._fps

        try:
            while self._running:
                t_cycle = time.monotonic()

                # grab/retrieve 는 락으로 직렬화.
                t0 = time.monotonic()
                ret = False
                frame = None
                with state.cap_lock:
                    cap = state._open_caps.get(self._dev)
                    info = state._cap_info.get(self._dev)
                    if cap is None or info is None:
                        break  # cap 이 release 된 경우 루프 종료
                    if self._type == "ar0234" and info.get("trigger"):
                        # 트리거 AR0234: 펄스 + 전체 동시 grab 으로 큐 동기화 유지.
                        _sync_pulse_grab_ar0234()
                    else:
                        cap.grab()
                    ret, frame = cap.retrieve()
                t_grab = (time.monotonic() - t0) * 1000

                if not ret or frame is None:
                    # 실패 시 짧은 쉼 후 재시도 (무한 루프 방지).
                    time.sleep(0.01)
                    continue

                # demosaic 은 락 밖에서.
                t0 = time.monotonic()
                if self._type == "ar0234":
                    h, w = info["h"], info["w"]
                    if state.use_gpu:
                        bgr, _ = demosaic_gpu(frame, h, w, buf_id=self._dev)
                    else:
                        bgr = demosaic(frame, h, w)
                else:
                    bgr = frame
                t_isp = (time.monotonic() - t0) * 1000

                t0 = time.monotonic()
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                t_encode = (time.monotonic() - t0) * 1000
                if not ok:
                    continue

                t_total = (time.monotonic() - t_cycle) * 1000

                with self._cond:
                    self._latest_jpeg = buf.tobytes()
                    self._seq += 1
                    self._stats.append({
                        "grab": t_grab, "isp": t_isp, "encode": t_encode,
                        "fps": 1000.0 / t_total if t_total > 0 else 0,
                    })
                    self._cond.notify_all()

                # fps 캡: 남은 시간만큼 sleep.
                elapsed = time.monotonic() - t_cycle
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if ctx is not None:
                ctx.pop()


# ---------------------------------------------------------------------------
# Flask 라우트
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    with state.lock:
        return jsonify(state.cameras)


@app.route("/api/cameras/refresh", methods=["POST"])
def api_cameras_refresh():
    _release_caps()  # 기존 세션 해제 후 재감지
    cameras = detect_all_cameras()
    with state.lock:
        state.cameras = cameras

    # AR0234 카메라가 있으면 trigger_mode 복원 + Default 프리셋 값을 미리 적용
    # (UI의 슬라이더 초기값과 센서 레지스터를 일치시켜 로드 시 flicker 방지).
    for c in cameras:
        if c["type"] == "ar0234":
            subprocess.run(
                [sys.executable, "trigger_mode_ctrl.py", "on"],
                capture_output=True, timeout=10
            )
            break
    for c in cameras:
        if c["type"] == "ar0234":
            _init_ar0234_defaults(c["dev"])

    return jsonify(cameras)


@app.route("/api/capture", methods=["POST"])
def api_capture():
    data = request.get_json(force=True)
    devices = data.get("devices", [])
    mode = data.get("mode", "single")

    if not devices:
        return jsonify({"error": "카메라를 선택해 주세요."}), 400

    with state.lock:
        cam_map = {c["dev"]: c for c in state.cameras}

    # 선택된 카메라가 실제 감지 목록에 있는지 확인
    for dev in devices:
        if dev not in cam_map:
            return jsonify({"error": f"cam{dev}은 감지되지 않은 카메라입니다."}), 400

    t_total_start = time.monotonic()

    if mode == "single":
        dev = devices[0]
        bgr, timing = capture_single(dev, cam_map[dev]["type"])
        if bgr is None:
            return jsonify({"error": f"cam{dev} 촬영 실패"}), 500

        t0 = time.monotonic()
        jpeg_b64 = _bgr_to_jpeg_b64(bgr)
        timing["encode_ms"] = (time.monotonic() - t0) * 1000
        timing["total_ms"] = (time.monotonic() - t_total_start) * 1000

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        h, w = bgr.shape[:2]

        with state.lock:
            state.captured = {
                dev: {"bgr": bgr, "jpeg_b64": jpeg_b64, "timestamp": timestamp}
            }

        return jsonify({
            "images": [{
                "dev": dev,
                "jpeg_b64": jpeg_b64,
                "width": w,
                "height": h,
                "type": cam_map[dev]["type"],
            }],
            "timing": _round_timing(timing),
        })

    else:  # multi
        cam_infos = {d: cam_map[d] for d in devices}
        results, error, timing = capture_multi(devices, cam_infos)

        if not results:
            return jsonify({"error": error or "촬영 실패"}), 500

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        images = []

        t0 = time.monotonic()
        with state.lock:
            state.captured = {}
            for dev, bgr in results.items():
                jpeg_b64 = _bgr_to_jpeg_b64(bgr)
                h, w = bgr.shape[:2]
                state.captured[dev] = {
                    "bgr": bgr, "jpeg_b64": jpeg_b64, "timestamp": timestamp
                }
                images.append({
                    "dev": dev,
                    "jpeg_b64": jpeg_b64,
                    "width": w,
                    "height": h,
                    "type": cam_map[dev]["type"],
                })
        timing["encode_ms"] = (time.monotonic() - t0) * 1000
        timing["total_ms"] = (time.monotonic() - t_total_start) * 1000

        resp = {"images": images, "timing": _round_timing(timing)}
        if error:
            resp["warning"] = error
        return jsonify(resp)


@app.route("/api/save", methods=["POST"])
def api_save():
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "captured_images"
    )
    os.makedirs(save_dir, exist_ok=True)

    with state.lock:
        captured = dict(state.captured)
        cam_map = {c["dev"]: c for c in state.cameras}

    if not captured:
        return jsonify({"error": "저장할 이미지가 없습니다."}), 400

    saved_files = []
    for dev, img_data in captured.items():
        cam_type = cam_map.get(dev, {}).get("type", "cam")
        filename = f"{cam_type}_dev{dev}_{img_data['timestamp']}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, img_data["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_files.append(filepath)

    return jsonify({"saved": saved_files})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """미리보기 및 저장 대상 캡처 데이터를 초기화한다."""
    with state.lock:
        state.captured = {}
    return jsonify({"status": "cleared"})


@app.route("/api/close", methods=["POST"])
def api_close():
    print("[INFO] 서버 종료 요청")
    _release_caps()
    threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)),
                     daemon=True).start()
    return jsonify({"status": "shutting down"})


@app.route("/api/trigger_status", methods=["GET"])
def api_trigger_status():
    """AR0234 카메라들의 trigger_mode 상태를 반환한다."""
    with state.lock:
        ar0234_devs = [c["dev"] for c in state.cameras if c["type"] == "ar0234"]
    result = {}
    for dev in ar0234_devs:
        # 이미 세션이 열려있으면 _cap_info의 trigger 값 사용
        # (v4l2 값은 grab() 후 0으로 리셋되지만, 세션 내에서는 트리거 유효)
        if dev in state._cap_info:
            result[str(dev)] = state._cap_info[dev]["trigger"]
        else:
            result[str(dev)] = check_trigger_mode(dev)
    return jsonify(result)


@app.route("/api/params", methods=["GET"])
def api_params_get():
    dev = request.args.get("dev", type=int)
    if dev is None:
        return jsonify({"error": "dev 파라미터 필요"}), 400

    exposure = v4l2_get(dev, "exposure")
    gain = v4l2_get(dev, "analogue_gain")
    trigger = v4l2_get(dev, "trigger_mode")

    return jsonify({
        "dev": dev,
        "exposure": exposure,
        "analogue_gain": gain,
        "trigger_mode": trigger,
    })


@app.route("/api/params", methods=["POST"])
def api_params_set():
    data = request.get_json(force=True)
    dev = data.get("dev")
    if dev is None:
        return jsonify({"error": "dev 필요"}), 400

    changed = []
    # Tegra 드라이버: override_enable=1 이어야 exposure/gain이 센서에 반영됨.
    # _ensure_cap_open에서 이미 켜지만, 세션 재오픈/드라이버 리셋 케이스를
    # 위해 매번 방어적으로 재적용 (idempotent).
    if "exposure" in data or "analogue_gain" in data:
        v4l2_set(dev, "override_enable", 1)
    if "exposure" in data:
        v4l2_set(dev, "exposure", data["exposure"])
        changed.append("exposure")
    if "analogue_gain" in data:
        v4l2_set(dev, "analogue_gain", data["analogue_gain"])
        changed.append("analogue_gain")

    return jsonify({"changed": changed})


@app.route("/api/resolutions", methods=["GET"])
def api_resolutions():
    """해당 카메라의 지원 해상도 목록 + 현재 선택된 해상도."""
    dev = request.args.get("dev", type=int)
    if dev is None:
        return jsonify({"error": "dev 필요"}), 400

    cam = next((c for c in state.cameras if c["dev"] == dev), None)
    # 실제 capture 시 사용하는 포맷에 맞춰 enumerate — 드롭다운에 뜬 해상도가
    # 전부 실제 적용되도록. AR0234=BA10, USB=MJPG (저해상도 포함 넓은 커버리지).
    if cam and cam["type"] == "ar0234":
        res = list_resolutions(dev, fourcc_filter="BA10")
        if not res:
            # 드라이버가 enumerate를 비정상 반환할 때를 대비한 fallback.
            res = [(1920, 1200), (1920, 1080), (1280, 720)]
    elif cam and cam["type"] == "usb":
        res = list_resolutions(dev, fourcc_filter="MJPG")
        if not res:
            res = list_resolutions(dev, fourcc_filter="YUYV")
        if not res:
            res = list_resolutions(dev)
    else:
        res = list_resolutions(dev)

    current = state._cam_res.get(
        dev, (state.default_width, state.default_height))
    return jsonify({"resolutions": res, "current": list(current)})


@app.route("/api/resolution", methods=["POST"])
def api_resolution_set():
    """카메라의 해상도를 변경한다. 열려있는 세션은 닫아서 다음 Capture 때 재오픈."""
    data = request.get_json(force=True)
    dev = data.get("dev")
    if dev is None or "width" not in data or "height" not in data:
        return jsonify({"error": "dev, width, height 필요"}), 400

    w, h = int(data["width"]), int(data["height"])
    state._cam_res[dev] = (w, h)

    # 해당 dev 스트림이 돌고 있으면 먼저 중지 (cap 을 release 하기 전에).
    stream_stopped = False
    if state.streamer is not None and state.streamer.dev == dev:
        state.streamer.stop()
        state.streamer = None
        stream_stopped = True

    cap = state._open_caps.pop(dev, None)
    state._cap_info.pop(dev, None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    # AR0234는 release 시 trigger_mode/override가 리셋될 수 있으므로 복원.
    cam = next((c for c in state.cameras if c["dev"] == dev), None)
    if cam and cam["type"] == "ar0234":
        subprocess.run(
            [sys.executable, "trigger_mode_ctrl.py", "on"],
            capture_output=True, timeout=10,
        )
        _init_ar0234_defaults(dev)

    return jsonify({
        "dev": dev, "width": w, "height": h,
        "stream_stopped": stream_stopped,
    })


@app.route("/api/stream/start", methods=["POST"])
def api_stream_start():
    """선택된 카메라 1대에 대해 백그라운드 스트리머를 시작한다.

    body: {"dev": int, "fps": int(optional, default=15)}
    """
    data = request.get_json(force=True) or {}
    dev = data.get("dev")
    fps = int(data.get("fps", 15))
    if dev is None:
        return jsonify({"error": "dev 필요"}), 400

    if state.streamer is not None:
        # 단일-cam 스코프: 이미 돌고 있으면 같은 dev 면 OK 로 간주, 다른 dev 면 409.
        if state.streamer.dev == dev:
            return jsonify({"dev": dev, "fps": fps, "already": True})
        return jsonify({"error": f"cam{state.streamer.dev} 스트림이 이미 실행 중"}), 409

    cam = next((c for c in state.cameras if c["dev"] == dev), None)
    if cam is None:
        return jsonify({"error": f"cam{dev} 감지 안 됨"}), 400

    # cap 을 미리 열어 warmup + trigger_mode 상태를 고정 (Streamer 는 이 cap 을 빌림).
    _ensure_cap_open(dev, cam["type"])

    streamer = Streamer(dev, cam["type"], fps=fps)
    streamer.start()
    state.streamer = streamer
    return jsonify({"dev": dev, "fps": fps})


@app.route("/api/stream/stop", methods=["POST"])
def api_stream_stop():
    """현재 활성 스트리머를 중지한다. 없으면 no-op."""
    streamer = state.streamer
    if streamer is None:
        return jsonify({"stopped": False})
    streamer.stop()
    state.streamer = None
    return jsonify({"stopped": True, "dev": streamer.dev})


@app.route("/api/stream/<int:dev>/mjpeg")
def api_stream_mjpeg(dev):
    """해당 카메라의 라이브 MJPEG multipart 스트림.

    브라우저는 <img src="/api/stream/N/mjpeg"> 으로 바로 렌더. 클라이언트
    연결이 끊겨도 Streamer 루프는 계속 돈다 — 명시적 /api/stream/stop 만이 중지.
    """
    streamer = state.streamer
    if streamer is None or streamer.dev != dev:
        return jsonify({"error": f"cam{dev} 스트림이 시작되지 않음"}), 404

    boundary = b"--frame"

    def gen():
        last_seq = 0
        while True:
            s = state.streamer  # 스트림이 중간에 교체/중지될 수 있음
            if s is None or s.dev != dev:
                return
            seq, jpeg = s.wait_frame(last_seq, timeout=1.0)
            if jpeg is None or seq == last_seq:
                continue
            last_seq = seq
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, private",
                 "Pragma": "no-cache"},
    )


@app.route("/api/stream/stats", methods=["GET"])
def api_stream_stats():
    """활성 스트리머의 이동평균 통계 (fps, grab/isp/encode ms)."""
    streamer = state.streamer
    if streamer is None:
        return jsonify({"running": False})
    stats = streamer.get_stats()
    stats["running"] = True
    stats["dev"] = streamer.dev
    return jsonify(stats)


# ---------------------------------------------------------------------------
# HTML 템플릿
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Camera Test GUI</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
    padding-bottom: 40px;  /* fixed status-bar가 preview를 가리지 않도록 */
  }

  header {
    padding: 12px 20px;
    background: #222;
    border-bottom: 1px solid #444;
    display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  #btn-close {
    background: #c62828; color: #fff; border: none; padding: 6px 16px;
    border-radius: 4px; cursor: pointer; font-size: 13px;
  }
  #btn-close:hover { background: #e53935; }

  .controls {
    padding: 12px 20px;
    background: #252525;
    border-bottom: 1px solid #333;
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  }
  .controls label { font-size: 13px; color: #aaa; }
  .controls select, .controls input[type="number"] {
    background: #333; color: #e0e0e0; border: 1px solid #555;
    padding: 4px 8px; border-radius: 3px; font-size: 13px;
  }
  .controls select:focus, .controls input:focus { outline: none; border-color: #4fc3f7; }

  .btn {
    padding: 6px 18px; border: none; border-radius: 4px;
    cursor: pointer; font-size: 13px; font-weight: 500;
  }
  .btn-primary { background: #1565c0; color: #fff; }
  .btn-primary:hover { background: #1976d2; }
  .btn-primary:disabled { background: #555; cursor: not-allowed; }
  /* Stop Stream 용: 스트리밍 중일 때 Capture 버튼이 Stop 역할을 하도록 색상만 교체.
     .btn-capture 의 크기/웨이트는 그대로 유지돼 위계도 유지된다. */
  .btn-danger { background: #c62828; color: #fff; }
  .btn-danger:hover { background: #d32f2f; }
  .btn-danger:disabled { background: #555; cursor: not-allowed; }
  /* Capture: primary 중에서도 위계를 한 단계 더 — 사용자가 가장 자주 누르는 메인 액션.
     Save 와 시각적으로 분명히 구분하려고 패딩/폰트 키움. */
  .btn-capture {
    padding: 9px 32px; font-size: 14px; font-weight: 700;
    letter-spacing: 0.3px;
  }
  .btn-success { background: #2e7d32; color: #fff; }
  .btn-success:hover { background: #388e3c; }
  .btn-success:disabled { background: #555; cursor: not-allowed; }
  /* Save 를 outline 으로 낮춰 Capture 와 시각적 무게를 분리. 위치도 프리뷰 쪽으로 이동. */
  .btn-success-outline {
    background: transparent; color: #81c784; border: 1px solid #2e7d32;
  }
  .btn-success-outline:hover { background: rgba(46, 125, 50, 0.18); color: #a5d6a7; border-color: #388e3c; }
  .btn-success-outline:disabled { color: #555; border-color: #444; cursor: not-allowed; background: transparent; }
  .btn-outline {
    background: transparent; color: #aaa; border: 1px solid #555;
  }
  .btn-outline:hover { color: #fff; border-color: #888; }
  .btn-outline:disabled { color: #555; border-color: #444; cursor: not-allowed; }

  .mode-toggle {
    display: flex; gap: 0;
  }
  .mode-toggle label {
    padding: 4px 12px; border: 1px solid #555; cursor: pointer;
    font-size: 13px; color: #aaa;
  }
  .mode-toggle label:first-child { border-radius: 3px 0 0 3px; }
  .mode-toggle label:last-child { border-radius: 0 3px 3px 0; }
  .mode-toggle input[type="radio"] { display: none; }
  .mode-toggle input[type="radio"]:checked + span {
    color: #4fc3f7; font-weight: bold;
  }
  .mode-toggle label:has(input:checked) {
    border-color: #4fc3f7; background: rgba(79, 195, 247, 0.1);
  }

  /* 카메라 체크박스 (다중 모드) — .controls 밖의 자체 행 */
  .cam-checkboxes {
    display: none; gap: 8px; flex-wrap: wrap;
    padding: 8px 20px;
    background: #252525;
    border-bottom: 1px solid #333;
  }
  .cam-checkboxes.active { display: flex; }
  .cam-checkboxes label {
    padding: 3px 10px; border: 1px solid #555; border-radius: 3px;
    cursor: pointer; font-size: 12px;
    max-width: 240px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }

  /* 액션 버튼 그룹 (controls 우측 고정) */
  .controls .action-group {
    display: flex; gap: 8px;
    margin-left: auto;
  }
  .cam-checkboxes input:checked + span { color: #4fc3f7; }
  .cam-checkboxes label:has(input:checked) {
    border-color: #4fc3f7; background: rgba(79, 195, 247, 0.1);
  }
  .cam-checkboxes input[type="checkbox"] { display: none; }

  /* 캡처 결과 액션 툴바 — 프리뷰 바로 위. Save/Clear 는 "찍은 이미지에 대한 조치"이므로
     프리뷰 근처에 배치해 Capture(상단) 와 위치로도 구분. */
  .preview-toolbar {
    padding: 8px 20px;
    display: flex; gap: 8px; justify-content: flex-end; align-items: center;
    background: #1e1e1e;
    border-bottom: 1px solid #2a2a2a;
  }
  .preview-toolbar .hint {
    margin-right: auto; color: #555; font-size: 12px; font-style: italic;
  }

  /* 미리보기 영역 */
  .preview-area {
    padding: 16px;
    min-height: 300px;
    display: flex; flex-wrap: wrap; gap: 12px;
    justify-content: center; align-items: flex-start;
  }
  .preview-area .placeholder {
    color: #555; font-size: 14px; text-align: center;
    padding-top: 120px; width: 100%;
  }
  .preview-card {
    background: #222; border: 1px solid #444; border-radius: 4px;
    overflow: hidden; max-width: 100%;
  }
  .preview-card .card-header {
    padding: 6px 12px; font-size: 12px; background: #2a2a2a;
    border-bottom: 1px solid #333; color: #aaa;
  }
  .preview-card img {
    display: block; max-width: 100%; height: auto; background: #000;
  }
  .preview-area.single .preview-card { max-width: 960px; }
  .preview-area.multi .preview-card { max-width: 640px; flex: 1 1 45%; }

  /* 파라미터 패널 (controls 아래, preview 위에 고정 위치) */
  .params-panel {
    padding: 12px 20px;
    background: #222;
    border-bottom: 1px solid #333;
    display: none;
  }
  .params-panel.active { display: block; }
  .params-panel h3 { font-size: 13px; margin-bottom: 8px; color: #aaa; }
  .param-sliders {
    display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 8px;
  }
  .param-sliders .param-row {
    flex: 1 1 340px; margin-bottom: 0;
  }
  .param-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 6px;
  }
  .param-row label { font-size: 12px; width: 120px; color: #ccc; }
  .param-row input[type="range"] {
    flex: 1; accent-color: #4fc3f7;
  }
  .param-row .param-value {
    font-size: 12px; color: #4fc3f7; min-width: 60px;
    font-family: monospace;
  }
  .preset-row {
    display: flex; align-items: center; gap: 6px; margin-top: 8px;
    flex-wrap: wrap;
  }
  .preset-row .preset-label {
    font-size: 12px; color: #888; width: 120px;
  }
  .preset-btn {
    background: #333; color: #e0e0e0; border: 1px solid #555;
    padding: 4px 12px; border-radius: 3px; cursor: pointer;
    font-size: 12px;
  }
  .preset-btn:hover { background: #3e5368; border-color: #4fc3f7; }
  .preset-btn.default { border-color: #4fc3f7; color: #4fc3f7; }

  /* 트리거 상태 표시 */
  .trigger-status {
    display: none; align-items: center; gap: 6px;
    font-size: 12px; padding: 4px 10px;
    border: 1px solid #555; border-radius: 3px;
    background: #2a2a2a;
  }
  .trigger-status.active { display: flex; }
  .trigger-status .badge {
    padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px;
  }
  .trigger-status .badge.on { background: #2e7d32; color: #fff; }
  .trigger-status .badge.off { background: #c62828; color: #fff; }
  .trigger-status .warn-icon {
    cursor: help; font-size: 16px; color: #ffa726;
    position: relative;
  }
  .trigger-status .warn-icon .tooltip {
    display: none;
    position: absolute; bottom: 130%; left: 50%; transform: translateX(-50%);
    background: #333; color: #eee; padding: 8px 12px; border-radius: 4px;
    font-size: 12px; white-space: nowrap; font-weight: normal;
    border: 1px solid #555; z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }
  .trigger-status .warn-icon .tooltip::after {
    content: ''; position: absolute; top: 100%; left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent; border-top-color: #333;
  }
  .trigger-status .warn-icon:hover .tooltip { display: block; }

  /* 타이밍 바 */
  .timing-bar {
    display: none;
    padding: 8px 20px;
    background: #1e1e1e;
    border-bottom: 1px solid #333;
    font-size: 12px; color: #aaa;
    font-family: 'Courier New', monospace;
    flex-wrap: wrap; gap: 18px;
  }
  .timing-bar.active { display: flex; }
  .timing-bar .metric { display: inline-flex; gap: 5px; }
  .timing-bar .label { color: #888; }
  .timing-bar .val { color: #4fc3f7; }
  .timing-bar .val.total { color: #fff; font-weight: bold; }
  .timing-bar .val.delta { color: #81c784; }
  .timing-bar .val.delta.warn { color: #ffa726; }

  /* Busy 상태: 세팅/촬영 진행 중 전체 인터랙션 영역을 디밍하고 클릭 차단. */
  body.busy { cursor: wait; }
  body.busy header,
  body.busy .controls,
  body.busy .cam-checkboxes,
  body.busy .params-panel {
    opacity: 0.5;
    pointer-events: none;
  }
  body.busy .status-bar { opacity: 1; }

  /* 스피너 — 상태바에서 진행 중 표시 */
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    margin-right: 8px;
    vertical-align: middle;
    border: 2px solid #444;
    border-top-color: #4fc3f7;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* 상태바 */
  .status-bar {
    padding: 8px 20px;
    background: #1e1e1e;
    border-top: 1px solid #333;
    font-size: 12px; color: #888;
    position: fixed; bottom: 0; width: 100%;
  }
  .status-bar .ok { color: #81c784; }
  .status-bar .error { color: #ef5350; }
  .status-bar .info { color: #4fc3f7; }
</style>
</head>
<body>

<header>
  <h1>Camera Test GUI</h1>
  <button id="btn-close" onclick="closeServer()">Close</button>
</header>

<div class="controls">
  <label>Camera:</label>
  <select id="cam-select"><option value="">-- loading --</option></select>
  <button class="btn btn-outline" id="btn-refresh" onclick="refreshCameras()">Refresh</button>

  <label>Resolution:</label>
  <select id="res-select" disabled>
    <option value="">-- select a camera --</option>
  </select>

  <div class="mode-toggle">
    <label><input type="radio" name="mode" value="single" checked><span>Single</span></label>
    <label><input type="radio" name="mode" value="multi"><span>Multi</span></label>
    <label><input type="radio" name="mode" value="stream"><span>Streaming</span></label>
  </div>

  <div class="trigger-status" id="trigger-status">
    Trigger: <span class="badge" id="trigger-badge">--</span>
    <span class="warn-icon" id="trigger-warn" style="display:none;">&#9888;
      <span class="tooltip">Multi mode requires trigger_mode=ON.<br>Run: python3 trigger_mode_ctrl.py on</span>
    </span>
  </div>

  <div class="action-group">
    <button class="btn btn-primary btn-capture" id="btn-capture" onclick="onMainButton()">Capture</button>
  </div>
</div>

<!-- Multi 모드 카메라 체크박스: 자체 행 (action 버튼이 밀려나지 않도록 분리) -->
<div class="cam-checkboxes" id="cam-checkboxes"></div>

<div class="params-panel" id="params-panel">
  <h3>Camera Parameters (AR0234)</h3>
  <div class="param-sliders">
    <div class="param-row">
      <label>Exposure (2-65535)</label>
      <input type="range" id="slider-exposure" min="2" max="65535" value="900">
      <span class="param-value" id="val-exposure">900</span>
    </div>
    <div class="param-row">
      <label>Gain (100-1200)</label>
      <input type="range" id="slider-gain" min="100" max="1200" value="100">
      <span class="param-value" id="val-gain">100</span>
    </div>
  </div>
  <div class="preset-row">
    <span class="preset-label">Presets</span>
    <button class="preset-btn" onclick="applyPreset(200, 100)">Outdoor</button>
    <button class="preset-btn" onclick="applyPreset(900, 100)">Indoor</button>
    <button class="preset-btn" onclick="applyPreset(1100, 500)">Low Light</button>
    <button class="preset-btn" onclick="applyPreset(1100, 1000)">Dark</button>
    <button class="preset-btn default" onclick="applyPreset(900, 100)">Default</button>
  </div>
</div>

<div class="timing-bar" id="timing-bar"></div>

<div class="preview-toolbar">
  <span class="hint">Captured image actions</span>
  <button class="btn btn-success-outline" id="btn-save" onclick="doSave()" disabled>Save</button>
  <button class="btn btn-outline" id="btn-clear" onclick="doClear()" disabled>Clear</button>
</div>

<div class="preview-area single" id="preview-area">
  <div class="placeholder">Select a camera and press Capture</div>
</div>

<div class="status-bar" id="status-bar">Ready</div>

<script>
let cameras = [];
let hasCaptured = false;
let busyState = false;

// 스트리밍 상태. streaming=true 동안 Capture 버튼은 Stop Stream 역할을 하고,
// cam/resolution 드롭다운은 잠긴다. statsTimer 는 /api/stream/stats 를 500ms 주기로
// polling 하는 setInterval 핸들.
let streaming = false;
let statsTimer = null;

// 수 초가 걸리는 작업(해상도 변경 / 카메라 재검출 / 촬영) 동안 모든 인터랙션
// 요소(버튼·드롭다운·슬라이더·라디오·체크박스·프리셋·Close)를 비활성화하고
// 상태바에 스피너를 띄워 "지금 세팅 중"임을 시각적으로 명확히 보여준다.
// body.busy CSS 클래스로 추가 디밍/pointer-events:none 중첩 차단.
function setBusy(on, msg) {
  busyState = on;
  document.body.classList.toggle('busy', on);

  if (on && msg) {
    const bar = document.getElementById('status-bar');
    bar.className = 'status-bar';
    bar.innerHTML = '<span class="spinner"></span><span class="info">' + msg + '</span>';
  }

  // 모든 인터랙티브 요소 잠금 (pointer-events:none 은 CSS 에서 처리하되,
  // disabled 속성도 함께 설정해 프로그램적 제출/키보드 Enter 도 차단).
  document.querySelectorAll(
    'button, select, input[type="range"], input[type="radio"], input[type="checkbox"]'
  ).forEach(el => { el.disabled = on; });

  if (!on) {
    // Save / Clear: 이전 캡처 존재 여부에 따라 복원.
    document.getElementById('btn-save').disabled = !hasCaptured;
    document.getElementById('btn-clear').disabled = !hasCaptured;
    // Resolution 드롭다운: 선택된 카메라 기준 enable 재계산.
    updateResolutionDropdown();
  }
}

// --- 타이밍 바 ---
function showTiming(t) {
  const bar = document.getElementById('timing-bar');
  if (!t) {
    bar.classList.remove('active');
    bar.innerHTML = '';
    return;
  }
  const metric = (label, val, cls) => {
    const c = cls ? (' ' + cls) : '';
    return '<span class="metric"><span class="label">' + label + '</span>' +
           '<span class="val' + c + '">' + val + '</span></span>';
  };
  const fmt = (v, d) => (v === null || v === undefined) ? '-' : v.toFixed(d) + ' ms';

  const parts = [];
  // drain은 실제 프레임을 비워낸 경우에만 표시
  if (t.drain_ms !== undefined && (t.drained || 0) > 0) {
    parts.push(metric('drain(' + t.drained + ')', fmt(t.drain_ms, 2)));
  }
  parts.push(metric('pulse', fmt(t.pulse_ms, 2)));
  parts.push(metric('grab', fmt(t.grab_ms, 2)));
  parts.push(metric('retrieve', fmt(t.retrieve_ms, 2)));
  parts.push(metric('isp', fmt(t.isp_ms, 2)));
  parts.push(metric('encode', fmt(t.encode_ms, 2)));
  parts.push(metric('total', fmt(t.total_ms, 2), 'total'));
  if (t.delta_grab_ms !== null && t.delta_grab_ms !== undefined) {
    // 2ms 초과 시 주황색 경고
    const warn = t.delta_grab_ms > 2.0 ? ' warn' : '';
    parts.push(metric('sync delta', fmt(t.delta_grab_ms, 3), 'delta' + warn));
  }
  bar.innerHTML = parts.join('');
  bar.classList.add('active');
}

// --- 상태바 ---
function setStatus(msg, cls) {
  const el = document.getElementById('status-bar');
  el.className = 'status-bar';
  el.innerHTML = '<span class="' + (cls || '') + '">' + msg + '</span>';
}

// --- 모드 ---
function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

document.querySelectorAll('input[name="mode"]').forEach(r => {
  r.addEventListener('change', async () => {
    // 스트리밍 중 다른 모드로 전환하려 하면 먼저 중지.
    if (streaming && getMode() !== 'stream') {
      await doStopStream();
    }
    applyModeUI();
    updateTriggerStatus();
    updateParamsPanel();
    updateResolutionDropdown();
  });
});

// 현재 모드에 맞춰 UI 를 재배치. stream 모드는 단일 카메라 전용 + Save/Clear
// 비활성 + 메인 버튼을 Start Stream 로 표기.
function applyModeUI() {
  const mode = getMode();
  const multi = mode === 'multi';
  const stream = mode === 'stream';
  document.getElementById('cam-select').style.display = multi ? 'none' : '';
  document.getElementById('cam-checkboxes').classList.toggle('active', multi);
  document.querySelector('.preview-toolbar').style.display = stream ? 'none' : '';
  const btn = document.getElementById('btn-capture');
  if (stream) {
    btn.textContent = streaming ? 'Stop Stream' : 'Start Stream';
    btn.classList.toggle('btn-danger', streaming);
    btn.classList.toggle('btn-primary', !streaming);
  } else {
    btn.textContent = 'Capture';
    btn.classList.add('btn-primary');
    btn.classList.remove('btn-danger');
  }
}

// --- 트리거 상태 ---
async function updateTriggerStatus() {
  const container = document.getElementById('trigger-status');
  const badge = document.getElementById('trigger-badge');
  const warn = document.getElementById('trigger-warn');
  const multi = getMode() === 'multi';
  const hasAR0234 = cameras.some(c => c.type === 'ar0234');

  if (!multi || !hasAR0234) {
    container.classList.remove('active');
    return;
  }

  container.classList.add('active');
  badge.textContent = '...';
  badge.className = 'badge';

  try {
    const resp = await fetch('/api/trigger_status');
    const data = await resp.json();
    const allOn = Object.values(data).every(v => v === true);
    const anyOn = Object.values(data).some(v => v === true);

    if (allOn) {
      badge.textContent = 'ON';
      badge.className = 'badge on';
      warn.style.display = 'none';
    } else {
      badge.textContent = anyOn ? 'PARTIAL' : 'OFF';
      badge.className = 'badge off';
      warn.style.display = '';
    }
  } catch (e) {
    badge.textContent = '?';
    badge.className = 'badge';
    warn.style.display = '';
  }
}

// --- 카메라 목록 ---
async function loadCameras(cams) {
  cameras = cams;
  const sel = document.getElementById('cam-select');
  sel.innerHTML = '';
  if (cams.length === 0) {
    sel.innerHTML = '<option value="">-- no cameras --</option>';
  } else {
    cams.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.dev;
      opt.textContent = c.label;
      opt.dataset.type = c.type;
      sel.appendChild(opt);
    });
  }

  // 체크박스 (다중 모드) — GPIO 트리거 동기 촬영은 AR0234에만 유효하므로
  // USB 카메라는 Multi 체크박스에서 제외한다 (Single 모드에서는 그대로 사용).
  const cb = document.getElementById('cam-checkboxes');
  cb.innerHTML = '';
  const multiCams = cams.filter(c => c.type === 'ar0234');
  if (multiCams.length === 0) {
    cb.innerHTML = '<span style="color:#888;font-size:12px;padding:3px 4px;">No AR0234 cameras for multi-sync capture</span>';
  } else {
    multiCams.forEach(c => {
      const lbl = document.createElement('label');
      lbl.innerHTML = '<input type="checkbox" value="' + c.dev + '" data-type="' + c.type + '"><span>' + c.label + '</span>';
      cb.appendChild(lbl);
    });
  }

  updateParamsPanel();
  updateTriggerStatus();
  updateResolutionDropdown();
}

async function refreshCameras() {
  if (busyState) return;
  // 서버가 cap 을 release 하면서 스트리머도 stop 하므로, UI 도 미리 정리.
  if (streaming) await doStopStream();
  setBusy(true, 'Scanning cameras...');
  try {
    const resp = await fetch('/api/cameras/refresh', { method: 'POST' });
    const cams = await resp.json();
    await loadCameras(cams);
    setStatus(cams.length + ' camera(s) detected', 'ok');
  } catch (e) {
    setStatus('Camera scan failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

// 페이지 로드 시
(async () => {
  try {
    const resp = await fetch('/api/cameras');
    const cams = await resp.json();
    await loadCameras(cams);
    setStatus(cams.length + ' camera(s) detected', 'ok');
  } catch (e) {
    setStatus('Failed to load cameras', 'error');
  }
})();

// 드롭다운 변경 시 파라미터 패널 & 해상도 목록 업데이트
document.getElementById('cam-select').addEventListener('change', () => {
  updateParamsPanel();
  updateResolutionDropdown();
});

// --- 파라미터 패널 ---
function getSelectedCamType() {
  if (getMode() === 'single') {
    const sel = document.getElementById('cam-select');
    const opt = sel.options[sel.selectedIndex];
    return opt ? opt.dataset.type : null;
  }
  return null;
}

function getSelectedDev() {
  if (getMode() === 'single') {
    return parseInt(document.getElementById('cam-select').value);
  }
  return null;
}

// 파라미터(exposure/gain) 적용 대상 AR0234 카메라 리스트.
// Single/Stream 모드: 선택된 AR0234 1대, Multi 모드: 체크된 AR0234 모두.
function getTargetDevsForParams() {
  const mode = getMode();
  if (mode === 'single' || mode === 'stream') {
    const v = document.getElementById('cam-select').value;
    if (!v) return [];
    const dev = parseInt(v);
    const cam = cameras.find(c => c.dev === dev);
    return (cam && cam.type === 'ar0234') ? [dev] : [];
  }
  const devs = [];
  document.querySelectorAll('#cam-checkboxes input:checked').forEach(cb => {
    const dev = parseInt(cb.value);
    const cam = cameras.find(c => c.dev === dev);
    if (cam && cam.type === 'ar0234') devs.push(dev);
  });
  return devs;
}

async function updateParamsPanel() {
  const panel = document.getElementById('params-panel');
  const devs = getTargetDevsForParams();

  if (devs.length === 0) {
    panel.classList.remove('active');
    return;
  }
  panel.classList.add('active');
  // 첫 번째 타겟 카메라 기준으로 현재 값 read-back
  try {
    const resp = await fetch('/api/params?dev=' + devs[0]);
    const p = await resp.json();
    if (p.exposure && p.exposure !== '?') {
      document.getElementById('slider-exposure').value = p.exposure;
      document.getElementById('val-exposure').textContent = p.exposure;
    }
    if (p.analogue_gain && p.analogue_gain !== '?') {
      document.getElementById('slider-gain').value = p.analogue_gain;
      document.getElementById('val-gain').textContent = p.analogue_gain;
    }
  } catch (e) {}
}

// Multi 모드에서 체크박스 변경 시에도 패널 & 해상도 목록 갱신
document.getElementById('cam-checkboxes').addEventListener('change', () => {
  updateParamsPanel();
  updateResolutionDropdown();
});

// 슬라이더 디바운스
let paramTimer = null;
function onSliderChange(name, slider, display) {
  display.textContent = slider.value;
  clearTimeout(paramTimer);
  paramTimer = setTimeout(() => {
    const devs = getTargetDevsForParams();
    if (devs.length === 0) return;
    devs.forEach(dev => {
      const body = { dev: dev };
      body[name] = parseInt(slider.value);
      fetch('/api/params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
    });
    setStatus(name + ' = ' + slider.value + ' (' + devs.length + ' cam)', 'ok');
  }, 200);
}

document.getElementById('slider-exposure').addEventListener('input', function() {
  onSliderChange('exposure', this, document.getElementById('val-exposure'));
});
document.getElementById('slider-gain').addEventListener('input', function() {
  onSliderChange('analogue_gain', this, document.getElementById('val-gain'));
});

// --- 해상도 드롭다운 ---
// Single/Stream: 선택된 1대, Multi: 체크된 전체.
function getResolutionTargets() {
  const mode = getMode();
  if (mode === 'single' || mode === 'stream') {
    const v = document.getElementById('cam-select').value;
    return v ? [parseInt(v)] : [];
  }
  const out = [];
  document.querySelectorAll('#cam-checkboxes input:checked')
    .forEach(cb => out.push(parseInt(cb.value)));
  return out;
}

async function updateResolutionDropdown() {
  const sel = document.getElementById('res-select');
  const devs = getResolutionTargets();
  if (devs.length === 0) {
    sel.innerHTML = '<option value="">-- select a camera --</option>';
    sel.disabled = true;
    return;
  }
  try {
    const r = await fetch('/api/resolutions?dev=' + devs[0]).then(r => r.json());
    const opts = (r.resolutions || []).map(
      ([w, h]) => '<option value="' + w + 'x' + h + '">' + w + ' x ' + h + '</option>'
    );
    sel.innerHTML = opts.length ? opts.join('') : '<option value="">(no resolutions)</option>';
    if (r.current && r.current.length === 2) {
      const key = r.current[0] + 'x' + r.current[1];
      // 일치하는 옵션이 있을 때만 선택 (없으면 첫 번째 옵션이 자동 선택됨).
      if ([...sel.options].some(o => o.value === key)) {
        sel.value = key;
      }
    }
    sel.disabled = opts.length === 0;
  } catch (e) {
    sel.innerHTML = '<option value="">(error)</option>';
    sel.disabled = true;
  }
}

document.getElementById('res-select').addEventListener('change', async function() {
  if (busyState) return;
  if (!this.value) return;
  const [w, h] = this.value.split('x').map(Number);
  const devs = getResolutionTargets();
  if (devs.length === 0) {
    setStatus('Select a camera first', 'error');
    return;
  }
  setBusy(true, 'Setting resolution ' + w + 'x' + h + '...');
  try {
    const responses = await Promise.all(devs.map(dev =>
      fetch('/api/resolution', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dev: dev, width: w, height: h })
      }).then(r => r.json())
    ));
    // 서버가 스트림을 강제 중지했으면 UI 도 반영.
    if (streaming && responses.some(r => r && r.stream_stopped)) {
      streaming = false;
      clearInterval(statsTimer);
      statsTimer = null;
      applyModeUI();
    }
    setStatus('resolution ' + w + 'x' + h + ' (' + devs.length + ' cam, applied on next Capture)', 'ok');
  } catch (e) {
    setStatus('Resolution change failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
});

// --- 프리셋 (Outdoor/Indoor/Low Light/Dark/Default) ---
// 슬라이더 디바운스와 별개로, 버튼 클릭 시 즉시 두 값을 브로드캐스트.
function applyPreset(exposure, gain) {
  document.getElementById('slider-exposure').value = exposure;
  document.getElementById('val-exposure').textContent = exposure;
  document.getElementById('slider-gain').value = gain;
  document.getElementById('val-gain').textContent = gain;

  const devs = getTargetDevsForParams();
  if (devs.length === 0) {
    setStatus('Select AR0234 camera(s) first', 'error');
    return;
  }
  devs.forEach(dev => {
    fetch('/api/params', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dev: dev, exposure: exposure, analogue_gain: gain })
    });
  });
  setStatus('preset: exposure=' + exposure + ' gain=' + gain +
            ' (' + devs.length + ' cam)', 'ok');
}

// --- 메인 버튼 (모드별 디스패처) ---
// Single/Multi: Capture. Stream: Start/Stop 토글.
function onMainButton() {
  if (getMode() === 'stream') {
    streaming ? doStopStream() : doStartStream();
  } else {
    doCapture();
  }
}

// --- 촬영 ---
async function doCapture() {
  if (busyState) return;
  const mode = getMode();
  let devices = [];

  if (mode === 'single') {
    const val = document.getElementById('cam-select').value;
    if (!val) { setStatus('Select a camera first', 'error'); return; }
    devices = [parseInt(val)];
  } else {
    document.querySelectorAll('#cam-checkboxes input:checked').forEach(cb => {
      devices.push(parseInt(cb.value));
    });
    if (devices.length === 0) { setStatus('Select at least one camera', 'error'); return; }
  }

  setBusy(true, 'Capturing...');
  try {
    const resp = await fetch('/api/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ devices: devices, mode: mode })
    });
    const data = await resp.json();

    if (data.error) {
      setStatus(data.error, 'error');
      return;
    }

    // 미리보기 표시
    const area = document.getElementById('preview-area');
    area.className = 'preview-area ' + (data.images.length > 1 ? 'multi' : 'single');
    area.innerHTML = '';

    data.images.forEach(img => {
      const card = document.createElement('div');
      card.className = 'preview-card';
      card.innerHTML =
        '<div class="card-header">cam' + img.dev + ' (' + img.type + ') &mdash; ' + img.width + 'x' + img.height + '</div>' +
        '<img src="data:image/jpeg;base64,' + img.jpeg_b64 + '">';
      area.appendChild(card);
    });

    hasCaptured = true;
    document.getElementById('btn-save').disabled = false;
    document.getElementById('btn-clear').disabled = false;
    showTiming(data.timing);

    let msg = data.images.length + ' image(s) captured';
    if (data.warning) msg += ' (warning: ' + data.warning + ')';
    setStatus(msg, 'ok');

  } catch (e) {
    setStatus('Capture failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

// --- 저장 ---
async function doSave() {
  if (!hasCaptured) return;
  setStatus('Saving...', 'info');

  try {
    const resp = await fetch('/api/save', { method: 'POST' });
    const data = await resp.json();

    if (data.error) {
      setStatus(data.error, 'error');
      return;
    }

    const files = data.saved.map(f => f.split('/').pop()).join(', ');
    setStatus('Saved: ' + files, 'ok');
    document.getElementById('btn-save').disabled = true;
    hasCaptured = false;

  } catch (e) {
    setStatus('Save failed: ' + e.message, 'error');
  }
}

// --- 화면 초기화 ---
async function doClear() {
  try {
    await fetch('/api/clear', { method: 'POST' });
  } catch (e) {}
  const area = document.getElementById('preview-area');
  area.className = 'preview-area single';
  area.innerHTML = '<div class="placeholder">Select a camera and press Capture</div>';
  hasCaptured = false;
  document.getElementById('btn-save').disabled = true;
  document.getElementById('btn-clear').disabled = true;
  showTiming(null);
  setStatus('Preview cleared', 'info');
}

// --- 스트리밍 ---
// 스트리밍 중에는 cam 전환과 resolution 변경을 잠근다. exposure/gain 은
// 라이브 튜닝을 위해 열어둔다.
function lockControls(on) {
  document.getElementById('cam-select').disabled = on;
  document.getElementById('res-select').disabled =
    on || document.getElementById('res-select').options.length === 0;
  document.getElementById('btn-refresh').disabled = on;
}

async function doStartStream() {
  if (busyState || streaming) return;
  const val = document.getElementById('cam-select').value;
  if (!val) { setStatus('Select a camera first', 'error'); return; }
  const dev = parseInt(val);

  setStatus('Starting stream...', 'info');
  try {
    const resp = await fetch('/api/stream/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dev: dev, fps: 15 })
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      setStatus(data.error || ('Stream start failed (' + resp.status + ')'), 'error');
      return;
    }
    streaming = true;
    const area = document.getElementById('preview-area');
    area.className = 'preview-area single';
    // 브라우저가 native 로 multipart/x-mixed-replace 를 렌더.
    // cache 우회 쿼리로 이전 스트림 잔존 방지.
    area.innerHTML =
      '<div class="preview-card">' +
        '<div class="card-header">cam' + dev + ' &mdash; live</div>' +
        '<img src="/api/stream/' + dev + '/mjpeg?t=' + Date.now() + '">' +
      '</div>';
    applyModeUI();
    lockControls(true);
    statsTimer = setInterval(pollStreamStats, 500);
    setStatus('Streaming cam' + dev, 'ok');
  } catch (e) {
    setStatus('Stream start failed: ' + e.message, 'error');
  }
}

async function doStopStream() {
  if (!streaming) return;
  clearInterval(statsTimer);
  statsTimer = null;
  try {
    await fetch('/api/stream/stop', { method: 'POST' });
  } catch (e) {}
  streaming = false;
  // img 태그를 지워 브라우저의 multipart 연결도 종료.
  const area = document.getElementById('preview-area');
  if (getMode() === 'stream') {
    area.innerHTML = '<div class="placeholder">Stream stopped. Press Start Stream.</div>';
  } else {
    area.innerHTML = '<div class="placeholder">Select a camera and press Capture</div>';
  }
  showTiming(null);
  applyModeUI();
  lockControls(false);
  setStatus('Stream stopped', 'info');
}

async function pollStreamStats() {
  try {
    const resp = await fetch('/api/stream/stats');
    const s = await resp.json();
    if (!s.running) return;
    // timing-bar 를 라이브 fps 위젯으로 재활용.
    showTiming({
      grab_ms: s.grab_ms, isp_ms: s.isp_ms, encode_ms: s.encode_ms,
      total_ms: s.fps > 0 ? 1000.0 / s.fps : 0,
    });
  } catch (e) {}
}

// --- 종료 ---
async function closeServer() {
  if (!confirm('Close the server?')) return;
  try {
    await fetch('/api/close', { method: 'POST' });
    setStatus('Server shutting down...', 'info');
    setTimeout(() => { document.body.innerHTML = '<div style="text-align:center;padding:100px;color:#888;font-size:18px;">Server closed. You can close this tab.</div>'; }, 500);
  } catch (e) {}
}
</script>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Camera Test GUI (Flask web)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cpu", action="store_true",
                        help="GPU 대신 CPU demosaic 사용")
    args = parser.parse_args()

    state.default_width = args.width
    state.default_height = args.height
    if args.cpu:
        state.use_gpu = False

    print("[INFO] 카메라 스캔 중...")
    state.cameras = detect_all_cameras()
    print(f"[INFO] {len(state.cameras)}개 카메라 감지: "
          f"{[c['label'] for c in state.cameras]}")

    # AR0234 카메라가 있으면 trigger_mode 자동 활성화 + Default 프리셋 사전 적용
    ar0234_devs = [c["dev"] for c in state.cameras if c["type"] == "ar0234"]
    if ar0234_devs:
        print("[INFO] trigger_mode 활성화 중...")
        subprocess.run(
            [sys.executable, "trigger_mode_ctrl.py", "on"],
            capture_output=True, timeout=10
        )
        print("[INFO] trigger_mode=ON 완료")
        # 브라우저에서 /api/params 초기 GET이 800/? 같은 센서 부팅값을 읽지 않도록
        # 서버 시작 시점에 override_enable=1, exposure=900, gain=100 을 써둔다.
        for dev in ar0234_devs:
            _init_ar0234_defaults(dev)
        print(f"[INFO] AR0234 기본값(exposure={_AR0234_DEFAULT_EXPOSURE}, "
              f"gain={_AR0234_DEFAULT_GAIN}) 적용")

    print(f"[INFO] ISP: {'GPU (PyCUDA)' if state.use_gpu else 'CPU'}")
    print(f"[INFO] Camera Test GUI: http://0.0.0.0:{args.port}/")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
