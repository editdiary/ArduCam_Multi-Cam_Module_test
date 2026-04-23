"""동기 캡처 엔진.

GPIO 트리거로 다중 카메라를 동시에 캡처하고,
ISP 처리 및 JPEG 인코딩까지 수행하는 파이프라인을 제공한다.

주요 흐름:
  GPIO 펄스 → 병렬 grab → 병렬 retrieve → ISP(demosaic) → JPEG 인코딩

SyncCaptureServer는 이 전체 루프를 백그라운드 스레드에서 실행하며,
외부(Flask 웹 서버 등)에서 최신 JPEG 프레임을 가져갈 수 있도록 제공한다.
사전에 trigger_mode=1이 설정되어 있어야 한다.
"""

import collections
import threading
import time

import cv2

from ar0234_cam.isp import demosaic, demosaic_gpu, _USE_GPU
from ar0234_cam.encoder import GstJpegEncoder
from ar0234_cam.gpio import gpio_pulse
from ar0234_cam.v4l2_utils import V4L2_BA10, check_trigger_mode

# PyCUDA 컨텍스트 (GPU 모드에서 스레드 간 컨텍스트 전환에 필요)
_cuda_ctx = None
if _USE_GPU:
    import pycuda.driver as cuda
    _cuda_ctx = cuda.Context.get_current()


def parallel_grab(caps, cam_ids):
    """여러 카메라에서 동시에 grab을 수행한다 (멀티스레드).

    Args:
        caps: {dev_num: cv2.VideoCapture} 딕셔너리
        cam_ids: 카메라 디바이스 번호 리스트

    Returns:
        {dev_num: {"ok": bool, "grab_ms": float}} 딕셔너리
    """
    results = {}

    def _grab(dev):
        t0 = time.monotonic()
        ret = caps[dev].grab()
        t1 = time.monotonic()
        results[dev] = {"ok": ret, "grab_ms": (t1 - t0) * 1000}

    threads = [threading.Thread(target=_grab, args=(d,)) for d in cam_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=6)
    return results


def parallel_retrieve(caps, cam_ids, grabs):
    """grab 성공한 카메라에서 동시에 프레임을 retrieve한다 (멀티스레드).

    Args:
        caps: {dev_num: cv2.VideoCapture} 딕셔너리
        cam_ids: 카메라 디바이스 번호 리스트
        grabs: parallel_grab()의 반환값

    Returns:
        (raws, errors) 튜플:
          raws: {dev_num: raw_frame} 딕셔너리
          errors: 실패한 디바이스 번호 리스트
    """
    raws = {}
    errors = []

    def _retrieve(dev):
        if not grabs.get(dev, {}).get("ok"):
            errors.append(dev)
            return
        ret, raw = caps[dev].retrieve()
        if not ret:
            errors.append(dev)
            return
        raws[dev] = raw

    threads = [threading.Thread(target=_retrieve, args=(d,)) for d in cam_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=6)
    return raws, errors


class SyncCaptureServer:
    """GPIO 트리거 동기화 캡처 + JPEG 프레임 서빙 엔진.

    캡처 루프를 별도 스레드에서 실행하며, 최신 JPEG 프레임을
    외부(Flask 등)에서 가져갈 수 있도록 제공한다.

    Args:
        cam_ids: 카메라 디바이스 번호 리스트
        width: 프레임 너비
        height: 프레임 높이
        fps: 트리거 FPS
        quality: JPEG 품질
        use_gpu: GPU ISP 사용 여부
    """

    def __init__(self, cam_ids, width, height, fps, quality, use_gpu):
        self.cam_ids = cam_ids
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.use_gpu = use_gpu

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frames = {cam_id: None for cam_id in cam_ids}
        self._frame_seqs = {cam_id: 0 for cam_id in cam_ids}

        self._timing_history = collections.deque(maxlen=100)

        self._running = False
        self._thread = None
        self._caps = {}
        self._encoders = {}

    def start(self):
        """카메라 초기화 + 캡처 스레드 시작.

        1. 모든 카메라의 trigger_mode=1 확인
        2. V4L2 디바이스 열기 (BA10 raw 모드)
        3. 초기 GPIO 펄스로 파이프라인 활성화
        4. 카메라별 GstJpegEncoder 초기화
        5. 워밍업 (5프레임)
        6. 캡처 루프 스레드 시작

        Raises:
            RuntimeError: trigger_mode가 비활성 상태인 카메라가 있을 때
        """
        for dev in self.cam_ids:
            if not check_trigger_mode(dev):
                raise RuntimeError(
                    f"cam{dev} trigger_mode!=1. 먼저: python3 trigger_mode_ctrl.py on")

        for dev in self.cam_ids:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, V4L2_BA10)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._caps[dev] = cap

        self.actual_w = int(self._caps[self.cam_ids[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self._caps[self.cam_ids[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 해상도: {self.actual_w}x{self.actual_h}")

        # 파이프라인 활성화
        gpio_pulse(duration_ms=1)
        init_result = parallel_grab(self._caps, self.cam_ids)
        for dev in self.cam_ids:
            r = init_result.get(dev, {})
            status = f"OK ({r.get('grab_ms', 0):.0f}ms)" if r.get("ok") else "FAIL"
            print(f"[INFO] cam{dev}: {status}")

        if not all(init_result.get(d, {}).get("ok") for d in self.cam_ids):
            print("[WARN] 일부 카메라 초기화 실패. 추가 펄스 시도...")
            for _ in range(3):
                gpio_pulse(duration_ms=1)
                time.sleep(0.05)
            parallel_grab(self._caps, self.cam_ids)

        for dev in self.cam_ids:
            self._encoders[dev] = GstJpegEncoder(
                self.actual_w, self.actual_h, quality=self.quality)

        # 워밍업
        print("[INFO] 워밍업 중...")
        for _ in range(5):
            gpio_pulse()
            grabs = parallel_grab(self._caps, self.cam_ids)
            for dev in self.cam_ids:
                if grabs.get(dev, {}).get("ok"):
                    self._caps[dev].retrieve()
            time.sleep(0.02)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[INFO] 동기화 캡처 시작 ({self.fps}fps, "
              f"{'GPU' if self.use_gpu else 'CPU'})")

    def stop(self):
        """캡처 루프 중지 + 인코더/카메라 리소스 해제."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        for enc in self._encoders.values():
            enc.close()
        for cap in self._caps.values():
            cap.release()
        print("[INFO] 캡처 정지")

    def get_frame(self, cam_id):
        """특정 카메라의 최신 JPEG 프레임을 반환한다."""
        with self._lock:
            return self._frames.get(cam_id)

    def wait_frame(self, cam_id, last_seq, timeout=1.0):
        """새 프레임이 올 때까지 블로킹 대기한다.

        MJPEG 스트리밍에서 사용: 마지막으로 전송한 seq 번호를 넘기면
        새 프레임이 도착할 때까지 대기 후 반환한다.

        Args:
            cam_id: 카메라 디바이스 번호
            last_seq: 마지막으로 수신한 시퀀스 번호
            timeout: 최대 대기 시간 (초)

        Returns:
            (seq, frame) 튜플. frame은 JPEG bytes 또는 None.
        """
        with self._cond:
            if self._frame_seqs.get(cam_id, 0) == last_seq:
                self._cond.wait(timeout=timeout)
            return self._frame_seqs.get(cam_id, 0), self._frames.get(cam_id)

    def get_stats(self):
        """최근 100프레임의 타이밍 통계(이동 평균)를 반환한다.

        Returns:
            dict: fps, pulse_ms, grab_ms, retrieve_ms, isp_ms,
                  encode_ms, total_ms, delta_grab_ms (카메라 간 grab 시간 차이)
        """
        with self._lock:
            entries = list(self._timing_history)
        if not entries:
            return {"fps": 0, "pulse_ms": 0, "grab_ms": 0, "retrieve_ms": 0,
                    "isp_ms": 0, "encode_ms": 0, "total_ms": 0, "delta_grab_ms": 0}

        n = len(entries)
        return {
            "fps": round(sum(e["fps"] for e in entries) / n, 1),
            "pulse_ms": round(sum(e["pulse"] for e in entries) / n, 2),
            "grab_ms": round(sum(e["grab"] for e in entries) / n, 2),
            "retrieve_ms": round(sum(e["retrieve"] for e in entries) / n, 2),
            "isp_ms": round(sum(e["isp"] for e in entries) / n, 2),
            "encode_ms": round(sum(e["encode"] for e in entries) / n, 2),
            "total_ms": round(sum(e["total"] for e in entries) / n, 2),
            "delta_grab_ms": round(sum(e["delta_grab"] for e in entries) / n, 3),
        }

    def _capture_loop(self):
        """백그라운드 캡처 루프 (데몬 스레드에서 실행).

        매 사이클: GPIO 펄스 → 병렬 grab → 병렬 retrieve → ISP → JPEG 인코딩.
        GPU 모드에서는 CUDA 컨텍스트를 이 스레드에 바인딩한다.
        """
        if self.use_gpu and _cuda_ctx is not None:
            _cuda_ctx.push()

        trigger_interval = 1.0 / self.fps

        try:
            while self._running:
                t_cycle = time.monotonic()

                t0 = time.monotonic()
                gpio_pulse()
                t_pulse = time.monotonic() - t0

                t0 = time.monotonic()
                grabs = parallel_grab(self._caps, self.cam_ids)
                t_grab = time.monotonic() - t0

                t0 = time.monotonic()
                raws, errors = parallel_retrieve(self._caps, self.cam_ids, grabs)
                t_retrieve = time.monotonic() - t0

                if errors:
                    continue

                t0 = time.monotonic()
                bgrs = {}
                for dev in self.cam_ids:
                    if self.use_gpu:
                        bgr, _ = demosaic_gpu(raws[dev], self.actual_h, self.actual_w,
                                              buf_id=dev)
                    else:
                        bgr = demosaic(raws[dev], self.actual_h, self.actual_w)
                    bgrs[dev] = bgr
                t_isp = time.monotonic() - t0

                t0 = time.monotonic()
                jpegs = {}
                for dev in self.cam_ids:
                    jpeg = self._encoders[dev].encode(bgrs[dev])
                    if jpeg:
                        jpegs[dev] = jpeg
                t_encode = time.monotonic() - t0

                t_total = time.monotonic() - t_cycle
                grab_times = [grabs[d]["grab_ms"] for d in self.cam_ids]
                delta_grab = max(grab_times) - min(grab_times)

                with self._cond:
                    for dev, jpeg in jpegs.items():
                        self._frames[dev] = jpeg
                        self._frame_seqs[dev] += 1
                    self._timing_history.append({
                        "pulse": t_pulse * 1000,
                        "grab": t_grab * 1000,
                        "retrieve": t_retrieve * 1000,
                        "isp": t_isp * 1000,
                        "encode": t_encode * 1000,
                        "total": t_total * 1000,
                        "delta_grab": delta_grab,
                        "fps": 1.0 / t_total if t_total > 0 else 0,
                    })
                    self._cond.notify_all()

                elapsed = time.monotonic() - t_cycle
                sleep_time = trigger_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if self.use_gpu and _cuda_ctx is not None:
                _cuda_ctx.pop()
