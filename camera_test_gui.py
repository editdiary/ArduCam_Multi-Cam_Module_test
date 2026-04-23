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
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, jsonify, render_template_string, request

from ar0234_cam.v4l2_utils import (
    V4L2_BA10, check_trigger_mode, v4l2_get, v4l2_set,
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
        self.width = 1920
        self.height = 1080
        self._open_caps = {}        # {dev: cv2.VideoCapture} 열린 카메라 세션
        self._cap_info = {}         # {dev: {"w": int, "h": int, "trigger": bool, "type": str}}


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


def _release_caps():
    """열려 있는 모든 카메라 세션을 해제한다."""
    for dev, cap in state._open_caps.items():
        try:
            cap.release()
        except Exception:
            pass
    state._open_caps.clear()
    state._cap_info.clear()


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

    cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
    if cam_type == "ar0234":
        cap.set(cv2.CAP_PROP_FOURCC, V4L2_BA10)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, state.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, state.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 워밍업 (파이프라인 활성화)
    for _ in range(3):
        if is_trigger:
            gpio_pulse()
        cap.grab()
        cap.retrieve()

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
    """
    cap, actual_w, actual_h, is_trigger = _ensure_cap_open(dev_num, cam_type)

    if is_trigger:
        gpio_pulse()
    ret = cap.grab()
    if not ret:
        return None
    ret, frame = cap.retrieve()
    if not ret:
        return None

    if cam_type == "ar0234":
        if state.use_gpu and _cuda_ctx is not None:
            _cuda_ctx.push()
            try:
                bgr, _ = demosaic_gpu(frame, actual_h, actual_w)
            finally:
                _cuda_ctx.pop()
        else:
            bgr = demosaic(frame, actual_h, actual_w)
    else:
        bgr = frame

    return bgr


def capture_multi(dev_nums, cam_infos):
    """다중 카메라 동시 촬영.

    AR0234 카메라는 GPIO 트리거로 동기 촬영하고,
    USB 카메라는 일반 grab/retrieve로 촬영한다.
    카메라 세션을 유지하여 trigger_mode 리셋을 방지한다.
    """
    ar0234_devs = [d for d in dev_nums if cam_infos[d]["type"] == "ar0234"]
    usb_devs = [d for d in dev_nums if cam_infos[d]["type"] == "usb"]

    results = {}
    error = None

    # AR0234 다중 촬영 (GPIO 트리거)
    if ar0234_devs:
        caps = {}
        for dev in ar0234_devs:
            cap, actual_w, actual_h, is_trigger = _ensure_cap_open(dev, "ar0234")
            if not is_trigger:
                return {}, (f"cam{dev} trigger_mode 비활성. "
                           "먼저: python3 trigger_mode_ctrl.py on")
            caps[dev] = cap

        info = state._cap_info[ar0234_devs[0]]
        actual_w, actual_h = info["w"], info["h"]

        # 본 촬영
        gpio_pulse()
        grabs = parallel_grab(caps, ar0234_devs)
        raws, errors = parallel_retrieve(caps, ar0234_devs, grabs)
        # release 하지 않음 — 세션 유지

        if errors:
            error = f"촬영 실패 카메라: {errors}"

        if state.use_gpu and _cuda_ctx is not None:
            _cuda_ctx.push()
        try:
            for dev in ar0234_devs:
                if dev in raws:
                    if state.use_gpu:
                        bgr, _ = demosaic_gpu(raws[dev], actual_h, actual_w,
                                              buf_id=dev)
                    else:
                        bgr = demosaic(raws[dev], actual_h, actual_w)
                    results[dev] = bgr
        finally:
            if state.use_gpu and _cuda_ctx is not None:
                _cuda_ctx.pop()

    # USB 카메라 촬영
    for dev in usb_devs:
        cap, _, _, _ = _ensure_cap_open(dev, "usb")
        ret = cap.grab()
        if ret:
            ret, bgr = cap.retrieve()
            if ret:
                results[dev] = bgr
                continue
        error = (error or "") + f" cam{dev} USB 촬영 실패."

    return results, error


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

    # AR0234 카메라가 있으면 trigger_mode 복원
    ar0234_exists = any(c["type"] == "ar0234" for c in cameras)
    if ar0234_exists:
        subprocess.run(
            [sys.executable, "trigger_mode_ctrl.py", "on"],
            capture_output=True, timeout=10
        )

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

    if mode == "single":
        dev = devices[0]
        bgr = capture_single(dev, cam_map[dev]["type"])
        if bgr is None:
            return jsonify({"error": f"cam{dev} 촬영 실패"}), 500

        jpeg_b64 = _bgr_to_jpeg_b64(bgr)
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
            }]
        })

    else:  # multi
        cam_infos = {d: cam_map[d] for d in devices}
        results, error = capture_multi(devices, cam_infos)

        if not results:
            return jsonify({"error": error or "촬영 실패"}), 500

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        images = []

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

        resp = {"images": images}
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
    if "exposure" in data:
        v4l2_set(dev, "exposure", data["exposure"])
        changed.append("exposure")
    if "analogue_gain" in data:
        v4l2_set(dev, "analogue_gain", data["analogue_gain"])
        changed.append("analogue_gain")

    return jsonify({"changed": changed})


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
  body { background: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }

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
  .btn-success { background: #2e7d32; color: #fff; }
  .btn-success:hover { background: #388e3c; }
  .btn-success:disabled { background: #555; cursor: not-allowed; }
  .btn-outline {
    background: transparent; color: #aaa; border: 1px solid #555;
  }
  .btn-outline:hover { color: #fff; border-color: #888; }

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

  /* 카메라 체크박스 (다중 모드) */
  .cam-checkboxes {
    display: none; gap: 8px; flex-wrap: wrap;
  }
  .cam-checkboxes.active { display: flex; }
  .cam-checkboxes label {
    padding: 3px 10px; border: 1px solid #555; border-radius: 3px;
    cursor: pointer; font-size: 12px;
  }
  .cam-checkboxes input:checked + span { color: #4fc3f7; }
  .cam-checkboxes label:has(input:checked) {
    border-color: #4fc3f7; background: rgba(79, 195, 247, 0.1);
  }
  .cam-checkboxes input[type="checkbox"] { display: none; }

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

  /* 파라미터 패널 */
  .params-panel {
    padding: 12px 20px;
    background: #222;
    border-top: 1px solid #333;
    display: none;
  }
  .params-panel.active { display: block; }
  .params-panel h3 { font-size: 13px; margin-bottom: 8px; color: #aaa; }
  .param-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 6px;
  }
  .param-row label { font-size: 12px; width: 120px; color: #ccc; }
  .param-row input[type="range"] {
    flex: 1; max-width: 400px; accent-color: #4fc3f7;
  }
  .param-row .param-value {
    font-size: 12px; color: #4fc3f7; min-width: 60px;
    font-family: monospace;
  }

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
  <button class="btn btn-outline" onclick="refreshCameras()">Refresh</button>

  <div class="mode-toggle">
    <label><input type="radio" name="mode" value="single" checked><span>Single</span></label>
    <label><input type="radio" name="mode" value="multi"><span>Multi</span></label>
  </div>

  <div class="trigger-status" id="trigger-status">
    Trigger: <span class="badge" id="trigger-badge">--</span>
    <span class="warn-icon" id="trigger-warn" style="display:none;">&#9888;
      <span class="tooltip">Multi mode requires trigger_mode=ON.<br>Run: python3 trigger_mode_ctrl.py on</span>
    </span>
  </div>

  <div class="cam-checkboxes" id="cam-checkboxes"></div>

  <button class="btn btn-primary" id="btn-capture" onclick="doCapture()">Capture</button>
  <button class="btn btn-success" id="btn-save" onclick="doSave()" disabled>Save</button>
</div>

<div class="preview-area single" id="preview-area">
  <div class="placeholder">Select a camera and press Capture</div>
</div>

<div class="params-panel" id="params-panel">
  <h3>Camera Parameters (AR0234)</h3>
  <div class="param-row">
    <label>Exposure (2-65535)</label>
    <input type="range" id="slider-exposure" min="2" max="65535" value="5000">
    <span class="param-value" id="val-exposure">5000</span>
  </div>
  <div class="param-row">
    <label>Gain (100-1200)</label>
    <input type="range" id="slider-gain" min="100" max="1200" value="400">
    <span class="param-value" id="val-gain">400</span>
  </div>
</div>

<div class="status-bar" id="status-bar">Ready</div>

<script>
let cameras = [];
let hasCaptured = false;

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
  r.addEventListener('change', () => {
    const multi = getMode() === 'multi';
    document.getElementById('cam-select').style.display = multi ? 'none' : '';
    const cb = document.getElementById('cam-checkboxes');
    cb.classList.toggle('active', multi);
    updateTriggerStatus();
    updateParamsPanel();
  });
});

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

  // 체크박스 (다중 모드)
  const cb = document.getElementById('cam-checkboxes');
  cb.innerHTML = '';
  cams.forEach(c => {
    const lbl = document.createElement('label');
    lbl.innerHTML = '<input type="checkbox" value="' + c.dev + '" data-type="' + c.type + '"><span>' + c.label + '</span>';
    cb.appendChild(lbl);
  });

  updateParamsPanel();
  updateTriggerStatus();
}

async function refreshCameras() {
  setStatus('Scanning cameras...', 'info');
  try {
    const resp = await fetch('/api/cameras/refresh', { method: 'POST' });
    const cams = await resp.json();
    await loadCameras(cams);
    setStatus(cams.length + ' camera(s) detected', 'ok');
  } catch (e) {
    setStatus('Camera scan failed: ' + e.message, 'error');
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

// 드롭다운 변경 시 파라미터 패널 업데이트
document.getElementById('cam-select').addEventListener('change', updateParamsPanel);

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

async function updateParamsPanel() {
  const panel = document.getElementById('params-panel');
  const camType = getSelectedCamType();
  const dev = getSelectedDev();

  if (camType === 'ar0234' && dev !== null && !isNaN(dev)) {
    panel.classList.add('active');
    // 현재 값 로드
    try {
      const resp = await fetch('/api/params?dev=' + dev);
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
  } else {
    panel.classList.remove('active');
  }
}

// 슬라이더 디바운스
let paramTimer = null;
function onSliderChange(name, slider, display) {
  display.textContent = slider.value;
  clearTimeout(paramTimer);
  paramTimer = setTimeout(() => {
    const dev = getSelectedDev();
    if (dev === null || isNaN(dev)) return;
    const body = { dev: dev };
    body[name] = parseInt(slider.value);
    fetch('/api/params', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(() => {
      setStatus(name + ' = ' + slider.value + ' (cam' + dev + ')', 'ok');
    });
  }, 200);
}

document.getElementById('slider-exposure').addEventListener('input', function() {
  onSliderChange('exposure', this, document.getElementById('val-exposure'));
});
document.getElementById('slider-gain').addEventListener('input', function() {
  onSliderChange('analogue_gain', this, document.getElementById('val-gain'));
});

// --- 촬영 ---
async function doCapture() {
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

  setStatus('Capturing...', 'info');
  document.getElementById('btn-capture').disabled = true;

  try {
    const resp = await fetch('/api/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ devices: devices, mode: mode })
    });
    const data = await resp.json();

    if (data.error) {
      setStatus(data.error, 'error');
      document.getElementById('btn-capture').disabled = false;
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

    let msg = data.images.length + ' image(s) captured';
    if (data.warning) msg += ' (warning: ' + data.warning + ')';
    setStatus(msg, 'ok');

  } catch (e) {
    setStatus('Capture failed: ' + e.message, 'error');
  }

  document.getElementById('btn-capture').disabled = false;
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

    state.width = args.width
    state.height = args.height
    if args.cpu:
        state.use_gpu = False

    print("[INFO] 카메라 스캔 중...")
    state.cameras = detect_all_cameras()
    print(f"[INFO] {len(state.cameras)}개 카메라 감지: "
          f"{[c['label'] for c in state.cameras]}")

    # AR0234 카메라가 있으면 trigger_mode 자동 활성화
    if any(c["type"] == "ar0234" for c in state.cameras):
        print("[INFO] trigger_mode 활성화 중...")
        subprocess.run(
            [sys.executable, "trigger_mode_ctrl.py", "on"],
            capture_output=True, timeout=10
        )
        print("[INFO] trigger_mode=ON 완료")

    print(f"[INFO] ISP: {'GPU (PyCUDA)' if state.use_gpu else 'CPU'}")
    print(f"[INFO] Camera Test GUI: http://0.0.0.0:{args.port}/")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
