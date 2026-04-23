#!/usr/bin/env python3
"""
FRSYNC 동기화 캡처 + 웹 모니터링

GPIO 트리거로 모든 카메라를 동시에 촬영하고, MJPEG 웹 스트리밍으로 실시간 모니터링한다.

사전 조건:
  python3 trigger_mode_ctrl.py on

사용법:
  python3 sync_monitor_ar0234.py
  python3 sync_monitor_ar0234.py --fps 15 --port 9090 --width 1280 --height 720
  python3 sync_monitor_ar0234.py -d 0 -d 1 --cpu

브라우저 접속:
  http://<jetson-ip>:9090/
"""

import argparse
import atexit
import json
import signal
import sys

from flask import Flask, Response, render_template_string

from ar0234_cam.v4l2_utils import detect_cameras
from ar0234_cam.isp import _USE_GPU
from ar0234_cam.sync import SyncCaptureServer

# ---------------------------------------------------------------------------
# Flask 웹 서버
# ---------------------------------------------------------------------------

app = Flask(__name__)
server: SyncCaptureServer = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FRSYNC Sync Monitor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a1a; color: #e0e0e0; font-family: monospace; }
  header { padding: 12px 20px; background: #222; border-bottom: 1px solid #444; }
  header h1 { font-size: 16px; font-weight: normal; }
  header .sync-stats { font-size: 13px; color: #aaa; margin-top: 4px; }
  header .sync-stats span { margin-right: 16px; }
  header .sync-stats .fps { color: #4fc3f7; }
  header .sync-stats .delta { color: #81c784; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
    gap: 8px;
    padding: 8px;
  }
  .cam-card {
    background: #222;
    border: 1px solid #444;
    border-radius: 4px;
    overflow: hidden;
  }
  .cam-card .info {
    padding: 6px 12px;
    font-size: 13px;
    background: #2a2a2a;
    border-bottom: 1px solid #333;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .cam-card .info .name { font-weight: bold; }
  .cam-card img {
    width: 100%;
    display: block;
    background: #000;
  }
</style>
</head>
<body>
  <header>
    <h1>FRSYNC Sync Monitor &mdash; {{ cam_ids|length }} camera(s)</h1>
    <div class="sync-stats" id="sync-stats">loading...</div>
  </header>
  <div class="grid">
    {% for cam_id in cam_ids %}
    <div class="cam-card">
      <div class="info">
        <span class="name">cam {{ cam_id }} &mdash; /dev/video{{ cam_id }}</span>
      </div>
      <img src="/stream/{{ cam_id }}" alt="cam{{ cam_id }}">
    </div>
    {% endfor %}
  </div>
  <script>
    async function updateStats() {
      try {
        const resp = await fetch('/api/stats');
        const s = await resp.json();
        const el = document.getElementById('sync-stats');
        el.innerHTML =
          '<span class="fps">' + s.fps + ' fps</span>' +
          '<span class="delta">delta ' + s.delta_grab_ms + 'ms</span>' +
          '<span>pulse ' + s.pulse_ms + 'ms</span>' +
          '<span>grab ' + s.grab_ms + 'ms</span>' +
          '<span>retrieve ' + s.retrieve_ms + 'ms</span>' +
          '<span>isp ' + s.isp_ms + 'ms</span>' +
          '<span>enc ' + s.encode_ms + 'ms</span>' +
          '<span>total ' + s.total_ms + 'ms</span>';
      } catch(e) {}
    }
    setInterval(updateStats, 2000);
    updateStats();
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, cam_ids=server.cam_ids)


@app.route('/api/stats')
def api_stats():
    return Response(json.dumps(server.get_stats()), mimetype='application/json')


@app.route('/stream/<int:cam_id>')
def stream(cam_id):
    if cam_id not in server._frames:
        return f"cam{cam_id} not found", 404

    def generate():
        last_seq = -1
        while True:
            seq, frame = server.wait_frame(cam_id, last_seq, timeout=1.0)
            if frame is None or seq == last_seq:
                continue
            last_seq = seq
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot/<int:cam_id>')
def snapshot(cam_id):
    if cam_id not in server._frames:
        return f"cam{cam_id} not found", 404
    frame = server.get_frame(cam_id)
    if frame is None:
        return "no frame available", 503
    return Response(frame, mimetype='image/jpeg')


# ---------------------------------------------------------------------------
# 종료 처리
# ---------------------------------------------------------------------------

def cleanup():
    if server:
        server.stop()


atexit.register(cleanup)


def _signal_handler(_sig, _frame):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global server

    parser = argparse.ArgumentParser(
        description="FRSYNC 동기화 캡처 + 웹 모니터링"
    )
    parser.add_argument("-d", "--device", type=int, action="append", default=None,
                        help="카메라 디바이스 번호 (여러 번 지정 가능, 생략 시 자동 감지)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=15,
                        help="트리거 FPS (default: 15)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090,
                        help="웹 서버 포트 (default: 9090)")
    parser.add_argument("--quality", type=int, default=70,
                        help="JPEG 품질 (default: 70)")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU demosaic 강제")
    args = parser.parse_args()

    print("=" * 60)
    print("  FRSYNC Sync Monitor")
    print("=" * 60)

    if args.device is None:
        args.device = detect_cameras()
    cam_ids = sorted(args.device)
    print(f"[INFO] 카메라: {cam_ids}")
    if not cam_ids:
        print("[ERROR] 카메라 없음")
        sys.exit(1)

    use_gpu = _USE_GPU and not args.cpu
    print(f"[INFO] ISP: {'GPU' if use_gpu else 'CPU'}")
    print(f"[INFO] 트리거: {args.fps}fps")

    try:
        server = SyncCaptureServer(
            cam_ids=cam_ids,
            width=args.width,
            height=args.height,
            fps=args.fps,
            quality=args.quality,
            use_gpu=use_gpu,
        )
        server.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[INFO] 모니터링 서버: http://0.0.0.0:{args.port}/")
    print(f"[INFO] 브라우저에서 http://<jetson-ip>:{args.port}/ 로 접속하세요")
    print("[INFO] Ctrl+C로 종료\n")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
