"""AR0234 Multi-Camera Vision System.

ArduCam AR0234 Bayer 센서 기반 다중 카메라 시스템의 핵심 모듈을 제공한다.

주요 컴포넌트:
  - v4l2_utils: V4L2 디바이스 감지 및 제어
  - isp: SW ISP (CPU/GPU demosaic)
  - auto_exposure: 자동 노출 제어
  - encoder: GStreamer HW JPEG 인코딩
  - gpio: GPIO 트리거 제어
  - sync: 동기 캡처 엔진

사용 예:
  from ar0234_cam import detect_cameras, demosaic, AutoExposure
  from ar0234_cam.encoder import GstJpegEncoder
  from ar0234_cam.sync import SyncCaptureServer
"""

from ar0234_cam.v4l2_utils import fourcc, V4L2_BA10, detect_cameras
from ar0234_cam.isp import demosaic, _USE_GPU, _GAMMA_POST_LUT

if _USE_GPU:
    from ar0234_cam.isp import demosaic_gpu

from ar0234_cam.auto_exposure import AutoExposure

try:
    from ar0234_cam.encoder import GstJpegEncoder
except ImportError:
    pass
