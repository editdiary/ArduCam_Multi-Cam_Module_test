"""GStreamer nvjpegenc HW 가속 JPEG 인코더.

appsrc로 BGR 프레임을 입력하고, NVIDIA nvjpegenc로
하드웨어 가속 JPEG 인코딩 후 결과를 반환한다.

이 모듈은 GStreamer(gi.repository.Gst)가 필요하다.
GStreamer가 설치되어 있지 않으면 import 시 ImportError가 발생한다.
"""

import sys

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp


class GstJpegEncoder:
    """GStreamer nvjpegenc 기반 HW JPEG 인코더.

    파이프라인 구조:
      appsrc (BGR 입력) → videoconvert (BGR→I420)
      → nvvidconv (CPU→NVMM) → nvjpegenc (HW JPEG) → appsink (JPEG 출력)

    Args:
        width: 프레임 너비
        height: 프레임 높이
        quality: JPEG 품질 (1-100, 기본 85)
    """

    def __init__(self, width, height, quality=85):
        Gst.init(None)

        pipeline_str = (
            f"appsrc name=src is-live=true format=3 "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate=30/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=I420 ! "
            f"nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=I420 ! "
            f"nvjpegenc quality={quality} ! "
            f"appsink name=sink emit-signals=false sync=false"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("sink")

        self.pipeline.set_state(Gst.State.PLAYING)

        ret, state, _ = self.pipeline.get_state(5 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                "GStreamer 파이프라인 시작 실패. "
                "nvjpegenc/nvvidconv 사용 불가능할 수 있습니다."
            )

    def encode(self, bgr_frame):
        """BGR 프레임을 JPEG 바이너리로 인코딩한다.

        Args:
            bgr_frame: 8-bit BGR numpy array (h, w, 3)

        Returns:
            JPEG 바이너리 데이터 (bytes) 또는 실패 시 None
        """
        data = bgr_frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)

        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            print(f"[ERROR] appsrc push-buffer 실패: {ret}", file=sys.stderr)
            return None

        sample = self.appsink.try_pull_sample(3 * Gst.SECOND)
        if sample is None:
            bus = self.pipeline.get_bus()
            msg = bus.pop_filtered(Gst.MessageType.ERROR)
            if msg:
                err, debug = msg.parse_error()
                print(f"[ERROR] GStreamer 에러: {err.message}", file=sys.stderr)
                print(f"[ERROR] 디버그: {debug}", file=sys.stderr)
            else:
                print("[ERROR] appsink pull-sample 타임아웃 (3초)", file=sys.stderr)
            return None

        buf_out = sample.get_buffer()
        result, map_info = buf_out.map(Gst.MapFlags.READ)
        if not result:
            print("[ERROR] buffer map 실패", file=sys.stderr)
            return None

        jpeg_data = bytes(map_info.data)
        buf_out.unmap(map_info)

        return jpeg_data

    def close(self):
        """GStreamer 파이프라인을 종료하고 리소스를 해제한다."""
        self.appsrc.emit("end-of-stream")
        self.pipeline.set_state(Gst.State.NULL)
