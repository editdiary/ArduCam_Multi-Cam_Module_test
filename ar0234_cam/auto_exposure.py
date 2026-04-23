"""V4L2 기반 자동 노출(AE) 제어기.

프레임 밝기를 목표값에 수렴시키기 위해 exposure → gain 순으로 조절한다.
exposure를 먼저 조절하고(노이즈 적음), 한계에 도달하면 gain을 올린다.

AR0234 센서의 HW ISP가 없으므로, 소프트웨어에서 직접 AE를 수행한다.
매 프레임의 채널 평균을 demosaic 결과에서 가져와 update()에 전달하면 된다.
"""

import fcntl

import v4l2


class AutoExposure:
    """V4L2 기반 자동 노출 제어기.

    제어 전략:
      1. exposure를 우선 조절 (노이즈 영향 없음)
      2. exposure가 EXPOSURE_MAX에 도달하면 gain을 올림
      3. 밝기 과다 시 gain을 먼저 낮추고, 그 다음 exposure를 줄임

    사용 후 반드시 close()를 호출하여 디바이스 파일 핸들을 해제해야 한다.

    Args:
        device_num: V4L2 디바이스 번호 (/dev/videoN의 N)
        target_luma: 목표 밝기 (0-255, 기본 120 = 중간 톤)
        speed: 수렴 속도 (0.0~1.0, 높을수록 빠르지만 진동 위험)

    Attributes:
        exposure (int): 현재 노출 값
        gain (int): 현재 아날로그 게인 값
    """

    EXPOSURE_MIN = 2       #: 센서 최소 노출 (v4l2-ctl 기준)
    EXPOSURE_MAX = 65535    #: 센서 최대 노출
    GAIN_MIN = 100          #: 센서 최소 아날로그 게인
    GAIN_MAX = 1200         #: 센서 최대 아날로그 게인

    def __init__(self, device_num, target_luma=120, speed=0.3):
        self.vd = open(f'/dev/video{device_num}', 'w')
        self.target = target_luma
        self.speed = speed
        self.exposure = self._get_ctrl(v4l2.V4L2_CID_EXPOSURE)
        self.gain = self._get_ctrl(0x009e0903)  # analogue_gain

    def _get_ctrl(self, ctrl_id):
        """V4L2 컨트롤 값을 읽는다."""
        ctrl = v4l2.v4l2_control()
        ctrl.id = ctrl_id
        try:
            fcntl.ioctl(self.vd, v4l2.VIDIOC_G_CTRL, ctrl)
            return ctrl.value
        except OSError:
            return None

    def _set_ctrl(self, ctrl_id, value):
        """V4L2 컨트롤 값을 설정한다."""
        ctrl = v4l2.v4l2_control()
        ctrl.id = ctrl_id
        ctrl.value = int(value)
        try:
            fcntl.ioctl(self.vd, v4l2.VIDIOC_S_CTRL, ctrl)
        except OSError:
            pass

    def update(self, avg_b, avg_g, avg_r):
        """프레임의 채널 평균값으로 노출을 조절한다.

        demosaic 후 WB 적용 전의 채널 평균을 입력받아
        ITU-R BT.601 가중 평균으로 밝기를 산출하고,
        목표값과의 오차에 따라 exposure/gain을 비례 조절한다.
        오차가 허용 범위(+-5) 이내이면 조절하지 않는다.

        Args:
            avg_b: Blue 채널 평균 (0-255)
            avg_g: Green 채널 평균 (0-255)
            avg_r: Red 채널 평균 (0-255)
        """
        # ITU-R BT.601 가중 평균
        luma = 0.114 * avg_b + 0.587 * avg_g + 0.299 * avg_r
        error = self.target - luma

        if abs(error) < 5:
            return

        if luma > 0:
            ratio = self.target / luma
        else:
            ratio = 2.0

        # 수렴 속도 적용
        ratio = 1.0 + (ratio - 1.0) * self.speed

        # exposure 먼저 조절
        new_exposure = int(self.exposure * ratio)
        new_exposure = max(self.EXPOSURE_MIN, min(self.EXPOSURE_MAX, new_exposure))

        if new_exposure != self.exposure:
            self.exposure = new_exposure
            self._set_ctrl(v4l2.V4L2_CID_EXPOSURE, self.exposure)
            return

        # exposure 한계 도달 시 gain 조절
        if self.gain is not None:
            new_gain = int(self.gain * ratio)
            new_gain = max(self.GAIN_MIN, min(self.GAIN_MAX, new_gain))
            if new_gain != self.gain:
                self.gain = new_gain
                self._set_ctrl(0x009e0903, self.gain)

    def close(self):
        """디바이스 파일 핸들을 해제한다."""
        self.vd.close()
