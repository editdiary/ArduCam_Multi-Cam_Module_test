"""소프트웨어 ISP (Image Signal Processing).

AR0234 센서의 BA10 raw Bayer 데이터를 BGR 이미지로 변환한다.
CPU(OpenCV) 및 GPU(PyCUDA) 두 가지 경로를 제공한다.

AR0234는 HW ISP를 거치지 않는 Bayer raw 출력 센서이므로,
색 보정(demosaic, WB, 감마)을 소프트웨어에서 직접 수행해야 한다.
PyCUDA가 없는 환경에서도 CPU 경로로 정상 동작한다.

모듈 변수:
    _USE_GPU (bool): PyCUDA 사용 가능 여부. True이면 demosaic_gpu() 사용 가능.
    _GAMMA_POST_LUT (np.ndarray): uint16→uint8 감마 보정 테이블 (65536 엔트리).
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# GPU 가속 (PyCUDA) — 선택 의존성
# ---------------------------------------------------------------------------
_USE_GPU = False
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    _USE_GPU = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 감마 LUT: uint16 → uint8 변환용 (모듈 로드 시 1회 계산, 64KB)
# black level 차감 후의 uint16 값(0~62784)을 감마 2.4 보정된 uint8(0~255)로 매핑.
# 62784 = 센서 최대 출력(65535) - black level(2688) - 여유분.
# ---------------------------------------------------------------------------
_GAMMA_POST_LUT = np.zeros(65536, dtype=np.uint8)
for _i in range(65536):
    _val = min(_i / 62784.0, 1.0)
    _GAMMA_POST_LUT[_i] = int(np.clip(_val ** (1.0 / 2.4) * 255 + 0.5, 0, 255))


# ---------------------------------------------------------------------------
# CPU Demosaic
# ---------------------------------------------------------------------------

def demosaic(frame, height, width):
    """BA10 raw Bayer 프레임을 BGR 이미지로 변환한다 (CPU).

    SW ISP 파이프라인:
      1. Black level 차감 (optical black = 2688)
      2. 16-bit Bayer demosaic (GRBG 패턴)
      3. R/B 채널 교환
      4. 감마 2.4 보정 (LUT)
      5. Gray-world 화이트밸런스

    Args:
        frame: V4L2에서 읽은 raw 프레임 (numpy array)
        height: 프레임 높이
        width: 프레임 너비

    Returns:
        8-bit BGR numpy array (h, w, 3)
    """
    frame = frame.flatten().view(np.uint16).reshape(int(height), int(width))

    # 1) Black level 차감
    frame = np.clip(frame.astype(np.int32) - 2688, 0, 65535).astype(np.uint16)

    # 2) Bayer demosaic (GRBG 패턴)
    bgr16 = cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2BGR)

    # 3) R/B 채널 교환
    bgr16 = bgr16[:, :, ::-1].copy()

    # 4) 감마 2.4 보정 (LUT)
    bgr = _GAMMA_POST_LUT[bgr16]

    # 5) Gray-world 화이트밸런스
    avgs = cv2.mean(bgr)
    for ch in range(3):
        if avgs[ch] > 0:
            lut = np.clip(np.arange(256) * (128.0 / avgs[ch]), 0, 255).astype(np.uint8)
            bgr[:, :, ch] = cv2.LUT(bgr[:, :, ch], lut)

    return bgr


# ---------------------------------------------------------------------------
# GPU Demosaic (PyCUDA)
# ---------------------------------------------------------------------------

if _USE_GPU:
    # CUDA 커널 3개를 컴파일한다 (모듈 로드 시 1회):
    #   demosaic_kernel  — BL차감 + GRBG Bayer 보간 + 감마 LUT → uint8 BGR
    #   channel_sum_kernel — BGR 채널별 합산 (parallel reduction, WB 계산용)
    #   wb_kernel — 채널별 WB 스케일 적용 (gray-world: 각 채널 평균 → 128)
    _CUDA_MODULE = SourceModule("""
    __constant__ unsigned char gamma_lut[65536];

    __global__ void demosaic_kernel(
        const unsigned short* __restrict__ raw,
        unsigned char* __restrict__ out,
        int width, int height, int black_level)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        #define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
        #define RAW(r, c) raw[CLAMP(r, 0, height-1) * width + CLAMP(c, 0, width-1)]
        #define BL(r, c) (RAW(r, c) > black_level ? RAW(r, c) - black_level : 0)

        int r_val, g_val, b_val;
        int phase_y = y & 1;
        int phase_x = x & 1;

        if (phase_y == 0 && phase_x == 0) {
            g_val = BL(y, x);
            r_val = (BL(y, x-1) + BL(y, x+1)) / 2;
            b_val = (BL(y-1, x) + BL(y+1, x)) / 2;
        } else if (phase_y == 0 && phase_x == 1) {
            r_val = BL(y, x);
            g_val = (BL(y, x-1) + BL(y, x+1) + BL(y-1, x) + BL(y+1, x)) / 4;
            b_val = (BL(y-1, x-1) + BL(y-1, x+1) + BL(y+1, x-1) + BL(y+1, x+1)) / 4;
        } else if (phase_y == 1 && phase_x == 0) {
            b_val = BL(y, x);
            g_val = (BL(y, x-1) + BL(y, x+1) + BL(y-1, x) + BL(y+1, x)) / 4;
            r_val = (BL(y-1, x-1) + BL(y-1, x+1) + BL(y+1, x-1) + BL(y+1, x+1)) / 4;
        } else {
            g_val = BL(y, x);
            b_val = (BL(y, x-1) + BL(y, x+1)) / 2;
            r_val = (BL(y-1, x) + BL(y+1, x)) / 2;
        }

        int idx = (y * width + x) * 3;
        out[idx + 0] = gamma_lut[CLAMP(b_val, 0, 65535)];
        out[idx + 1] = gamma_lut[CLAMP(g_val, 0, 65535)];
        out[idx + 2] = gamma_lut[CLAMP(r_val, 0, 65535)];

        #undef CLAMP
        #undef RAW
        #undef BL
    }

    __global__ void channel_sum_kernel(
        const unsigned char* __restrict__ img,
        unsigned long long* __restrict__ partial_sums,
        int total_pixels)
    {
        __shared__ unsigned long long sdata[3][256];

        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[0][tid] = 0;
        sdata[1][tid] = 0;
        sdata[2][tid] = 0;

        for (int idx = i; idx < total_pixels; idx += blockDim.x * gridDim.x) {
            int base = idx * 3;
            sdata[0][tid] += img[base + 0];
            sdata[1][tid] += img[base + 1];
            sdata[2][tid] += img[base + 2];
        }
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[0][tid] += sdata[0][tid + s];
                sdata[1][tid] += sdata[1][tid + s];
                sdata[2][tid] += sdata[2][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial_sums[blockIdx.x * 3 + 0] = sdata[0][0];
            partial_sums[blockIdx.x * 3 + 1] = sdata[1][0];
            partial_sums[blockIdx.x * 3 + 2] = sdata[2][0];
        }
    }

    __global__ void wb_kernel(
        unsigned char* __restrict__ img,
        int total_pixels,
        float scale_b, float scale_g, float scale_r)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= total_pixels) return;

        int idx = i * 3;
        float b = img[idx + 0] * scale_b;
        float g = img[idx + 1] * scale_g;
        float r = img[idx + 2] * scale_r;

        img[idx + 0] = (unsigned char)(b > 255.0f ? 255 : (b < 0.0f ? 0 : b));
        img[idx + 1] = (unsigned char)(g > 255.0f ? 255 : (g < 0.0f ? 0 : g));
        img[idx + 2] = (unsigned char)(r > 255.0f ? 255 : (r < 0.0f ? 0 : r));
    }
    """)

    _demosaic_kernel = _CUDA_MODULE.get_function("demosaic_kernel")
    _channel_sum_kernel = _CUDA_MODULE.get_function("channel_sum_kernel")
    _wb_kernel = _CUDA_MODULE.get_function("wb_kernel")

    # 감마 LUT → constant memory
    _gamma_lut_gpu = _CUDA_MODULE.get_global("gamma_lut")[0]
    cuda.memcpy_htod(_gamma_lut_gpu, _GAMMA_POST_LUT)

    _gpu_bufs = {}


def _get_gpu_bufs(h, w, buf_id=0):
    """해상도+ID별 GPU/CPU 버퍼를 캐싱하여 재사용한다.

    Args:
        h: 프레임 높이
        w: 프레임 너비
        buf_id: 버퍼 식별자 (다중 카메라 시 카메라별 독립 버퍼)
    """
    key = (h, w, buf_id)
    if key not in _gpu_bufs:
        num_reduce_blocks = 128
        _gpu_bufs[key] = {
            'raw': cuda.mem_alloc(h * w * 2),
            'out': cuda.mem_alloc(h * w * 3),
            'partial_sums': cuda.mem_alloc(num_reduce_blocks * 3 * 8),
            'partial_sums_host': np.zeros(num_reduce_blocks * 3, dtype=np.uint64),
            'num_reduce_blocks': num_reduce_blocks,
            'block': (16, 16, 1),
            'grid': ((w + 15) // 16, (h + 15) // 16),
            'raw_host': np.empty((h, w), dtype=np.uint16),
            'out_host': np.empty((h, w, 3), dtype=np.uint8),
        }
    return _gpu_bufs[key]


def demosaic_gpu(frame, height, width, buf_id=0):
    """GPU 가속 demosaic (PyCUDA).

    전체 ISP 파이프라인을 GPU에서 처리한다:
      1. BL차감 + Bayer보간 + 감마 LUT → uint8 BGR
      2. 채널별 합산 (parallel reduction)
      3. WB 스케일 적용
      4. 최종 결과만 CPU로 복사

    Args:
        frame: V4L2에서 읽은 raw 프레임
        height: 프레임 높이
        width: 프레임 너비
        buf_id: 버퍼 식별자 (다중 카메라용)

    Returns:
        (bgr, avgs) 튜플:
          bgr: 8-bit BGR numpy array (h, w, 3)
          avgs: 채널 평균 [B, G, R]
    """
    h, w = int(height), int(width)
    bufs = _get_gpu_bufs(h, w, buf_id)
    total_pixels = h * w
    n_blocks = bufs['num_reduce_blocks']

    np.copyto(bufs['raw_host'], frame.flatten().view(np.uint16).reshape(h, w))
    cuda.memcpy_htod(bufs['raw'], bufs['raw_host'])

    _demosaic_kernel(bufs['raw'], bufs['out'],
                     np.int32(w), np.int32(h), np.int32(2688),
                     block=bufs['block'], grid=bufs['grid'])

    _channel_sum_kernel(bufs['out'], bufs['partial_sums'], np.int32(total_pixels),
                        block=(256, 1, 1), grid=(n_blocks, 1))

    cuda.memcpy_dtoh(bufs['partial_sums_host'], bufs['partial_sums'])
    sums = bufs['partial_sums_host'].reshape(n_blocks, 3).sum(axis=0).astype(np.float64)
    avgs = sums / total_pixels

    if avgs[0] > 0 and avgs[1] > 0 and avgs[2] > 0:
        _wb_kernel(bufs['out'], np.int32(total_pixels),
                   np.float32(128.0 / avgs[0]),
                   np.float32(128.0 / avgs[1]),
                   np.float32(128.0 / avgs[2]),
                   block=(256, 1, 1),
                   grid=((total_pixels + 255) // 256, 1))

    cuda.memcpy_dtoh(bufs['out_host'], bufs['out'])

    return bufs['out_host'], avgs
