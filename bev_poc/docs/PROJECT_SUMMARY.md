# BEV POC — 프로젝트 요약

어안 듀얼 카메라(AR0234)를 사용한 Bird's Eye View(BEV) 변환 파이프라인 구축 기록.  
Intrinsic 캘리브레이션 → Extrinsic 캘리브레이션 → LUT 기반 BEV 변환의 세 단계로 구성된다.

---

## 목차
1. [시스템 구성](#1-시스템-구성)
2. [Intrinsic 캘리브레이션](#2-intrinsic-캘리브레이션)
3. [Extrinsic 캘리브레이션](#3-extrinsic-캘리브레이션)
4. [BEV 변환](#4-bev-변환)
5. [캘리브레이션 결과 해석](#5-캘리브레이션-결과-해석)
6. [파일 구조](#6-파일-구조)
7. [실행 명령 정리](#7-실행-명령-정리)

---

## 1. 시스템 구성

### 카메라 하드웨어

| 항목 | 사양 |
|------|------|
| 센서 | ON Semiconductor AR0234 |
| 해상도 | 1920 × 1200 px |
| 픽셀 피치 | 3.0 µm |
| 렌즈 초점거리 | 1.56 mm |
| 렌즈 모델 | 어안(Fisheye) — 등거리 투영(Equidistant) |
| 수평 FOV | ~214° (등거리 투영 기준) |
| 수직 FOV | ~134° |

> **등거리 투영(Equidistant projection)이란?**  
> 일반 핀홀 카메라는 `r = f·tan(θ)` 이지만, 어안 렌즈는 `r = f·θ` 를 따른다.  
> θ가 커질수록 핀홀 모델에서는 tan(θ)가 발산하지만, 등거리 모델은 선형으로 증가하므로
> 180° 이상의 시야각을 하나의 평면 센서에 담을 수 있다.

### 카메라 배치

```
        전방(+x)
           ↑
    [Left cam]   [Right cam]
    ~45° 좌향     ~45° 우향
         \         /
          \       /
           원점(O)
```

- 두 카메라가 각각 약 45° 바깥쪽을 향하도록 장착
- 원점(World origin): 카메라 리그 중앙 바닥 지점
- 좌표계: **x=전방(+), y=왼쪽(+), z=위(+)** (오른손 좌표계)

### 캘리브레이션 패턴

- 패턴: 체커보드
- 내부 코너: **가로 10개(y방향) × 세로 7개(x방향)**
- 칸 크기: **36 mm**

---

## 2. Intrinsic 캘리브레이션

**파일:** `intrinsic_calibration.py`  
**저장:** `calib_params/left_intrinsic_result.npz`, `calib_params/right_intrinsic_result.npz`

### 2.1 왜 어안 모델을 써야 하는가

핀홀 모델(`cv2.calibrateCamera`)로 어안 렌즈를 캘리브레이션하면 재투영 오차가 8~17 px에 달해 실용 수준이 되지 않는다. 실제로 첫 시도에서 Mean RMS 8.60 px가 나왔으며, 이후 `cv2.fisheye.calibrate`로 전환하자 **0.305 px**로 급감했다.

### 2.2 초기값 문제와 해결

`cv2.fisheye.calibrate`는 초기 K를 영행렬로 시작하면 `InitExtrinsics`에서 크래시한다.

**물리 기반 초기 fx 계산:**
```python
# r = f·θ (등거리)  →  f = r / θ
# 이미지 엣지(r = W/2)에서 θ = hfov/2
f_init = focal_length_mm / (pixel_pitch_um * 1e-3)
# 예: 1.56mm / 0.003mm = 520 px
```

코드에서는 `--focal_length_mm 1.56` 인자로 전달하며, 없을 경우 180° HFOV 가정 fallback을 사용한다.

```python
# intrinsic_calibration.py:69
if f_init is None:
    f_init = image_size[0] / np.pi  # fallback: 180° HFOV
K = np.array([[f_init, 0., image_size[0]/2.],
              [0., f_init, image_size[1]/2.],
              [0., 0., 1.]], dtype=np.float64)
flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
         cv2.fisheye.CALIB_FIX_SKEW            |
         cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
```

### 2.3 반복적 아웃라이어 제거

캘리브레이션 후 재투영 오차가 `outlier_thresh`(기본 5.0 px)를 초과하는 이미지를 한 장씩 제거하고 재캘리브레이션한다. 한 번에 전부 제거하면 불안정하므로 최악 이미지 하나씩 제거하는 방식을 사용한다.

```python
# intrinsic_calibration.py:140
worst = max(bad_local, key=lambda j: errors[j])
removed_files.append(valid_files[indices[worst]])
indices.pop(worst)
```

### 2.4 결과

| | Left (dev1) | Right (dev0) |
|--|-------------|--------------|
| 사용 이미지 | 55 / 63 장 | 57 / 112 장 |
| fx | 513.25 px | 512.95 px |
| fy | 512.03 px | 511.80 px |
| cx | 964.33 px | 953.64 px |
| cy | 599.11 px | 611.38 px |
| Mean RMS | **0.305 px** | **0.611 px** |
| H FOV | 214.34° | 214.46° |

두 카메라의 **fx가 513 px 근방으로 거의 동일**한 것은 동일 렌즈를 사용하고 있다는 물리적 일관성을 확인해준다.

### 2.5 저장 키

```
camera_matrix   : (3,3) float64  — K 행렬
dist_coeffs     : (4,1) float64  — k1, k2, k3, k4 (등거리 왜곡 계수)
image_size      : (2,)  int      — [width, height]
rms             : float          — Overall RMS (px)
per_image_errors: (N,)  float    — 이미지별 재투영 오차
model           : str            — "fisheye"
```

---

## 3. Extrinsic 캘리브레이션

**파일:** `extrinsic_calibration.py`  
**저장:** `calib_params/left_extrinsic_result.npz`, `calib_params/right_extrinsic_result.npz`

### 3.1 세계 좌표 정의

체커보드를 바닥(z=0 평면)에 놓고 각 교차점의 세계 좌표를 직접 측정해 정의한다.

```
세계 원점: 카메라 리그 중앙 바닥
가장 가까운 교차점(전방 우측): (x=0.30, y=-0.16, z=0.003)  ← z는 보드 두께
보드는 +x(전방)와 +y(왼쪽) 방향으로 확장
```

```python
# extrinsic_calibration.py:55
x = origin_x_m + (board_h - 1 - row) * s   # row=0 → 가장 먼 행(큰 x)
y = origin_y_m + (board_w - 1 - col) * s   # col=0 → 가장 왼쪽(큰 y)
```

### 3.2 어안 렌즈에서 solvePnP 하는 방법

`cv2.solvePnP`는 어안 모델을 직접 지원하지 않는다. 표준 접근법:

1. `cv2.fisheye.undistortPoints(corners, K, D, P=K)` — 코너 좌표만 핀홀 공간으로 변환 (전체 이미지 변환 불필요)
2. `cv2.solvePnP(obj_pts, undist_pts, K, zeros(4))` — 이미 왜곡 제거됐으므로 dist=0

이 방식의 장점: 이미지 전체를 리샘플링하지 않아 보간 오차 없음.

```python
# extrinsic_calibration.py:95
undist = cv2.fisheye.undistortPoints(
    img_corners.reshape(-1, 1, 2), K, D, R=None, P=K
)
```

### 3.3 평면 타깃의 켤레 해(Conjugate Solution) 문제

모든 점이 한 평면에 있으면 solvePnP는 재투영 오차가 동일한 두 가지 해를 가진다:
- **정답**: 카메라가 보드 위에서 아래를 바라봄 (z > 0)
- **오답**: 카메라가 보드 아래에서 위를 바라봄 (z < 0, 거울 해)

`cv2.SOLVEPNP_IPPE`를 사용하면 두 해를 모두 반환받을 수 있고, 물리적 조건으로 정답을 선택한다.

```python
# extrinsic_calibration.py:100
n_sol, rvecs_out, tvecs_out, _ = cv2.solvePnPGeneric(
    obj_pts, undist, K, zeros4, flags=cv2.SOLVEPNP_IPPE
)
# 물리적 조건으로 필터링
cam_z     = float((-R.T @ tv).ravel()[2])   # 카메라 높이 > 0
cam_fwd_x = float(R[2, 0])                  # 카메라 전방 x 성분 > 0
if cam_z > min_height and cam_fwd_x > 0.0:
    # 이 해가 정답
```

### 3.4 코너 순서 자동 보정 (4-way flip)

`cv2.findChessboardCorners`는 카메라 방향에 따라 코너를 다른 순서로 반환할 수 있다. 특히 ±45° 기울어진 카메라에서는 이미지의 좌우가 뒤집혀 보여 단순 역순만으로는 부족하다.

4가지 순서를 모두 시도하고, 물리 조건을 통과하는 해 중 최소 RMS를 선택한다:

```python
# extrinsic_calibration.py:151
c = corners.reshape(board_h, board_w, 1, 2)
candidates = [
    ("normal",  c.reshape(-1, 1, 2)),
    ("flip-y",  c[:, ::-1].reshape(-1, 1, 2)),   # 열 좌우 반전
    ("flip-x",  c[::-1, :].reshape(-1, 1, 2)),   # 행 상하 반전
    ("flip-xy", c[::-1, ::-1].reshape(-1, 1, 2)), # 전체 역순
]
```

**물리 유효 해를 항상 fallback보다 우선**: RMS가 낮더라도 물리 조건(z>0, fwd_x>0)을 위반하는 해는 기각된다.

### 3.5 결과

| | Left (dev1) | Right (dev0) |
|--|-------------|--------------|
| 카메라 위치 x | +0.011 m | +0.016 m |
| 카메라 위치 y | +0.064 m (6.4cm 좌) | −0.050 m (5.0cm 우) |
| 카메라 높이 z | **+0.193 m** ✅ | **+0.195 m** ✅ |
| Pitch (하향각) | −40.8° | −48.2° |
| 카메라 전방 방향 | (+0.654, +0.748, −0.110) | (+0.746, −0.662, −0.075) |
| Extrinsic RMS | 1.034 px | 1.086 px |
| Solution | OK (valid) ✅ | OK (valid) ✅ |

카메라 전방 방향에서:
- Left: x=+0.65, y=+0.75 → 전방 약 49° 좌향 ✅
- Right: x=+0.75, y=−0.66 → 전방 약 41° 우향 ✅

> **Yaw 값 해석 주의**: Right 카메라의 ZYX Euler Yaw가 −174°로 표시되는데, 이는 Roll ≈ ±90°일 때 발생하는 짐벌락(Gimbal Lock) 현상으로 수학적 아티팩트이다. 실제 카메라가 바라보는 방향은 R 행렬의 3행(`R[2,:]`)으로 확인해야 한다.

### 3.6 저장 키

```
rvec          : (3,1) float64  — 회전 벡터 (Rodrigues)
tvec          : (3,1) float64  — 이동 벡터 (world→camera, 카메라 프레임)
R_mat         : (3,3) float64  — 회전 행렬 (world→camera)
T_cam_world   : (4,4) float64  — 4×4 변환 행렬 (world→camera)
cam_pos       : (3,)  float64  — 카메라 위치 (world 프레임)
roll/pitch/yaw: float          — ZYX Euler 각도 (deg)
rms           : float          — 재투영 RMS (px)
camera_matrix : (3,3) float64  — 사용된 K
dist_coeffs   : (4,1) float64  — 사용된 D
```

---

## 4. BEV 변환

**파일:** `bev_transform.py`  
**저장 (LUT):** `calib_params/left_bev_lut.npz`, `calib_params/right_bev_lut.npz`

### 4.1 LUT 기반 IPM의 원리

LUT(Look-Up Table) 방식은 BEV 각 픽셀이 원본 이미지의 어느 픽셀에 대응하는지를 사전 계산해두고, 매 프레임마다 `cv2.remap()`으로 고속 변환하는 방법이다.

```
BEV 픽셀 (row, col)
    ──→  세계 좌표 (X, Y, Z=0)          [BEV 해상도로 역산]
    ──→  원본 이미지 픽셀 (u, v)        [fisheye 투영]
    ──→  LUT에 저장
매 프레임: cv2.remap(img, LUT) → BEV
```

### 4.2 BEV 파라미터

```
x: [0.0, 3.0] m  (전방 거리)
y: [-1.5, +1.5] m  (좌우)
해상도: 0.02 m/px (2 cm/px)
BEV 크기: 150 × 150 px
```

BEV 픽셀 ↔ 세계 좌표 변환:
```python
# bev_transform.py:61
X = x_max - row * res   # row=0 → 전방 3m (이미지 상단)
Y = y_max - col * res   # col=0 → 왼쪽 +1.5m
Z = 0.0                 # 지면 가정
```

### 4.3 어안 투영을 이용한 LUT 계산

```python
# bev_transform.py:100
img_pts, _ = cv2.fisheye.projectPoints(world_pts, rvec, tvec, K, D)
map_x = img_pts[:, 0, 0].reshape(H, W)  # 원본 이미지 x좌표
map_y = img_pts[:, 0, 1].reshape(H, W)  # 원본 이미지 y좌표
```

이미지 범위 밖(시야각 이탈)은 −1로 설정 → `cv2.remap`에서 black으로 처리된다.

### 4.4 품질 맵 (Quality Map)

각 BEV 픽셀에서 해당 카메라의 광축과의 각도 θ를 계산하고, `cos(θ)`를 품질 가중치로 사용한다. 광축에 가까울수록(θ 작을수록) 이미지가 선명하고 왜곡이 적다.

```python
# bev_transform.py:109
P_cam = (R @ world_pts.reshape(-1, 3).T + t[:, None]).T  # 카메라 좌표
cos_theta = P_cam[:, 2] / (np.linalg.norm(P_cam, axis=1) + 1e-9)
quality = np.clip(cos_theta, 0.0, 1.0)
```

### 4.5 품질 가중 블렌딩 (Quality-Weighted Blending)

두 카메라 BEV를 합칠 때 단순 평균을 쓰면, 한 카메라에서 흐린(극단 각도) 픽셀이 다른 카메라의 선명한 픽셀을 오염시킨다. 품질 가중치를 사용하면 이를 방지할 수 있다.

```python
# bev_transform.py:162
wl = left_quality  * l_valid   # 유효하지 않으면 가중치=0
wr = right_quality * r_valid
combined = (left_bev * wl + right_bev * wr) / (wl + wr)
```

이 방식은 **알파 블렌딩(Alpha Blending)의 일종**으로, α값이 수동 지정이 아닌 카메라 물리 특성(광축 각도)에서 자동으로 유도되는 점이 특징이다. 서라운드뷰 카메라 시스템의 표준 기법이다.

---

## 5. 캘리브레이션 결과 해석

### 5.1 Intrinsic 결과 검증 기준

| 항목 | 정상 범위 | 비고 |
|------|----------|------|
| Mean RMS | < 1.0 px (이상: < 0.5 px) | 어안 기준 |
| fx ≈ fy | 비율 < 1% | 정사각형 픽셀 가정 |
| cx ≈ W/2, cy ≈ H/2 | ±50 px 이내 | 주점이 이미지 중앙 근처 |
| k1 | 보통 음수 (barrel distortion) | 어안에서 전형적 |

### 5.2 Extrinsic 결과 검증 기준

| 항목 | 확인 방법 | 정상 |
|------|----------|------|
| 카메라 높이 z | `cam_pos[2]` | > 0 (양수) |
| 전방을 향하는지 | `R_mat[2,0]` | > 0 |
| 실제 장착 높이와 일치 | z값 확인 | ~0.19 m |
| Solution 상태 | 출력 메시지 | "OK (physically valid)" |

### 5.3 BEV 결과 시각적 검증

- **체커보드 격자**가 BEV에서 **직선**으로 보이면 캘리브레이션 정확 ✅
- 격자선이 굽어 있으면 intrinsic 또는 extrinsic 오차 존재 ❌
- combined BEV에서 두 카메라 경계가 자연스러우면 블렌딩 성공 ✅

### 5.4 주의 사항

- **LUT는 캘리브레이션 파일을 반영**한다. `calib_params/*.npz`가 바뀌면 `--save_lut`로 LUT를 재생성해야 한다.  
  `--load_lut` 사용 시 BEV 파라미터(범위, 해상도)가 다르면 자동으로 재계산한다.
- **카메라를 이동/회전**하면 extrinsic 재캘리브레이션이 필요하다.  
  렌즈 교체 시에는 intrinsic도 다시 수행해야 한다.
- **Right 카메라 BEV 품질**이 Left보다 낮은 경우, Right intrinsic의 Mean RMS(0.611 px)가 Left(0.305 px)보다 높은 것이 주원인이다. 더 다양한 위치에서 촬영된 intrinsic 이미지(50장 이상, FOV 전 영역 커버)로 재캘리브레이션하면 개선된다.

---

## 6. 파일 구조

```
bev_poc/
│
├── intrinsic_calibration.py    # Intrinsic 캘리브레이션
├── extrinsic_calibration.py    # Extrinsic 캘리브레이션
├── bev_transform.py            # BEV 변환 (LUT 생성 + 적용)
├── export_cal_data.py          # npz → YAML 내보내기 / 수치 검증
│
├── calib_imgs/
│   ├── intrinsic/
│   │   ├── left/      # Left 카메라 intrinsic 촬영 이미지
│   │   └── right/     # Right 카메라 intrinsic 촬영 이미지
│   └── extrinsic/     # Extrinsic 촬영 이미지 (동시각 2장)
│
├── calib_params/                   # 캘리브레이션 결과 (핵심 산출물)
│   ├── left_intrinsic_result.npz   # Left K, D
│   ├── right_intrinsic_result.npz  # Right K, D
│   ├── left_extrinsic_result.npz   # Left R, t
│   ├── right_extrinsic_result.npz  # Right R, t
│   ├── left_bev_lut.npz            # Left BEV LUT (재사용용)
│   ├── right_bev_lut.npz           # Right BEV LUT
│   └── *.yaml                      # 위 파일들의 사람이 읽을 수 있는 버전
│
└── bev_output/
    ├── left_bev.jpg
    ├── right_bev.jpg
    └── combined_bev.jpg
```

---

## 7. 실행 명령 정리

### Intrinsic 캘리브레이션
```bash
# Left 카메라
python intrinsic_calibration.py \
  --img_dir calib_imgs/intrinsic/left

# Right 카메라
python intrinsic_calibration.py \
  --img_dir calib_imgs/intrinsic/right
```

### Extrinsic 캘리브레이션
```bash
python extrinsic_calibration.py \
  --left_img  calib_imgs/extrinsic/ar0234_left_cam.jpg \
  --right_img calib_imgs/extrinsic/ar0234_right_cam.jpg \
  --left_npz  calib_params/left_intrinsic_result.npz \
  --right_npz calib_params/right_intrinsic_result.npz \
  --origin_y -0.16 \
  --origin_z 0.003
```

### BEV 변환
```bash
# 처음 실행 — LUT 생성 및 저장
python bev_transform.py \
  --left_img  <이미지> \
  --right_img <이미지> \
  --save_lut

# 이후 실행 — 저장된 LUT 재사용 (빠름)
python bev_transform.py \
  --left_img  <이미지> \
  --right_img <이미지> \
  --load_lut

# 고해상도 BEV (0.5 cm/px)
python bev_transform.py \
  --left_img  <이미지> \
  --right_img <이미지> \
  --res 0.005
```

### 캘리브레이션 데이터 검증 및 내보내기
```bash
python export_cal_data.py
# → calib_params/*.yaml 생성 + 콘솔에 수치 검증 출력
```

---

## 핵심 기술 요약

| 기술 | 적용 목적 | 코드 위치 |
|------|----------|----------|
| `cv2.fisheye.calibrate` | 어안 렌즈 intrinsic 추정 | `intrinsic_calibration.py:82` |
| `CALIB_USE_INTRINSIC_GUESS` | 물리 기반 초기값으로 수렴 안정화 | `intrinsic_calibration.py:80` |
| 반복적 아웃라이어 제거 | 나쁜 이미지 자동 제거 | `intrinsic_calibration.py:113` |
| `fisheye.undistortPoints` | 어안 왜곡 제거 (포인트만) | `extrinsic_calibration.py:95` |
| `SOLVEPNP_IPPE` | 평면 타깃 켤레 해 모두 획득 | `extrinsic_calibration.py:100` |
| 물리 조건 필터 (z>0, fwd_x>0) | 켤레 해 중 정답 선택 | `extrinsic_calibration.py:121` |
| 4-way 코너 순서 탐색 | 45° 기울어진 카메라 대응 | `extrinsic_calibration.py:151` |
| `fisheye.projectPoints` + LUT | BEV 변환 사전 계산 | `bev_transform.py:100` |
| cos(θ) 품질 가중 블렌딩 | 선명한 영역 우선 합성 | `bev_transform.py:151` |
