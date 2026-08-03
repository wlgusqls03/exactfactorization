# Single-GPU Multi-Component Exact Factorization

모든 ``--outdir`` 결과는 서버 현지 실행일에 따라 자동으로
``results/YYYYMMDD/<지정한 이름>`` 아래에 저장된다. 예를 들어
``--outdir results/gpu_mixed_1fs``를 2026년 7월 30일에 실행하면 실제 경로는
``results/20260730/gpu_mixed_1fs``다.

기존 CPU 코드를 기준으로 남겨두고, 큰 3차원 direct-EF 전파만 CuPy GPU에서
수행한다. 결과 NPZ의 key와 shape은 CPU 버전과 같아서 기존
`visualize`, `dynamics_analysis`, `visualize_3d`를 그대로 사용할 수 있다.

## 서버 환경

제공된 서버는 NVIDIA driver 460.84, CUDA 11.2, 약 11 GiB GPU 두 장이다.
여기서는 GPU 0 한 장만 사용한다. 약 11 GiB라는 점에서 장치는 RTX 2080 Ti일
가능성이 높지만 실제 이름은 다음 명령으로 확인한다.

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## 설치

Repository를 clone하고 기존 Python 환경과 분리된 환경을 권장한다.

```bash
git clone https://github.com/wlgusqls03/exactfactorization.git
cd exactfactorization

conda create -n mcef-gpu python=3.9 -y
conda activate mcef-gpu
pip install -r multi_component_exact_factorization_gpu/requirements-cuda112.txt
```

CUDA 11.2--11.8용 공식 CuPy wheel 이름은 `cupy-cuda11x`다. `cupy`와
`cupy-cuda11x`를 동시에 설치하면 안 된다. 서버 module 환경에서 CUDA toolkit이
별도로 관리된다면 먼저 `module avail cuda`와 `module load cuda/11.2`를 확인한다.

## 1. GPU와 DST-I 검증

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.validate_gpu
```

우리 전자 hard wall은 DST-I를 사용한다. CuPy 내장 DST가 type I을 지원하지
않으므로 `gpu_core.py`는 odd-extension FFT로 orthonormal DST-I를 구현한다.
위 명령은 이를 SciPy DST-I와 `complex128`, `complex64`에서 비교하고 비주기
5점 유한차분도 CPU와 비교한다. `GPU 기본 검증: PASS`가 나와야 전파한다.

`CUDA_VISIBLE_DEVICES=0`을 쓰면 물리 GPU 0만 프로그램에 보이며, 프로그램
내부에서는 그 장치가 다시 index 0이 된다. 따라서 보통 `--device 0`을 그대로
둔다.

## 2. 짧은 세 정밀도 시험

먼저 같은 조건으로 아주 짧게 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --precision double --t-final-fs 0.01 --dt-au 0.005 \
  --save-every 20 --outdir results/gpu_double_001fs

CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --precision mixed --t-final-fs 0.01 --dt-au 0.005 \
  --save-every 20 --outdir results/gpu_mixed_001fs

CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --precision single --t-final-fs 0.01 --dt-au 0.005 \
  --save-every 20 --outdir results/gpu_single_001fs
```

정밀도의 의미는 다음과 같다.

| option | 큰 배열 | norm/inner-product reduction |
|---|---|---|
| `double` | complex128 | FP64 |
| `single` | complex64 | FP32 |
| `mixed` | complex64 | FP64로 합한 뒤 propagation field는 FP32 |

세 precision 모두 CPU와 같은 product-preserving nested tangent correction을
RK4의 네 stage마다 적용한다. `mixed`에서는 norm과 expectation을
FP64/complex128로 합산한 뒤 correction field에 곱하는 `gamma`만 float32로
되돌린다. 결과
archive의 `max_abs_support_gamma_*`와 `max_abs_support_gamma_*_dt`가 각각
점유 support 안의 correction rate와 한 step의 dimensionless tangent load다.
빈 tail까지 포함하는 `max_abs_gamma_*`는 보조 디버깅용으로만 남긴다.
이 값들과 `max_raw_rate_*`, `max_corrected_rate_*`,
`pnc_projection_correction`은 각 저장 간격 전체의 최댓값이다.

기본값은 `mixed`다. 처음부터 single 결과를 신뢰하지 말고 CPU double 또는 GPU
double과 비교한다.

## 3. CPU/GPU 결과 비교

CPU와 GPU 계산은 `dt`, grid, 초기상태, `save-every`가 완전히 같아야 한다.

```bash
python -m multi_component_exact_factorization_gpu.compare_precision \
  results/cpu_double/multi_component_direct_ef.npz \
  results/gpu_mixed_001fs/multi_component_direct_ef_gpu.npz
```

Full-Psi fidelity, 세 marginal L1 오차와 평균 위치 오차를 출력한다. 0.01 fs가
괜찮으면 0.1 fs, 1 fs 순서로 늘린다. Precision 차이가 `dt` 또는 grid를 바꿨을
때의 convergence 차이보다 작아야 한다.

## 4. 긴 계산

검증 후 예를 들어 1 fs mixed 계산은 다음처럼 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --precision mixed --dt-au 0.005 --t-final-fs 1.0 \
  --save-every 80 --progress-every 500 --check-every 100 \
  --outdir results/gpu_mixed_1fs
```

`--check-every`는 GPU에서 CPU로 매 step 동기화하는 것을 피하기 위한 검사
간격이다. 저장 frame과 마지막 step에서는 항상 검사한다. `--save-every`는
출력 크기와 CPU 전송 횟수를 줄이지만 time step 수는 줄이지 않는다.

공용 서버에서 평균 GPU 부하를 낮추려면 `--gpu-util-limit`을 사용한다. 예를
들어 다음 옵션은 20 step 단위의 계산 시간 뒤에 짧게 대기하여 계산/대기
duty cycle을 약 60%로 맞춘다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --device 0 --precision double --gpu-util-limit 60 \
  --electron-excitation 1 --nR 60 \
  --dt-au 0.005 --t-final-fs 5.0 \
  --save-every 400 --progress-every 2000 --check-every 200 \
  --outdir mcef_nR60_double_5fs
```

이는 `nvidia-smi`의 순간 측정값을 고정하는 하드웨어 제한이 아니라 평균
duty-cycle 제한이다. 따라서 표시되는 순간 사용률은 목표보다 높거나 낮을 수
있지만 장시간 평균 부하와 발열은 감소한다. 계산 결과에는 영향을 주지 않고
wall time은 대략 `100/limit`배로 증가한다. 기본값 100은 대기 없는 기존
동작이다. 더 짧은 부하 burst가 필요할 때만 `--gpu-throttle-every 10`처럼
조절한다.

계산이 정상 종료되어 NPZ 저장까지 성공한 뒤 모든 그림·영상·분석·3D HTML을
곧바로 만들려면 ``--render-after``를 덧붙인다. 렌더링 시간과 파일 크기를
줄이는 preview 설정은 ``--render-fast``를 함께 사용한다. GPU 전파 함수가
반환되어 계산 배열이 해제된 다음 CPU 렌더링을 시작한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --device 0 --precision double --electron-excitation 1 \
  --dt-au 0.005 --t-final-fs 10.0 \
  --save-every 400 --progress-every 1000 --check-every 200 \
  --outdir mcef_exc1_double_10fs --render-after --render-fast
```

렌더링 중 오류가 나더라도 그 전에 계산 archive는 이미 저장되어 있다.

## 5. 기존 그림 생성

완료된 GPU 계산의 폴더 이름만 지정해 모든 그림·영상·상태 분석·3D HTML을
한 번에 생성할 수 있다.

```bash
python -m multi_component_exact_factorization.render_all gpu_mixed_1fs
```

명령은 날짜별 ``results/YYYYMMDD`` 폴더에서 가장 최근의 동일 이름을 찾고,
NPZ metadata로 초기 ground/excited state와 분석 상태 수를 자동 판별한다.
기본 분석은 residual 판정을 위해 BO 상태 6개를 사용하고, occupied-support
potential, 인접 gap/NAC, gauge-invariant current/force 진단도 생성한다. 기본
실행은 archive를 RAM에 한 번만 풀고 BO decomposition도 공유한다. 빠른
preview에는 ``--fast``를, RAM이 부족할 때만 ``--low-memory``를 사용한다.

```bash
python -m multi_component_exact_factorization.visualize \
  results/gpu_mixed_1fs/multi_component_direct_ef_gpu.npz

python -m multi_component_exact_factorization.dynamics_analysis \
  results/gpu_mixed_1fs/multi_component_direct_ef_gpu.npz
```

## 코드 읽는 순서

1. `gpu_core.py`: `dst1_ortho`, `derivative`
2. `gpu_core.py`: `geometric_fields`, `instantaneous_functionals`
3. `gpu_core.py`: `coupled_rhs`, `full_step`
4. `propagate_gpu.py`: GPU 상주, 저장 시 CPU 전송, 정밀도 선택

GPU 코드는 한 trajectory를 두 GPU로 분할하지 않는다. 두 장을 쓸 때는 GPU 0과
GPU 1에서 서로 다른 dt/grid 계산을 독립적으로 실행하는 편이 효율적이다.
