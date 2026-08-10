# Single-GPU Multi-Component Exact Factorization

모든 ``--outdir`` 결과는 서버 현지 실행일에 따라 자동으로
``results/YYYYMMDD/<지정한 이름>`` 아래에 저장된다. 예를 들어
``--outdir results/mcef_exc1_1fs``를 2026년 7월 30일에 실행하면 실제 경로는
``results/20260730/mcef_exc1_1fs``다.

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
위 명령은 이를 SciPy DST-I와 `complex128`, `complex64`에서 비교하고 periodic
5점 유한차분도 CPU와 비교한다. `GPU 기본 검증: PASS`가 나와야 전파한다.

`CUDA_VISIBLE_DEVICES=0`을 쓰면 물리 GPU 0만 프로그램에 보이며, 프로그램
내부에서는 그 장치가 다시 index 0이 된다. 따라서 보통 `--device 0`을 그대로
둔다.

## 2. 짧은 double 시험

Production CLI는 누적 오차가 작은 `complex128`/FP64 double precision으로
고정되어 있다. `--precision`을 입력할 필요가 없으며, 예전 명령과의 호환을
위해 `--precision double`만 계속 허용한다. 먼저 아주 짧게 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --t-final-fs 0.01 --dt-au 0.005 \
  --outdir results/mcef_double_001fs
```

Double 전파는 CPU와 같은 product-preserving nested tangent correction을
RK4의 네 stage마다 적용한다.

또한 finite difference가 연속 Leibniz rule을 정확히 만족하지 않는 defect는
각 RK stage에서 factor product RHS와 Hermitian periodic nuclear `D2`의 차이를
계산해 occupied nested tangent에 되돌리는 discrete product projection으로
제거한다. Tail correction은 기존 joint/heavy support mask와 같은 smooth
weight로 감쇠한다.

archive의 `max_abs_support_gamma_*`와 `max_abs_support_gamma_*_dt`가 각각
점유 support 안의 correction rate와 한 step의 dimensionless tangent load다.
빈 tail까지 포함하는 `max_abs_gamma_*`는 보조 디버깅용으로만 남긴다.
이 값들과 `max_raw_rate_*`, `max_corrected_rate_*`,
`pnc_projection_correction`은 각 저장 간격 전체의 최댓값이다.

Factor-level 발산을 분리하기 위해 archive에는 다음 진단도 저장한다.
큰 factor reduction의 production overhead를 피하기 위해 세부 RHS/PNC 진단은
`--verbose-diagnostics`를 지정한 실행에서 활성화된다.

- `max_support_pnc_*_projection_load`: 빈 conditional tail을 감쇠한 PNC load
- `max_weighted_rms_pnc_*_projection_load`: 실제 nuclear density로 평균한 PNC load
- `max_*_rhs_*_before/after_product_projection`: product projection 전후 factor RHS
- `max_*_rhs_*_after_product_projection_dt`: 한 step의 dimensionless factor 변화량
- `max_rk_stage_amplification_*`: RK4의 첫 RHS에 대한 후속 stage RHS 증폭률

전파가 regular save 사이에서 non-finite가 되면 실패 배열 자체를 trajectory에
섞지 않는다. 대신 마지막 finite check-point를 추가 frame으로 저장하고
`saved_steps`, `failure_last_finite_check_step`,
`failure_nonfinite_counts`, `failure_max_finite_abs`를 기록한다. 따라서 실패
지점에 더 가까운 checkpoint가 필요하면 `--check-every`를 줄이면 된다.

## 3. CPU/GPU 결과 비교

CPU와 GPU 계산은 `dt`, grid, 초기상태, `save-every`가 완전히 같아야 한다.

```bash
python -m multi_component_exact_factorization_gpu.compare_precision \
  results/cpu_double/multi_component_direct_ef.npz \
  results/mcef_double_001fs/multi_component_direct_ef_gpu.npz
```

Full-Psi fidelity, 세 marginal L1 오차와 평균 위치 오차를 출력한다. 0.01 fs가
괜찮으면 0.1 fs, 1 fs 순서로 늘린다. Precision 차이가 `dt` 또는 grid를 바꿨을
때의 convergence 차이보다 작아야 한다.

## 4. 긴 계산

검증 후 예를 들어 1 fs 계산은 다음처럼 실행한다. 저장 간격은 약 200 frame,
진행 문구는 약 20회, finite 검사는 약 500회가 되도록 trajectory 길이에 따라
자동 설정된다. 계산이 끝나면 빠른 report와 동영상도 자동 생성된다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --electron-excitation 1 --dt-au 0.005 --t-final-fs 1.0 \
  --outdir results/mcef_exc1_1fs
```

필요할 때만 `--save-every`, `--progress-every`, `--check-every`를 직접 지정한다.
`--check-every`는 GPU에서 CPU로 매 step 동기화하는 것을 피하기 위한 검사
간격이고 저장 frame과 마지막 step에서는 항상 검사한다. `--save-every`는 출력
크기와 CPU 전송 횟수를 줄이지만 time step 수는 줄이지 않는다.

기본 ``--gpu-optimization fused``는 물리식과 RK4 순서를 바꾸지 않고 한 RK
stage에서 이미 계산한 ``D_q Phi``, ``D_R Phi``, ``D_q Lambda``,
``D_R Lambda``, ``D_R chi``를 covariant square와 logarithmic derivative가
공유한다. 또한 동일한 곱미분식을 ``Xi=Lambda*chi``로 먼저 묶어 full-product
RHS의 큰 3차원 임시 배열을 하나 줄인다. Periodic 5-point stencil도 네 번의
``cp.roll``과 여러 임시 배열 대신 같은 계수를 한 번의 CUDA memory pass에서
계산한다. 검증용 ``--gpu-optimization reuse``는 CuPy-roll stencil과 stage
재사용을, ``baseline``은 CuPy-roll stencil과 기존 반복 계산을 사용한다.

서버에서 실제 grid의 가속률과 두 경로의 수치 동등성을 확인하려면 다음을
실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.validate_gpu \
  --device 0 --nx 174 --nq 174 --nR 60 \
  --step-benchmark-repeats 3
```

``baseline/reuse RHS error``와 ``baseline/fused RHS error``가 double roundoff
수준이고 마지막 ``fused full-step speedup``이 1보다 커야 한다. 실제 trajectory
archive에는 선택한 실행 경로가 ``gpu_optimization`` metadata로 기록된다.

공용 서버에서 평균 GPU 부하를 낮추려면 `--gpu-util-limit`을 사용한다. 예를
들어 다음 옵션은 20 step 단위의 계산 시간 뒤에 짧게 대기하여 계산/대기
duty cycle을 약 60%로 맞춘다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --device 0 --gpu-util-limit 60 \
  --electron-excitation 1 --nR 60 \
  --dt-au 0.005 --t-final-fs 5.0 \
  --outdir mcef_nR60_double_5fs
```

이는 `nvidia-smi`의 순간 측정값을 고정하는 하드웨어 제한이 아니라 평균
duty-cycle 제한이다. 따라서 표시되는 순간 사용률은 목표보다 높거나 낮을 수
있지만 장시간 평균 부하와 발열은 감소한다. 계산 결과에는 영향을 주지 않고
wall time은 대략 `100/limit`배로 증가한다. 기본값 100은 대기 없는 기존
동작이다. 더 짧은 부하 burst가 필요할 때만 `--gpu-throttle-every 10`처럼
조절한다.

### Weak derivative와 weighted tangent projection

기존 pointwise/nested-inverse backend는 기본값으로 그대로 유지된다. 새
안정화 backend는 다음처럼 명시적으로 선택한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --electronic-representation grid \
  --log-derivative-backend weak \
  --product-projection-backend weighted_tikhonov \
  --weak-log-delta 1e-10 --weak-log-smoothing 0.04 \
  --projection-tau-phi 1e-10 --projection-tau-lam 1e-10 \
  --projection-tau-chi 1e-10 \
  --electron-excitation 1 --dt-au 0.025 --t-final-fs 0.1 \
  --outdir mcef_weak_weighted_smoke
```

Weak backend는 ``Xi=Lambda*chi``의 q/R amplitude logarithmic derivative와
``chi``의 R derivative를 density-weighted periodic PCG로 구한다. Weighted
projection은 ``1/Xi``와 ``1/chi``를 직접 사용하지 않고 occupied residual과
inverse-support factor penalty 사이의 Tikhonov 해를 사용한다.
강한 PNC tangent gauge에서 전자/양성자 수직 블록과 heavy 평행 블록이
직교하므로, 거대한 전역 행렬 대신 동일한 structured minimum-norm 해를
세 개의 닫힌형 block으로 계산한다.

Weak PCG의 전처리기는 각 conditional line에서 density를 그 평균으로 바꾼

```text
P = <rho> + delta - ell^2 D2
```

를 사용한다. ``D2``가 periodic 5점 operator이므로 ``P``는 FFT 공간에서
정확히 대각화된다. 실제 variable-density weak operator는 바꾸지 않고 PCG의
탐색 방향만 개선하며, Jacobi 전처리보다 적은 iteration으로 같은 해에
수렴한다. 기본 최대 iteration은 안전 여유를 위해 80이지만 residual tolerance를
만족하면 즉시 종료한다.

### Deep-tail exact-zero gate와 전체 핵 좌표 범위

기본 ``--deep-tail-zero-threshold 1e-12``는 full-Psi에서 계산한 실제 qR/R
marginal의 상대밀도를 사용한다. 상대밀도 ``1e-13`` 이하는 phase ratio,
log-amplitude ratio, product inverse correction과 PNC gauge transfer를 정확히
0으로 만들고, ``1e-11`` 이상은 정확히 1로 보존한다. 그 사이는 log-density
축의 C2 quintic gate로 연결한다. Vector potential과 factor 자체는 자르지
않는다. 완전한 기존 동작을 재현하려면 다음 옵션을 쓴다.

```bash
--deep-tail-zero-threshold 0
```

q와 R box가 좁아서 생기는 boundary/vector-potential noise를 분리해 검사할
때는 ``--full-nuclear-range``를 추가할 수 있다. 이 옵션은 q와 R 범위를
전자 hard-wall 범위 ``[left-position,x-max)``와 같게 만들지만 점 수는 자동으로
늘리지 않는다. 따라서 동일한 공간 해상도를 유지하려면 점 수도 box 길이에
맞게 늘려야 하며, direct-grid 메모리 비용이 ``nx*nq*nR``로 증가한다. 먼저
작은 점 수와 짧은 시간으로 boundary 진단만 수행한 뒤 production에 사용한다.

전자 box 자체를 대칭인 ``[-L,+L]``로 바꾸려면
``--symmetric-box-half-width L``을 쓴다. 오른쪽 벽에 실제 고정 중심을 둘
때만 ``--right-charge Z_R``를 함께 지정한다. 이 전하는 전자 attraction과
proton/heavy repulsion 및 좌우 고정 중심 상호작용을 Hamiltonian에 추가하므로
수치적인 absorbing boundary가 아니며, 기본값 0은 기존 물리를 그대로 보존한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --electronic-representation grid --full-nuclear-range \
  --deep-tail-zero-threshold 1e-12 \
  --nx 174 --nq 174 --nR 120 \
  --dt-au 0.025 --t-final-fs 0.1 --no-render-after \
  --outdir mcef_full_nuclear_range_smoke
```

### Electronic-only Born--Huang backend

``paper/MCEF_revised.pdf``의 Eqs. (71)--(86)에 따라 ``Phi``만 local BO
basis로 전개할 수 있다. q/R의 ``Lambda``와 ``chi`` grid는 바뀌지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --electronic-representation born_huang --bo-states 6 \
  --log-derivative-backend weak \
  --product-projection-backend weighted_tikhonov \
  --electron-excitation 1 --nq 174 --nR 120 \
  --dt-au 0.025 --t-final-fs 0.1 \
  --no-render-after --outdir mcef_bh6_weak_weighted_smoke
```

시간 loop에는 ``C_j(q,R)``, BO energy와 q/R 1·2차 NAC만 GPU에 남는다.
따라서 큰 동적 전자 배열의 원소 수는 ``nx*nq*nR``에서
``N_BO*nq*nR``로 줄어든다. ``--bo-save-basis-states``를 추가하면 정적 BO
eigenvector도 archive에 저장하여 full-Psi reference 비교가 가능하지만,
archive가 커지므로 convergence run에서만 권장한다.

같은 전자 Hamiltonian의 BO diagonalization은 기본적으로
``results/bo_basis_cache``에 한 번 저장한다. cache key는 ``N_BO``뿐 아니라
전체 x/q/R 격자와 electronic potential의 SHA-256 fingerprint를 포함하므로,
``dt``, 최종 시간, mask만 바꾼 계산은 즉시 재사용하고 전하·softening·box·grid
중 하나라도 달라지면 자동으로 별도 basis를 생성한다.

```bash
--bo-basis-cache-dir results/bo_basis_cache
```

정상 cache는 두 번째 실행에서 ``BO basis cache HIT: 재사용``으로 표시된다.
현재 조건만 강제로 다시 만들 때는 ``--rebuild-bo-basis-cache``, cache 자체를
쓰지 않을 때는 ``--no-bo-basis-cache``를 사용한다. 큰 full-box BH6 cache는
real64 eigenstate와 NAC를 포함하므로 수 GB의 디스크 공간이 필요하지만, 매
실행마다 수십만 개의 전자 고유값 문제를 다시 푸는 시간을 없앤다.

Born--Huang archive도 direct-grid와 같은 이름과 역할의 compact report를
만든다. 저장 frame마다 exact electron marginal을 R-block으로 합성하고,
``|C_n Lambda chi|^2``의 q/R marginal도 저장한다. 따라서 거대한 static
``bo_basis_states`` tensor를 archive에 넣지 않고도 electron/proton/heavy
marginal, q-R density, momentum/current/force, exact potentials 및 논문식 BO
surface 위 state-resolved nuclear wave packet을 그릴 수 있다. 이 추가 저장량은
``O(nt*(nx+N_BO*(nq+nR)))``이다. 합성 시간이 필요 없는 계산만 원하면
``--no-bo-save-electron-density``를 쓸 수 있으며, 그 archive의 electron panel은
BO-state composition으로 대체된다.

정적 출력은 direct-grid와 같은 핵심 4장에 BO 전용 1장이 추가된다.

* ``01_particle_motion.png``: electron/proton/heavy motion과 marginal
* ``02_electronic_transitions.png``: BO population 및 state-projected packet
* ``03_exact_potentials.png``: scalar/vector potential, momentum, force
* ``04_numerical_reliability.png``: norm/tangent/product 진단
* ``05_born_huang_surface_dynamics.png``: q/R BO surfaces와 colored wave packets

영상 이름도 ``mcef_dynamics_overview``, ``mcef_exact_potentials``,
``mcef_physical_interpretation``으로 direct-grid report와 맞춘다.

```bash
python -m multi_component_exact_factorization.render_all \
  results/YYYYMMDD/run_name --fast
```

출력은 계산 폴더의 ``report/`` 아래에 저장된다. 이번 변경 이전 archive도
``electronic_coefficients``를 한 번 읽어 state-resolved q/R density를 복원할 수
있지만, 저장되지 않은 electron marginal은 BO composition panel로 표시한다.

BO truncation은 같은 짧은 시간에서 ``--bo-states 4``, ``6``, ``8``을
순서대로 실행하여 ``bo_populations``와 full-Psi fidelity가 수렴하는지
확인한다. Full-TDSE와 직접 비교할 한 run에만 ``--bo-save-basis-states``를
추가하고 다음 비교기를 사용한다.

```bash
python -m multi_component_exact_factorization.compare \
  results/REFERENCE/multi_component_reference.npz \
  results/BO/multi_component_born_huang_ef_gpu.npz \
  --progress-every 10
```

가장 높은 retained BO state의 population이 계속 커지면 state 수가 부족한
것이므로 장시간 run으로 넘어가지 않는다. 짧은 기준을 통과한 뒤에만
``--t-final-fs``를 2, 5, 10, 20 fs 순서로 늘린다.

계산이 정상 종료되어 NPZ 저장까지 성공하면 빠른 report와 동영상을 자동으로
만든다. GPU 전파 함수가 반환되어 계산 배열이 해제된 다음 CPU 렌더링을
시작한다. 계산만 원하면 ``--no-render-after``, full 품질 렌더링이 필요하면
``--render-full``을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_gpu.propagate_gpu \
  --device 0 --electron-excitation 1 \
  --dt-au 0.005 --t-final-fs 10.0 \
  --outdir mcef_exc1_double_10fs
```

렌더링 중 오류가 나더라도 그 전에 계산 archive는 이미 저장되어 있다.
CLI의 terminal 출력은 별도 ``tee`` 명령 없이 계산 폴더의
``propagation.log``에도 동시에 기록된다.

Non-finite가 검출되거나 ``Ctrl-C``로 중단되면 실패한 GPU state는 버리고,
마지막 정상 저장 frame까지를 표준 NPZ 이름으로 부분 저장한다. NPZ의
``propagation_completed=false``, ``failure_detected_step``,
``last_saved_step``, ``failure_reason``으로 완료 결과와 구별할 수 있다.
``propagation_status.log``에는 이 정보와 마지막 정상 norm/PNC 진단을 짧게
기록하고, 자동 report와 동영상 제목에는 ``PARTIAL trajectory``를 표시한다.
부분 렌더링 후 process는 성공으로 위장하지 않고 exit code 2로 종료된다.
강제 ``kill -9``, 전원 차단, GPU driver/process 자체의 즉시 종료는 Python이
정리 코드를 실행할 수 없으므로 이 복구 경로의 대상이 아니다.

기본 ``report`` 폴더에는 핵심 PNG 4장과 목적이 분리된 동영상 3개가 생긴다.

* ``mcef_dynamics_overview``: joint nuclear density, BO population과 핵심
  전자·양성자 dynamics의 시간적 연결
* ``mcef_exact_potentials``: 두 TDPES와 connection을 mechanical momentum,
  current, drive로 이어서 해석
* ``mcef_physical_interpretation``: 전자·양성자·heavy marginal을 동시에 보여준
  뒤, ``a``가 포함된 proton current/drive와 ``b``가 ``alpha``를 통해 포함되는
  heavy current/drive를 같은 frame에서 연결

세 번째 영상의 위쪽 A--C는 모두 다른 좌표를 적분해 얻은 실제 marginal이고,
아래쪽 D--F는 gauge-dependent connection 하나를 독립적으로 해석하는 대신
gauge-invariant 조합으로 실제 transport와 힘의 방향을 보여준다. 회색 cell은
확률이 거의 없어 phase ratio를 물리적으로 해석하지 않는 영역이다.

### Probability-budget flat-top coupling mask

기존 ``rational_deep_tail`` mask는 기본값으로 유지된다. 실험적인
``flat_top`` backend는 초기 joint/heavy physical density의 suppressed-mass
budget으로 고정 ``r_on``을 한 번 정한다. ``r >= r_on``에서는 coupling
coefficient가 정확히 보존되고, 지정한 log-density 폭에서 C2 smootherstep으로
감소한 뒤 정확히 0이 된다. PNC gate와 product projection은 변경하지 않는다.

```bash
--coupling-mask-backend flat_top \
--flat-top-budget-phi 1e-9 \
--flat-top-budget-lam 1e-9 \
--flat-top-transition-decades 3
```

재현 가능한 production scan은 pilot이 출력한 onset을 직접 고정할 수 있다.

```bash
--coupling-mask-backend flat_top \
--flat-top-on-phi 2.5e-11 \
--flat-top-on-lam 1.8e-10 \
--flat-top-transition-decades 3
```

고정 threshold는 미래 frame의 budget을 자동 보장하지 않으므로 archive의
``suppressed_probability_phi``와 ``suppressed_probability_lam``을 확인한다.
Flat-top은 gauge-invariant first-order 조합 ``P+A-iL``에 적용하고,
covariant derivative와 square의 vector potential은 그대로 유지한다.

## 5. 기존 그림 생성

완료된 GPU 계산의 폴더 이름만 지정해 모든 그림·영상·상태 분석·3D HTML을
한 번에 생성할 수 있다.

```bash
python -m multi_component_exact_factorization.render_all mcef_exc1_1fs
```

명령은 날짜별 ``results/YYYYMMDD`` 폴더에서 가장 최근의 동일 이름을 찾고,
NPZ metadata로 초기 ground/excited state와 분석 상태 수를 자동 판별한다.
기본 분석은 residual 판정을 위해 BO 상태 6개를 사용하고, occupied-support
potential, 인접 gap/NAC, gauge-invariant current/force 진단도 생성한다. 기본
실행은 archive를 RAM에 한 번만 풀고 BO decomposition도 공유한다. 빠른
preview에는 ``--fast``를, RAM이 부족할 때만 ``--low-memory``를 사용한다.

```bash
python -m multi_component_exact_factorization.visualize \
  results/mcef_exc1_1fs/multi_component_direct_ef_gpu.npz

python -m multi_component_exact_factorization.dynamics_analysis \
  results/mcef_exc1_1fs/multi_component_direct_ef_gpu.npz
```

## 코드 읽는 순서

1. `gpu_core.py`: `dst1_ortho`, `derivative`
2. `gpu_core.py`: `geometric_fields`, `instantaneous_functionals`
3. `gpu_core.py`: `coupled_rhs`, `full_step`
4. `propagate_gpu.py`: GPU 상주, 저장 시 CPU 전송, 정밀도 선택

GPU 코드는 한 trajectory를 두 GPU로 분할하지 않는다. 두 장을 쓸 때는 GPU 0과
GPU 1에서 서로 다른 dt/grid 계산을 독립적으로 실행하는 편이 효율적이다.
