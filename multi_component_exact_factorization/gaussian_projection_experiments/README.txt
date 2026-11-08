고정-R Gaussian projection 진단 / K 검사 / 실험적 에너지 제약
===========================================================

범위와 보존
-----------
이 모듈은 별도 연구 실험이다. 기존 gaussian_trajectory, MCEF, TDSE,
MTE/TSH/CTMQC의 전파 코드, 기본값, checkpoint 형식을 수정하지 않는다.
일반적인 복소 C_k(q)를 그대로 유지한다. 위상을 다항식으로 제한하지 않는다.
heavy trajectory는 움직이지 않고, quantum momentum/MLS/force/hopping도 없다.
따라서 이 모듈의 에너지 조건을 움직이는 핵의 내부 에너지에 적용하면 안 된다.
실공간 MCEF adapter에서는 x가 전자, q가 proton, 고정 R가 heavy이다.
PEN benchmark adapter에서는 두 diabatic 전자성분, q가 photon, 고정 R가 핵이다.
PEN을 full-electronic-grid 모델로 바꾸었다고 주장하지 않는다.

수식 및 기존 구현과의 관계
-------------------------
paper/MCEF_Gaussian_Trajectory_Codex_Prompt.md, Reference.pdf, 11_3.pdf와
기존 gaussian_trajectory/gaussian.py의 P3/P8/P15를 따른다.
새 에너지 제약은 사용자가 이번에 승인한 실험적 추가 조건이며 원 논문의
검증된 전파법으로 이름 붙이지 않는다.

  G = Lambda_K C;       F = -i H_disc G  (현재 근사 상태에서 계산)
  r = Re sum_k C_k* F_k
  Cdot = (F - C r)/Lambda_K              (기존 P3 그대로)
  A = sqrt(w) dLambda_K/dz; y = sqrt(w) r

standard: min_v ||A v - y||^2           (기존 함수를 직접 호출)
energy:   위 최소제곱 + h^T A v = h^T y
  h = sqrt(w) Re sum_k C_k*[(H_disc-E)G]_k / ||G||^2

실제 Gdot = F + (Lambdadot_K-r) C.
dE/dt = 2 h^T(A v-y).
에너지 제약은 parameter 속도를 고르는 조건이다. 전파 후 에너지를 맞추거나,
상태를 scaling하거나, 결과에서 drift를 빼는 보정이 아니다.
A=U S V^T의 SVD 좌표에서 선형 제약을 풀어 normal equation의 조건수 제곱을
피한다. 독립적인 KKT solve와 일치하는지 테스트한다. 이미 만족된 조건의
허용오차(power_atol)도 기록하며, 미해결 rank/에너지 방향은 실패로 보고한다.
기본 density mask/floor, truncated SVD, ridge, 실패 시 무제약 fallback은 없다.

진단의 구별
-----------
1. requested r vs actual Lambdadot 및 전체/occupied/tail EOM residual.
   occupied 기준 rho > 1e-6 rho_max는 진단에만 쓰고 전파에는 쓰지 않는다.
2. 같은 H_disc와 각 K의 실제 초기 G0에서 출발한 spectral reference.
   원래 물리 초기상태와 Gaussian 초기 fitting의 차이는 별도로 기록한다.
3. 복소 상태 오차, global-phase aligned 오차, amplitude/density 오차,
   energy drift, projection power 적분, PNC, Fourier high-k mass, 조건수.
4. 적분 잔차의 연속시간 error bound. 유한 RK 적분/적분구적 오차는 남으므로
   저장된 bound 숫자를 엄밀한 이산 계산 인증이라고 해석하지 않는다.
5. offline_capacity.json은 매 시점 독립 fitting일 뿐 자율 전파가 아니다.
   분리된 center bounds 안의 local fit이며 전역 최적 표현력의 증명이 아니다.
   K별 fitting 오차의 비단조성을 표현 공간 자체의 한계로 해석하면 안 된다.

reference는 옵션이다. RHS에는 reference 또는 reference와의 차이가 들어가지
않는다. --no-reference on/off 전파 결과가 bitwise 동일한지 테스트한다.

초기화와 시간적분
----------------
--initialization fit (기본): 분리된 Hermite seed에서 초기 amplitude fitting.
--initialization hermite: fitting 없이 Gaussian convolution의 Hermite quadrature
seed 사용. 초기 오차는 더 클 수 있으나 초기 성분 퇴화를 분리해서 검사한다.
두 경우 모두 실제 초기 오차와 tangent rank/조건수를 저장한다. 같은 kernel을
단순히 복제하지 않으며 K 증가가 전파 수렴을 보장한다고 가정하지 않는다.

--integrator sphere-rk4 (기본): 기존 homogeneous-coordinate RK4를 재사용.
--integrator adaptive-sphere-rk4: 동일 RHS의 full step vs 두 half step 오차 제어.
모든 C(q), transformed parameter, passive integral에 rtol/atol을 적용한다.
accepted solution은 두 half step이며 Richardson/energy 사후 보정은 없다.
기본적으로 한 outer interval 내 2^16 step 축소 또는 2048회 시도 한계를 넘으면 실패한다.
--adaptive-max-halvings / --adaptive-max-attempts로 이 수치적 재시도 한도를
명시적으로 늘릴 수 있다. rtol/atol 또는 energy/rank 허용오차를 완화하는 옵션이
아니다. 중간/최종 RK 시험 상태의 Gaussian 폭 등이 표현 가능 범위를 벗어나면
작은 step으로 재시도하며, 초기화/좌표/shape 설정 오류는 그대로 보고한다.
폭이나 가중치를 clipping하거나 유효하지 않은 상태를 저장하지 않는다.
time-step 축소도 rank 손실/거의 사라지는 에너지 제약 방향을 해결하지 못할 수 있다.

실행 (저장소 루트, MCEF 환경)
---------------------------
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

1) 수식/신규/기존 Gaussian 회귀검사
python -m unittest discover -s tests -p 'test_gaussian*.py' -b
python -m unittest discover -s tests -p 'test_pen_perspectives_2026.py' -b
python -m unittest discover -s tests -p 'test_discrete_mcef.py' -b

2) 완주 확인한 짧은 coupled K=1 실험, PNG/PDF/MP4 자동 생성
python -m multi_component_exact_factorization.gaussian_projection_experiments.run \
  --out results/20261108/gaussian_projection_server_v1/pilot \
  --components 1 --mode energy --g .01 --integrator adaptive-sphere-rk4 \
  --dt-au .02 --end-fs .4 --every-fs .01 --render --movies

3) 원래 projection vs energy, K=1,2,3,4, g=0/.01 (순차 실행)
python -m multi_component_exact_factorization.gaussian_projection_experiments.sweep \
  --out results/20261108/gaussian_projection_server_v1/K_study \
  --K 1 2 3 4 --modes standard energy --couplings 0 .01 \
  --dt-au .00125 --end-fs .4 --every-fs .01 --render

기본 q: FD2, 64점, [-14,14), PEN on-resonance omega=0.17, DSE off.
이들은 이전 고정-R 실패를 분리하는 benchmark 설정이지 기존 TDSE 기본값의
변경이 아니다. --q-operator fft/fd4, --nq, --qmin, --qmax, --fixed-R,
--omega, --dse 옵션으로 명시적으로 바꿀 수 있다.
새 출력 폴더에서 --initialization hermite를 넣으면 초기화 민감도를 비교한다.
--integrator adaptive-sphere-rk4로 정확한 시간적분이 가능한지 별도로 검사한다.
rank/시간적분 guard로 일부 case가 실패하면 sweep은 나머지를 계속 수행하고
전체 종료코드 2를 반환한다. 이는 실패를 숨기지 않기 위한 동작이다.

장시간 목표 + 다중 Gaussian 시험 (완주 보장 명령이 아님):
python -m multi_component_exact_factorization.gaussian_projection_experiments.sweep \
  --out results/20261108/gaussian_manyK_server_v1 \
  --K 4 6 8 12 --modes energy --couplings .01 --nq 128 \
  --initialization hermite --separation .8 \
  --integrator adaptive-sphere-rk4 --dt-au .01 --end-fs 10 --every-fs .05 \
  --adaptive-max-halvings 24 --adaptive-max-attempts 8192 --render
실제 로컬 시험의 도달 시간/중단 이유는 LONG_K_TEST_REPORT.txt에 기록한다.
초기 성분을 잘 분리하는 것과 초기 vacuum을 정확하게 표현하는 것 사이에
조건수/초기오차 tradeoff가 있다. initialization.json을 함께 확인할 것.

4) reference 없이 자율 전파: 위 run 명령의 새 --out에 --no-reference 추가.
5) 실공간 MCEF adapter의 매우 짧은 fixed-heavy 검사 (production 아님)
python -m multi_component_exact_factorization.gaussian_projection_experiments.run \
  --out results/20261108/gaussian_projection_server_v1/mcef_smoke \
  --model mcef --nq 48 --qmin -6 --qmax -2 --nx 17 \
  --mode energy --dt-au .0025 --end-fs .02 --every-fs .005 --render

6) 기존 진단 재렌더링 (계산 없음)
python -m multi_component_exact_factorization.gaussian_projection_experiments.render \
  --case results/20261108/gaussian_projection_server_v1/pilot --movies

7) offline capacity만 검사 (전파 상태에 절대 반영하지 않음)
python -m multi_component_exact_factorization.gaussian_projection_experiments.capacity \
  --case results/20261108/gaussian_projection_server_v1/pilot \
  --K 1 2 3 4 --times-fs 0 .1 .2 .4

출력 / resume / 자원
-------------------
각 case: config.json, metadata.json(기존+신규 source hash), initialization.json,
status.json, checkpoint.npz, diagnostics.npz, 실패 시 failure.json.
plots/: diagnostic_summary.png/pdf, state_XXXXX.png, state_diagnostics.mp4.
sweep/: sweep.json, K_comparison_g_*.png/pdf.
checkpoint에는 last accepted state와 동일 시점의 history를 함께 atomic 저장.
--resume은 동일 source/config에서 end_fs만 늘릴 수 있다. 다른 dt/K/모델은
새 폴더에서 실행한다. 다른 기존 폴더를 자동으로 덮어쓰지 않는다.
생성한 그림만 render 명령으로 덮어쓸 수 있다.
H action/potential은 고정 R에서 캐시한다. dense diagonalization은 reference를
켰을 때에만 수행하며 기본 dimension 상한 2048, --no-reference로 생략 가능.
GPU나 장시간 서버 계산을 실행했다고 주장하지 않는다. CPU 1 thread 순차
실험이며 수십~수백 프레임의 작은 내부 상태만 저장한다.

중요 결론
---------
2026-11-08 검사에서 짧은 coupled pilot은 완주했지만 K 수렴이나 10--40 fs
안정 전파는 확인되지 않았다. TEST_REPORT.txt를 먼저 읽을 것.
에너지 보존은 상태 정확도 보장이 아니다. COMPLETE_NOT_CERTIFIED는 실행
완료 상태일 뿐, 물리 검증 통과나 production 승인을 뜻하지 않는다.
