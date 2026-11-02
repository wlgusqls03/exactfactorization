PEN perspectives 2026: 독립 EL / POL CTMQC 실행
================================================

0. 범위와 한계 (랩미팅에서 반드시 구분)
-------------------------------------
이 모듈은 기존 pen_perspectives_2026의 2-electronic-state 모델을 사용하는
별도 CTMQC 구현이다. 기존 실공간 PEN, MCEF, TDSE 전파/재시작은 변경하지 않는다.
기존 TDSE의 결과를 CTMQC라고 바꿔 표시하거나 CTMQC 전파 중 TDSE를 호출하지 않는다.

EL: (R,q_c)는 classical Wigner ensemble, 전자 2상태는 양자 계수.
POL: R만 classical Wigner ensemble, 전자 2 × photon N_Fock은 양자 계수.
기본 N_Fock=2는 n=0,1이며, POL에서만 이 양자 절단을 적용한다.
EL photon은 연속적인 classical 변수이다. EL을 'n=0,1로 제한된 광자'라고
설명하면 안 된다. 서로 다른 근사/표현 공간이라는 점이 비교 목적의 일부다.

요청에 따라 복소 전체 파동함수를 임의 재구성하지 않는다. 따라서 exact
a,b,alpha,epsilon1,epsilon2,GI/GD, Lambda, S_q/S_adv/S_rel은 출력하지 않는다.
대신 직접 얻는 계수/힘/분포/조건부 확률/상태가중 에너지를 출력한다.
특히 <E_CBO>, <E_Pol>은 geometric/GD 성분을 포함한 exact TDPES가 아니다.
POL 후처리의 local density matrix는 혼합 상태이다. 밀도의 제곱근에 임의
위상을 넣어 pure Psi라고 취급하지 않는다. capabilities.json에도 명시한다.

이것은 저자 G-CTMQC 실행 파일을 그대로 돌리는 wrapper가 아니다. 공개 수식에
근거한 독립 구현이며, 저자 수치 결과와의 동일성/장시간 수렴은 아직 인증되지 않았다.
status.json의 COMPLETE_NOT_CERTIFIED는 이 의미다. 발표에서 '독립 CTMQC 구현의
결과'라고 쓰고, 검증 전 '논문 곡선 완전 재현'이라고 쓰지 않는다.

1. 근거, 방정식, 명시적 수치 선택
--------------------------------
모델/관점/초기조건:
  Magi et al., Different perspectives on the exact factorization for
  photon-electron-nuclear systems, PCCP (2026), Table 2, Eqs21-27, Appendix A.
  https://doi.org/10.1039/D6CP02020H
  paper/molecular vibration polariton/Different perspectives on the exact
  factorization for photon–electron–nuclear systems.pdf

표준 CTMQC 계수/힘 및 보정 quantum momentum:
  Villaseco Arribas, Agostini, Maitra, Molecules 27, 4002 (2022),
  Eqs16-18,26-27. https://doi.org/10.3390/molecules27134002

참고한 저자 공개 코드:
  https://gitlab.com/agostini.work/g-ctmqc
  HEAD b7a2a6d95656e4e0535e744b102670d85b07fb88 (검토 당시)
  src/coherence_corrections.f90, coefficients_evolution.f90,
  classical_evolution.f90. 코드는 복사/포팅하지 않았다.

Q=-grad(log probability density)/2. Q는 실제 trajectory momentum이 아니다.
f_k는 adiabatic force의 시간 적분. 공통 평균 힘을 빼고 적분하여 큰 공통값을
제거한다. 상태쌍 차이 f_k-f_l에는 영향이 없다(활성 구간 내부).
상태 population이 threshold보다 작거나 1-threshold보다 크면 그 상태의
force integral을 reset한다. 기본 threshold=0.001, 0.005 비교 가능.

상태쌍마다 df=f_l-f_k, p_k=|C_k|^2를 쓰면
  r_l += p_k sum_d Q_lk,d df_d/M_d
  r_k -= p_l sum_d Q_lk,d df_d/M_d
  F_CT += 2 p_l p_k df * sum_d Q_lk,d df_d/M_d
  dot(C)_CT = r*C (adiabatic basis에서 계산 후 fixed basis로 변환)
이다. 2026 논문 Appendix A에 인쇄된 force prefactor는 표준 CTMQC의 2/M과
같은 형태로 적혀 있지 않다. 이 구현은 위 2022 표준 식을 사용하며, 해당
차이를 숨기거나 인쇄된 식과 bitwise 동일한 구현이라고 주장하지 않는다.

핵/광자 mean-field force는 -c^dagger (dV/dX)c로 직접 계산한다.
fixed diabatic/product basis의 -iV c 전파는 adiabatic basis의 NAC 포함
mean-field 방정식과 동등하다. 작은 gap으로 나누는 NAC 계산은 필요 없다.
CT 항의 force-integral label은 eigenprojector overlap으로 추적하며, 출력
Pol/CBO population은 매 snapshot 에너지 순서로 계산한다.

기본 corrected pair Q:
  Q_lk,d^I=(X_d^I-center_lk,d)/(2 h_d^2)
  center=sum_I X_d^I p_l^I p_k^I (f_l,d^I-f_k,d^I)
            /sum_I p_l^I p_k^I (f_l,d^I-f_k,d^I)
활성 상태쌍으로만 가중한다. 이 선택은 CT-only ensemble population change가
0이라는 제약을 상태쌍/좌표별로 만족시킨다. 순간 trajectory별 norm도 보존한다.

중요한 안전 조건:
  denominator의 cancellation 또는 center가 ensemble 범위에서
  --intercept-limit-sigma * h 이상 벗어나면 그 상태쌍/방향 전체의 Q를 0으로
  두고 기록한다(기본 10). 개별 trajectory의 Q를 clipping하지 않는다.
  이것은 저자 코드의 M-parameter/uncorrected-density fallback과 같은 처리가
  아니다. 실험에서 확인된 초기 EL 수치 발산에 대한 명시적 guard이다.
  guard가 많이 작동하면 '그냥 문제없음'으로 판단하지 말고 5/10/20, dt,
  threshold, Ntraj, seed 민감도를 비교해야 한다. 논문의 정확한 실행 input이
  제공되지 않아 저자의 Gaussian/fallback 설정까지 동일하다고 할 수 없다.
  횟수는 모든 RK stage에서 누적하여 status/observables에 저장한다.

h_R=sigma_R/sqrt(2), h_q=1/sqrt(2 omega), sigma_R=(M*k)^(-1/4).
h는 density Gaussian의 표준편차이며 --bandwidth-scale로 배율을 바꿀 수 있다.
기본 corrected 계산은 O(Ntraj * Nstate^2), Ntraj*Ntraj 행렬을 저장하지 않는다.
--quantum-momentum density는 미보정 Gaussian-mixture log derivative를 직접
계산하는 작은 reference 모드이다. O(N^2)이므로 Ntraj<=1024로 제한한다.

시간 적분은 모든 ensemble/force/coefficient를 RK4 각 stage에서 다시 평가한다.
기본 dt=0.1 a.u.; 저자 코드의 Verlet/RK-Gill 조합과는 다르다. 유한 dt 오차를
반드시 확인한다. 매 step의 normalization 전 오차를 기록하고 1e-5를 넘으면
중단한다. 에너지를 강제로 맞추지 않는다. 원래 CTMQC 자체가 엄밀한 에너지
보존법은 아니므로 energy drift를 dt 오차와 구분해서 검사한다.

2. 모델, 초기 상태, 횟수
-----------------------
공통: M=20000, k=.02, a=3, b=.01, R2=2, R3=3.875, Z=1, mu12=-1.
nonadiabatic: R1=6, delta=0; omega_on=.17, omega_off=.097.
rabi: R1=R2=2, delta=.17; omega_on=.17, omega_off=.12.
strong g=.01; weak g=.001은 추가 control, zero g=0.
DSE 기본 off(논문 baseline), on도 지원. V와 DSE는 기존 모델과 동일하다.

--initial paper (기본): EL은 local upper CBO(Eq25), POL은 bare S1|0>(Eq27).
--initial bare: EL/POL 모두 bare S1. 기존 finite-Fock TDSE의 Eq27 초기 상태와
가급적 공정하게 비교하려면 이 옵션으로 별도 campaign을 만든다.
paper EL과 bare TDSE를 완전히 같은 초기 상태라고 비교하지 않는다.

R/P는 Gaussian Wigner sampling. 평균 momentum=0; sampled P는 0으로 고정하지
않는다. 같은 seed면 EL/POL/coupling case의 초기 R/P 표본은 동일하다.
POL photon은 진짜 Fock vacuum이므로 N(0)=0. EL photon은 vacuum Wigner 표본이며
N=omega*q^2/2+p^2/(2*omega)-1/2를 ensemble 평균한다. 유한 sampling 때문에
초기 평균이 조금 음수일 수도 있다. 0으로 clipping하지 않는다.
photon_number_se는 naive ensemble SE이지, coupled ensemble의 엄밀한 신뢰구간은
아니다. 실제 통계오차는 독립 seed로 확인한다.

기본 campaign = nonadiabatic/rabi × on/off × EL/POL = 8개.
--case all --dse both까지 선택하면 48개이므로 주의한다.
--end-fs 미지정이면 paper plot window: 30/40/30/3.5 fs.
--end-fs 40이면 8개 모두 40 fs로 한다. 계산 시간은 perspective마다 다르며
단순히 TDSE wall time의 2배라는 뜻은 아니다.

3. 서버 실행 순서 (CPU; GPU 점유하지 않음)
----------------------------------------
cd /home/hbji/exactfactorization
git pull
conda activate MCEF

아래 script는 BLAS thread=1, 로그, 디스크 확인을 자동 설정한다.
기존 결과를 덮어쓰지 않는다. ROOT는 새로운 경로를 사용한다.

# 계획만 출력, 파일/TDSE 쓰기 없음
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh plan \
  results/20261102/ctmqc_paper_server_v1 --jobs 2

# 8개 × 64 trajectories × 0.2 fs, PNG/MP4 자동 생성
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh smoke \
  results/20261102/ctmqc_smoke_server_v1 --jobs 2

# 먼저 8개 × 1000 trajectories × 1 fs (sampling/convergence 인증은 아님)
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh run \
  results/20261102/ctmqc_pilot_server_v1 --ntraj 1000 --end-fs 1 \
  --every-fs .1 --png-every-fs .5 --jobs 2

# 본 계산: 8개, 10000 trajectories, 논문 on/off frequency와 시간 창
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh run \
  results/20261102/ctmqc_paper_server_v1 --ntraj 10000 --every-fs .1 \
  --png-every-fs 5 --jobs 2

# 하나만: nonadiabatic/on/POL, 40 fs (별도 ROOT)
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh run \
  results/20261102/ctmqc_pol_on_server_v1 --model nonadiabatic --resonance on \
  --perspective pol --end-fs 40 --jobs 1

# 같은 ROOT/config/seed/cadence로 이어서 전파 (모든 case가 존재하는 경우)
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh run \
  results/20261102/ctmqc_paper_server_v1 --ntraj 10000 --every-fs .1 \
  --end-fs 40 --resume --jobs 2

# 재전파 없이 PNG/MP4 재생성: 3 fs마다 PNG, 30 fs까지
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh render \
  results/20261102/ctmqc_paper_server_v1 --png-every-fs 3 --t-max-fs 30 --jobs 2
# PNG만 재생성은 --no-movies. 기존 MP4는 삭제하지 않는다.

# 이미 있는 TDSE paper_campaign과 읽기 전용 비교
bash multi_component_exact_factorization/pen_ctmqc_2026/run_server.sh compare \
  results/20261102/ctmqc_paper_server_v1 \
  --tdse-root results/20261030/pen2026_paper_v1

TDSE overlay는 Hamiltonian/NF가 같은 경우에만 추가한다. EL 초기 준비 차이는
그림과 JSON에 표시한다. 정확한 matched-bare 비교는 처음 run에서
--initial bare를 사용하고 별도 ROOT에 저장한다.
각각 --model, --resonance, --perspective, --case, --dse로 부분 실행 가능.
--algorithm ehrenfest는 CT 항이 없는 control이며 별도 ROOT를 사용한다.
--no-render는 전파만 한다. 이후 render/compare로 다시 만들 수 있다.

4. 수렴 확인 / 장시간 계산 전 주의
--------------------------------
pilot의 status.json/numerical_checks.png에서 다음을 먼저 본다.
  - FAILED/failure.json, normalization 전 step error
  - energy drift, box 외부 trajectory 비율
  - quantum momentum guard disabled/attempted 비율
  - corrected-Q의 CT-only ensemble population derivative
그림이 예쁜 것이 수렴 증거가 아니다. 최소 dt=.1/.05, Ntraj=1000/2000/10000,
seed=2026/2027, population_threshold=.001/.005, bandwidth_scale=.5/1,
intercept_limit_sigma=5/10/20 민감도를 관심 사건의 시간 범위에서 비교한다.
sampling/Q guard 민감도가 크면 논문과의 일치를 주장하지 않는다.

5. 결과와 용량
-------------
ROOT/<nonadiabatic|rabi>/<on|off>/dse_<off|on>/<case>/<el|pol>/
  config.json, trajectory_config.json, provenance.json, scenario.json
  capabilities.json, status.json, observables.json
  restart.npz                 마지막 전체 상태; atomic replace
  frames/trajectories_*.npz    얇은 관측 snapshot; force-integral history는 생략
  plots/
    dynamics_overview/        density, BO/CBO/Pol, photon number, kinetic energy
    CTMQC_fields/             joint density, conditional mean CBO energy,
                              CBO character, 실제 MF/CT force, state density
    slice_diagnostics/        R/q CBO energy와 local population; R-energy 확대
    force_decomposition/      pair Q, CT force RMS, CT-only population rate,
                              energy drift
    numerical_checks.png
    render_report.json
각 family는 PNG와 MP4. PNG는 가장 가까운 실제 snapshot, endpoints도 포함한다.
movie도 실제 snapshot만 사용한다. PNG 선택은 전파 간격을 바꾸지 않는다.
2D만 density support를 따라 이동; line/color scales는 전체 선택 시간에 고정.
ROOT/<model>/<resonance>/dse_<...>/<case>/perspective_comparison/
  EL_POL_comparison.png, comparison.json
ROOT/ctmqc_campaign_report.json: case별 실제 시간/용량/상태.

기본 8개, Ntraj=10000, every=.1fs의 원시 frame 추정은 약 2.7 GiB(ZIP 이전).
압축에 따라 작아지지만 PNG/MP4/restart가 추가된다. 여유 4~6 GiB를 권장하며,
실제 용량은 plan과 pilot 파일 크기로 확인한다. grid full Psi/field archive는
생성하지 않는다. jobs는 전체 process pool의 상한이고 중첩되어 늘지 않는다.

개발 로컬 10,000 trajectories / 10-step(.0242fs) timing 참고:
  EL 약 .061 s/step, POL 약 .163 s/step (초기 정적 구간; 저장/렌더 제외).
  기본 paper 시간 창 8개를 단순 외삽하면 전파 합 약 2.7 CPU-worker hours.
  장시간 CT stiffness, CPU, rendering, jobs에 따라 변한다. 서버 측정이 아니며
  이 값으로 서버 완료 시각을 보장하지 않는다. 8개를 40fs로 통일하면 더 길다.

6. 구현 위치
-----------
core.py: Settings, hamiltonian, Engine.initial/evaluate/step, observables
quantum_momentum.py: density_quantum_momentum, coupled_terms
workflow.py: simulate, identity, build_tasks/preflight, CLI
diagnostics.py: mixed ensemble deposition, prepare_frame
render.py: overview/fields/slices/corrections/numerical_checks/render_case/comparison
tests/test_pen_ctmqc_2026.py: analytic force, norm, pair constraint, guard,
  Wigner sampling, fixed-R exact reference, dt, resume, density semantics

실행 예:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ctmqc-test-mpl \
  python -m unittest tests.test_pen_ctmqc_2026 -v
