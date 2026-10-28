PEN perspectives 2026 — two-diabatic-state / finite-Fock benchmark
================================================================

1. 무엇을 구현했는가
-------------------
기존 2024 model_a, 실공간 PEN, MCEF 코드는 수정하지 않았다. 2024 model A와
같은 두 parabolic diabatic potential 계열을 사용하되, 다음 논문의 2026 설정을
새 namespace에 구현했다.

Magi, Schürger, Lauvergnat, Agostini, "Different perspectives on the exact
factorization for photon–electron–nuclear systems", PCCP (2026).
https://doi.org/10.1039/D6CP02020H
https://arxiv.org/html/2604.20538v2
저장소 paper/molecular vibration polariton/Different perspectives ...pdf:
6쪽 Table 2 및 Eq.21–27; 7쪽 양자 계산 격자 설명; 9쪽 POL 초기화 설명.

반드시 구분:
* 논문의 quantum reference: q와 R 각각 500-point grid, dt=0.1 a.u.
* 논문의 POL trajectory 표현: 전자 2 × photon n=0,1 -> 4 states.
* 이번 요청: TDSE 자체를 전자 2 × photon n=0,1에 제한한 양자 계산.
  따라서 "2026 Hamiltonian의 2-Fock TDSE"이지, photon real-grid benchmark를
  수치적으로 완전히 재현했다고 주장하지 않는다. CTMQC/TSH/MTE도 재구현하지 않았다.
* DSE off + strong가 논문 파라미터에 해당한다. DSE on, weak, zero는 추가 대조군.
* 실제 전자 좌표 x도 Shin–Metiu proton 질량도 없는 diabatic model이다.
  질량 20000은 이 모델의 일반화된 핵 좌표 R에 대한 값이다.

2. 기본 물리/격자 설정
--------------------
nonadiabatic preset (Table 2):
  M=20000, k=0.02, a=3, b=0.01, R1=6, R2=2, R3=3.875,
  Delta=0, Z=1, mu12=-1, omega(on)=0.17, omega(off)=0.097.
rabi preset: R1=R2=2, Delta=0.17; omega(on)=0.17, omega(off)=0.12.
모든 값 atomic units. omega=0.17을 bare gap≈0.16으로 임의 재조정하지 않는다.

핵 격자: [-2,10), 500점, dR=0.024. periodic FFT, 끝점 제외.
논문은 범위/점수는 명시하나 끝점 포함 여부는 명시하지 않으므로 이 FFT convention은
본 구현의 명시적 선택이다. 시간간격 0.1 a.u.=0.0024188843266 fs.
propagation: static 4x4 potential exponential + nuclear FFT, Strang splitting.
photon: n=0,1 (두 상태, 최대 광자수 1). number 자체를 평균 1로 고정하는 것은 아니다.
q_c는 propagation grid가 아니다. 기본 그림에서는 [-10,10) 160점으로 HO 함수를
해석적으로 복원한다. --nq-display 500은 논문 q-grid의 점수/간격을 표시용으로 사용한다.
q방향 미분은 HO 함수의 해석적 미분; R방향 미분은 native FFT.

초기상태: 논문 Eq.27의 bare adiabatic S1(R) × real nuclear Gaussian × |0>.
  nuclear Gaussian 중심 R2=2, 추가 momentum kick 없음.
  amplitude exp[-(R-R2)^2/(2 sigma_R^2)], sigma_R=(M k)^(-1/4)=0.22360679775.
  density 표준편차는 sigma_R/sqrt(2), 초기 <N>=0.
Eq.25의 q-dependent electronic state와 Eq.27의 bare-state 표기는 같은 초기준비인지
주의가 필요하다. 이 코드는 Eq.27을 명시적으로 선택하며 초기 LP/UP 반반을 가정하지 않는다.

Hamiltonian:
  H_BO(R) = [[k/2 (R-R1)^2, b exp(-a(R-R3)^2)],
             [b exp(-a(R-R3)^2), k/2 (R-R2)^2+Delta]]
  D(R) = Z R I + mu12 sigma_x
  H = T_R + H_BO tensor I + I tensor omega(N+1/2)
      + omega*g*D tensor (P q_c P) + DSE
  DSE = (g^2/2)(D @ D) tensor I (on일 때).
g=zero:0, weak:0.001, strong:0.01 (paper g convention, atomic units).
weak=0.001은 논문 수치가 아니라 10배 작은 coupling 대조군이다.
DSE를 elementwise square로 계산하지 않는다. 2 Z R mu12 교차항도 포함한다.
이 정의는 모델이 지정한 2x2 dipole 연산자의 DSE이다. 미지의 full-electron
dipole-square를 축약한 것과 동일하다고 주장하지 않는다.

3. photon number와 polaritonic state
----------------------------------
<N>는 정확한 diag(0,1) expectation. 연산자 검증에서는
  P q^2 P != (P q P)^2
를 지킨다. 최고 Fock 상태에서 잘린 q 행렬을 제곱하면 photon number가 틀린다.
논문의 Eq.28과 같은 bare Pauli-Fierz oscillator number를 출력한다.
검출기의 gauge-transformed photon count와 무조건 동일하다는 주장은 하지 않는다.

각 R에서 4x4 H-T_R를 대각화하고 에너지순 Pol0~Pol3로 표시한다.
이는 |S0,0>, |S0,1>, |S1,0>, |S1,1>를 다시 대각화한 eigenstates이며,
Pol1/Pol2를 모든 R에서 고정 LP/UP character라고 가정하지 않는다.
polaritonic_composition.png는 각 Pol의 bare BO x Fock 성분을 실제 계산한다.
nf>2 config는 truncation audit용으로 지원; 기본은 반드시 nf=2.

4. MCEF 진단의 중요한 제한 및 처리
---------------------------------
Psi(R,q_c,e)를 finite-Fock 계수에서 복원한 후 positive marginal gauge:
  Psi = Phi_e(q_c,R) Lambda_R(q_c) chi(R), F=sqrt(sum_e |Psi|^2).
기존 model_a/factorization.py 및 cavity/fields.py와 같은 quotient derivative,
instantaneous Psi_t=-i H Psi를 사용한다. sparse phase unwrap을 사용하지 않는다.

finite-Fock projection은 연속 q_c에서 nonlocal 연산자이므로
  delta H Psi = (H_Fock - H_local_grid) Psi
를 따로 계산한다. epsilon1은 그 실수 conditional quotient를 포함한 effective
scalar이며, GI components 그림에서 projection correction을 별도 표시한다.
local CBO curves는 unprojected local-q 전자 Hamiltonian의 비교 기준선이다.
exact scalar가 CBO와 동일하거나 CBO를 따를 필요는 없다.

조건부 밀도 수송:
  partial_t |Lambda_R|^2 = S_q + S_adv + S_rel + S_projection
  S_q = -partial_q(a |Lambda_R|^2)
  S_adv = -(alpha/M) partial_R |Lambda_R|^2
  S_rel = -1/(M |chi|^2) partial_R[|chi|^2 |Lambda_R|^2 (b-alpha)]
  S_projection = 2 |Lambda_R|^2 Im[sum_e Psi_e* (delta H Psi)_e / sum_e |Psi_e|^2].
S_projection은 residual에 맞춰 정의한 값이 아니라 H action에서 독립 계산한다.
기존 세 항만 더해 놓고 exact continuity라고 표시하지 않는다.
이것은 i partial_t Lambda가 아니라 partial_t |Lambda|^2의 분해다.

outer epsilon2, alpha, force=-partial_R epsilon2+partial_t alpha는 native Fock
coefficient의 R derivatives로 직접 구한다. bridge 검사는
  alpha = integral |Lambda|^2 b dq
  epsilon2 = integral |Lambda|^2 epsilon1 dq + K_q + G_R^Lambda
를 사용한다. positive-real Lambda에서 GD bridge의 실수부는 0이다.
K_q에는 |partial_q Lambda|^2 및 a^2 |Lambda|^2가, G_R^Lambda에는
|partial_R Lambda|^2 및 (b-alpha)^2 |Lambda|^2가 들어간다.
단순 평균 epsilon1만으로 outer scalar를 재구성하지 않는다.

q plot box [-10,10]의 작은 HO tail 누락으로 bridge quadrature residual이
~1e-7일 수 있다. audit는 [-16,16] 재적분과 비교한다. 이는 TDSE grid 변경이 아니다.
near-node ratio들은 일반적으로 불안정할 수 있다. 2D는 occupied support(1e-5)+margin
moving window, 선은 density support mask, 2D background는 window 전체에 표시한다.
fixed scales는 영상 전체에서 유지; GI zoom은 별도 정적 그림이다.

5. 실행: 저장소 루트, conda MCEF
-----------------------------
새 계산과 기존 real-grid PEN checkpoint는 호환되지 않는다. 새 폴더를 사용한다.

가벼운 전체 경로 smoke (6개, 0.1 fs, 그림/영상까지):
  bash multi_component_exact_factorization/pen_perspectives_2026/run_server.sh smoke \
    results/20261028/pen2026_smoke_v1 --jobs 3

기본 40 fs, 총 6개, 자동 PNG/MP4, CPU 3개 작업 동시 실행:
  bash multi_component_exact_factorization/pen_perspectives_2026/run_server.sh run \
    results/20261028/pen2026_40fs_v1 --end-fs 40 --every-fs 0.5 --jobs 3

50 fs로 계산하려면 --end-fs 50. 모든 작업을 순차 실행하려면 --jobs 1.
하나만 계산하려면 --case strong --dse off (새 output root 권장).
--weak-g, --strong-g로 g 변경. --preset rabi, --resonance off도 지원.
--config my_config.json으로 Config의 모든 분자/격자 값을 변경할 수 있다.
case별 g와 dse는 위 CLI case 옵션이 config의 g/dse를 override한다.

40 -> 50 fs 연장 (같은 model/config/save cadence만 허용):
  bash multi_component_exact_factorization/pen_perspectives_2026/run_server.sh run \
    results/20261028/pen2026_40fs_v1 --end-fs 50 --every-fs 0.5 --jobs 3 --resume

그림만 다시 생성, 40 fs까지 자르기 (재전파 없음):
  bash multi_component_exact_factorization/pen_perspectives_2026/run_server.sh render \
    results/20261028/pen2026_40fs_v1 --t-max-fs 40 --jobs 3

빠른 영상 preview: --movie-stride 10 (중간 프레임 생략임을 render_report에 기록).
모든 실제 저장 프레임: 기본 --movie-stride 1. 시간 보간으로 가짜 프레임을 만들지 않는다.
더 촘촘한 새 계산: --every-fs 0.25 (저장량/영상 생성량 약 2배).
PNG만: --no-movies; 전파만: --no-render. 렌더링 시 DPI 기본85, snapshot최소140.
각 case의 PNG는 시작/중간/끝 3시점만 저장하여 대량 PNG 낭비를 피한다.

정확도 검사 (40 fs, 여섯 조건 dt/R; strong는 nf=2/4/8 비교):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m \
    multi_component_exact_factorization.pen_perspectives_2026.audit \
    --end-fs 40 --out results/20261028/pen2026_audit.json

6. 출력
-------
ROOT/dse_{off,on}/{zero,weak,strong}/
  config.json, provenance.json, status.json, observables.json
  restart.npz (원자적 갱신, Hamiltonian/config/source hash 검사)
  references.npz (BO/Pol eigenpairs)
  coefficients/wave_XXXXXXXX.npz (작은 원시 Fock 계수만 저장)
  plots/                         (time cut시 plots_to_40fs/)
    A_diagnostics_clean/         epsilon1,a,b / epsilon2 / slice / b,alpha
    B_diagnostics_clean/         source decomposition + projection source
    epsilon1_slice_analysis/     R,q cuts + photon number + CBO populations
    dynamics_overview/           핵/photon density, BO/Pol populations 분리
    outer_nuclear_effective/     epsilon2, force, Pol wavepacket, Pol populations
    bridge_consistency/          direct/reconstructed/residual PNG
    GI_components/               GI components 및 reference zoom PNG
    polaritonic_composition.png
    consistency.json, render_report.json
ROOT/comparison/six_case_dynamics.png, polaritonic_surfaces.png
ROOT/cost_report.json, server.log

product_R_ge_4 = integral_4^Rmax |chi|^2 dR / norm;
signed_flux_R4 = Im sum_en Psi* partial_R Psi / (M norm), R=4에서 평가.
일반화된 핵 좌표의 분할 확률/flux이다. 항상 irreversible reaction yield는 아니다.
R>=4 point mask 합은 경계 한 셀 때문에 O(dR) bias가 생겨, trigonometric
interpolant의 구간 적분으로 partial-cell contribution까지 포함한다.

7. 비용 / 검증 범위
------------------
실측은 MEASURED_VALIDATION.txt 및 결과 cost_report.json을 참고한다.
Fock coefficient 1 frame = 500*2*2*16 bytes = 32,000 bytes, 압축 전.
40 fs / 0.5 fs cadence: 81 frames (dt 격자 반올림), 6개 원자료 약 15.6 MiB.
초기 run은 지정 end를 넘는 가장 가까운 dt까지 진행할 수 있다 (~0.00242 fs 이내).
40 fs까지 render cut하면 엄밀히 40 fs를 넘는 마지막 frame은 제외한다.
checkpoint는 하나만 유지, history도 작은 coefficient snapshots만 저장한다.
비용 대부분은 matplotlib/MP4이며 GPU가 필요하지 않다.
11 GB GPU를 사용하지 않는 4-channel CPU 계산이다. jobs3에서는 CPU/RAM만 증가한다.

COMPLETE_NOT_CERTIFIED는 해당 finite-Fock TDSE 실행 완료라는 뜻이다.
수치 norm/energy conservation 통과 != photon basis 수렴 != 논문 곡선 완전 재현.
strong nf2 vs nf4 차이는 따로 보고하며, 두 Fock 상태 제한을 몰래 완화하지 않는다.
