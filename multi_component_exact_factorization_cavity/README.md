# 실공간 MCEF를 계승한 cavity 확장

이 패키지만 새로 추가한다. 기존 MCEF·VSC·model-A의 코드를 수정하거나 결과를 덮어쓰지 않는다.
전자 2-state TDSE가 아니다. `Psi[R,x,Q]` 전체를 complex128로 전파한다.
`Q=sqrt(omega_c)*q_c`, 전자 `x`는 DST-I/Dirichlet, 양성자 `R`와 광자 `Q`는 FFT/periodic이다.
기존 heavy는 `+9.5 a0`에 고정하며, **기존 proton q를 새 R로 이름만 바꾼다**.

## 반드시 알아야 하는 모델 점검 결과

기존 실공간 설정의 heavy를 고정하면 ground BO surface는 **단일 우물**이다.
native `nx=359,nR=750`에서 내부 최소는 약 `R=-2.604 a0`이며 내부 이중우물 saddle은 없다.
따라서 기본 `R=0`은 전달을 계수하는 **관측면**이지 장벽이 아니다.
이 모델의 coherent transfer를 Li 논문의 double-well thermal reaction rate라고 부르면 안 된다.

사용자 선택에 따라 VSC 기본값은 실제 **단일 우물의 v=1**이다.

- `local_state_method="single_well"`: 실제 단일 우물의 v=1. 핵 운동량 kick 없이 시작한다.
- `local_wall_R=0.0`: 왼쪽 영역에 준비용 Dirichlet 벽을 두어 v=1을 구하고, 전파에서 벽을 제거한다.
  벽 제거 자체가 추가 초기조건 효과다. 이 준비법을 자연적인 이중우물 고유상태라고 해석하지 않는다.
- `local_state_method=null`로 선택을 미정 상태로 두면 VSC를 거부한다. PEN은 영향을 받지 않는다.

smoke도 같은 단일 우물 준비법을 쓰지만, 거친 격자의 수치 실행경로 검증일 뿐이다.

## Hamiltonian과 실제 초기조건

원자단위에서

\[
H=T_x+T_R+V_{\rm old}(x,R;R_{\rm heavy}=9.5)
 +\frac{\omega_c}{2}(-\partial_Q^2+Q^2)
 +\sqrt{\omega_c}\,g_{\rm cavity}\mu Q+\frac12g_{\rm cavity}^2\mu^2.
\]

`g_cavity`는 dipole-gauge 결합계수 lambda이며 ERF smoothing의 strong/weak와 무관하다.
`mu=R-x+left_fixed+heavy_fixed`; 대칭 고정 site에서는 `mu=R-x`다.
고정 이온의 dipole와 이온–이온 상수항은 각각 한 번 포함한다. 기본 DSE는 ON이다.
`dse=false`는 비물리적 불안정성을 일으킬 수 있는 비교/debug 옵션이지 production 기본값이 아니다.

PEN은 실제 기존 archive의 `args`를 추적했다:

`results/20260916/erf_fullgrid_BOcurvature_compact_qR_analysisBO2_100fs/multi_component_discrete_tdse_gpu.npz`

- 전자 bare S1, 핵 중심 −4, **밀도 표준편차** 0.3208418789467482, 질량 1836, 운동량 0.
- Gaussian 진폭은 `exp[-(R-R0)^2/(4 sigma^2)]`다.
- 전자 359개 내부점, box [−22,30], dx=52/360; 핵 750점, [−12.012,11.988), dR=.032.
- ERF 길이 lx=3.1, qx=5, Rx=4, qR=1; 고정 왼쪽 site −9.5.
- photon vacuum은 `pi^(-1/4) exp(-Q^2/2)`이며 실제 grid에서 정규화한다.

PEN 공명은 **정확한 R0**에서 bare S1–S0 gap, VSC는 선택한 bare local v1–v0 gap이다.
LP/UP splitting을 omega로 사용하지 않는다. `detuning`은 bare resonance에 대한 상대 변화다.
weak/strong은 각각 목표 splitting/omega=.01/.1로 보정하고 실제 작은 분석 Hamiltonian의
splitting, product character 및 Fock basis 확대 결과를 기록한다. 닫힌계이므로 linewidth 기준의
실험적 strong-coupling 판정은 아니다. 정확한 공명에서는 weak도 50:50 혼합일 수 있다.

native 보정 예: PEN omega=0.1010673388 Ha, VSC omega=0.00197425326 Ha.
VSC strong의 두 상태 교환 주기 추정은 약 770 fs이므로 40 fs는 초기 반응을 확인하는 구간이다.
긴 VSC 진동–광자 교환을 보려면 먼저 짧은 검증을 통과한 후 400–800 fs 등으로 확장하고
그 시간 범위의 격자·box 수렴을 별도 확인해야 한다. 실제 예상 실행시간은 서버 pilot에서 측정한다.
초기 bare vacuum은 결합된 cavity의 dressed ground state가 아니다. 영구쌍극자에 따른
sudden-coupling displacement도 photon number를 증가시킬 수 있어,
`photon_coherent_number=(<Q>^2+<P_Q>^2)/2`와 나머지 fluctuation 성분을 함께 저장한다.
photon number 증가만으로 진동 에너지가 광자로 전환되었다고 단정하지 않는다.

### 논문과 같은 점 / 다른 점

`paper/`의 *Exact factorization of the photon–electron–nuclear wavefunction:
Formulation and coupled-trajectory dynamics*, JCP **161**, 084112 (2024),
Sec. III, Eq. (27)–(28)와 Fig. 1/3/4를 확인했다. S1 준비, bare FC gap 공명,
cavity vacuum, CBO 비교 및 EF 분석 방향을 참고한다.
하지만 그 논문의 두 diabatic 상태, M=20000, 초기 R=2, DSE 생략을 복제하지 않는다.
본 확장은 기존 실공간 MCEF의 전자와 질량·Gaussian을 유지한다.
Li et al., *Quantum dynamical effects of vibrational strong coupling in chemical reactivity*
(2021)의 thermal-rate ensemble도 이번 coherent v1 준비와 다르다.

## 실행: smoke → short validation → production

저장소 root, MCEF 환경에서 실행한다. 새 output 이름을 사용한다.
자동 MP4 생성에는 **ffmpeg가 필수**다. `run_server.sh`는 모든 단계 시작 전에
`ffmpeg -version`을 확인하고, 없으면 계산을 시작하지 않는다. 필요한 경우 현재
conda 환경에 `conda install -c conda-forge ffmpeg`로 설치한 뒤 확인한다.

기존 미추적 연구 파일이 없는 **clean checkout 서버에서는 `smoke-portable`을 사용한다**:

```bash
conda activate MCEF
git pull
bash multi_component_exact_factorization_cavity/run_server.sh smoke-portable \
  results/20261023/cavity_smoke_server_v1
```

`smoke-portable`은 새 cavity 테스트 전체, 기존 MCEF 테스트, 기존 reference CLI를
검사한다. **기존 VSC 연구 테스트는 검사하지 않는 명시적 모드**이며, 해당 VSC 검증을
PASS로 기록하지 않는다. 서버의 `regression.json`에도 실행한 MCEF/CLI 검사만 남는다.
일부 기존 VSC 테스트는 미추적 `phase6_resume.py` 등의 연구 파일에 의존하여 clean
checkout에서는 import가 실패할 수 있다. 새 cavity TDSE 자체는 이 파일에 의존하지 않는다.
로컬 전체 연구 작업공간의 VSC **237개 PASS**와 portable 서버의 검증 범위는 다르다.

기존 VSC 연구 파일까지 모두 있는 서버에서는 원래의 전체 검사 모드를 사용한다:

```bash
bash multi_component_exact_factorization_cavity/run_server.sh smoke \
  results/20261023/cavity_smoke_full_research_v1
```

`smoke`는 기존 VSC 검사를 생략하지 않으며, 누락·실패하면 중단한다. 두 모드 모두
테스트 gate 이후 작고 거친 격자 6개 case를 실제 전파하며 자동 그림/MP4를 만든다.
그림의 시간은 매우 짧다. 물리 현상을 이미 재현했다는 결과가 아니다.

PEN용 수정 가능한 JSON을 만들고 bare calibration을 확인한다:

```bash
python -m multi_component_exact_factorization_cavity.workflow init \
  --out pen_native.json --set t_final_fs=40 --set field_every_fs=0.5 \
  --set validation_dir='"results/20261023/cavity_native_v1/validation"' \
  --set legacy_regression_file='"results/20261023/cavity_smoke_server_v1/regression/regression.json"'
python -m multi_component_exact_factorization_cavity.workflow calibrate \
  --config pen_native.json --out results/20261023/cavity_native_v1
bash multi_component_exact_factorization_cavity/run_server.sh validate \
  pen_native.json results/20261023/cavity_native_v1 pen all
bash multi_component_exact_factorization_cavity/run_server.sh production \
  pen_native.json results/20261023/cavity_native_v1 pen all 1
```

`field_every_fs=.5`는 용량 우선 81개 실제 프레임이다. 더 자연스러운 영상은 `.2`로
변경하여 **validation부터 새 config로** 실행한다. 보간 프레임을 물리 데이터인 것처럼 만들지 않는다.
dt 기본 .025 a.u.; 40fs는 약 66,147 step이다. 실행시간은 GPU pilot 측정 전에는 확정하지 않는다.

단일 case / 동시 3개:

```bash
bash multi_component_exact_factorization_cavity/run_server.sh production \
  pen_native.json results/20261023/cavity_native_v1 pen strong 1
# 위와 같은 output의 이미 완료된 case에는 다시 쓰지 않는다.
# 세 case를 동시에 시작할 새 output을 잡고 validation_dir도 그 경로로 맞춘 뒤:
bash multi_component_exact_factorization_cavity/run_server.sh production \
  pen_parallel.json results/20261023/cavity_parallel_v1 pen all 3
```

11GB GPU에 native 3개 동시 실행이 들어간다고 가정하지 않는다. 합산 GPU 예약량과 모든
case의 디스크량을 먼저 검사하며 넘으면 실행을 거부한다. **jobs=1 순차 실행이 기본**이다.
프로세스 3개가 같은 GPU에서 반드시 더 빠른 것도 아니다.

VSC는 새 JSON에서 기본값 `local_state_method="single_well"`을 유지하고 같은 순서에서
`pen`을 `vsc`로 바꾼다. 준비벽 비교실험은 `local_state_method="local_wall"`와
`local_wall_R=0.0`을 모두 지정해야 한다. component 격자, 질량, 위치, smoothing,
폭, analysis state 수는 JSON 또는 `--set KEY=JSON`으로 조절한다. 각 변경은 새 calibration과
short validation을 요구한다. snapshot 간격만 변경해도 config identity를 새로 기록한다.

재시작 / 기존 작은 캐시 재렌더:

```bash
python -m multi_component_exact_factorization_cavity.workflow run \
  --config pen_native.json --campaign pen --case strong --gpu --resume \
  --out results/20261023/cavity_native_v1/pen/strong
python -m multi_component_exact_factorization_cavity.workflow render \
  --out results/20261023/cavity_native_v1/pen/strong
```

## 자동 산출물과 저장 정책

각 `.../pen/strong/` 또는 `.../vsc/strong/` 아래:

- `config.json`, `metadata.json`, `status.json`: 확정 설정, source SHA256, git HEAD, 진행상태.
- `restart.npz`: 마지막 checkpoint 한 개. 원자적 임시파일 교체이므로 기록 중 잠시 2개 공간 필요.
- `observables.npz`: norm/energy/kinetic/광자수/밀도/폭/BO·pol·vib population/누락분/flux.
- `observables/obs_*.npz`: 프레임별 작은 journal. 전체 history는 checkpoint/종료 시에만 합쳐
  매 프레임 반복 재압축하는 이차적 I/O 병목을 피한다.
- `fields/`: 압축된 작은 EF 필드. 검증은 complex128/float64, 영상용 2D는 float32라고 명시.
- `metrics/`: float64로 평가한 PNC/bridge/복소 scalar/연속방정식 잔차와 support budget.
- `analysis_cache.npz`, `references.npz`: 한 번 계산하는 정적 분석 basis와 CBO/POL 기준면.
- `report/` 하위: 자동 PNG/PDF와 스트리밍 MP4. 실제 목록은 `render_manifest.json`.
- campaign root `cache/`, `schematics/`, coupling 비교 그림.

매 프레임의 full Psi는 기본 저장하지 않는다. `save_wave_every`를 0보다 크게 설정하면
그 step 간격으로 추가 저장하지만 비용을 스스로 늘리는 옵션이다. 임의의 과거 시점에서
새 고정밀 분석을 하려면 raw snapshot 또는 checkpoint에서 재전파가 필요하다.
`debug_fields=true`는 T1…T8 등을 모두 저장해 매우 커지므로 짧은 run에만 권장한다.
압축은 lossless deflate level1이며 저장 시간이 계산을 잠식하지 않게 했다.

## 그림 해석 / 정합성의 범위

positive marginal gauge에서 첫 관점은 `a,b,epsilon1`, 바깥 관점은 `alpha,epsilon2`다.
`Sq+Sadv+Srel`은 **partial_t |Lambda|^2**이며 complex Schrödinger RHS가 아니다.
별도의 `lambda_dt`와 T1…T8/중간 EOM 잔차도 내부에서 평가한다.

bridge는 기존 `PHASE7_NESTED_EF_DERIVATION.md` 및 `perspective_bridge_data.py`의 식:

\[
\alpha=\int dq_c\,|\Lambda|^2b,\qquad
\epsilon_2=\langle\epsilon_1\rangle_\Lambda+K_q+G_R^\Lambda+C_t,
\]
\[
K_q=\tfrac12\int(|\partial_q\Lambda|^2+a^2|\Lambda|^2)dq,
\quad G_R^\Lambda=\tfrac1{2M}\int(|\partial_R\Lambda|^2+(b-\alpha)^2|\Lambda|^2)dq,
\quad C_t=\int-i\Lambda^*\partial_t\Lambda\,dq.
\]

시간미분은 sparse-time unwrap이 아니라 그 순간의 **동일 Hamiltonian의 H Psi**로 얻는다.
몫의 해석적 미분에서 continuum product rule로 얻는 항등식 잔차와 native FFT current의
product-rule 오차는 따로 기록한다. 전자가 작다고 공간 격자 수렴이 입증되는 것은 아니다.
진짜 node는 제외 확률과 함께 기록하며 epsilon으로 채워 가짜 유한값을 만들지 않는다.
이번 패키지의 동적 출력은 PG이다. 전역 zero-vector gauge까지 완성했다고 주장하지 않는다.

핵 force는 `-partial_R epsilon2 + alpha_t`이며, 밀도의 양자포텐셜까지 포함한 국소 가속도와
구별한다. 정미 flux와 누적 backflow도 별개이며 개별 입자의 재교차 횟수라는 뜻이 아니다.

`epsilon1`의 CBO 비교 기준은 fixed(R,q_c)에서 전자를 대각화해 얻는다.
`epsilon2`의 fixed-R electron+photon PES는 전자–광자 branch이다.
**VSC 진동 polariton은 T_R까지 포함한 global level**이므로 R 의존 PES를 LP/UP로 개명하지 않는다.
주요 4개 상태로 닫혀 있다고 가정하지 않고 실제 outside plotted/basis fraction을 저장한다.
공명에서도 초기 LP/UP가 1/2라고 가정하지 않는다. 분석용 BO/Fock/vib 수렴은 따로 확인한다.
CBO reference는 정적 sample 격자에서 대각화한 뒤 표시용으로 보간한다. raw sample도 저장하고
`reference_R_points`, `reference_Q_points`로 해상도를 높일 수 있다. TDSE/EF 미분에는 이 보간을 쓰지 않는다.
보간 전 알려진 핵 scalar와 photon harmonic 항을 제거하고, native grid에서 정확하게 다시
더하므로 조화 퍼텐셜이나 Coulomb pole 자체를 선형보간해 생기는 가짜 굴곡을 피한다.
local CBO character는 retained BO 부분공간에서 계산한 분석용 투영이다. full-x CBO eigenvector에
대한 완전한 투영과 구별하며 outside fraction과 sampled full-x/reference 에너지 차이를 함께 저장한다.
VSC polariton 분석의 기본 matter matrix는 `phi_S0(x;R) chi_v(R)` 전체에 native nuclear kinetic을
작용시켜 투영하므로 BO derivative/DBOC와 off-diagonal 기여를 포함한다. 초기 bare-v1 및 공명 gap은
그대로 유지하고, 이 분석용 corrected gap은 따로 기록한다.

## 검증과 아직 필요한 계산

```bash
python -m unittest discover -s tests -p 'test_cavity_*.py' -v
python -m multi_component_exact_factorization_cavity.audit \
  --capture --outdir results/20261023/cavity_preservation_server_v1
python -m multi_component_exact_factorization_cavity.audit \
  --regression --outdir results/20261023/cavity_regression_server_v1
```

short validation은 같은 공간 격자의 dt/dt2 fidelity·핵 밀도·에너지·경계 확률 검사이며,
장시간 scientific PASS가 아니다. production 상태명도 `COMPLETE_NOT_CERTIFIED`이다.
장시간 결론에는 x/R/Q 해상도, Q-box, dt, 분석 BO/Fock/vib 수, local-state 경계, support를
각각 바꾼 수렴 검증이 필요하다. 반사/recrossing을 노이즈라고 자동 삭제하지 않는다.
오래된 protected manifest와의 기존 불일치는 기록만 하고 원본을 갱신하지 않는다.
