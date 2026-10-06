# Model A 및 광자수 구현 점검 — 2026-10-06

## 범위와 현재 판정

새 `model_a/` package만 추가했다. 두 diabatic 전자 상태, DSE 없음, positive
marginal gauge이다. 원래 MCEF와 VSC propagation/postprocessing source는 수정하지 않았다.
**구현 및 small tests 통과이지, 논문 완전 재현 또는 production 수렴 PASS는 아니다.**
Hamiltonian/초기상태/출처/논문 표기의 모호성은 README에 명시했다.

## 계산 함수 위치, 수식, shape, 단위, 검증

- `model.py lines 12–40`: `Config`, `grids`; 논문 수치와 endpoint-excluded
  `(NR,), (Nq,)` grids, atomic units. invalid box/dt 검사.
- `model.py lines 43–56`: `potential`; 논문 Eq.27 및 Eq.5의 `omega*g*q*mu`,
  `(NR,Nq,2,2)`, Ha. 저자 QuantumModelLib의 q 의존 LM과 대조, Hermiticity test.
- `model.py lines 59–74`: `initial`; bare S1 times real Gaussian,
  `(NR,Nq,2)`, `sum |Psi|² dR dq=1`. 초기 P_S1와 norm test.
- `model.py lines 77–81`: `derivative`; `(ik)^order` FFT symbol, 입력 shape 유지,
  좌표 차수에 따른 inverse-length 단위. TDSE/EF에서 동일 연산을 재사용.
- `model.py lines 84–106`: `Propagator`; V/2–T–V/2 second-order split 및 H Psi,
  `(NR,Nq,2)`, dt au, action Ha*Psi. Hermiticity, dense expm/order,
  stationary-density test; 서버 GPU 사용 전 CPU/GPU 비교.
- `model.py lines 109–134`: `observables`; 전자 population, norm, energy,
  R>=4 population, signed flux, photon moments. scalar dict, atomic units.
  `p2=integral |Dq Psi|²`, `n_ph=omega*q2/2+p2/(2*omega)-norm/2`.
  진공/1광자/같은 밀도의 momentum-boost 상태 테스트.
- `factorization.py lines 13–16`: `ratio`; exact-zero denominator는 NaN, 임의 floor 없음.
- `factorization.py lines 20–83`: `analyze`; Psi=chi Lambda Phi,
  epsilon1=cond+geo_q+geo_R+GD, outer force=-dR epsilon2+alpha_t.
  qR field `(NR,Nq)`, outer `(NR,)`; scalar Ha, connection momentum,
  force Ha/a0. 순간 `dt Psi=-iH Psi`. 재구성/PNC/두 route/continuity,
  electronic constant-unitary invariance, alpha_t probe 검사.
- `factorization.py lines 86–101`: `diagnostics`; 상대밀도 threshold 세 개에서
  weighted residual 및 excluded mass. threshold는 probability budget과 다르다.
- `factorization.py lines 104–122`: `compact`; 미분이 끝난 map만 stride2로
  줄이며 Fig.4 단면과 outer line은 native resolution 보존.
- `run.py lines 18–29`: `atomic_json`, `atomic_npz`, `identity`; atomic checkpoint,
  source/settings SHA 검증. 데이터 함수가 아니므로 물리 단위 없음.
- `run.py lines 32–50`: `backend_check`; 같은 Hamiltonian/grid의 CPU/GPU
  action, wave, observables 오차. 16 steps; basis convergence와 구별.
- `run.py lines 53–117`: `simulate`; scalar histories + compact fields + six event
  waves; 체크포인트 history와 Psi를 함께 보존. 재시작/덮어쓰기 거부 테스트.
- `run.py lines 120–133`: `main`; 서버/로컬 CLI, 입력 수치 검증.
- `campaign.py lines 18–26`: `settings`; 실행 전 고정된 여섯 numerical controls.
- `campaign.py lines 29–60`: `compare`; 시간별 scalar 오차 및 event별 native EF
  common-grid 비교. interpolation은 비교에만 사용, potential offset fitting 없음.
  공통 격자 밖 확률질량과 field 비교 support 질량도 기록.
- `campaign.py lines 63–94`: `main`; 여섯 case 순차 실행, gate/그림/영상/광자수 비교.
  전체 시간창이 아닌 실행에는 PASS_NUMERICAL을 주지 않는다.
- `postprocess.py lines 11–21`: `main`; 저장 event waves만 read-only 재분석,
  새 폴더 compact fields 출력. 전파 재실행 아님.
- `photon_number.py lines 16–30`: `model_series`; Model A scalar history를 norm으로
  나눠 `<N>, <q²>, <p²>, n_q, n_p` 배열 `(Nt,)`로 로딩.
- `photon_number.py lines 33–66`: `vsc_series`; VSC 원본 `rho_Q`, photon energy,
  plan omega, status Q grid로 moments를 복원한다. `Q=sqrt(omega)q`,
  `n_q=<Q²>/2`, `n_p=Eph/omega-n_q`. 저장 nph와 대수적 cross-check.
  원래 Hamiltonian의 photon kinetic이 이미 Eph에 포함되어 있다.
- `photon_number.py lines 69–106`: `draw`, `main`; N/DeltaN/quadrature budget의
  세 패널, NPZ, provenance JSON. density-only archive는 거부한다.

## 시각화 함수

- `plot.py lines 21–22`: `load`, NPZ 안전 로딩.
- `plot.py lines 25–40`: `fig1`, 논문 Fig.1의 q-fixed/R-fixed static cuts.
- `plot.py lines 43–53`: `map_panel`, 상대 joint-density support와 contour.
- `plot.py lines 56–63`: `snapshot_maps`, Fig.3-style six times. GI와 full PG
  epsilon1을 별도 출력한다. GI를 total TDPES라고 부르지 않는다.
- `plot.py lines 66–84`: `draw_cuts`, Fig.4 cond/geo와 두 adiabatic references,
  실제 joint density는 독립 y축에 표시하며 cut별로 재정규화하지 않는다.
- `plot.py lines 87–152`: `render`, 4종 MP4, PNG, clipping metadata,
  photon plot. 2.5 au 간격 실제 frames 사용; sparse events로 movie 만들기 거부.

## 실행한 검증과 한계

- `test_model_a.py lines 13–134`: **14 tests PASS**. source 구현 테스트이지
  전체 시간창/모든 grid의 수렴 증명은 아니다.
- CPU 256×256, dt=.05, 0–1250 au diagnostic: 약 263초(전파+event 분석,
  dense movie 미포함). 최종 S1=.775360, N=.155196. norm drift 1.65e-12,
  energy drift 9.59e-8 Ha. q-edge 5.20e-7 및 continuity 6.35e-5 때문에
  **FAIL_INTERNAL**. 실패 자료를 보존하며 tolerance를 바꾸지 않았다.
- 종전 VSC 414-frame archive photon diagnostics: N(0)=5.82342567584,
  N(final)=7.49469017562; 0–39.959969 fs, min=4.09621920859,
  max=7.54459909477. 원본 status COMPLETE_NOT_CERTIFIED 그대로 유지.
- final smoke 128×128 0–5 au: photon 분해/새 파일/모든 movie 생성 경로 검사.
  이 작은 grid도 q-boundary/continuity 때문에 scientific gate는 FAIL이다.
- 보호 source SHA: baseline과 일치, regression 중 변화 없음.
- 기존 MCEF **184 tests 중 1개 기존 실패**:
  `tests.test_tdse_report.test_final_visualization_uses_shared_point_one_percent_focus`.
  renderer의 density-focus 기준과 test 기대값 차이가 새 package import 없이도
  재현된다. 기존 파일 보호를 위해 수정하지 않았다. 나머지 183 tests 통과.
- 기존 VSC **186 tests PASS**, 역사적 smoke PASS.
  일부 오래된 pytest-style function은 unittest count에 포함되지 않으므로,
  이 수치를 저장소의 모든 가능한 테스트 개수라고 주장하지 않는다.
- 따라서 **GLOBAL PASS 또는 완전 재현 완료라고 보고하지 않는다**.

원문 Fig.3과 full PG total scalar의 초기 모양 차이, 정확한 author initial
coefficient input 부재를 먼저 구분해야 한다. 수식과 GI/GD를 바꿔서 그림을
맞추지 않는다. 서버 campaign은 이 문제를 숨기지 않고 raw 비교 자료를 만든다.
