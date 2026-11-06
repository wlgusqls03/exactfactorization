# Gaussian bridge implementation — 실제 검증 기록 (2026-11-06)

## 범위와 판정

이 작업은 **공통 수학 코어 + 지원되는 MTE pilot**을 구현했다. 기존
MCEF/TDSE/CTMQC/G-CTMQC 파일과 기본값은 수정하지 않았다.
Gaussian CTMQC/TSH 및 autonomous EF-G/MLS 전파는 아직 지원하지 않는다.
원본에 없는 scalar/force/gauge/hopping closure를 발명하지 않았다.

실행 호스트는 로컬 `20235256-p01`, Python은
`/home/hbji/miniconda3/envs/MCEF/bin/python`이다. CPU/complex128,
BLAS/OMP 각 1 thread를 사용했다. 원격 서버 접속·GPU 실행은 하지 않았다.
`COMPLETE_NOT_CERTIFIED`는 실행/guard 통과이지 물리적 수렴 인증이 아니다.

## 확인한 자료와 코드

수정 전에 읽은 계약: `paper/MCEF_Gaussian_Trajectory_Codex_Prompt.md`,
reference `.tex` 전체 및 PDF 19쪽, handwritten
`CTMQC and another approximation of MCEF_261106_165152.pdf` 16쪽,
추가된 `paper/11_3.pdf` 5쪽. 현재 `sources/` 디렉터리는 없다.
수식 registry의 explicit equations를 우선했고, handwritten sign/번호와
달라지는 곳은 README의 transcription 항목에 기록했다.

특히 `11_3`의 amplitude projection을 density least squares로 바꾸지 않았다.
PNC를 유지하는 real contraction, 독립 Gaussian 변수의 weighted matrix,
outer residual operator의 minus-alpha, 물질미분의 한 번만 적용되는
R-advection을 별도 테스트했다. basis derivative의 D를 d 원소의 제곱으로
대체하지 않는다. MLS common gauge는 호출자 계약이며, 임의 phase alignment를
물리적 gauge 변환인 것처럼 수행하지 않는다.

새 파일의 구현 위치(이 커밋 기준):

| 파일 | 줄 범위 | 역할 |
|---|---:|---|
| `gaussian.py` | 80–188 | 독립/변환 파라미터, discrete-normalized density/Jacobian |
| `gaussian.py` | 189–334 | support, complex C update, rank-aware amplitude projection, 실제 G 변화율 |
| `mls.py` | 19–220 | current-R factorial polynomial fit, covariance/scatter 및 실패 진단 |
| `heavy_density.py` | 87–177 | population-weighted density, posterior log derivative, 별도 uncoupled intercept |
| `fields.py` | 31–113 | 현재 G의 a,b,alpha,b-alpha, complex/quotient/bridge 검증 |
| `fields.py` | 116–186 | full-positive-gauge scalar 및 continuity reference 평가만 |
| `reference.py` | 16–87 | R1/R4/R3, internal basis projection/lift, unsupported backend guard |
| `models.py` | 69–265 | 기존 분자 potential, B1 Hamiltonian/force, 초기상태, observables |
| `run.py` | 33–250 | 설정, 초기 fit, stage별 갱신, DOP853/RK4, raw guards |
| `run.py` | 273–444 | 진단/저장, source hash, resource estimate, checkpoint/resume |
| `run.py` | 447–483 | 독립 CLI |
| `renderer.py` | 87–456 | 공통 scales, summary/snapshots/movie/manifest |
| `validation.py` | 27–253 | fixed-R reference, box/MLS/manufactured-dt 검증, 그림 |
| `tests/test_gaussian_trajectory_*.py` | 각 파일 전체 | 9개 새 테스트 파일 |

기존 입력 모델 연결: `pen_perspectives_2026/model.py`, `resonance.py`,
`pen_ctmqc_2026/core.py:hamiltonian`, `multi_component_exact_factorization_cavity/core.py`의
실공간 potential/kinetic. **PEN은 기존 두 diabatic 전자상태 benchmark**이며
photon은 real q grid다. MCEF는 x real grid를 유지한다.

## 실제 테스트

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_gaussian_trajectory_*.py' -b
python -m unittest discover -s tests -p 'test_pen_perspectives_2026.py' -b
python -m unittest discover -s tests -p 'test_pen_ctmqc_2026.py' -b
python -m unittest discover -s tests -p 'test_discrete_mcef.py' -b
```

- 새 core/MLS/density/fields/models/driver/render/reference/snapshot: **69 tests**.
- 기존 PEN perspectives: **17 tests** 통과.
- 기존 CTMQC: **12 tests** 통과.
- 기존 discrete MCEF: **7 tests** 통과. 실제 Shin–Metiu 모델 구성과
  factor recombination, TDSE Hermiticity/짧은 RK step을 포함한다.
- 기존 regression은 총 **36 tests**이며, 기존 코드 파일의 diff는 없다.
- restart: 고정 RK4 bitwise, DOP853 tolerance 기반 비교
  (동일 state의 weighted G L2 차이 < 1e-10).
- 초기 underflow rank regression: density floor 없이 representable support에서
  rank를 검사. 초기화 실패 시 config/status/failure가 남고 checkpoint는 없음.

수식 검사에서 K=1 matrix 오차 약 1.8e-15, moment rate 1.7e-16,
실제 projected G-rate 항등식 1.1e-16, PNC rate 1.2e-15였다.
모델 action은 Hermitian이며 matched force는 독립 finite difference로 검사했다.
동일 real-q Hamiltonian을 기존 Fock 공간에 투영한 read-only 추가 검사에서
action 최대 차이 7.5e-13, force 차이 1.1e-15였다.

## 짧은 self-consistent pilot과 비교 기준

`results/20261106/gaussian_bridge_pilot_v2/`에 실제 완료한 비교가 있다:

| 경로 | 실제 종료 fs | 전체 에너지 최대 변화 Ha | saved norm/PNC 최대 | PNG/MP4 포함 wall s | 폴더 MiB |
|---|---:|---:|---:|---:|---:|
| `pen_native` | 0.100142 | 5.6e-17 | 8.9e-16 | 12.1 | 3.59 |
| `pen_gaussian_K1` | 0.100142 | 5.4e-10 | 2.2e-10 | 34.4 | 3.66 |
| `pen_gaussian_K2` | 0.050313 | 3.8e-8 | 1.0e-9 | 62.0 | 3.65 |
| `mcef_native` | 0.020077 | 6.9e-17 | 4.4e-16 | 13.0 | 4.46 |
| `mcef_gaussian_K1` | 0.020077 | 5.5e-11 | 1.1e-15 | 16.7 | 4.50 |

여기서 native는 **full-internal MTE**, full quantum heavy TDSE가 아니다.
PEN의 native/K1 matched trajectory-averaged G L2 차이는 최대 1.02e-5,
MCEF는 3.51e-7이었다. K2는 초기 fit 자체에 L2 오차 0.00493과
진공 photon number 약 9.89e-5가 있으므로 K1보다 좋다고 판정하지 않는다.
이 표는 압축 추가비용을 숨기지 않는다. 현재 작은 PEN pilot에서는
projection 경로가 native보다 느리며 장시간 속도 우위는 입증하지 않았다.

최종 rank/실패기록 보완 후 새 코드로 재실행한 경로는
`results/20261106/gaussian_bridge_final_v2/{pen,mcef,rabi}/`다.
최종 실행의 실제 wall/폴더 크기는 PEN 35.56 s / 3.64 MiB,
MCEF 17.56 s / 4.48 MiB, Rabi 16.06 s / 3.23 MiB였다.
세 결과 모두 metadata source hash가 최종 Python 소스와 일치함을 확인했다.
각 폴더의 `status.json`은 **모든 accepted step**의 norm/PNC 최대치를
저장하므로, 일부 저장시각만 보는 summary보다 엄격하다. 예를 들어 PEN
0.1 fs에서 saved error는 약 2.16e-10이지만 accepted-step 최대는 약
2.07e-7이다. guard 1e-5 이하라는 사실을 수렴 인증으로 혼동하지 말 것.
Rabi trajectory pilot은 약 0.02 fs의 실행검사이며 Rabi period 검증은 아니다.

직접 확인한 출력 종류:

- `plots/summary.png`: population, photon, energy, projection/PNC/MLS, Gaussian 변수.
- `plots/snapshot_*.png`: bridge/quantum q/heavy density, a,b,alpha,b-alpha,
  raw quantum momentum, time marker.
- `plots/diagnostics.mp4`: H.264, 1610×1610 고정 프레임. 최종 PEN은 5 frames,
  0.625 s. 이는 short pilot의 시각화 검사이며 긴 발표 영상이 아니다.
- 각 PNG를 열어 글자/범례/축을 확인했고 ffprobe로 movie encoding을 확인했다.

## 별도 reference/convergence 그림

실제 실행 위치: `results/20261106/gaussian_reference_validation_v1/`.
`metrics.json`, `convergence.png`, `convergence.pdf`를 생성했다.

1. 고정 R=2, omega=0.17, 128차원 실제 real-q Hamiltonian의 eigensolution
   기준으로 6.5162 fs를 평가했다. g=0에서 최대 S0 population 1.61e-27,
   g=0.02에서 0.997807. norm 오차 1.07e-14, energy drift 4.05e-15 Ha.
   이는 고정핵 참조이며 6.5 fs heavy-trajectory 실행이 아니다.
2. photon half-box 8/12/16/20에서 vacuum action 결함은
   8.26e-3/1.15e-5/1.04e-9/6.49e-15 Ha.
3. manufactured phase-only RK4의 dt 0.5→0.0625에서 state 오차
   1.12e-6→2.74e-10. 이것을 stiff full-grid PEN의 dt 수렴으로 부르지 않는다.
4. nonuniform current-R MLS의 첫째/둘째 도함수 수렴 그림과 known-Gaussian
   QD snapshot 테스트를 별도로 실행했다. 미래 QD 상태는 전파 RHS에 쓰지 않았다.

## 실패 기록과 남은 검증

`gaussian_bridge_pilot_v1/`의 실패한 RK4 결과·checkpoint는 삭제하지 않았다.
저밀도 꼬리에서는 G가 작더라도 C=G/Lambda가 민감하고, off-PNC mode에는
deltaN_dot=-2(r/Lambda)deltaN가 생긴다. DOP853 및 동일 guard로 재시험했지만
이 사실은 **더 긴 구간에서의 안정성을 보장하지 않는다**.

아직 미완료:

- 물리적 branching/reflection 길이의 MCEF/PEN 및 장시간 수렴;
- 동일 long-time 조건에서 dt/Nq/Ntraj/MLS degree-neighbors/K/initial fit의
  독립 수렴 및 비용 비교. 현재는 formula/manufactured/reference/초단기 검사;
- verified full-q CTMQC coefficient/force/channel-history/gauge adapter;
- TSH active surfaces, hop probability, rescale, frustrated-hop 처리의 검증된 adapter;
- EF-G/MLS의 matched epsilon2, chi phase, heavy force closure;
- CPU 외의 GPU 경로, 더 일반적인 질량/분자 파라미터 변경 interface.

따라서 exact epsilon1/epsilon2 및 Sq/Sadv/Srel 영상을 허위로 채우지 않는다.
F6/F8 함수는 **이미 해당 gauge의 exact/reference 입력이 있을 때**만 사용한다.
지금 출력한 a,b,alpha 역시 current-R MLS와 공통 gauge에 의존하는 근사장이다.
heavy density 그림은 실제 복소 chi의 재구성이 아니다. 여러 history가 같은
R에 오는 multistream 영역에서 단일 exact field가 성립한다고 주장하지 않는다.

## 재현/서버와 git

실행 명령은 같은 디렉터리 `README.md`에 있다. 우선 거기에 있는
0.1 fs PEN / 0.02 fs MCEF를 서버에서 재현한 뒤 검증 구간을 늘린다.
현재 측정은 setup/render 비중이 큰 CPU short pilot이므로 40 fs/10000
trajectory 시간으로 선형 외삽하지 않는다. 큰 메모리/저장량은 `--estimate`로
먼저 점검한다. 기존 결과 폴더를 덮어쓰지 않으며 resume는 source hash까지
일치해야 한다. 소스 수정 전후 결과를 한 checkpoint에 연결하지 않는다.

작업 브랜치: `codex/gaussian-bridge-trajectories`.
작업 시작 HEAD: `8914c568b403361acba982e1df5e7cc049e5b517`.
pilot metadata의 commit은 실행 당시 HEAD이며 실제 코드는 `source_sha256`으로
따로 기록했다. commit 이후 동일 .py 파일의 hash는 변하지 않는다.
이 커밋에는 새 namespace와 새 9개 테스트만 포함하고, 사용자 `paper/`,
기존 untracked VSC 파일, 계산 결과는 포함하지 않는다. push는 수행하지 않는다.
