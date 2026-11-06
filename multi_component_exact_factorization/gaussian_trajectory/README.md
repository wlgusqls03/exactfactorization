# Gaussian bridge trajectory pilot — 구현 및 검증 기록

이 모듈은 기존 MCEF/TDSE/CTMQC/G-CTMQC를 바꾸지 않는 별도 연구 경로다.
`q`는 공통 양자 격자이고, 고전 표본은 `R,P_R`뿐이다. 기본 저장 상태는
`R,P_R,z,C(q,k)`이며 `G=Lambda*C`는 매 RHS에서 재구성한다.

## 구현 범위와 현재 지원 상태

| 경로 | 현재 상태 |
|---|---|
| 공통 Gaussian 밀도·복소 계수 갱신·parameter 투영 | 구현, 수식 단위검사 완료 |
| MTE + Gaussian bridge projection | 구현, 짧은 self-consistent pilot 수행 |
| 무압축 full-internal MTE | 같은 Hamiltonian/힘/초기 준비의 비교 기준 |
| 전자–proton–heavy MCEF | 기존 실공간 potential, 전자 x-grid/proton q-grid, heavy만 trajectory |
| PEN nonadiabatic / Rabi | 기존 2026 **두 diabatic 전자 상태** 모델, photon은 실제 q-grid. real-electron PEN과 혼동 금지 |
| native 내부기저 변환 B3–B5 | round-trip 및 basis-motion 검사 함수 제공 |
| 새 Gaussian CTMQC / TSH 전파 | 별도 `channel_run` CLI: CTMQC 채널식 / FSSH + 선택적 Gaussian 투영. 아래 제한사항 참조 |
| EF-G/MLS exact outer operator | R1/R4 식 평가와 manufactured 검사만 지원. 미정 scalar/gauge/force를 만들어 전파하지 않음 |
| exact epsilon1/epsilon2, exact Sq/Sadv/Srel | trajectory density만으로 생성하지 않음. F6/F8는 별도 reference 함수 |

기존 G-CTMQC의 native 실행 경로는 그대로 남아 있다. 기존 감사
`results/20261104/gctmqc_paper_equation_audit_v1/README.txt`에서 보고한
계수/힘 prefactor와 componentwise RK 문제를 무시하고 새 공간의 검증된
backend라고 선언하지 않았다. 새 경로는 저자 실행파일과 구분한 독립 수식 구현이다.
TSH는 Tully FSSH의 명시적 baseline을 사용한다. 기존 `run.py`는 계속 MTE 전용이다.

## CTMQC / TSH 추가 경로 (2026-11-06)

`channel_run.py`는 같은 초기상태·Hamiltonian·P3/P8 Gaussian 모듈을 사용한다.
`internal_channels.py`는 **q kinetic을 포함한 전체 내부 Hamiltonian**을 대각화한다.
작은 시험 격자의 모든 채널을 유지하고 round-trip을 검사한다. 4-state truncation이나
고전 photon 대체는 없다. MCEF의 전자는 x-grid이고 PEN benchmark는 기존 두 diabatic
전자 상태다. 완전 기저에서 material NAC와 basis-motion 항을 함께 소거하여,
공통 primitive G에 `-i H G + U*(CT correction)`을 적용한다.

- CTMQC: Agostini et al. (2016), DOI `10.1021/acs.jctc.5b01180` 식 (28),(29);
  Villaseco Arribas et al. (2022), DOI `10.3390/molecules27134002` 식 (14)–(18).
  계수의 CT항과 힘의 CT항은 같은 force history를 쓴다. `fdot=-dE/dR`, 임의 reset 없음.
- 기본 `quantum_momentum_mode=density`: state-resolved heavy Gaussian density의
  posterior-weighted log derivative. `pair_intercept` 선택시 2022 식 (26),(27)의
  pair별 corrected momentum을 따로 저장한다. denominator cancellation은 기본 중단;
  `density_fallback`은 명시 선택만 허용하고 zero-transfer 조건 상실을 기록한다.
- TSH: Tully (1990), DOI `10.1063/1.459170` FSSH. native NAC outgoing flux만 hopping에
  사용한다. active state는 초기 channel population에서 sampling. 1D 에너지 보존
  momentum rescaling, frustrated hop 기본 `keep`/선택 `reverse`, decoherence/collapse 없음.
  확률 합>1이면 중단한다. Gaussian 압축으로 생긴 population 변화는 hop 확률에 넣지 않는다.
- 채널 label은 공통 R 중심에서 overlap continuation하고 매 step 이어 간다.
  모호한 overlap 또는 derivative-coupled degeneracy는 중단한다. 완전한 degenerate-subspace
  전파를 구현했다고 주장하지 않는다.
- RK4의 매 stage에서 전체 ensemble의 Q/history/Gaussian rate를 재계산한다.
  TSH event는 각 dt 경계, 확률은 interval 시작점 flux다. event dt와 ensemble 수렴은 별도 필요.
- Gaussian 압축을 켜면 **실험적 CTMQC/FSSH + Gaussian projection**이다. native 저자
  G-CTMQC와 동등하거나 정확한 EF-G/MLS closure라고 부르지 않는다.

짧은 구현 검사 명령(장시간 물리 계산용 수렴 격자가 아님):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m multi_component_exact_factorization.gaussian_trajectory.channel_run \
  --method ctmqc --model pen-nonadiabatic --compression gaussian \
  --ntraj 7 --nq 32 --dt-au 0.01 --end-fs 0.01 --every-fs 0.005 \
  --out results/20261106/channel_ctmqc_pilot --movies
python -m multi_component_exact_factorization.gaussian_trajectory.channel_run \
  --method tsh --model mcef --compression gaussian \
  --ntraj 7 --nx 9 --nq 16 --qmin -6 --qmax -2 \
  --dt-au 0.01 --end-fs 0.01 --every-fs 0.005 \
  --out results/20261106/channel_tsh_mcef_pilot --movies
```

무압축 비교는 새 outdir에서 `--compression native`로 실행한다. CT corrected-Q는
`--quantum-momentum-mode pair_intercept`. component 수는 `--components`로 선택하며
K>1 초기 fitting 오차/rank는 별도 검사한다. mask 실험은
`--support-relative 1e-8 --max-excluded-mass 1e-5`를 명시한다.
**mask 밖 C 고정은 방정식을 바꾼다. 작은 excluded mass만으로 정확하다고 보장하지 않는다.**
실제 g=0 시험에서는 이 정책으로 photon number/energy 오차가 발생했고
threshold 증가가 항상 안정화하지 않았다. 기본 support=0은 유지한다.

출력: `frames/`, `checkpoint.npz`, `status.json`, `metadata.json`, `timeseries.npz`,
기존 형식 `plots/`, 새 `channels/summary.png`, `channels/snapshot_*.png`,
`channels/channels.mp4`(`--movies`). raw/pair Q와 force history, active states,
hop counts, Gaussian 압축의 channel-population-rate 기여를 저장한다.
TSH active-surface energy와 quantum expectation energy는 다르게 표시한다.
resume은 같은 source/config에서 `--out 기존경로 --resume --end-fs ...`;
channel basis, force history, active state, RNG, 누적 audit까지 복구한다.

현재 complete dense diagonalization은 작은 CPU pilot 용도다. 기본 dimension 한도256;
한도를 넘으면 중단하고 state를 자동으로 버리지 않는다. production MCEF x×q 격자의
효율적인 대규모 eigensolver/backend는 아직 없다. `--max-internal-dimension`을 늘릴 수
있지만 자원/시간 추정 없이 큰 격자를 돌리는 것은 권장하지 않는다. 장시간 안정성,
Gaussian K/grid/mask 수렴, CTMQC 채널 및 sampling 수렴은 미인증이다.

실제 추가 시험에서는 위 PEN q32 설정의 Gaussian 전파가 dt=.01에서 .06749fs에
PNC guard로 중단됐고, **dt=.0025에서는 CTMQC와 TSH 모두 .1fs를 통과**했다.
MCEF compact pilot은 dt=.01로 .1fs를 통과했다. 장시간 안정성 인증은 아니다.
PEN .1fs 시험을 재현하려면 위 PEN 명령의 `--dt-au 0.0025 --end-fs 0.1
--every-fs 0.025`와 새로운 outdir을 사용한다. 실패한 run에서 dt를 바꾸는 것은
동일-config resume가 아니므로 새 outdir에서 다시 시작한다.

## 읽은 자료와 수식 → 코드

현재 체크아웃에는 `sources/`가 없고 사용자가 제공한 자료는 `paper/`에 있다.
MD 전체, reference TeX/PDF 19쪽, 원본 handwritten PDF 16쪽,
`11_3.pdf` 5쪽을 확인했다. 제공 문서는 사용자의 파일이므로 이 commit에
무단으로 함께 포함하지 않는다.

| Reference 식 | 구현 파일 / 함수 |
|---|---|
| S3–S5, P11–P13, N1 | `gaussian.py:build_bridge`, `pack_parameters`, `unpack_parameters` |
| P3: real contraction, 복소 위상 유지 | `gaussian.py:split_complex_rate` |
| P6–P14: amplitude metric / weighted SVD | `gaussian.py:project_amplitude_rate` |
| P15: 실제 projected G 변화율 | `gaussian.py:split_and_project` |
| D1–D2: factorial-scaled local polynomial | `mls.py:build_mls_operator`, `MLSOperator.apply` |
| Q1–Q5, Q6 | `heavy_density.py:reconstruct_heavy_density`, `reconstruct_kde_density` |
| Q8–Q10: 별도 two-channel intercept | `heavy_density.py:two_channel_intercept_correction` |
| F2–F5/F7: a,b,alpha | `fields.py:analyze_fields` |
| F6/F8: reference only | `fields.py:bridge_scalar_reference`, `bridge_continuity_reference` |
| R1/R4, R3 | `reference.py:outer_compact`, `outer_expanded`, `material_rate` |
| B3–B5 | `reference.py:project_internal`, `lift_internal_rate` |
| B1: H action 및 matched mean force | `models.py:InternalModel.action`, `force` |
| 모든 stage에서 P3/P8 재계산 | `run.py:Engine.evaluate`, `advance` |

중요한 transcription 정리:

- Gaussian은 `|Lambda|^2`의 **밀도 혼합**이지 복소 Gaussian wavepacket 합이 아니다.
- `C_dot`에는 추가 `dt`가 없고, 빼는 항은 **real** contraction이다.
- `C_dot`에서 requested amplitude rate를 projected `Lambda_dot`로 바꾸지 않는다.
- outer R1의 마지막 residual operator는 `(-i partial_R - alpha)`다.
- 양의 bridge gauge는 양의 heavy marginal gauge를 의미하지 않는다.
- `alpha`는 표본 `P_R`이 아니다. density quantum momentum도 이 둘과 다르다.
- heavy mixture의 log derivative는 posterior weight를 사용하며,
  독립 전파 population으로 대체하지 않는다.
- Gaussian 압축 오차, 적분 오차, MLS 오차, QD와의 차이는 서로 다르다.

## 모델 및 초기 준비

`models.ModelConfig`와 `run.RunConfig`는 JSON으로 수정할 수 있다.
`--config config.json`으로 전체 설정을 읽고 CLI에서 지정한 값만 덮어쓴다.

PEN은 `pen_perspectives_2026/model.py`, `resonance.py`와
`pen_ctmqc_2026/core.py:hamiltonian`의 물리 potential/미분을 재사용한다.
기본 `g=0.01`, DSE **off**, on-resonance `omega=0.17`이다. 이는 기존 논문
benchmark의 선택이지 DSE가 불필요하다는 주장이 아니다. `--g`, `--dse`,
`--omega`, 또는 `model.resonance`로 명시적으로 바꾼다. photon Fock 수는
전파에 제한하지 않는다. q kinetic은 mass=1의 FFT다.

MCEF 초기값은 실제 보존 archive의 `args`에서 추적했다:
`results/20260916/erf_fullgrid_BOcurvature_compact_qR_analysisBO2_100fs/multi_component_discrete_tdse_gpu.npz`.
`M=12000`, `m_q=1836`, `R0=9.5`, `sigma_R=0.13650604313545334`,
`q0=-4`, `sigma_q=0.3208418789467482`, 전자 S1,
ERF 길이 `3.1,5,4,1`, heavy trap `0.03*(R-9.5)^2`를 보존했다.
이 sigma들은 **밀도 표준편차**다. R/P는 지정 seed의 Gaussian Wigner
분포에서 뽑으며 `sigma_P=1/(2*sigma_R)`다. 표본 가중치 `eta_I`는 기본 균등,
`RunConfig.sample_weights`로 명시적으로 지정할 수 있다.

MCEF의 x primitive 계수는 `sqrt(dx)*Phi(x)`이므로 전자 PNC에는 추가 dx가
없다. 초기 BO 대각화는 상태 준비에만 사용하며 이후 x-grid 전체가 전파된다.
아래 compact q-box/x-grid는 **짧은 구현검사용**이다. 기존 장시간 TDSE의
격자 수렴이나 반사·분기를 검증했다는 뜻이 아니다.

K>1은 서로 다른 center window를 가진 deterministic amplitude fit을 쓴다.
fit 오차·rank를 metadata에 기록한다. 초기 진공을 K=2로 근사한 이번 예는
amplitude L2 오차 약 0.00493, 초기 photon number 약 9.89e-5를 가진다.
따라서 K 증가만으로 정확도가 좋아졌다고 해석하면 안 된다.

## 적분·support·성능

기본은 DOP853, `rtol=1e-10`, `atol=1e-12`, `max_step=dt_au=0.02`다.
모든 accepted step에서 raw PNC/norm과 q edge mass를 검사한다. 상태를
재정규화하지 않는다. Gaussian rank 부족은 기본 실패, truncated SVD는
`rank_policy="truncate"`를 명시했을 때만 허용한다.

기본 production RHS support는 모든 representable positive point다.
`support_relative>0`이면 해당 구간 밖 **C를 고정하는 별도 수치 정책**이며
버려진 mass/RHS norm을 기록한다. plotting mask와 혼동하지 말 것.
Gaussian kernel은 유한 q-grid에서 정규화하고 그 정규화 상수의 미분도
Jacobian에 포함한다.

초기 RK4 pilot에서 PEN 저밀도 꼬리 PNC 검사가 실패했다. raw total norm만
좋다고 이를 숨기지 않았다. 원인은 periodic FFT seam 오차를 `1/Lambda`가
증폭하는 조건부 계수의 경직성이다. 예를 들어 q-box ±12의 초기 진공
Hamiltonian 잔차는 weighted L2 약 1.15e-5, ±14에서는 약 1.53e-7이었다.
off-PNC 오차 deltaN에는 `deltaN_dot=-2*(r/Lambda)*deltaN`이므로 작은
꼬리 오차가 증폭될 수 있다. DOP853에서 동일 guard를 유지한 pilot은
통과했지만, 공간/시간 수렴 인증과는 다르다. RK4는 명시적 옵션과 회귀
검사 목적으로 남긴다.

MLS는 MTE RHS에 필요 없으므로 저장시각의 진단에서만 계산한다. 실제
phase를 독립적으로 align하지 않으며 current-R 이웃을 사용한다.
같은 R에 다른 history의 상태가 있으면 scatter를 기록한다. 생성된 a,b는
이 재구성에 의존하는 **근사장**이다. 확정된 exact EF 장으로 부르지 않는다.

GPU backend는 구현/검증하지 않았다. CPU/complex128만 사용한다.
dense x-q-R 전체 Hamiltonian은 만들지 않지만, 상태 메모리는 여전히
`16*Ntraj*Nq*Nelectronic` bytes다. 큰 MCEF x-grid에서 Ntraj=10000을
무조건 쓰면 절약이 아니라 매우 큰 메모리를 요구한다. `--estimate`로
먼저 확인한다. 별도의 long-time 속도 우위를 주장하지 않는다.

## 실행 순서 (로컬 또는 서버의 MCEF 환경)

서버 접속/자원은 이번 환경에서 검증되지 않았다. 아래는 서버용으로
준비한 명령이며 서버에서 실제 실행했다고 주장하지 않는다.

```bash
cd /home/hbji/exactfactorization
conda activate MCEF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_gaussian_trajectory_*.py' -b

# 전체 common core/고정-R 참조의 작은 검증
python -m multi_component_exact_factorization.gaussian_trajectory.validation \
  --out results/20261106/gaussian_validation_server_v1

# 먼저 필요한 메모리/저장량 확인 (출력 파일 생성/전파 없음)
python -m multi_component_exact_factorization.gaussian_trajectory.run \
  --out results/20261106/gaussian_pen_server_v1 --estimate

# PEN 짧은 Gaussian pilot: 전파 뒤 PNG/MP4 자동 생성
python -m multi_component_exact_factorization.gaussian_trajectory.run \
  --out results/20261106/gaussian_pen_server_v1 \
  --model pen-nonadiabatic --g 0.01 --dse off \
  --ntraj 12 --nq 64 --qmin -14 --qmax 14 --components 1 \
  --dt-au 0.02 --end-fs 0.1 --every-fs 0.025 --movies

# 동일한 초기 모델의 무압축 MTE 비교: 새 폴더 필수
python -m multi_component_exact_factorization.gaussian_trajectory.run \
  --out results/20261106/gaussian_pen_native_server_v1 \
  --model pen-nonadiabatic --compression native \
  --ntraj 12 --nq 64 --qmin -14 --qmax 14 \
  --dt-au 0.02 --end-fs 0.1 --every-fs 0.025 --movies

# 기존 실공간 MCEF 모델의 매우 짧은 검증
python -m multi_component_exact_factorization.gaussian_trajectory.run \
  --out results/20261106/gaussian_mcef_server_v1 --model mcef \
  --ntraj 7 --nx 33 --nq 64 --qmin -6 --qmax -2 \
  --dt-au 0.01 --end-fs 0.02 --every-fs 0.005 --movies

# 이미 저장된 결과의 그림만 다시 생성 (전파 없음)
python -m multi_component_exact_factorization.gaussian_trajectory.renderer \
  results/20261106/gaussian_pen_server_v1 --movies
```

Rabi pilot은 `--model pen-rabi`. coupling zero는 `--g 0`.
기존 폴더의 덮어쓰기는 거부한다. 재시작은 동일 코드/config에서만:

```bash
python -m multi_component_exact_factorization.gaussian_trajectory.run \
  --out results/20261106/gaussian_pen_server_v1 --resume --end-fs 0.2 --movies
```

위 0.2 fs는 **다음 단계 명령 예**이며 검증 완료 길이가 아니다. source hash가
달라지면 기존 checkpoint에 새 알고리즘을 섞지 않고 거부한다.
DOP853 restart는 tolerance 기반 재현, 고정 RK4는 bitwise 재현을 검사한다.
저장 주기 fs는 dt의 정수배로 반올림되므로 실제 저장시각을 NPZ에서 읽는다.

## 출력

- `config.json`, `metadata.json`: 물리 설정, seed, basis/gauge, 사용 코드 hash,
  실행 host, 초기 fit, resource estimate.
- `checkpoint.npz`: 유일한 독립 상태, step, config/source hash. 원자적 교체.
- `frames/frame_*.npz`: 현재 G/밀도/장/에너지/population/Q_raw와 eta.
- `frames/audit_*.json`: Gaussian rank/condition, MLS scatter/fit,
  제외 mass, imaginary defects, alpha bridge residual.
- `timeseries.npz`, `status.json`, 실패 시 `failure.json`.
- `plots/summary.png`, `snapshot_*.png`, `diagnostics.mp4`, `render_manifest.json`.

동영상 중간 PNG는 임시 폴더에서만 만들고 인코딩 후 회수한다. ffmpeg가
없으면 PNG는 남기고 영상 미생성을 명시한다. 필드 색상은 프레임 공통,
검은 등고선은 **시각화용 R-KDE × conditional density**임을 명시한다.
표본 사이 공간을 임의의 exact 장으로 채우지 않는다. MCEF primitive-grid
population은 BO population으로 표시하지 않는다.

검증 세부 수치와 실제 산출물은 `VERIFICATION.md`에 정리한다.
