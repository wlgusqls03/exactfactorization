# Model A: two diabatic electronic states + photon + nuclear coordinate

별도 추가 모듈입니다. 기존 MCEF propagation, VSC Phase 1–7, 저장 결과를
수정하거나 import-time 실행하지 않습니다. **DSE는 이번 범위에서 제외**합니다.
실제 전자 x-grid도 추가하지 않습니다. 전자 두 상태 모델 안의 full TDSE를
풀고, 그 파동함수를 positive gauge에서 두 번 exact factorization합니다.

## 출처와 재현 범위

- E. Sangiogo Gil, D. Lauvergnat, F. Agostini, JCP **161**, 084112 (2024),
  DOI: https://doi.org/10.1063/5.0224779, Sec. III A/B, Eqs. 4/5/27/28.
- 로컬 원문: `paper/Exact factorization of the photon–electron–nuclear wavefunction Formulation and coupled-trajectory dynamics.pdf`.
- 저자 구현: https://github.com/lauvergn/QuantumModelLib/blob/73f2ef3d322b253a4ef39e08216261bd974180f6/SRC/QML/OneD_Photons_m.f90
  (2026-10-06 조회; 해당 커밋은 2024년 QD 입력 파일 자체가 아님).
  모델 독립 구현이며 외부 Fortran 소스를 vendoring하지 않음.

**아직 논문 완전 재현 PASS를 주장하지 않습니다.** author raw QD/gauge input과
digitized reference arrays가 없는 상태입니다. Figure resemblance, conservation,
factorization algebra, numerical convergence는 서로 다른 검사입니다.

### 원문과 코드 대조에서 발견한 사항

1. Printed Eq.28의 비대각 LM 항에는 q가 빠져 보입니다. Eq.5 및 저자의
   `OneD_Photons` 실제 구현에는 q가 있습니다. 여기서는 그 구현을 따릅니다.
2. Sec.III의 photon kinetic 표기와 Eq.4가 다릅니다. Eq.4 및 저자의 mass=1을
   따라 `Tq=-1/2 d_q^2`를 사용합니다.
3. 저자 default R1/R2는 논문과 순서가 반대입니다. 같은 diagonal 두 개를
   교환한 것으로, 여기서는 인쇄된 R1=6, R2=2와 positive bare S1 eigenvector를 사용합니다.
4. cavity frequency는 인쇄값 omega=.17 Ha를 고정합니다. R=2의 bare gap은
   약 .16 Ha여서 수학적으로 완전 공명은 아닙니다. 맞추기 위해 omega를 .16으로
   바꾸지 않습니다. dipole sign은 논문 mu12=-1 및 저자의 electronic-position
   Mu12=+1에 앞의 minus sign이 붙는 구현과 일치합니다.
5. Gaussian 폭은 amplitude `exp[-(z-z0)^2/(2 sigma^2)]` convention으로 선언합니다.
   sigma_q=2.425는 1/sqrt(.17)에, sigma_R=.223은 (M*k)^(-1/4)에 가깝습니다.
   density 표준편차는 sigma/sqrt(2)입니다. 반올림된 paper width를 유지하므로
   g=0에서도 아주 작은 photon breathing이 있을 수 있습니다.
6. 초기 전자는 R별 **bare adiabatic S1**입니다. 초기 q Gaussian에 photon을
   추가하거나 초기 nuclear momentum을 주지 않습니다. 저자의 정확한 초기
   coefficient input이 확보되지 않았으므로 cavity-adiabatic S1 초기화와의 차이를
   재현 불확실성으로 남깁니다. 결과를 맞추려고 바꾸지 않습니다.

## Hamiltonian 및 초기상태

모든 수치 atomic units. 파동함수 shape `(NR,Nq,2)`, 정규화
`sum(abs(Psi)^2)*dR*dq=1`. 전자 합은 quadrature weight 없이 두 orthonormal
diabatic label에 대한 합입니다.

```
H = -d_R^2/(2 M) - d_q^2/2 + V(R,q)
V11 = k/2 (R-6)^2 + omega^2 q^2/2 + Z omega g R q
V22 = k/2 (R-2)^2 + omega^2 q^2/2 + Z omega g R q
V12 = V21 = b exp[-a(R-3.875)^2] + omega g mu12 q
M=20000, k=.020, a=3, b=.01, Z=1, mu12=-1, omega=.17
g=.01 (coupled), g=0 (control); DSE absent in both.
Psi_j(0) = S1_j(R) N exp[-(R-2)^2/(2*.223^2)-q^2/(2*2.425^2)]
```

`q`는 physical photon quadrature이며 VSC의 dimensionless Q가 아닙니다.
`R`는 질량 20000인 generalized nuclear reaction coordinate이지 proton mass가 아닙니다.
0–1250 au = 약 0–30.236 fs; 과거 VSC의 40-fs 실험과 다릅니다.

## Nested EF와 gauge

```
joint=sum_j |Psi_j|^2, F=sqrt(joint)
rho_R=integral dq joint, chi=sqrt(rho_R)
Lambda=F/chi, Phi_j=Psi_j/F
Psi_j=chi Lambda Phi_j
a=Im sum_j Phi_j* d_q Phi_j
b=Im sum_j Phi_j* d_R Phi_j
alpha=integral dq Lambda^2 b                  (positive gauge)
epsilon1 = <V> + geo_q + geo_R + GD1
geo_q=(<d_q Phi|d_q Phi>-a^2)/2
geo_R=(<d_R Phi|d_R Phi>-b^2)/(2M)
GD1=-i<Phi|dt Phi>
epsilon2 = <Xi|Tq+V|Xi> + geo_outer - i<Xi|dt Xi>, Xi=Psi/chi
F_outer=-d_R epsilon2+dt alpha
```

`dt Psi=-i H Psi`는 순간 native action입니다. phase unwrap이나 저장 시각 차분을 쓰지 않습니다.
파동함수 미분은 propagation과 같은 FFT symbol을 사용합니다. 유한 격자의 quotient/product-rule
오차는 별도의 continuity 및 resolution 비교로 확인합니다. 두 PNC, 재구성,
epsilon1 expectation/inversion, epsilon2 expectation/inversion/nested integral을 저장합니다.
정확한 node는 NaN으로 유지하고 phase를 채우지 않습니다. 전자 unitary basis 변환 불변성도 시험합니다.

### Fig.3의 중요한 주의사항

positive F gauge에서 초기 real factor에 대해
`epsilon1 = (Fqq/F-a^2)/2 + (FRR/F-b^2)/(2M)`입니다.
따라서 초기 중심의 total PG epsilon1은 약 -0.0855 Ha이며, cavity-adiabatic
S1의 약 +0.16 Ha와 같지 않습니다. 원문 Fig.3의 초기 harmonic surface 설명과
positive gauge의 전체 scalar를 그대로 동일시할 수 없습니다.

**GD를 몰래 빼고 total이라고 부르지 않습니다.**
`fig3_GI.png`와 `fig3_epsilon1.png`를 둘 다 출력하고 명칭을 구분합니다.
Fig.4 caption은 `<V_PEN>`과 `geo`를 따로 표시합니다. 그에 대응하는 그림에 더해
PG total이 포함된 cut movie를 출력합니다. 저자 raw data 없이 이 차이의 원인을
gauge 또는 그림 정의 중 하나로 확정했다고 주장하지 않습니다.

## 서버 실행

코드는 git으로 전달됩니다. 이전 VSC input packet이나 추가 tar는 필요 없습니다.
Python >=3.9, NumPy, SciPy, Matplotlib, ffmpeg가 필요하고 GPU에는 해당 환경의 CuPy가 필요합니다.
500*500*2 complex128 파동함수 자체는 8 MB로 x-grid VSC보다 훨씬 작습니다.
CPU도 지원하며 GPU는 각 setting에서 CPU/GPU action 및 16-step 비교 후 사용합니다.

```bash
cd /home/hbji/exactfactorization
git pull
CUDA_VISIBLE_DEVICES=1 bash \
  multi_component_exact_factorization/vsc_polariton/model_a/run_server.sh
```

중단 시 같은 명령으로 restart합니다. source/settings hash가 다르면 정지하며 기존 run을
보호합니다. 코드를 바꾼 경우 새로운 root를 지정하세요.

```bash
CUDA_VISIBLE_DEVICES=1 bash \
  multi_component_exact_factorization/vsc_polariton/model_a/run_server.sh \
  results/vsc_polariton/model_a/server_v2
```

먼저 짧게 실행하려면 (production 결과가 아닙니다):

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 python -m \
  multi_component_exact_factorization.vsc_polariton.model_a.run \
  --backend gpu --end 10 --out results/vsc_polariton/model_a/gpu_probe_v1
```

### 순서대로 실행하는 사전 고정 campaign

|run|NR,Nq|R box|q box|dt au|저장|
|---|---|---|---|---|---|
|coupled|500,500|[0,8)|[-10,10)|.05|maps/cuts + event waves|
|free|500,500|같음|같음|.05|maps/cuts + event waves|
|coupled_grid|600,600|같음|같음|.05|observables + event waves|
|coupled_dt|500,500|같음|같음|.025|observables + event waves|
|coupled_box|624,800|[-.992,8.992)|[-16,16)|.05|observables + event waves|
|free_grid|600,600|[0,8)|[-10,10)|.05|observables + event waves|

모든 run은 1250 au, 관측 저장 간격 2.5 au, movie 501 frame입니다.
**native 미분이 끝난 표시 배열만** stride2로 줄입니다. full wave는
0/250/500/750/1000/1250 au의 6개와 restart만 저장합니다.
movie compact field로 다시 미분하지 마세요. convergence는 event full wave를
native grid에서 재분석한 뒤 common-grid에서 비교합니다.

전체 campaign은 보수적으로 약 7 GiB 이하를 예상하므로 **최소 8 GiB를 확보하세요**.
lossless NPZ 압축을 사용하며 모든 step의 wave나 대량 PNG를 저장하지 않습니다.
실제 시간은 `backend_check.json`의 GPU 실측으로 propagation 시간을 추정합니다.
EF/압축/동영상 시간은 별도이며 아직 측정하지 않은 GPU 총시간을 보장하지 않습니다.

## 합격 기준과 한계

`run.LIMITS`는 실행 전 고정: norm1e-8, energy drift1e-6 Ha, R/q edge1e-8,
reconstruction1e-10, EF route1e-7 Ha, continuity1e-7。
**논문 box에서도 초기 photon edge가 1e-8을 넘을 수 있으므로** FAIL을 숨기지 않고 box 비교를 진행합니다.
절대 edge PASS와 observable/field convergence는 별개입니다. 작은 solver 잔차를 basis PASS로 부르지 않습니다.
campaign status는 `PASS_NUMERICAL` 또는 `NOT_CERTIFIED`입니다.
논문 재현 인증 flag는 author-data 비교가 없으므로 false로 유지합니다.

## 출력 및 전달할 파일

root는 `results/vsc_polariton/model_a/server_v1/campaign/`입니다.

- `campaign_validation.json`, `plan.json`
- 각 run의 `backend_check.json`, `status.json`, `observables.json`, `ef_diagnostics.json`
- `coupled_plots/`, `free_plots/`의 PNG, MP4, manifest
- `campaign.log` (한 단계 위)

raw `waves/`와 `restart.npz`는 서버에 보관하세요. 처음부터 보낼 필요는 없습니다.
그림: fig1_reference, fig3_GI, fig3_epsilon1, fig4_components.
동영상: tdpes1_positive, vectors_positive(a,b,alpha), outer_positive(epsilon2/force), cuts_positive.
다른 데이터나 이전 실패 run을 삭제하는 명령은 없습니다.

## 광자수: VSC와 Model A 공통

`N = omega*<q²>/2 + <p²>/(2*omega) - 1/2`를 사용합니다.
Model A는 native FFT의 `Dq Psi`로 `<p²>=integral |Dq Psi|²`를 계산하며,
`observables.json`에 `q2,p2,n_q,n_p,photon_energy,n_ph,norm`을 저장합니다.
원시 적분값은 norm을 포함하고, 그림에서는 모두 recorded norm으로 나눕니다.
`n_q=omega*<q²>/2`, `n_p=<p²>/(2*omega)` 각각은 광자수가 아닙니다.
진공에서는 두 값 모두 1/4이고, 합에서 1/2을 빼면 N=0입니다.

server campaign은 `photon_comparison/photon_number.png`에서 coupled/free를
같이 그리고, 각 case plot에도 `photon/`을 만듭니다. N, N(t)-N(0), 두 quadrature
기여의 세 패널입니다. NPZ와 provenance JSON도 함께 저장합니다.

기존 VSC는 이미 `photon_real_grid.observe()`에서 photon kinetic을 포함한
`nph`를 저장합니다. 재전파 없이 아래 adapter로 그릴 수 있습니다.
**기존 VSC source를 수정하지 않았습니다.** 서버의 실제 폴더에 맞춰 `--vsc-run`만
지정하며, 바로 그 run의 `status.json`, `observable_*.npz`와 원래
`campaign_plan.json`이 필요합니다. 상위 폴더에서 plan을 찾지 못하면
`--plan /실제/경로/campaign_plan.json`을 추가합니다.

```bash
OPENBLAS_NUM_THREADS=1 python -m \
  multi_component_exact_factorization.vsc_polariton.model_a.photon_number \
  --vsc-run results/vsc_polariton/photon_real_grid/barrier_v1/full/base_Q384/full \
  --out results/vsc_polariton/photon_real_grid/photon_number_v1
```

VSC에서는 Q=sqrt(omega)q이므로 저장된 rho_Q로 `<Q²>/2`를 계산하고,
독립적으로 저장된 photon energy로 운동량 기여를 복원합니다.
**density로 momentum을 추정하지 않습니다.** 이 확인은 energy/number 저장값의
대수적 일관성 검사이며 새로운 basis convergence 검증은 아닙니다.
밀도만 있는 `fields_*.npz`로는 실행할 수 없습니다.

이 N은 현재 Hamiltonian convention의 **bare oscillator occupation**입니다.
상호작용 cavity에서 검출되는 방출 광자수와 같다고 해석하지 않습니다.
VSC 초기 displaced vacuum은 bare N이 이미 0이 아닐 수 있습니다.

## server_v1 결과를 받은 뒤: event audit + 그림/동영상 개선

2026-10-07 review에서 전체 gate는 NOT_CERTIFIED였다. 큰 box의 internal gate는
PASS였지만, 기본 box의 continuity와 epsilon1 convergence, P(R>=4)의 경계
quadrature를 분리해서 확인해야 한다. 따라서 다음 명령은 새 TDSE를 돌리지 않고
기존 여섯 case의 **1000,1250 au wave 12개**를 재분석한다. 이어서 coupled/free의
기존 501-frame fields로 그림과 동영상을 다시 만든다. 전파나 GPU 계산은 없다.

```bash
cd /home/hbji/exactfactorization
git pull
conda activate MCEF
bash multi_component_exact_factorization/vsc_polariton/model_a/run_event_audit_server.sh
```

GPU/CUDA 선택이 필요 없는 CPU 후처리이다. 실제 결과는 홈 디렉터리 내부
`results/vsc_polariton/model_a/server_v1/next_review_v1/`에 저장된다.
로그는 옆의 `next_review_v1.log`이다. 유닛 테스트의 임시 fixture만 /tmp에서
자동 생성·정리되며 과학 분석 결과를 /tmp에 보관하지 않는다.

**이 단계에서 원래 `run_server.sh`를 다시 실행하지 않는다.** 새 .py 파일 추가로
전체-package source hash가 달라지므로 기존 전파의 restart가 거부될 수 있다.
event audit은 원래 numerical kernel인 model.py/factorization.py의 원본 hash를
검사하고, 원자료를 수정하지 않는 별도 경로이다.

event audit은 대략 수 분~10분, 8개 영상 렌더링은 추가로 수십 분을 예상한다.
이는 실측 총시간이 아니며 CPU/I/O 부하에 따라 더 길어질 수 있다.
추가 디스크 최소 1 GiB, 시스템 RAM은 보수적으로 4 GiB 여유를 권한다.
실제 256×256 저장 wave를 native+2x 평가한 로컬 시험은 약 1.16초,
peak RAM 287 MiB였다. 가장 큰 624×800 wave는 그보다 큰 메모리가 필요하다.
GPU의 11 GiB VRAM과는 무관하다. 모든 case를 직렬로 처리한다.

끝나면 아래 파일 **하나만** 로컬로 보내면 된다.

```
results/vsc_polariton/model_a/server_v1/next_review_v1/model_a_next_review.tar.gz
```

파동함수 12개, identity/관측/검증 JSON, 재분석 JSON, 원자료 SHA256,
새 PNG/MP4와 표시 설정을 포함한다. wave의 비압축 원소 크기 합은 약 120 MiB이다.
동영상의 최종 크기는 인코딩에 따라 달라지며 원본과 return archive가 함께 저장된다.
전체 fields나 restart는 포함하지 않는다. 별도 입력 압축파일 불필요.
서버의 기존 wave/fields는 계속 보존해야 한다.

중단된 audit 폴더를 덮어쓰지 않는다. 재실행이 필요하면 새 출력명을 지정한다.

```bash
bash multi_component_exact_factorization/vsc_polariton/model_a/run_event_audit_server.sh \
  results/vsc_polariton/model_a/server_v1/campaign \
  results/vsc_polariton/model_a/server_v1/next_review_v2
```

### 그림에서 달라진 점

- `plots/coupled/`, `plots/free/`에 Fig.1/3/4 형식 PNG와
  `vectors_positive.mp4`, `tdpes1_positive.mp4`, `outer_positive.mp4`,
  `cuts_positive.mp4`를 각각 만든다. 원래 그림은 덮어쓰지 않는다.
- 밀도 등고선은 매 프레임 joint density 최대값 대비 **1e-5~0.9의 20단계** 검은 점선.
  색상 마스크는 기존 상대밀도 1e-6을 유지한다. 표시 기준이지 수렴 인증 기준이 아니다.
- coupled/free 및 전체 시간에 **동일한 고정 축·색상 범위**를 적용한다.
  b/alpha의 과거 ±60 고정 제한을 없앴다. alpha/force/epsilon2 line 범위는
  occupied finite 값 전체를 포함한다. 맵과 단면은 밀도 가중 0.1~99.9% 분위수의
  시간·case 전체 envelope를 사용한다. 색상 포화 확률질량을 프레임마다 저장한다.
- 단면의 색상은 고정 좌표(q=0/약1.5 또는 R=2/4), 선 종류는
  total, conditional potential, GI, GD, geo=geo_q+geo_R를 뜻한다.
  cond는 점 marker를 추가하여 GI와 겹쳐도 구분한다. static S0/S1은 옅은 실선/파선.
  밀도는 독립적인 오른쪽 축 점선이며 각 cut을 따로 정규화하지 않는다.
- energy 범위 밖 값은 경계 삼각형과 전체 occupied extrema로 알린다.
  큰 geometric peak를 삭제하거나 smoothing하지 않는다. `manifest.json`에 범위를 기록한다.
- R=4 수직 점선은 product population 분할선이다. 자동으로 static barrier 위치를 뜻하지 않는다.
- 501개 **실제 저장 프레임**을 24 fps로 출력한다. 시간 보간이나 재전파는 하지 않는다.

이 수정은 표시 오류를 줄이지만, epsilon1의 grid convergence 실패를 해결했다고
주장하지 않는다. 영상에도 diagnostic / not certified를 명시한다.

### 새 코드 위치와 검증 범위

- `event_audit.py lines 27–31`: SHA256 streaming; 원자료 전후 동일성 검증.
- `event_audit.py lines 34–66`: `integral_fourier`, `populations`.
  `(NR,Nq,2)` wave에서 rho_R를 구하고 기존 rectangle 합, 경계 half-weight,
  주기 Fourier 보간함수의 정확한 구간 적분, 2NR wave-first density 적분을 비교한다.
  probability는 무차원. 동일 Gaussian의 두 grid에서 생기는 가짜 차이 제거를 시험.
- `event_audit.py lines 69–84`: `fields`. 기존 `analyze`와 `derivative`를 재사용.
  `epsilon1 = Fqq/(2F) - a²/2 + FRR/(2MF) - b²/(2M)`의 4개 PG QHJ 항,
  `(NR,Nq)`, Ha. expectation decomposition cond/geo/GD도 그대로 비교한다.
- `event_audit.py lines 87–132`: `stats`, `compare_fields`. 상대밀도 threshold
  1e-2/1e-4/1e-6/1e-8, R/q 영역별 RMS·최대 오차·위치·제외 영역·nonfinite
  확률질량을 기록한다. potential offset fitting을 하지 않는다.
- `event_audit.py lines 135–142`: `same_wave`. 같은 Fourier wave의 두 축을
  2배 촘촘하게 평가한 후 native derivatives를 다시 구한다. 이는 후처리 sampling
  diagnostic이지 새 TDSE propagation/basis convergence가 아니다. norm도 기록.
- `event_audit.py lines 145–209`: `preflight`, `audit`. 입력 존재/shape/time/source
  검사, 직렬 분석, wave hash 재확인, 한 개의 tar.gz 생성. 기존 gate 변경 없음.
- `event_audit.py lines 212–218`: `main`, 명령행 옵션.
- `test_event_audit.py lines 13–70`: 새 6개 테스트. Model A의 기존 14개와 합쳐
  패키지 생성·원본 보존·출력 덮어쓰기 거부도 테스트한다.
- `review_plot.py lines 24–68`: `quantile`, `scan`. 전체 시각/case 공통 표시 범위;
  native derivative 이후 compact된 기존 map `(NR/2,Nq/2)`와 native line만 읽는다.
  에너지 Ha, a/b/alpha momentum a.u., 밀도는 원래 quadrature convention.
- `review_plot.py lines 71–119`: `map_axis`, `cuts`. 20단계 등고선, 포화질량,
  component cut 및 독립 밀도축. synthetic b=120/alpha=95/geo=15 테스트로
  예전 축 제한 회피와 clipping 경고를 검증한다. raw field 수정 없음.
- `review_plot.py lines 122–188`: `preflight`, `render`. 기존 `plot.load`, `plot.fig1`,
  `model.potential` 재사용; ffmpeg 확인, 원본 시각 사용, 새 그림과 MP4 저장.
- `next_review.py lines 11–33`: `main`. event audit → rerender → 원본 hash 재검사
  → 한 개 return archive. 단위 테스트에서 archive 내용/해시/원본 보존 검증.
- `test_review_plot.py lines 17–77`: 5개 표시·preflight·패키징 테스트.
  기존 14 + event 6 + 표시/연결 5 = 25개 테스트.

## 단위 테스트 실행

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest \
  multi_component_exact_factorization.vsc_polariton.model_a.test_model_a \
  multi_component_exact_factorization.vsc_polariton.model_a.test_event_audit \
  multi_component_exact_factorization.vsc_polariton.model_a.test_review_plot -v
```

Hamiltonian Hermiticity, dense expm과 split 비교 및 수렴 차수, stationary eigenstate 밀도,
PNC/reconstruction, 독립 scalar routes, bare photon vacuum, 전자 basis unitary 불변성,
native cuts, restart/overwrite 보호를 검사합니다. 짧은 테스트 PASS를 1250 au 수렴으로 부르지 않습니다.
