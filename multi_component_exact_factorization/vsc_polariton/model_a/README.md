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

## 단위 테스트

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest \
  multi_component_exact_factorization.vsc_polariton.model_a.test_model_a -v
```

Hamiltonian Hermiticity, dense expm과 split 비교 및 수렴 차수, stationary eigenstate 밀도,
PNC/reconstruction, 독립 scalar routes, bare photon vacuum, 전자 basis unitary 불변성,
native cuts, restart/overwrite 보호를 검사합니다. 짧은 테스트 PASS를 1250 au 수렴으로 부르지 않습니다.
