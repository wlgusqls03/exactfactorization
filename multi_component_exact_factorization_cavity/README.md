# 실공간 MCEF를 계승한 cavity 확장

## 50 fs 계산에서 0–40 fs만 다시 그리기

`--t-max-fs 40`은 **렌더링 전용 옵션**이다. 50 fs의 fields/observables,
checkpoint, config는 수정하지 않으며 TDSE 재실행이 필요 없다.
실제 저장 시간 `t <= 40 fs`인 자료만 사용한다. 정확히 40 fs 자료가 없으면
그보다 앞선 마지막 저장 시각에서 끝나고, 보간 프레임은 만들지 않는다.

```bash
for coupling in zero weak strong; do
  python -m multi_component_exact_factorization_cavity.workflow render \
    --out "results/20261026/pen10_native_v1/pen/$coupling" \
    --t-max-fs 40 || break
done
```

기본 출력: 각 case의 `report_40fs/`. 기존 50 fs `report/`는 유지된다.
`--report-out`을 지정하면 그 경로에 쓴다. 구간이 다른 보고서에 덮어쓰면
옛 snapshot 파일이 남을 수 있으므로 구간별 별도 폴더를 권장한다.
같은 `report_40fs/`에 동일한 옵션으로 다시 실행하면 같은 파일을 갱신한다.

범위 제한은 A/B·GI·slice·consistency·timeline·outer force·density overview·
BO/POL/CBO population·photon number·에너지·Pol 핵 파동묶음 모두에 적용된다.
축/색상 범위와 대표 snapshot도 제한된 시간 구간에서 다시 결정한다.
정적 BO/CBO/Pol 기준면·상태 구성 자체는 시간이 없으므로 변경하지 않는다.
`render_manifest.json`과 `polaritonic_dynamics_manifest.json`에 요청한 제한과
실제 사용한 시간 범위를 기록한다. 새 Pol 영상만 만들 때도 같은 옵션을 쓴다:

```bash
python -m multi_component_exact_factorization_cavity.polaritonic_dynamics \
  --case results/20261026/pen10_native_v1/pen/strong --t-max-fs 40
```

`--no-movies` 추가 시 정적 그림만 생성한다. 시간 옵션을 생략하면 기존 전체 구간 동작이다.
zero/weak/strong 통합 비교 그림도 40 fs로 제한하려면:

```bash
python - <<'PY'
from multi_component_exact_factorization_cavity.render import render_comparison
render_comparison('results/20261026/pen10_native_v1', t_max_fs=40)
PY
```

이는 `pen_coupling_comparison_40fs.png/.pdf`를 저장한다.

## Population 분리·핵 파동묶음·Pol 구성 (2026-10-28)

기존 `workflow render`와 계산 완료 후 자동 렌더링에 함께 적용된다.
TDSE / Hamiltonian / gauge / projection 계산은 변경하지 않는다.

- 밀도 실루엣은 더 진한 회색과 경계선으로 표시한다. 축 높이의 30%를 쓰는
  **형태 비교용 배경**이며, 에너지 축 단위로 읽지 않는다.
- `dynamics_overview.png/.pdf/.mp4`: BO와 POL population을 별도 패널로 분리.
  기존 에너지 교환·보존 진단은 `energy_diagnostics.png/.pdf`로 옮긴다.
- `outer_force/snapshot_*.png/.pdf`: A는 TDPES2/Pol 기준면, B는 EF force,
  C는 Pol0–3 population 시간 이력. 세로 점선은 해당 snapshot의 시간이다.
  일반 MCEF에서는 존재하지 않는 Pol population을 추가하지 않고 기존 2개 패널을 유지한다.
- `polaritonic_wavepacket/snapshot_*.png/.pdf`, `polaritonic_wavepacket.mp4`:
  위 4개 패널은 각 Pol 곡선과 그 상태로 **투영한 핵 밀도**다. 리본 높이는
  모든 상태·프레임에 같은 배율을 쓰며 추가 에너지가 아니다. 아래 패널은
  실제 핵 밀도와 상태별 밀도, 전체 population을 표시한다.
  이는 표면 위의 고전 궤적이나 단일 상태의 force 해석을 뜻하지 않는다.
- `polaritonic_composition/product_weights.png/.pdf`:
  Pol0–3 각각에 대해 `|S0,0>`, `|S0,1>`, `|S1,0>`, `|S1,1>`의 제곱 계수를
  R의 함수로 표시한다. 나머지 product 성분을 회색으로 포함하며 네 상태로 재정규화하지 않는다.
  `initial_composition.json`에는 초기 R0에 가장 가까운 격자점에서의 가중치와
  복소 계수(실수·허수 분리)를 저장한다. 위상은 고유벡터 선택에 의존하고,
  축퇴 상태 내부에서는 고유벡터 회전의 자유가 있다. Pol 번호만으로 LP/UP를 지정하지 않는다.

사용 입력: `observables.npz`의 `pol_local_density`, `pol_population`, `rho_R`와
`analysis_cache.npz`의 `pol_vectors`, `pol_energy`, `fock`.
고정 R에서 **전자+photon**을 대각화한 분석 상태이며 VSC 진동 polariton으로 바꿔 부르지 않는다.
상태별 핵 밀도를 R로 적분한 값이 저장된 population과 일치하는지 확인한다.
별도 refined reference로 재렌더링하더라도 새 상태별 영상은 **원래 투영 basis의 고유값**을 쓴다.
캐시가 없거나 불일치하면 새 산출물을 건너뛰고 `polaritonic_dynamics_manifest.json`에
이유를 기록한다. 다른 그림은 계속 생성한다. 새 TDSE나 대각화를 몰래 실행하지 않는다.

계산 완료 후 전체 그림·동영상을 기존 `report/`에 덮어쓰기:

```bash
for coupling in zero weak strong; do
  python -m multi_component_exact_factorization_cavity.workflow render \
    --out "results/20261026/pen10_native_v1/pen/$coupling" || break
done
```

새 Pol 구성·파동묶음 그림/영상만 빠르게 만들기:

```bash
python -m multi_component_exact_factorization_cavity.polaritonic_dynamics \
  --case results/20261026/pen10_native_v1/pen/strong
```

두 명령 모두 `--no-movies`를 추가하면 정적 그림만 만든다.
새 전용 CLI는 `--out`, `--dpi`, `--fps`도 지원한다.
영상은 실제 저장 시간별로 스트리밍 인코딩하며, 중간 PNG 전체나 원시 wave 복사본을 만들지 않는다.

## 표기 통일 및 기존 그림 덮어쓰기

- `epsilon1_qgeo`, `epsilon1_Rgeo`는 그림에서 각각
  `epsilon^(1)_(geo,q_c)`, `epsilon^(1)_(geo,R)`로 표시한다.
  두 항의 합은 `epsilon^(1)_geo`이며 별도 추가 항이 아니다.
- `epsilon2_geo`는 `epsilon^(2)_geo`로 표시한다.
- 설명 없는 `<h>_x` 대신 `<T_x+V>_Phi`를 표시한다.
  Outer 기댓값은 `<T_x+T_qc+V>_Gamma`이다. V는 분자·photon 조화·LM·DSE
  퍼텐셜의 합이며 각 그림 아래에 정의한다. 전자 운동에너지가 포함되어 있다.
- `bare molecular`도 `<T_x+V_mol>_Phi`로 명확히 표시한다.
  저장 필드 이름과 수치, TDSE, gauge는 변경하지 않는다.

기존 `report_GI_v3`의 같은 이름 PNG/PDF/MP4를 덮어쓰려면 다음을 실행한다.
새 폴더를 추가하거나 이전 결과를 삭제할 필요는 없다. 계산 완료 후에만 실행한다.

```bash
for coupling in zero weak strong; do
  python -m multi_component_exact_factorization_cavity.workflow render \
    --out "results/20261026/pen10_native_v1/pen/$coupling" \
    --report-out "results/20261026/pen10_native_v1/pen/$coupling/report_GI_v3" || break
done
```

원래 자동 생성 `report/` 폴더를 덮어쓰려면 `--report-out`을 생략한다.
`--no-movies`는 PNG/PDF만 갱신하므로 기존 MP4 표기까지 갱신하려면 붙이지 않는다.

## GI 전용 분해와 확대 (2026-10-26)

기존 six-panel slice와 TDSE는 유지한다. `workflow render` 및 계산 완료 후 자동 보고서에
다음 PNG/PDF와 MP4가 추가된다. 원시 파동함수나 새로운 propagation은 필요 없다.

- `outer_force/`: A에 epsilon2 total/GI/GD 및 Pol 0–3 기준면, B는 기존 EF force,
  C는 Pol 0–3 population 시간 이력.
- `outer_GI_decomposition/`: GI2 = conditional Hamiltonian expectation + nuclear geometry.
  왼쪽은 GI/expectation/Pol, 오른쪽은 geometry만 별도 에너지 축으로 확대.
- `epsilon1_GI_decomposition/`: 두 방향 slice. 위는 GI1/conditional expectation/CBO,
  아래는 photon geometry/nuclear geometry/두 항의 합. 같은 행은 공통 y축.
- `GI_reference_zoom/`: 위는 모든 곡선에서 같은 reference 0를 뺀 meV 차이.
  아래는 초기 GI에 가장 가까운 두 기준선을 시간 전체에서 고정해서 확대.
  아래 y범위는 선택한 기준선으로 정하고, GI가 벗어나면 그림에 명시한다.

원래 absolute-energy 그림은 보존된다. 공간 의존 기준면의 차감은 **gauge 변환이 아니며**,
차감한 그림의 기울기를 물리적 force로 읽으면 안 된다. 비교용 두 기준선 선택은
population이나 LP/UP 판정이 아니다. 값이 실제로 거의 같으면 억지로 수직 이동시키지 않는다.
GI 범위는 모든 저장 프레임의 occupied support에서 구하고 total/GD를 제외한다.
원래 정의된 GI 구성항만 사용하며 누락 항은 unavailable로 표시한다.

기존 결과를 새 폴더로 렌더링하는 예:

```bash
python -m multi_component_exact_factorization_cavity.workflow render \
  --out results/20261026/pen10_native_v1/pen/strong \
  --report-out results/20261026/pen10_native_v1/pen/strong/report_GI_v3
```

PNG/PDF만 확인할 때는 `--no-movies` 추가. 현재 report 전체를 생성하므로 디스크 여유를
확인하며, 완료된 TDSE에 대해서만 실행한다. 구 보고서는 자동 삭제하지 않는다.

## PEN slice 보고서 갱신 (2026-10-26)

`epsilon1_slice_analysis`는 cavity 데이터에서 2행×3열이다. 왼쪽 두 열은
R 방향 2개 및 photon 방향 2개 단면, 오른쪽은 photon number와 global CBO
population 시간 이력이다. 점선은 현재 프레임 시각이며 시간축/확률축은 고정된다.
CBO population은 저장된 유한 bare-BO 분석 공간의 투영 확률이고, 단면별 확률이나
독립 full-x CBO 기준면으로 다시 계산한 확률이 아니다. Outside basis도 표시하고
재정규화하지 않는다. 관측량이 없는 이전 캐시는 unavailable로 표시한다.
MCEF에는 photon 관측량을 추가하지 않으며 기존 4-panel slice를 유지한다.
Pol ket 범례는 해당 고유상태의 **에너지 곡선**을 표시한다.

TDSE가 완료된 뒤 새 보고서 폴더로 재렌더링(전파/검증 재실행 불필요):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mcef-cavity-mpl XDG_CACHE_HOME=/tmp/mcef-cavity-cache
for coupling in zero weak strong; do
  python -m multi_component_exact_factorization_cavity.workflow render \
    --out "results/20261026/pen10_native_v1/pen/$coupling" \
    --report-out "results/20261026/pen10_native_v1/pen/$coupling/report_slices_v2" || break
done
```

PNG/PDF만 먼저 확인하려면 `--no-movies`를 추가한다. 이 명령은 A/B를 포함한 전체
보고서를 새 폴더에 생성하므로 추가 디스크 공간이 필요하다. 이전 보고서는 삭제하지 않는다.

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

## 표시 개선 및 저장된 결과의 재렌더

밀도 등고선은 frame peak에 대한 상대값 1e-5부터 .95까지 28개다.
2D 맵은 occupied support+margin을 따라가며, 1D 축은 시간열 전체의 occupied
window 합집합으로 고정한다. total/GI/GD를 보기 좋게 만들기 위해 임의로 이동시키지 않는다.
저장 시각 간격이 1 fs를 넘으면 camera smoothing을 초기화한다. 따라서 0/40/80 fs
스냅샷은 이전 밀도 영역을 끌고 오지 않으며, 촘촘한 영상 프레임만 부드럽게 추적한다.
별도 `reference_focus/`는 absolute GI와 가까운 기준면 두 개를 확대하고,
차이는 명시적으로 meV로 표시한다. '가까운 기준면' 선택은 population이나 LP/UP 판정이 아니다.

현재 positive-marginal gauge에서는 정규화된 밀도가중 평균 GD가 -<H>다.
이는 초기만의 성질이 아니며, 각 R에서 GD가 동일한 상수라는 의미도 아니다.
따라서 raw total의 절대 높이를 static BO/CBO/POL 높이와 곧바로 비교하여
장벽 변화로 해석하지 않는다. force는 여전히 -partial_R epsilon2+alpha_t다.
고정 heavy의 trap은 alpha_h*(heavy_fixed-heavy_trap_center)^2라는 상수다.
기본값 둘 다 9.5이면 0이며, 새 proton R에 harmonic trap을 추가한 것이 아니다.

VSC의 outer reference 라벨 `E^{x+q_c}_k`는 fixed-R 전자–광자 에너지다.
zero coupling에서 낮은 네 개는 S0와 photon n=0,1,2,3의 ladder일 수 있다.
T_R를 포함한 global vibrational LP/UP와 같은 상태들이 아니다.
smoke의 Fock 분석 basis 4개는 coupling된 reference에도 충분하지 않을 수 있다.
다음은 전파하지 않고 static reference만 확대하는 예다. 원래 population은 바꾸지 않는다.

```bash
python -m multi_component_exact_factorization_cavity.reference_refine \
  --case-dir results/20261023/cavity_smoke_v3/vsc/strong \
  --out results/20261023/my_static_refs_v1 --compare-fock 8 --fock 16
python -m multi_component_exact_factorization_cavity.workflow render \
  --out results/20261023/cavity_smoke_v3/vsc/strong \
  --references results/20261023/my_static_refs_v1/refined_refs_vsc_strong.npz \
  --report-out results/20261023/my_vsc_report_v1 --no-movies
```

`--no-movies`를 빼면 기존 실제 프레임으로 MP4도 생성한다. reference 계산은 기존
config/metadata/analysis 및 molecular cache가 필요하고 physics source SHA를 검사한다.
새 `--report-out` 경로를 지정하면 과거 report를 그대로 보존한다.

일반 MCEF는 기존 소형 source snapshot과 postprocessed TDPES member만 읽어 연결한다:

```bash
python -m multi_component_exact_factorization_cavity.mcef_preview \
  --run-dir results/20260916/erf_fullgrid_BOcurvature_compact_qR_analysisBO2_100fs \
  --out results/20261023/my_mcef_display_cache_v1 --times-fs 0 40 80
python -m multi_component_exact_factorization_cavity.workflow render \
  --out results/20261023/my_mcef_display_cache_v1 \
  --report-out results/20261023/my_mcef_report_v1 --no-movies
```

이 경우 q=proton(가로), R=heavy(세로), 기준면은 ordinary BO다.
기존 external-harmonic convention에 맞춰 effective TDPES를 복원하고,
BO에 이미 포함된 trap을 중복으로 더하지 않는다. source는 저장된 density continuity
진단을 그대로 사용하며, 원시 TDSE wave를 다시 계산하지 않는다.
선택 시각은 가까운 실제 저장시각을 표시한다(이번 예: 0,40.153,80.065 fs).
일치하는 independent bridge/force cache가 없으면 값을 만들어서 채우지 않는다.

## 별도 double-well 분자 파라미터 탐색 (2026-10-23)

기존 기본값/TDSE/결과를 바꾸지 않고 `double_well_search.py`로 ground BO를 탐색했다.
외부 quartic potential이나 인위적인 반응 장벽은 추가하지 않았다.
고정 site ±9.5, 질량 1836, 기존 Shin–Metiu potential 식을 유지한다.
아래 값은 논문 파라미터를 그대로 복제한 것이 아니라 실제 수치 탐색 결과다.

| 후보 | 변경값 | 우물 위치 a0 | bare 장벽 meV | local v01 meV | 장벽−local v1 meV |
|---|---|---|---|---|---|
| 최소 변경/깊은 우물 | erf_r_lx=4.0 (기존 3.1), qx=5 유지 | ±2.20361 | 195.748 | 48.719 | 122.396 |
| 얕은 우물/반응 분석 후보 | erf_r_lx=4.0, erf_r_qx=2.8 (기존 5) | ±2.36929 | 50.8175 | 28.8520 | 6.0927 |

두 경우 모두 erf_r_Rx=4 유지. 좌우 전자 smoothing은 같아지지만 기존 핵–핵 항은
왼쪽 bare/right softened 그대로이므로 전역 대칭을 강제로 부여한 모델은 아니다.
반응 영역에서 우물 에너지가 거의 같다. 기존 default는 single_well 그대로 보존했다.

검증: nx=359→719, R sampling 401→801, 전자 box [-22,30]→[-28.5,36.5]
(확대 box nx=899, 같은 fine dx). 매번 최소점 2개/장벽 1개와 양쪽 양의 장벽을 확인했다.
최대 장벽 높이 변화는 깊은 후보 1.80e-8 Ha, 얕은 후보 3.27e-11 Ha였다.
이 기준은 BO topology에 대한 검증이지 real-time TDSE convergence 인증이 아니다.

얕은 후보의 DSE-on fixed-R 전자+광자 ground reference도 zero/weak/strong 모두
double well을 유지했다. 국소 BO 진동 basis로 보정한 strong coupling에서 장벽은
약 51.49 meV다. BO4→6 비교 최대 차이 8.17e-7 Ha, Fock32→64 차이 약 2.3e-16 Ha.
이 reference를 global vibrational LP/UP surface로 부르지 않는다. 국소 6-state 진동
보정은 장시간/전체 양자 동역학을 검증하지 않으며 strong의 의미도 linewidth 기준이 아니다.

초기 local v1은 saddle의 Dirichlet preparation wall로 준비한다. 이 wall은 full TDSE에
추가하지 않는다. 표의 주파수는 정확히 R=0에 wall을 둔 분석 값이다. 실제 native workflow의
격자에서는 preparation endpoint가 R=0.02라서 얕은 후보 gap은 0.00105799553 Ha
(표의 0.00106029026 Ha와 약 0.22% 차이)다. production은 저장된 실제 calibration을
사용해야 하며 wall/초기상태 수렴도 확인해야 한다. p0=0, photon vacuum, S0 준비다.

깊은 후보는 v1이 장벽보다 매우 낮아 짧은 시간의 반응 관찰에 불리하다.
얕은 후보도 즉시 전달을 보장하지 않는다. BO global v1 doublet으로만 계산한
pi*hbar/DeltaE는 약 789 fs이며, 이는 순수한 해당 doublet 조합의 교환 시간 척도이지
실제 준비 상태/전체 전자–광자–핵 반응률이나 TDSE 예측은 아니다.

실제 결과: `results/20261023/double_well_search_v1/`의 `validation/`(깊은 후보),
`shallow_validation/`, `cavity/`, `shallow_cavity/`. 각각 PNG/PDF, 숫자 JSON, 작은 NPZ.
새 폴더로 재현하려면:

```bash
OPENBLAS_NUM_THREADS=1 python -m multi_component_exact_factorization_cavity.double_well_search \
  scan --out results/20261023/my_double_well_scan
OPENBLAS_NUM_THREADS=1 python -m multi_component_exact_factorization_cavity.double_well_search \
  validate --moving-range 2.8 --out results/20261023/my_double_well_validation
OPENBLAS_NUM_THREADS=1 python -m multi_component_exact_factorization_cavity.double_well_search \
  cavity --validation-dir results/20261023/my_double_well_validation \
  --out results/20261023/my_double_well_static_cavity
```

저장소에 포함된 `configs/double_well_{deep,shallow}_proposal.json`은 **분자 설정 제안**이다.
기존 실행 CLI에 연결 가능하지만 production 인증 preset은 아니다. 실제 격자 초기상태 보정:

```bash
OPENBLAS_NUM_THREADS=1 python -m multi_component_exact_factorization_cavity.workflow calibrate \
  --config multi_component_exact_factorization_cavity/configs/double_well_shallow_proposal.json \
  --out results/20261023/my_double_well_calibration
```

`--preset smoke`는 초기상태 method를 single_well로 override하므로 이중 우물 smoke에서
사용할 경우 `--set 'local_state_method="local_wall"'`을 명시해야 한다.
이번 탐색에서는 장시간 전파를 하지 않았고 원래 model을 대체하지 않았다.

## PEN polaritonic surface 전체 비교 (zero / weak / strong)

기존 `references.npz`의 fixed-R 전자–광자 고유값을 사용한다. TDSE 재전파,
새 대각화, gauge 변환은 하지 않는다. 완료된 캠페인의 비교 보고서 생성 시
자동으로 추가되며, 이미 계산한 서버 결과에서는 다음 명령만 실행하면 된다.

```bash
python -m multi_component_exact_factorization_cavity.polaritonic_surfaces \
  --root results/20261026/pen10_native_v1
```

입력: `<root>/pen/{zero,weak,strong}/config.json`, `references.npz`.
`--root`에 `pen` 디렉터리 자체를 주어도 된다. `--out`으로 출력 위치를 바꿀 수 있고,
`--zoom-half-width 1.5`로 초기 R0 주변의 확대 반폭(a0)을 지정할 수 있다.
기본 출력은 `<root>/polaritonic_surfaces/`이며, 반복 실행은 같은 그림만 덮어쓴다.

- `PEN_polaritonic_surfaces.png/.pdf`: 열은 zero/weak/strong. 위는 전체 R 범위,
  아래는 초기 R0 주변 확대. 각 행의 세 패널은 동일한 x/y 범위를 사용한다.
  실선은 저장된 낮은 네 고유값 `|Pol_0>`–`|Pol_3>`, 점선은 bare product
  `|S0,0>`, `|S1,0>`, `|S0,1>`, `|S1,1>`의 에너지
  `E_BO_j(R)+(n+1/2)*omega_c`다. 공통 photon zero-point energy를 포함한다.
- `PEN_polaritonic_pair_zoom.png/.pdf`: 초기 영역에서 저장된 product-character
  지표로 선택한 두 energy index를 R0 주변에서 확대한다. 세 경우 모두 **같은**
  bare `|S1,0>`/`|S0,1>` 에너지 평균을 빼서 meV로 표시한다. 이는 시각화용
  R-dependent 기준 제거이며, gauge 변환도 force를 읽는 potential도 아니다.
  표시된 간격은 실제 표기된 격자 R에서의 값이지 fitted minimum Rabi splitting이
  아니다. 격자가 공명 위치를 정확히 포함하지 않으면 zero에서도 간격이 남는다.
- `surface_manifest.json`: 읽은 경로, 실제 coupling/주파수, basis 크기, 공통 축,
  선택한 pair index와 측정 R/간격, missing-cache 안내를 저장한다.

각 패널에는 `g_cavity` 수치(a.u.), `omega_c`(Ha), DSE on/off가 들어간다.
이 저장소의 정의는 `H_LM=omega_c*g_cavity*mu*q_c`, `H_DSE=g_cavity^2*mu^2/2`다.
`g_cavity` 자체는 Rabi splitting 에너지가 아니다. 값은 config에서 읽으며,
weak/strong에 특정 숫자를 하드코딩하지 않는다. 기존 동역학 비교 범례와
cavity diagnostics 제목에도 같은 `g_cavity` 표기를 사용한다.

네 bare product는 비교선일 뿐, 네 개로 TDSE를 잘랐다는 뜻이 아니다.
`Pol_k`는 에너지 순서이므로 Pol3가 반드시 `|S1,1>`은 아니며, Pol1/Pol2도
전역에서 자동으로 LP/UP라고 부르지 않는다. 고유값은 기존 BO×Fock **분석용**
투영 계산에서 가져오며, 이 plotting만으로 그 basis의 수렴이 인증되지는 않는다.
누락 case/pair는 unavailable로 표시하고, 서로 다른 BO/grid/frequency를 섞은
coupling 비교는 거부한다. 캠페인 자동 렌더에서는 이 실패를 경고로 남기고
이미 완성한 동역학 보고서는 보존한다.

로컬 출력 예시: `results/20261025/local_double_well_preview_v1/pen_5fs/polaritonic_surfaces/`.
이는 기존 5 fs coarse PEN preview의 실제 캐시이며 서버 production 결과는 아니다.
