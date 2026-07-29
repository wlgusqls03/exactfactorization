# 1D Multi-Component Exact Factorization

이 디렉터리는 기존 `exact_factorization/`과 완전히 분리된 교육용 구현이다.
왼쪽 고정 양전하 중심, 전자 하나, 양성자 하나, 오른쪽의 무거운 핵 하나를
모두 1차원 실공간에서 다룬다. 움직이는 세 입자의 full wavefunction은

```text
Psi(x,q,R,t) = Phi_{R,q}(x,t) Lambda_R(q,t) chi(R,t)
```

로 두 번 exact factorization한다.

- `x`: 전자 좌표
- `q`: 양성자(수소 핵) 좌표
- `R`: 오른쪽 무거운 핵 좌표
- 왼쪽 중심: 고정되어 있으므로 좌표축 없이 potential에만 포함

기본 Gaussian/Hermite 초기화와 direct dynamics에는 BO 전자 고유상태나
surface hopping을 사용하지 않는다. 단, 사용자가 명시적으로 선택하면
local `H_BO(x;q,R)` eigenstate를 **초기상태를 만드는 용도에만** 쓸 수 있다.

## 기존 2-component EF와 다른 점

| 항목 | 기존 EF | Multi-component EF |
|---|---|---|
| Factorization | `Psi=Phi_R chi` | `Psi=Phi_{R,q} Lambda_R chi` |
| 독립 field | `Phi`, `chi` | `Phi`, `Lambda`, `chi` |
| 배열 차원 | `(r,R)` | `(x,q,R)` |
| Scalar potential | `epsilon(R)` | `epsilon_1(q,R)`, `epsilon_2(R)` |
| Vector potential | `A(R)` | `a(q,R)`, `b(q,R)`, `alpha(R)` |
| PNC 수 | 1개 | 2개 |
| 기본 dynamics 영상 | 4분할 | 6분할 |

## 파일 구성

```text
core.py        격자, soft-Coulomb model, 미분, PNC, EF potential
propagate.py   Phi/Lambda/chi 세 coupled equation의 direct 전파
reference.py   full 3D TDSE 전파 후 nested factorization
compare.py     두 독립 계산의 full-Psi fidelity와 density 비교
visualize.py   snapshot, factor profile, wave/density/gauge-potential 동영상
visualize_3d.py full |Psi(x,q,R,t)|^2의 회전 가능한 standalone HTML
excited_state_analysis.py local electronic-state population과 분해 동영상
```

소스의 주석과 docstring에는 각 물리량의 의미와 배열 shape을 한국어로
적어 놓았다. 처음 읽을 때는 `core.py`의 `initial_factors`,
`geometric_fields`, `instantaneous_functionals`를 읽은 뒤 `propagate.py`의
`coupled_rhs`, `full_step`, `run` 순서로 보는 것이 좋다.

## 주요 배열 shape

```text
nx = 전자 격자점 수
nq = 양성자 격자점 수
nR = 무거운 핵 격자점 수
nt = 저장 frame 수
```

| NPZ key | Shape | 의미 |
|---|---:|---|
| `phi` | `(nt,nx,nq,nR)` | 조건부 전자 factor |
| `lambda_wavefunction` | `(nt,nq,nR)` | 조건부 양성자 factor |
| `chi` | `(nt,nR)` | 무거운 핵 marginal factor |
| `a`, `b` | `(nt,nq,nR)` | 첫 번째 vector potential |
| `alpha` | `(nt,nR)` | 두 번째 vector potential |
| `epsilon_1` | `(nt,nq,nR)` | 첫 번째 scalar potential |
| `epsilon_2` | `(nt,nR)` | 무거운 핵이 느끼는 두 번째 TDPES |
| `theta_1` | `(nt,nq,nR)` | 첫 번째 명시적 gauge function |
| `theta_2` | `(nt,nR)` | 두 번째 명시적 gauge function |
| `epsilon_gd_1` | `(nt,nq,nR)` | 첫 time-Berry connection 진단 |
| `epsilon_gd_2` | `(nt,nR)` | composite 두 번째 time connection 진단 |
| `psi` | `(nt,nx,nq,nR)` | 선택 저장하는 full wavefunction |

## 기본 초기상태

기본 계산은 BO 고유상태를 고르는 방식이 아니라 다음 세 Gaussian EF factor를
직접 지정해서 시작한다.

```text
chi(R,0)                 center R0 =  5.20, sigma_R = 0.38, p_R = 0.0
Lambda_R(q,0)            center q_c(R) = -0.40 + 0.08 (R-R0)
Phi_{R,q}(x,0)           center x_c(q,R)
  = -0.60 + 0.55 (q-q0) + 0.05 (R-R0)
```

양성자의 `sigma_q=0.65`, `p_q=3.0`이고 전자의 `sigma_x=1.0`, `p_x=0.7`이다.
질량은 원자단위로 `m_p=1836 m_e`, `M_H=12000 m_e`이다. 따라서 기준
configuration `(q0,R0)=(-0.40,5.20)`에서 전자 중심은 `x0=-0.60`이지만,
조건부 중심의 작은 좌표 의존성 때문에 초기상태에도 전자-양성자-heavy 핵
상관관계가 들어 있다. 모든 `follow` 계수를 0으로 바꾸면 좌표에 무관한
product형 Gaussian 초기상태가 된다.

전자 초기상태 option은 세 종류다.

- `gaussian`(기본): 위 Gaussian conditional packet
- `hermite`: BO 계산 없이 node를 가진 Hermite-Gaussian packet. 이것은
  실제 molecular Hamiltonian의 energy eigenstate라는 뜻은 아니다.
- `local-eigenstate`: 각 `(q,R)`에서 `H_BO(x;q,R)`를 대각화하여 선택한
  상태로 시작한다. 초기화만 BO형이고 이후 전파는 direct EF이다.

## 두 gauge의 선택

기본 dynamics는 두 단계 parallel-transport gauge를 쓴다.

```text
<Phi|-i d_t Phi>_x = 0
<Gamma_R|-i d_t Gamma_R>_{p,e} = 0,  Gamma_R=Lambda_R Phi
```

따라서 이 representation을 기준으로 기본 `theta_1=theta_2=0`이다. 코드는
이를 scalar functional `epsilon_1=<Phi|H_el|Phi>`와
`epsilon_2=<Lambda|H_pr|Lambda>`로 구현하고, 저장 frame의 시간차분으로
`epsilon_gd_1`, `epsilon_gd_2`가 0에 가까운지 진단한다.

PDF의 gauge 변환을 직접 시험하거나 다른 gauge로 결과를 저장하려면

```text
theta_1 = c_q(q-q0) + c_R(R-R0) + omega_1 t
theta_2 = d_R(R-R0) + omega_2 t
```

를 다음 option으로 지정한다.

```bash
--theta1-q-gradient 0.2 --theta1-R-gradient -0.1 \
--theta1-frequency 0.03 \
--theta2-R-gradient 0.15 --theta2-frequency -0.02
```

이때 factor와 potential은 PDF의 식대로 함께 변환되며 full `Psi`는 변하지
않는다. frequency는 atomic-unit energy, 공간 gradient는 대응하는 vector
potential 단위다. 임의의 비선형 gauge도 이론적으로 가능하지만 현재 CLI는
미분을 해석적으로 정확히 적용할 수 있는 선형형을 제공한다.

## 1. 아주 짧은 학습용 실행

먼저 작은 격자에서 전체 흐름과 shape을 확인한다.

```bash
cd /home/jubjhbjey5/Shin-Metiu

python -m multi_component_exact_factorization.propagate \
  --nx 24 --nq 20 --nR 16 \
  --dt-au 0.002 --t-final-fs 0.002 --save-every 5 \
  --outdir results/multi_component_exact_factorization/study_direct
```

## 2. Full-TDSE reference와 비교

두 명령은 같은 초기조건에서 서로 독립적으로 전파된다.

```bash
python -m multi_component_exact_factorization.reference \
  --nx 24 --nq 20 --nR 16 \
  --dt-au 0.002 --t-final-fs 0.002 --save-every 5 \
  --outdir results/multi_component_exact_factorization/study_reference

python -m multi_component_exact_factorization.compare \
  results/multi_component_exact_factorization/study_reference/multi_component_reference.npz \
  results/multi_component_exact_factorization/study_direct/multi_component_direct_ef.npz
```

Scalar/vector potential은 gauge에 따라 달라질 수 있으므로 `compare.py`는
gauge-invariant한 full-Psi fidelity와 marginal density를 비교한다.

## 3. 논문형 그림과 6분할 dynamics

```bash
python -m multi_component_exact_factorization.visualize \
  results/multi_component_exact_factorization/study_direct/multi_component_direct_ef.npz \
  --outdir results/multi_component_exact_factorization/study_direct/figures
```

생성 파일은 다음과 같다.

```text
initial_state_summary.png            t=0 marginal, 중심, 폭, 운동량, 질량
multi_component_snapshots.png       TDPES와 세 conditional/joint density
factor_wavefunction_profiles.png    peak configuration의 Re/Im/density
multi_component_wavefunction_dynamics.mp4  Re/Im/density 6분할 영상
multi_component_density_dynamics.mp4       논문식 colormap 6분할 영상
multi_component_gauge_potential_dynamics.mp4  gauge/TDPES/vector dynamics
*.gif                                      ffmpeg이 없을 때 자동 대체
```

첫 번째 6분할 영상의 panel은 다음 의미를 가진다.

1. peak `(q,R)`에서 전자 `Phi`의 실수부·허수부·밀도
2. peak `R`에서 양성자 `Lambda`의 실수부·허수부·밀도
3. heavy 핵 `chi`의 실수부·허수부·밀도
4. `int dq |Phi Lambda chi|^2`: full electron-heavy joint density
5. `epsilon_1(q,R,t)`: 첫 번째 factorization의 2D TDPES
6. `epsilon_2(R,t)`: 무거운 핵이 느끼는 두 번째 TDPES

전자 profile 제목의 `(q=..., R=...)`는 실제로 조건부 slice를 뜻한다. 각
frame에서 `|chi(R,t)|^2`가 최대인 `R_peak`를 먼저 고르고, 그 `R_peak`에서
`|Lambda_R(q,t)|^2`가 최대인 `q_peak`를 골라
`Phi_{R_peak,q_peak}(x,t)`를 그린다. 양성자 profile도 같은 `R_peak`에
조건부인 `Lambda_{R_peak}(q,t)`이다. 반면 heavy `chi(R,t)`는 marginal이다.

전자·양성자·heavy 핵의 1D wavefunction profile 세 panel은 전체 격자의
합집합 범위를 사용하므로 같은 물리적 위치 눈금에서 직접 비교할 수 있다.
반면 `(x,R)`, `(q,R)`처럼 서로 다른 두 좌표를 쓰는 heatmap과 TDPES는
실제 계산 grid extent를 사용한다. 표시 범위만 강제로 늘려 데이터가 없는
흰 여백을 만드는 대신 각 map이 panel을 온전히 채우도록 한 것이다.

두 번째 영상은 위쪽 세 factor를 다음 colormap으로 바꾼다.

- `int dq |Lambda|^2 |Phi|^2`: 조건부 전자 밀도 `(x,R)`
- `|Lambda_R(q,t)|^2`: 조건부 양성자 밀도 `(q,R)`
- `|chi(R,t)|^2`: heavy 핵의 1D density color strip

Full wavefunction panel은
`rho_xR(x,R,t)=int dq |Psi(x,q,R,t)|^2`를 사용한다. 이것은 임의의 q slice가
아니라 양성자 자유도를 정확히 적분한 marginal density이며, 2-component
논문의 total electron-nuclear density 그림을 자연스럽게 일반화한 것이다.

모든 colormap에는 고정된 colorbar가 있으며, 영상의 모든 frame에서 같은
색이 같은 수치를 뜻한다. `1e-2` 이상은 `0.00` 형식으로 표시하고,
`1e-3` order 이하처럼 두 자리 고정소수점에서 `0.00`으로 뭉개지는 값은
`1.00e-03` 형식으로 표시한다.

`epsilon_1` 그림은 이제 계산된 `(q,R)` 격자 전체를 표시한다. Low-density
tail의 값도 흰색으로 자르지 않지만, 이 영역은 `1/Lambda`, `1/chi`가 들어간
logarithmic derivative를 regularization하여 얻은 값이므로 점유된 영역보다
물리적 해석의 신뢰도가 낮다. 제목의 `regularized tails`가 이를 표시한다.

복소 3D wavefunction 자체는 2D 화면에 직접 표시할 수 없기 때문에, 논문의
conditional-density 방식처럼 물리적으로 해석 가능한 reduced density를 쓴다.

세 번째 분석 영상은 raw `epsilon_1`, raw `epsilon_2`, `a`, `b`, `alpha`,
`theta_1`, `theta_2`를 같은 frame에서 보여준다. `|Lambda chi|^2` panel을
함께 두어 potential의 어느 영역이 실제 wavepacket support인지 판단할 수
있다. 기존 wavefunction 영상의 TDPES는 매 frame의 peak에서 0으로 shift한
모양 비교용인 반면, 이 영상은 선택한 gauge의 시간 offset까지 유지한다.

## Excited-state dynamics와 population 분석

BO 계산 없이 node가 있는 전자 packet부터 시작하려면 다음처럼 실행한다.

```bash
python -m multi_component_exact_factorization.propagate \
  --electron-initial-state hermite --electron-excitation 1 \
  --outdir results/multi_component_exact_factorization/hermite_excited
```

실제 local electronic excited state `n=1`에서 시작하려면

```bash
python -m multi_component_exact_factorization.propagate \
  --electron-initial-state local-eigenstate --electron-excitation 1 \
  --electron-momentum 0 \
  --outdir results/multi_component_exact_factorization/local_excited
```

`local-eigenstate` mode에서는 electron center/sigma/momentum option을 쓰지
않는다. 각 configuration의 eigenvector phase는 이웃 상태와 overlap이 양의
실수가 되도록 맞춰 q/R derivative의 임의 부호 jump를 줄인다.

Full density가 변하는 모습만으로 `n=1 -> n=0` 전이를 주장할 수는 없다.
다음 분석은 각 시간의 조건부 `Phi`를 local `H_BO` 상태에 투영하여
`P_n(t)`와 state-resolved `(q,R)` density를 만든다.

```bash
python -m multi_component_exact_factorization.excited_state_analysis \
  results/multi_component_exact_factorization/local_excited/multi_component_direct_ef.npz \
  --n-states 3 --format gif
```

생성물은 `electronic_state_populations.png`,
`electronic_state_population_dynamics.gif`, `electronic_state_analysis.npz`다.
여기서 BO basis는 사후 population 분석에만 사용하며 surface hopping은 없다.

## 4. Interactive 3D configuration-space density

Full density `|Psi(x,q,R,t)|^2`를 독립적인 standalone HTML로 만든다.

```bash
python -m multi_component_exact_factorization.visualize_3d \
  results/multi_component_exact_factorization/direct/multi_component_direct_ef.npz
```

생성되는 `figures/multi_component_full_density_3d.html`을 브라우저로 열면
인터넷 연결 없이 자동 재생되며 다음 조작이 가능하다.

- 왼쪽 마우스 드래그: 회전
- 마우스 휠: 확대/축소
- 오른쪽 드래그: 평행 이동
- `Play`/`Pause`: 시간 재생과 정지
- 아래 slider: 원하는 저장 시간 선택
- surface hover: `(x,q,R)`와 `|Psi|^2` 수치 확인

편집기나 채팅의 HTML 미리보기는 보안상 JavaScript를 막아 정적인 `t=0`
그림만 보여줄 수 있다. 그런 경우 실제 browser에서 열거나 다음처럼 local
HTTP server를 사용한다.

```bash
cd results/multi_component_exact_factorization/direct_demo/figures
python -m http.server 8000
```

그 후 browser에서
`http://localhost:8000/multi_component_full_density_3d.html`을 연다.

세 축은 실제 3차원 공간의 `(x,y,z)`가 아니라 전자, 양성자, heavy 핵의
1D 좌표로 이루어진 configuration space임에 주의한다. HTML 크기와 browser
부하를 줄이기 위해 시각화에서만 downsampling하며 원본 NPZ는 바꾸지 않는다.

```bash
# 더 가벼운 HTML
python -m multi_component_exact_factorization.visualize_3d ARCHIVE.npz \
  --max-axis-points 18 --max-frames 30

# 더 촘촘한 3D surface
python -m multi_component_exact_factorization.visualize_3d ARCHIVE.npz \
  --max-axis-points 30 --surface-count 10
```

`visualize_3d.py`에는 `plotly`가 추가로 필요하다. 현재 검증 환경은
`plotly 5.6.0`이다.

## 5. 조금 더 매끄러운 결과

다음은 그림 확인용 권장 시작점이다. 계산 자원에 따라 격자와 최종 시간을
단계적으로 늘린다.

```bash
python -m multi_component_exact_factorization.propagate \
  --nx 48 --nq 40 --nR 36 \
  --dt-au 0.005 --t-final-fs 0.05 --save-every 10 \
  --outdir results/multi_component_exact_factorization/direct
```

고품질 결과를 주장하기 전에는 최소한 다음 scan이 필요하다.

```text
dt-au: 0.005 -> 0.0025
nx:    48 -> 64
nq:    40 -> 56
nR:    36 -> 48
box:   각 density가 양 끝에서 충분히 작은지 확인
```

다음 진단을 함께 확인한다.

- full molecular norm
- Phi와 Lambda의 두 PNC 오차
- reference 대비 full-Psi fidelity
- heavy/proton-heavy/electron-heavy density L1 오차
- wavepacket의 경계 도달 여부
- `density-threshold` 변화에 대한 민감도

## 수치적 주의점

Direct EF에는 `(-i d chi)/chi`, `(-i d Lambda)/Lambda`가 있어 density node와
tail에서 매우 불안정할 수 있다. 이 코드는 `--density-threshold`로 분모를
regularize하지만, 이는 물리적 근사가 아니라 수치적 안전장치다. 최종 결과가
threshold, 격자, time step에 안정적인지 반드시 확인해야 한다.

또한 기본 model은 q와 R 격자 범위를 분리하여 초기 공간 순서를 유지한다.
양성자와 무거운 핵의 crossing이나 같은 위치 configuration까지 연구하려면
configuration domain과 Coulomb regularization을 별도로 검토해야 한다.
