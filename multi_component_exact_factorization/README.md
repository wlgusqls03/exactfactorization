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
- 왼쪽 중심: 고정되어 potential에 포함되며 전자 hard-wall의 왼쪽 끝

전자 초기상태는 모든 `(q,R)`에서 local `H_BO(x;q,R)`를 풀어 얻은
고유상태로 만든다. BO 계산은 물리적인 **초기 전자상태 구성**에만 사용하며,
그 뒤 시간 전파는 세 coupled exact-factorization 방정식을 직접 푼다.
Surface hopping은 사용하지 않는다.

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
dynamics_analysis.py 실제 1D marginal, 전자 이동, BO-state 종합 분석
```

RTX GPU 전파기는 CPU 기준 구현과 분리된
`multi_component_exact_factorization_gpu/`에 있다. GPU 서버 설치, 정밀도
검증 및 실행 순서는 해당 폴더의 `README.md`를 따른다.

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
| `pnc_error` | `(nt,)` | 저장된 factor의 실제 PNC 잔차 |
| `pnc_projection_correction` | `(nt,)` | RK substep 뒤 PNC 투영 전 최대 이탈 |
| `psi` | `(nt,nx,nq,nR)` | 선택 저장하는 full wavefunction |

## 기본 초기상태

전자 초기상태는 모든 nuclear configuration에서 local BO 고유상태다.

```text
H_BO(x;q,R) phi_n(x;q,R) = E_n(q,R) phi_n(x;q,R)
Phi_{R,q}(x,0) = phi_n(x;q,R)
```

Nuclear 초기상태는 사용자가 지정한 `q0`, `R0`를 각각 Gaussian 중심으로
사용한다. 이전처럼 경험적인 `follow` 계수를 더하지 않으며, q와 R 초기
Gaussian은 서로 독립적이다.

```text
Lambda_R(q,0) = N_q exp[-(q-q0)^2/(4 sigma_q^2) + i p_q(q-q0)]
chi(R,0)      = N_R exp[-(R-R0)^2/(4 sigma_R^2) + i p_R(R-R0)]
```

선택한 전자 BO 표면 `E_n(q,R)`에서 다른 좌표를 중심값에 고정한 두 개의
1차원 energy slice를 만든다.

```text
E_q(q) = E_n(q,R0)
E_R(R) = E_n(q0,R)

k_q = d^2 E_q(q)/dq^2 at q=q0
k_R = d^2 E_R(R)/dR^2 at R=R0

sigma_q = (1/(4 m_p k_q))^(1/4)
sigma_R = (1/(4 M_H k_R))^(1/4)
```

각 이차미분은 해당 1차원 slice에서 중심에 가장 가까운 세 energy를 이차식으로
fit하여 구한다. `d^2E/(dq dR)` 혼합미분과 Hessian 대각화는 사용하지 않는다.
`Lambda_R(q,0)`의 중심과 폭은 모든 R에서 동일하다.

기본 중심은 `(q0,R0)=(2.0,4.2)`이며 `--q0`, `--R0`로 바꾼다. 중심이 grid와
정확히 일치하지 않으면 고정하는 다른 좌표 방향으로 energy를 선형보간한 뒤
1차원 fit을 수행한다.

이전 기본점 `(-0.4,5.2)`은 현재 ground BO 표면에서 두 대각 곡률이 음수라
자동 조화 폭을 정의할 수 없으므로 기본값에서 제외했다. 그 위치를 꼭 써야
한다면 별도 물리 모델에서 정당화한 양의 force constant를 함께 지정해야 한다.

`sigma_q`, `sigma_R`는 각각 `|Lambda|^2`, `|chi|^2`의 표준편차다. 유한
box에서는 각 1차원 wavefunction을 격자 적분으로 다시 정규화한다. 자동
곡률 대신 알려진 양의 값을 쓰려면 `--proton-force-constant`,
`--heavy-force-constant`를 지정한다. 어느 한 곡률이라도 0 이하이면 해당
1차원 harmonic 폭이 정의되지 않으므로 코드가 중단된다.

질량 기본값은 `m_p=1836 m_e`, `M_H=12000 m_e`, 두 nuclear momentum 기본값은
`p_q=p_R=0`이다. Momentum 위상 기능은 남아 있으므로 필요할 때만 예를 들어
`--proton-momentum 1.0 --heavy-momentum -0.2`처럼 지정한다. 전자는
`--electron-excitation n`으로 local BO 상태 번호를 선택하며, Gaussian/Hermite
전자 초기화 option은 제거되었다.

## 기본 격자와 전자 hard wall

기본 spacing은 세 좌표에서 모두 `0.08 a0`로 맞췄다.

| 좌표 | 물리적 범위 | 점 수 | spacing | 경계 |
|---|---:|---:|---:|---|
| electron `x` | `(-6,8)` interior | 174 | 0.08 | 양쪽 Dirichlet hard wall |
| proton `q` | `[-3.36,3.6)` | 87 | 0.08 | periodic numerical box |
| heavy `R` | `[3.0,5.4)` | 30 | 0.08 | periodic numerical box |

전자 왼쪽 경계는 항상 `--left-position`과 같으며 기본값은 `-6 a0`이다.
경계점 자체에서는 `Phi=0`이고 배열에는 174개 interior point만 저장한다.
전자 kinetic은 FFT가 아니라 DST-I sine basis를 사용하므로 왼쪽 벽을 넘어
오른쪽으로 wrap-around하지 않는다. `--x-min` option은 없으며 오른쪽 벽만
`--x-max`로 바꾼다.

왼쪽 경계 위치와 고정 중심 전하는 서로 독립적인 option이다.

```bash
--left-position -6.0   # hard wall과 왼쪽 고정 중심의 위치
--left-charge 0.0      # 왼쪽 고정 중심 전하 Z_L
```

`Z_L`의 기본값은 `0`이다. 따라서 기본 계산에서 `x=-6` hard wall은 유지되지만
왼쪽 중심이 만드는 electron attraction, proton repulsion, left-heavy repulsion
soft-Coulomb 항은 모두 0이다. 예전처럼 `Z_L=+1` 중심을 사용하려면
`--left-charge 1.0`을 명시한다.

Heavy 핵은 질량이 크고 짧은 dynamics에서 이동이 작으므로 기존보다 R box를
좁히고 점 수도 줄였다. 단, 더 긴 시간 전파에서는 density가 R 경계에
도달하지 않는지 확인하고 필요하면 범위를 넓혀야 한다.

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

먼저 짧은 시간 동안 전체 흐름과 shape을 확인한다. 자동 계산되는 nuclear
폭이 좁으므로 기본 격자보다 점 수를 크게 줄이지 않는 편이 좋다.

```bash
cd /home/jubjhbjey5/Shin-Metiu

python -m multi_component_exact_factorization.propagate \
  --nx 174 --nq 87 --nR 30 \
  --dt-au 0.002 --t-final-fs 0.002 --save-every 5 \
  --outdir results/multi_component_exact_factorization/study_direct
```

## 2. Full-TDSE reference와 비교

두 명령은 같은 초기조건에서 서로 독립적으로 전파된다.

```bash
python -m multi_component_exact_factorization.reference \
  --nx 174 --nq 87 --nR 30 \
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

실제 local electronic excited state `n=1`에서 시작하려면 다음처럼 실행한다.

```bash
python -m multi_component_exact_factorization.propagate \
  --electron-excitation 1 \
  --outdir results/multi_component_exact_factorization/local_excited
```

각 configuration의 eigenvector phase는 이웃 상태와 overlap이 양의 실수가
되도록 맞춰 q/R derivative의 임의 부호 jump를 줄인다. 선택한 excited BO
표면의 q/R 독립 local curvature로 nuclear 폭도 다시 계산한다. Excited
표면에서 폭이 더 좁아져 grid-resolution 경고가 나오면 세 축 spacing을 함께
줄여 convergence를 확인한다.

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

## 실제 이동을 보는 dynamics 분석

조건부 wavefunction 그림과 별도로, 저장된 trajectory에서 세 입자의 실제
marginal, 전자 이동, BO-state 분해를 한 번에 만든다.

```bash
python -m multi_component_exact_factorization.dynamics_analysis \
  results/multi_component_exact_factorization/local_excited/multi_component_direct_ef.npz
```

기본 출력 폴더는 archive 옆의 `dynamics_analysis/`이고 다음 파일이 생긴다.

```text
marginal_dynamics.png       rho_e(x,t), rho_p(q,t), rho_H(R,t), 평균±폭
electron_transfer.png       difference density, 좌/우 population, 이동량
nonadiabatic_summary.png    BO gap, rho_qR, BO population, state별 rho_n
dynamics_observables.npz    그림에 사용한 모든 수치 배열
```

전자 좌/우 경계는 기본적으로 `(q0+R0)/2`이다. 다른 경계를 쓰려면
`--electron-divider 3.0`처럼 지정한다. 마지막 frame 대신 중간 snapshot은
`--frame 10`으로 고른다. BO projection 없이 marginal과 전자 이동만 빠르게
만들려면 `--no-bo`를 사용한다.

## 긴 시간 전파와 계산 비용

필요한 step 수는 대략

```text
n_steps = t_final_fs * 41.341 / dt_au
```

이다. 예를 들어 `dt_au=0.005`이면 1 fs에 약 8,269 step, 10 fs에 약
82,683 step이 필요하다. `--save-every`는 저장 용량과 후처리 시간을 줄이지만
전파 step 수 자체는 줄이지 않는다.

긴 계산은 다음 순서로 준비하는 것이 안전하다.

1. 짧은 구간에서 `dt_au`를 두세 값으로 바꾸어 norm, PNC, observable이
   수렴하는 최대 time step을 찾는다.
2. 각 grid 점 수를 줄인 시험과 원래 grid를 비교해 허용 가능한 최소 격자를
   찾는다. 총 비용은 대략 `nx*nq*nR*n_steps`에 비례한다.
3. `save-every`를 늘려 저장 frame 수를 100--300개 정도로 제한한다.
4. 검증된 설정으로 1 fs를 먼저 수행한 뒤 10 fs로 확장한다.

현재 구현은 NumPy/SciPy CPU 코드다. 여러 CPU core는 서로 다른 convergence
계산을 동시에 돌리는 데에는 유용하지만, 한 trajectory는 메모리 대역폭의
영향이 커서 core 수만큼 빨라지지 않는다. 전자 DST는 기본적으로
`--fft-workers -1`로 사용 가능한 CPU core를 모두 쓰며, 다른 작업과 core를
나눠야 할 때는 `--fft-workers 2`처럼 제한할 수 있다. GPU를 쓰려면 CuPy/JAX 같은 backend로
큰 3D 배열 연산과 DST를 함께 옮겨야 하며, 현재 코드에 GPU option만 켜는 것으로는
동작하지 않는다. 정확도를 낮추는 임의의 큰 `dt_au`보다 grid/time-step 수렴을
먼저 확인해야 한다.

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
`plotly 5.6.0`이다. 전자 hard-wall DST와 tridiagonal local BO solver에는
`scipy`가 필요하며 현재 검증 환경은 `scipy 1.7.3`이다.

## 5. 조금 더 매끄러운 결과

다음은 그림 확인용 권장 시작점이다. 계산 자원에 따라 격자와 최종 시간을
단계적으로 늘린다.

```bash
python -m multi_component_exact_factorization.propagate \
  --nx 174 --nq 87 --nR 30 \
  --dt-au 0.005 --t-final-fs 0.05 --save-every 10 \
  --outdir results/multi_component_exact_factorization/direct
```

고품질 결과를 주장하기 전에는 최소한 다음 scan이 필요하다.

```text
dt-au: 0.005 -> 0.0025
spacing: 0.08 -> 0.06 -> 0.04 a0 (세 축을 함께 조밀하게)
box:   각 density가 양 끝에서 충분히 작은지 확인
```

다음 진단을 함께 확인한다.

- full molecular norm
- Phi와 Lambda의 저장 PNC 오차 및 substep projection 보정량
- reference 대비 full-Psi fidelity
- heavy/proton-heavy/electron-heavy density L1 오차
- wavepacket의 경계 도달 여부
- 자동 force constant와 Gaussian 표준편차의 격자 수렴
- `density-threshold` 변화에 대한 민감도

## 수치적 주의점

Direct EF에는 `(-i d chi)/chi`, `(-i d Lambda)/Lambda`가 있어 density node와
tail에서 매우 불안정할 수 있다. 이 코드는 `--density-threshold`로 분모를
regularize하지만, 이는 물리적 근사가 아니라 수치적 안전장치다. 최종 결과가
threshold, 격자, time step에 안정적인지 반드시 확인해야 한다.

`pnc_projection_correction`은 점유율이 거의 0인 조건부 tail까지 포함한
substep 전역 최댓값이라 크게 보일 수 있다. 저장 factor 자체의 PNC 잔차는
`pnc_error`이며, 물리적 정확도는 full norm과 reference fidelity도 함께 보고
판단해야 한다.

또한 기본 model은 q와 R 격자 범위를 분리하여 초기 공간 순서를 유지한다.
양성자와 무거운 핵의 crossing이나 같은 위치 configuration까지 연구하려면
configuration domain과 Coulomb regularization을 별도로 검토해야 한다.
