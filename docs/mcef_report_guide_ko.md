# MCEF 핵심 report 읽는 법

기본 렌더링은 `report/` 아래 네 PNG와 두 동영상을 만든다. 각 파일은
서로 다른 질문에 답하며, 패널 문자는 왼쪽 위부터 A, B, C 순서이다.

## 1. `01_particle_motion.png`: 무엇이 움직였는가?

- A의 가로축은 시간(fs), 세로축은 전자 위치 `x`(보어)다. 색은 전자
  marginal probability density이며 흰 선은 평균 위치다.
- B는 같은 방식의 양성자 위치 `q`, C는 무거운 핵 위치 `R`이다.
- D의 실선은 각 평균 위치가 초기값에서 얼마나 이동했는지, 점선은 분포의
  표준편차가 얼마나 변했는지를 보여준다. 점선 증가는 packet spreading이다.
- 붉은 세로 점선은 `|gamma_Phi| > 1`이 처음 검출된 저장 시점이다. 물리적
  전이선이 아니라 local-norm correction이 order-one이 된 수치 경고선이다.

밝은 띠가 이동하면 평균 위치가 움직인 것이고, 띠가 넓어지면 wavepacket이
퍼진 것이다. 경고선 이후에만 급격한 줄무늬가 나타난다면 물리 해석 전에
시간·공간 격자 수렴을 확인해야 한다.

## 2. `02_electronic_transitions.png`: 전자상태 혼합은 있었고 가능한 통로는 무엇인가?

- A의 가로축은 시간, 세로축은 BO population이다. `P_n`은 local BO 상태
  `n`으로 투영한 전역 확률이다. `outside basis`가 작아야 사용한 BO basis가
  충분하다.
- B는 양성자 평균 이동, 양성자 폭 변화, 초기 대비 전자 rearranged density를
  같은 시간축에서 비교한다. 전자 population 변화와 핵 운동이 같은 시점에
  시작하는지 보는 패널이다.
- C와 D의 가로축은 `R`, 세로축은 `q`다. 색은 각각 BO gap `E1-E0`, `E2-E1`이고
  노란 등고선은 q 방향 derivative coupling `|d^q_01|`, `|d^q_12|`다.
- 흰 외곽선 안쪽만 실제 nuclear packet이 점유한 영역이다. 흰 영역 밖의
  gap이나 coupling은 현재 dynamics가 방문하지 않았으므로 직접적인 원인으로
  해석하지 않는다.

작은 gap과 큰 NAC를 nuclear density가 실제로 점유하면서 population이 변하면
비단열 혼합의 물리적 통로가 존재한다. 그러나 이것은 가능성을 보여주는
조건이다. population 변화가 수렴된 물리 결과인지는 4번 그림과 grid scan으로
별도 검증해야 한다.

## 3. `03_exact_potentials.png`: exact potential은 dynamics에 어떻게 작용하는가?

Nested exact factorization은

`Psi(x,q,R,t) = Phi_qR(x,t) Lambda_R(q,t) chi(R,t)`

로 쓴다.

- A의 첫 번째 TDPES `epsilon1(q,R,t)`은 전자 factor `Phi`를 적분해 얻는
  q-R exact scalar surface다. 이는 하나의 BO surface가 아니며 전자 운동,
  electron-nuclear coupling, 시간 의존 효과가 포함된다. 색 단위는 Hartree다.
- B의 `a(q,R,t)=<Phi|-i d_q|Phi>_x`는 q 방향 vector connection이다. 핵
  운동량과 결합하며 단위는 `a0^-1`이다.
- C는 A와 B를 gauge-invariant하게 결합한 proton force
  `Fq = -d_q epsilon1 + d_t a`다. A나 B의 모양은 gauge에 따라 변하지만 이
  조합은 물리적으로 비교할 수 있다.
- D의 두 번째 TDPES `epsilon2(R,t)`은 `Lambda`까지 적분한 바깥 R-only exact
  scalar surface다. `alpha(R,t)`는 그 단계의 vector connection이다. 따라서
  epsilon2는 두 번째 전자상태나 두 번째 BO surface가 아니다.

Scalar potential은 gauge에 따라 시간별 상수만큼 이동할 수 있으므로 그림은
점유 peak의 값을 빼서 경사와 구조를 보여준다. 흰 부분은 density가 peak의
`1e-3`보다 작은 곳이며, 그곳의 ratio-based potential은 해석하지 않는다.

## 4. `04_numerical_reliability.png`: 이 trajectory를 믿을 수 있는가?

- A는 PNC projection이 factor를 얼마나 재배분했는지와 local norm generator
  `gamma_Phi`, `gamma_Lambda`의 최대 크기를 보여준다. 작고 grid refinement에
  따라 감소해야 한다.
- B는 실제 점유영역에서 vector connections와 exact forces의 density-weighted
  RMS다. 갑작스러운 여러 자릿수 증가나 좁은 spike는 미분 해상도 문제의
  신호다.
- C는 보정 전/후 local norm rate와 full norm 오차다. corrected rate와 norm이
  작아도 raw rate가 크면 correction이 큰 오류를 숨기고 있는 것이다.
- D는 q와 R 격자의 바깥 5점에 들어간 확률이다. 값이 충분히 작으면 box 범위
  부족은 주원인이 아니다. 상자 안에서 점이 너무 성기다는 해상도 문제는
  `sigma_q/dq`, `sigma_R/dR`로 따로 본다.

“개선”은 단순히 norm이 작아진다는 뜻이 아니다. 같은 물리 시간에서 dt, nq,
nR을 바꾸었을 때 A-B의 correction과 spike가 감소하고, 동시에 density,
평균·폭, BO population 같은 물리 관측량이 서로 수렴해야 한다.

## 동영상

### `mcef_dynamics_overview.mp4`

A 전자 marginal, B q-R nuclear density, C 전 시간 BO population과 현재 시간선,
D 첫 번째 TDPES를 동기화한다. “핵 packet이 어느 exact surface 영역을 지나며
전자 density와 BO composition이 함께 어떻게 변했는가?”를 보는 영상이다.

### `mcef_exact_potentials.mp4`

위 행 A-C는 전자 factor에서 생기는 `epsilon1`, `a`, `b`이고, 아래 D-E는
proton-heavy/outer factorization의 `epsilon2`, `alpha`다. F는 같은 순간의
nuclear support다. Potential의 요동이 F의 점유영역 안인지 밖인지 반드시
함께 확인한다.

## 생성 명령

```bash
python -m multi_component_exact_factorization.render_all RUN_DIRECTORY \
    --n-states 6 --format mp4
```

기존의 모든 개발용 단면, factor profile, gauge-function 영상, 3D HTML까지
필요한 경우에만 `--all-products`를 추가한다.
