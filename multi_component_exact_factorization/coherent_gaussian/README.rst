고정 폭 복소 Gaussian 실험 경로
================================

기존 계산과 기본값은 수정하지 않았다. 이 모듈은 production MCEF/CTMQC의
대체물이 아니라, full complex G를 Gaussian으로 제한하는 별도 prototype이다.
기존 positive bridge density mixture 방식과 수학적으로 다른 방법이다.

현재 판단 (2026-11-10)
---------------------

* 고정핵 PEN의 complex-G 전파는 40 fs 시험을 통과하는 설정이 있다.
  에너지뿐 아니라 독립 reference와의 complex-state/density/population 오차를 기록한다.
* Gaussian 수 증가의 단조 수렴은 확인되지 않았다. K, 초기 배치, regularization,
  시간적분 tolerance를 함께 비교해야 한다. 에너지 보존은 정확도의 증명이 아니다.
* 움직이는 핵의 8-trajectory / 10 fs 시험에서 Gaussian-POL은 native POL과
  비슷한 오차였으나, 16-trajectory / 40 fs 시도는 18 fs 이후 RHS 예산에 도달했다.
  EL보다 조금 비싸고 POL보다 빠르다는 목표는 아직 달성하지 못했다.
* 원래 MCEF에는 1/|q+9.5|의 비적분 Coulomb pole이 있다. 무한지지 Gaussian을
  그대로 대입하지 않았다. MCEF는 명시적 LOCAL polynomial surrogate만 지원한다.
  40 fs 탐색은 13 fs부터 domain/reference 범위를 벗어나 정확도 해석이 무효다.
* Real-electronic PEN은 모든 nx 전자 성분을 유지한다. BO K=8의 40 fs coarse
  nx=17 시험은 최대 L2 0.0037 수준이었다. production nx=359 수렴은 미검증이다.
  같은 real-electronic CBO는 local quadrature가 미수렴하여 현재 preflight가 거부한다.
  2026 two-state PEN의 CBO 시험과 구분할 것.

모듈
----

core.py: complex coherent TDVP, square-root Schur, 선택적 norm/energy-rate 제약.
models.py: PEN analytic BO, CBO weak kinetic, real-electronic PEN, MCEF surrogate.
experiment.py: fixed-R propagation, 독립 Hermite reference, residual/오차 기록.
compare.py: 기존 Python EL/POL CTMQC와 QD, experimental Gaussian-projected POL.
benchmark.py: 동일 ensemble의 RHS 비용, 공통 시간구간 오차 및 품질 audit.
render.py: conditional density/current, population, conservation/error PNG/PDF/MP4.
derive_report.py: 수식 유도·증명 및 실행 receipt를 합친 한국어 PDF.

정확한 수학 및 한계는 ../../docs/coherent_gaussian_derivation_ko.pdf 참조.
이 문서의 상대경로는 저장소 루트의 docs/ 아래를 뜻한다.

전파 변수와 q 점
---------------

전파 변수는 z[K]와 d[Ne,K]이다. C[Ne,Nq] 또는 G[Ne,Nq]는 전파하지 않는다.
BO PEN의 Hamiltonian 적분은 해석적이다. CBO/분자에서는 Gaussian pair별
local Gauss-Hermite 적분점으로 matrix elements를 계산한다. 이 점은 전파 격자가 아니다.
진단용 Hermite/Fock reference와 plot q 점은 별도로 있으며 RHS에 reference를 넣지 않는다.

Gaussian-POL은 native POL CT source/force를 재사용하기 위해 일시적인 Fock 투영을
수행한다. 따라서 polaritonic diagonalization과 integrated-force histories 비용이
남는다. 이 경로를 독립적인 저비용 EF-G/MLS closure 또는 저자 G-CTMQC라고 부르지 않는다.
R-dependent BO/CBO trajectory force closure 및 exact outer EF fields는 미구현이다.

실행 순서: 저장소 루트에서
-------------------------

기존 MCEF conda 환경을 사용한다. 아래 명령은 CPU 한 thread로 실행한다.
GPU를 사용하지 않는다. 기존 GPU TDSE 경로는 변경하지 않았다. 새 output만 허용한다::

  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python -m unittest discover -s tests -p 'test_coherent_gaussian.py' -b

먼저 작게 시험한다. PNG/PDF는 자동 생성되고 --movies를 주면 MP4도 생성된다::

  python -m multi_component_exact_factorization.coherent_gaussian.experiment \
    --system pen-na --basis bo --components 4 --g 0.01 \
    --end-fs 1 --every-fs 0.1 --shape-regularization 1e-7 \
    --energy-constraint --movies \
    --out results/20261110/coherent_server/pen_pilot

고정핵 40 fs 검증 (이것은 핵이 움직이는 40 fs TDSE/CTMQC와 다름)::

  python -m multi_component_exact_factorization.coherent_gaussian.experiment \
    --system pen-na --basis bo --components 4 --g 0.01 \
    --end-fs 40 --every-fs 0.5 --nf-reference 48 --max-step 0.5 \
    --shape-regularization 1e-7 --energy-constraint --movies \
    --out results/20261110/coherent_server/pen_bo40_K4

--basis cbo로 바꾸면 CBO 전개를 비교한다. --system pen-rabi는 Rabi benchmark다.
--components 1/2/4/8/12를 각각 새 outdir에 실행한다. --energy-constraint를 빼서
보존 제약 on/off를 비교할 때 나머지 tolerance/max-step 설정은 동일하게 둔다.
--g 0, --g 0.001, --g 0.01 및 --dse를 명시적으로 선택할 수 있다.
2026 모델의 g는 splitting이 아니며, 약/강 coupling을 모든 모델에서 같은 g로
해석하지 않는다.

기존 real-electronic PEN의 고정핵 시험::

  python -m multi_component_exact_factorization.coherent_gaussian.experiment \
    --system pen-real --R -4 --nx 17 --basis bo --components 8 \
    --g 0.001 --dse --end-fs 1 --nf-reference 32 \
    --out results/20261110/coherent_server/real_pen_pilot

이 명령의 명시적 g=0.001은 API의 g=None 자동 calibration과 다른 설정이다.
RealPen(g=None)은 bare S1-S0 resonance와 2-state splitting/omega=0.1 추정으로
g를 정한다. production과 비교하려면 반드시 기존 production config의 g/omega/grid를
맞춰야 한다. 기존 nx=359를 nx=17로 바꾼 결과는 물리적 수렴 결과가 아니다.

원래 MCEF의 국소 benchmark (명칭에 local을 반드시 유지)::

  python -m multi_component_exact_factorization.coherent_gaussian.experiment \
    --system mcef-local --basis primitive --nx 17 --components 4 \
    --end-fs 10 --every-fs 0.5 --nquad 24 --nf-reference 48 \
    --shape-regularization 1e-5 --energy-constraint --movies \
    --out results/20261110/coherent_server/mcef_local10

PEN의 native EL/POL/QD 비교부터 재현한다::

  python -m multi_component_exact_factorization.coherent_gaussian.compare \
    --out results/20261110/coherent_server/ct_compare10 \
    --model nonadiabatic --resonance on --end-fs 10 --every-fs 0.5 \
    --ntraj 8 --seed 2026 --nf 8 --nr 128 --dt 0.05 --K 4 \
    --regularization 1e-5

--algorithm ehrenfest는 기존 native mean-field force/source를 사용한 분리 시험이다.
--model rabi, --resonance off, --g 및 --dse 옵션도 있다. 초기 preparation은 공통
bare S1 x vacuum이다. EL은 vacuum Wigner sampling이며 논문의 upper-CBO 준비와 다르다.
소수 궤적은 성능/정합성 pilot이며 sampling 수렴 결과가 아니다.
40 fs Gaussian CT 시도는 비용 폭증이 확인되어 아직 production으로 권하지 않는다.

비용 측정::

  python -m multi_component_exact_factorization.coherent_gaussian.benchmark \
    --ntraj 16 --nf 12 --components 1 2 4 8 --repeats 20 \
    --out results/20261110/coherent_server/cost.json

재전파 없이 그림/동영상 재생성::

  python -m multi_component_exact_factorization.coherent_gaussian.render \
    --run results/20261110/coherent_server/pen_bo40_K4 --movies

산출물 / 실패 처리
-----------------

config.json, metrics.json, summary.json, states.npz, checkpoint.npz,
plots/conditional_*.png, conditional_summary.pdf, conditional_dynamics.mp4.
비교는 qd/el/pol/gaussian 하위 디렉터리와 plots/method_comparison.png/pdf,
accuracy_audit.json을 만든다. 기본 comparison movie는 만들지 않으며 fixed-R
conditional movie와 구분한다.

checkpoint.npz는 상태 snapshot이다. 자동 restart CLI는 아직 없다.
실패 CLI는 exit code 2이며, 저장된 부분 결과와 error를 보존한다.
norm/energy를 결과에서 덮어쓰지 않는다. quadrature, reference-tail, local-domain
gate가 실패하면 계산을 신뢰 가능한 성공으로 표시하지 않는다.

PDF 생성 (reportlab 4.3.x 필요, propagation dependency와 분리 가능)::

  python -m multi_component_exact_factorization.coherent_gaussian.derive_report \
    --out docs/coherent_gaussian_derivation_ko.pdf \
    --roots results/20261109/coherent_gaussian_v1 \
            results/20261109/coherent_gaussian_v2 \
            results/20261110/coherent_gaussian_final

회귀 보존
---------

기존 gaussian_trajectory, gaussian_projection_experiments, pen_ctmqc_2026,
pen_perspectives_2026, real-space MCEF/cavity TDSE의 파일/기본값/체크포인트는
수정하지 않았다. 새 파일과 별도 결과 폴더만 추가했다.
