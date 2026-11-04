저자 G-CTMQC 실행기: 독립적인 Python CTMQC와 분리한 비교 경로
===========================================================

중요한 결론
-----------
G-CTMQC는 저자들의 generalized CTMQC 소프트웨어다. G가 붙었다고
CTMQC-E(에너지 보존 보정)나 무조건 더 정확한 적분기를 의미하지 않는다.
이 adapter는 저자 Fortran 소스를 실제로 빌드/실행한다. 기존 Python
pen_ctmqc_2026의 이름만 바꾸거나 그 결과를 재사용하는 방식이 아니다.
기존 TDSE/MCEF/Python CTMQC 전파기 및 이전 결과는 수정하지 않는다.

기준 자료:
  https://arxiv.org/html/2604.20538v1 (2026 PEN 논문, IV 및 Appendix A)
  https://gitlab.com/agostini.work/g-ctmqc
  https://github.com/lauvergn/QuantumModelLib
공개 G-CTMQC commit: b7a2a6d95656e4e0535e744b102670d85b07fb88.
이는 공개 소스의 고정 버전이지 논문 계산에 사용된 정확한 commit/input이라는
뜻은 아니다. paper의 모든 seed/M_parameter/substep 설정은 확인되지 않았다.
QML 및 두 의존 라이브러리도 build.py에 commit과 archive SHA256을 고정했다.
부모 archive에 포함된 낡은 의존 라이브러리 트리는 제외하고 각각 별도 추출한다.

현재 Python과 정확히 다른 점
-----------------------------
1. 전파 basis와 갱신 순서
   Python core.py Engine.step: 고정 diabatic/product basis에서 X,P,C,f를
   동시에 RK4 전파. 각 RK stage에서 ensemble Q와 force를 다시 계산한다.
   G-CTMQC src/time_evolution.f90 evolution: 핵은 velocity-Verlet,
   전자는 coefficients_evolution.f90 RK4_coeff/cdot 경로. Q/k_li는
   ensemble step 뒤 갱신하여 다음 단계에 사용한다.
   저자는 adiabatic CBO/POL basis, QML의 eigenvector phase following/NAC를 쓴다.
   RK4_coeff는 Runge–Kutta–Gill이라고 명명되어 있지만 cdot의 내부 stage에서
   해당 계수만 바꾸고 나머지 계수는 고정한다. Python의 전체 결합 RK4와
   같은 stage algorithm/수렴 차수라고 단정하지 않는다.

2. quantum momentum과 intercept 보호
   Python quantum_momentum.py coupled_terms: 보정 중심이 ensemble 범위에서
   10 density bandwidth 이상 벗어나면 해당 상태쌍/좌표 전체의 Q를 끈다.
   G-CTMQC coherence_corrections.f90 quantum_momentum: Gaussian-mixture를
   계산하고, 각 trajectory의 corrected center 거리 검사 -> density center
   fallback -> 0 순서. 기본 M_parameter=100, DoubleIntercept=false.
   부호 convention이 다르므로 Q 자체 부호만 비교하지 말고 최종 계수/힘을
   비교해야 한다. Python의 보호 알고리즘과 저자 알고리즘은 동일하지 않다.

3. force-history reset
   Python 기본 population threshold=0.001; 저자 hard-coded threshold=0.005.
   둘 다 threshold 밖에서 적분력을 초기화한다. 저자는 state force와
   평균 force의 차이가 numerical zero일 때도 초기화한다.
   따라서 G-CTMQC라고 불연속적인 threshold/reset 자체가 사라지는 것은 아니다.

4. norm 검사
   Python: 정규화 전 norm 오차가 1e-5를 넘으면 중단하는 기본 guard.
   저자: 각 electronic step 뒤 항상 정규화하며 동일한 중단 guard는 없다.
   adapter는 원래 정규화 동작을 바꾸지 않고, 정규화 전 오차의 interval 최대를
   binary frame/validation.json에 추가한다. 오차가 크면 경고를 남긴다.
   norm=1이나 COMPLETE_NOT_CERTIFIED는 장시간 수렴 증명이 아니다.

5. 비용
   Python 기본 corrected Q: Ntraj*Nstate^2 규모, NxN Gaussian 저장 안 함.
   저자 기본 Gaussian 처리: O(Ntraj^2), NxN float64 matrix 저장.
   Ntraj=1000이면 matrix만 7.6 MiB, 10000이면 763 MiB / 실행 case.
   Fortran이라는 이유만으로 항상 더 빠르거나 GPU를 쓴다고 주장하지 않는다.

물리 설정과 어댑터 수정 범위
----------------------------
대상은 2026 비단열/Rabi, on/off, zero/weak/strong. 기본은 strong 8개
(2 model x 2 resonance x EL/POL). --case all은 총 24개다.
  nonadiabatic: omega_on=.17, omega_off=.097 Ha
  rabi:        omega_on=.17, omega_off=.12 Ha
  zero/weak/strong: g=0/.001/.01
PEN Config/resonance.py의 실제 값을 모두 명시적 QML namelist로 전달한다.
model_potential='read_model', read_nml=true가 중요하다. 이를 빠뜨리면
라이브러리의 compiled defaults를 쓸 위험이 있다.
QML의 off-diagonal dipole은 -Mu12, Python Config는 signed mu12다.
따라서 Mu12_QML=-mu12_Config로 넣는다(현재 +1 대 -1).
QML POL 순서는 [d0,n0; d1,n0; d0,n1; d1,n1], Python은 electronic-major.
입출력 시 [0,2,1,3] permutation으로 일치시킨다.
여기서 d0,d1은 고정 diabatic labels이며 S0,S1은 bare BO eigenstates다.

초기상태는 기존 Engine.initial의 같은 Wigner sample을 파일로 전달한다.
  --initial paper: EL upper local CBO; POL bare S1 x photon vacuum.
  --initial bare: EL도 bare S1 (동일 준비조건을 비교하는 별도 control).
초기 POL은 단순히 특정 diabatic basis vector를 선택하지 않고,
실제 bare S1 coefficient를 QML Vec0에 투영하여 넣는다.
EL photon number는 Wigner/classical estimator이다. 유한 sampling으로
진공에서도 평균이 0에서 벗어날 수 있다. 초기 offset을 임의로 빼지 않는다.

build.py의 instrument()가 바꾸는 부분은 다음뿐이다:
  - 외부 initial coefficient 읽기
  - binary snapshot 출력(X,P,C,energy,MF/CT force,history,k_li)
  - 정규화 전 norm 오차 관찰
actual unified diff는 build directory/adapter.patch에 남긴다.
전파식, quantum momentum, threshold, coupling force는 교체하지 않는다.
snapshot의 MF/CT force는 갱신된 author k_li에서 직접 다시 평가한 진단량이며,
직전 Verlet substep의 acceleration 자체라고 부르지 않는다.
원래 dense text output도 보존한다. RPE의 미초기화 Vdiadiag 열은 사용하지
않고, binary와 독립 Hamiltonian 검사를 사용한다.

기본 nesteps=1을 고정한다. 이 source의 nesteps>1 interpolation에는
integer iestep/nesteps가 있어, 그대로 substep 기능을 노출하지 않는다.
CTMQC-E와 DoubleIntercept는 이번 adapter의 기본/지원 경로가 아니다.
현재 저자 OneD_Photons2 경로는 DSE OFF, photon n=0,1 두 상태만 지원한다.
DSE ON/더 큰 Fock basis는 기존 Python/QD 코드에서 계속 사용할 수 있다.

설치/빌드 (서버에서 한 번)
---------------------------
cd /home/hbji/exactfactorization
conda activate MCEF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gfortran --version
python -m multi_component_exact_factorization.pen_gctmqc_2026.build \
  --out results/_dependencies/gctmqc_b7a2 --download

필수: gfortran, GNU make, 시스템 LAPACK/BLAS. ffmpeg는 동영상에만 필요.
의존 라이브러리를 설치하거나 sudo하지 않는다. 없으면 관리자에게 요청한다.
시스템 runtime liblapack.so.3도 지원하며 NumPy의 private BLAS를 가져오지 않는다.
빌드 폴더는 이 환경에서 약 1.6 GiB(compiler/object/library 포함), 다운로드
archive는 별도 약 73 MiB다. 소스와 라이선스를 유지한다.
빌드 중단 후에는 --retry-build로 source hash를 검사하고 재시도할 수 있다.
--archive-cache EXISTING_ARCHIVES로 네트워크 없이 동일 archive를 재사용한다.

서버 1 fs pilot (먼저 확인할 것)
--------------------------------
python -m multi_component_exact_factorization.pen_gctmqc_2026.workflow \
  --build results/_dependencies/gctmqc_b7a2 \
  --out results/20261104/gctmqc_server_1fs_v1 \
  --model all --resonance both --perspective both --case strong \
  --ntraj 1000 --dt 0.05 --end-fs 1 --every-fs 0.1 --jobs 2 --movies

각 case의 validation.json과 dynamics_summary.png를 먼저 확인한다.
그 다음 40 fs를 새 폴더에서 실행한다. 기존 Python restart는 가져오지 않는다.
native restart도 아직 지원하지 않으므로 반드시 t=0에서 시작한다.

python -m multi_component_exact_factorization.pen_gctmqc_2026.workflow \
  --build results/_dependencies/gctmqc_b7a2 \
  --out results/20261104/gctmqc_server_40fs_v1 \
  --model all --resonance both --perspective both --case strong \
  --ntraj 1000 --dt 0.05 --end-fs 40 --every-fs 0.25 --jobs 2 --movies

하나만 돌리려면 --model nonadiabatic --resonance on --perspective pol --jobs 1.
--m-parameter 50/100/200, dt=.05/.025, Ntraj와 seed를 바꾼 별도 폴더로 수렴검사.
paper의 10000 trajectories를 곧바로 8개 병렬 실행하지 말 것. 메모리뿐 아니라
N^2 비용이 증가한다. 각 case 실패는 격리되어 다른 case는 계속 실행된다.
기존 결과 덮어쓰기/부분 resume는 의도적으로 거절한다.

출력 구조
----------
ROOT/{nonadiabatic,rabi}/{on,off}/dse_off/{zero,weak,strong}/{el,pol}/
  config.json, author_settings.json, input.in, initial_*.dat
  run.log, QuantumModelLib.out, status.json, validation.json
  model_check/model_check.json : 독립 compiled QML V/gradient/NAC 검사
  adapter_frames/frame_STEP.bin : time, raw norm error, 실제 trajectory 자료
  native/ : 저자 원래 text output, energy, STC, dEdt 등
  observables.json : 직접 계산 가능한 양만
  plots/dynamics_summary.png
  plots/trajectory_00000.png, trajectory_middle.png에 해당하는 인덱스, 마지막 PNG
  plots/trajectory_diagnostics.mp4 (--movies)
ROOT/campaign.json : 전체 성공/실패 목록

재렌더(전파 없음):
python -m multi_component_exact_factorization.pen_gctmqc_2026.render \
  --root results/20261104/gctmqc_server_40fs_v1 --movies

QD 및 기존 Python CTMQC 비교:
python -m multi_component_exact_factorization.pen_gctmqc_2026.compare \
  --author-root results/20261104/gctmqc_server_40fs_v1 \
  --python-root results/20261103/ctmqc_pilot_server_v1 \
  --qd-root results/20261030/pen2026_paper_v1 \
  --out results/20261104/gctmqc_three_method_comparison_v1 --t-max-fs 40

사용자 서버의 Python 결과가 results/20261102/...에 있다면 그 경로로 바꾼다.
Python failed 사례와 Rabi/off 1fs 자료를 포함하면 공통 시각까지만 비교한다.
--require-through-fs 40을 추가하면 부족한 그룹은 SKIP, exit code 2를 반환한다.
두 CTMQC와 QD 그림은 direct observables만 비교하며 QD는 finite-Fock reference다.

시간간격 비교(전파 없음):
python -m multi_component_exact_factorization.pen_gctmqc_2026.audit \
  --coarse COARSE_ROOT --fine HALF_DT_ROOT --out timestep_audit.json
공통 timestamp의 차이와 저장시각 보간을 포함한 차이를 분리한다.
공간/시간 interpolation 오차를 적분기 오차라고 단정하지 않는다.

검증/현재 한계
--------------
로컬에서는 40 fs를 수행하지 않았다.
  - EL/POL x 두 모델 x on/off 8개: 128 trajectories, 1 fs
  - dt=.05 및 .025 a.u. 두 세트
  - zero/weak/strong 전체 24개 조합: 각 16 trajectories, .01 fs 테스트
  - 같은 초기 ensemble의 기존 Python 1fs 결과 및 기존 QD와 비교
  - 현재 테스트 실행: unittest discover -s tests -p 'test_pen*.py'
    GCTMQC_BUILD를 지정하면 native 실행기 integration test도 포함한다.
결과: results/20261104/gctmqc_validation_v1/
  dt005/.../plots/ : 8 MP4, 32 PNG
  comparison/.../observables_comparison.png : 4 PNG
  timestep_audit.json : 수치 비교

1 fs 기준 최대 normalized norm error ~1e-15, 최대 energy drift 약2.5e-6 Ha.
동일 초기 ensemble의 author/Python 1fs photon-number 차이는 최대 약2.2e-5,
조건부 state population 차이는 약1.1e-5. 초기 구간에서 가깝다는 뜻이며,
25--33fs 부근 실패/에너지 drift 문제가 해결되었다는 증명은 아니다.
원래 Python의 실패 원인을 숨기거나 norm guard를 낮추어 완료 처리하지 않았다.
서버에서 author run의 raw norm, energy drift, dt/Ntraj/M 민감도를 검사해야 한다.

1000 trajectories, .2fs 로컬 비용측정(전파/기록/모델검사 포함, 렌더 제외):
  nonadiabatic/on EL 약2.35초, POL 약2.48초.
단순 선형 외삽으로 같은 CPU에서 40fs 약8분/case, 8개 순차 약65분이다.
서버 CPU와 장시간 ensemble 상태, I/O가 달라 보장할 수 없으며, 10000개로
늘릴 때 이 숫자를 그대로 쓰면 안 된다. 먼저 서버의 1fs pilot을 측정한다.
1000개/40fs/8 cases는 every=.25fs라면 대략0.5 GiB, every=.1fs라면 약1.3 GiB
raw+native 저장 예상(그림/영상/한 번의 빌드 용량은 별도). 강결합 8개 기준이다.

CTMQC에는 핵의 복소 wavefunction이 없으므로 exact a,b,alpha,epsilon1,
epsilon2 전체를 고유하게 추출할 수 없다. 이 구현은 이를 날조하지 않는다.
그림의 conditional mean energy는 exact TDPES total/GI 전체와 다르다.
EL에서 Pol/Fock population도 만들지 않는다. POL만 양자 photon population이다.

라이선스/재현성
----------------
G-CTMQC: Copyright Federica Agostini and contributors, LGPL-3.0-or-later.
다운로드한 source/license/COPYING.LESSER.txt 및 모든 원본 notice를 보존한다.
adapter_io.inc와 저자 코드에 적용되는 patch는 LGPL-3.0-or-later로 제공한다.
QuantumModelLib 및 의존성의 원본 LICENCES/license도 build source에 보존한다.
third-party source/binary는 git에 넣지 않는다. build_manifest.json에 archive
checksum, commit, compiler, build command, executable hash를 저장한다.
배포 시 modified source/patch와 해당 LGPL 의무를 함께 지켜야 한다.
