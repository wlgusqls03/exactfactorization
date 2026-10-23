# 구현·검증 인계 기록 — 2026-10-23

## 실제 완료한 범위

기존 MCEF와 별도 namespace의 real-space PEN/VSC TDSE, 초기상태/공명 보정,
PG nested-EF, 분석용 BO/CBO/POL 투영, compact checkpoint/field 저장,
자동 PNG/PDF/MP4, 서버 실행 driver를 구현했다. 로컬에서는 **CPU**로 검증했다.
GPU 실행이나 장시간 물리 결과가 완료됐다는 보고가 아니다.

작업 시작 HEAD: `9041323bfa7799b6b62228026a6f805e199b502b`.
최종 물리 소스 SHA256:
`15a838dc1a066d25712d70d630febc667c41679e10176ad8dc03c3c76c03cc05`.
물리 파일과 표시/driver 파일의 hash는 구분한다. 실행 archive에는 둘 다 기록한다.

## 검증 결과와 실제 산출물

모든 상대 경로는 저장소 root 기준이다.

| 검사 | 실제 결과 | 기록 |
|---|---|---|
| 기존 MCEF 회귀 | 184 tests PASS | `results/20261023/cavity_extension_audit_v4_final/regression.json` |
| 기존 VSC 회귀 | 237 tests PASS | 위 파일, `vsc.log` |
| 기존 reference CLI | PASS, norm 최대오차 8.993e-15 | 위 폴더의 `legacy_reference_smoke/` |
| 새 cavity tests | 40 tests PASS, 실제 checkpoint 재시작 포함 | `results/20261023/cavity_smoke_v3/new_tests_final.log` |
| PEN/VSC × zero/weak/strong | 6 cases 완료, CPU 3 worker 실행 | `results/20261023/cavity_smoke_v3/` |
| 자동 동영상 | 36 MP4 생성, 모두 ffprobe 검사 통과 | 각 case의 `report/` |
| 시간간격 비교 | 6 cases SHORT_TIME_CHECK_PASS | `results/20261023/cavity_final_validation/` |
| 기존 파일 보존 | 승인된 테스트 외 변경 없음 | `results/20261023/cavity_preservation_final/preservation_after.json` |

최종 smoke는 nx=96, nR=96, nQ=24, dt=.01 a.u., 9 step,
실제 최종시각 0.0021769958286 fs, 저장 4 frame이다.
최대 norm 오차 3.3306691e-15, 최대 energy drift 6.1228800e-14 Ha.
전체 최종 smoke 폴더는 약 **75 MB**다.
짧은 영상이 거의 정지해 보이는 것은 실제 시간 범위가 매우 짧기 때문이다.
이를 긴 물리 동역학의 재현이나 공간 수렴으로 해석하지 않는다.

별도 dt/dt2 검사는 같은 거친 공간 격자에서 0.0200767393 fs까지 전파했다.
6 cases 최저 fidelity 0.9999999999999876, 최대 핵밀도 L1 차이 2.43e-14,
최대 energy drift 5.18e-12 Ha다. 공간·분석 basis·장시간 수렴은 포함하지 않는다.

별도 workflow test에서는 PEN strong을 실제 2 step 전파한 뒤 4 step까지 resume하고,
처음부터 4 step 연속 전파한 결과와 비교했다. 최종 Psi/norm/energy는 절대오차
2e-14 이내로 일치하며 저장 step은 중복 없이 [0,1,2,3,4]다.

smoke 전체 support-weighted RMS의 최댓값은 epsilon2 bridge 9.19e-17,
alpha bridge 8.19e-19, continuum quotient continuity 1.24e-18이다.
독립 native-current continuity residual은 3.17e-8로 별도 기록한다.
이 차이는 유한 격자의 product-rule 잔차를 항등식 closure와 혼동하지 않기 위해 드러낸다.

대표 파일:

- `results/20261023/cavity_smoke_v3/pen/strong/report/A_diagnostics_clean/A_diagnostics_clean.mp4`
- `results/20261023/cavity_smoke_v3/vsc/strong/report/B_diagnostics_clean/B_diagnostics_clean.mp4`
- 각 case `report/bridge_consistency/`, `epsilon1_slice_analysis/`, `bridge_wavefunction/`
- 각 case `report/dynamics_overview.mp4` 및 `dynamics_overview.png`
- 각 case `report/same_geometry_CBO/`, `outer_force/`, `epsilon1_component_slices/`의 정적 그림
- `results/20261023/cavity_smoke_v3/schematics/S1_resonance_tuning.png`
- 같은 폴더의 `S2_PEN_state_correlation.png`, `S3_VSC_state_correlation.png`와 PDF
- 전체 실제 그림 목록: 각 case `render_manifest.json`

개발 중의 `cavity_smoke_v1`, `cavity_smoke_v2`는 최종 결과로 사용하지 않는다.
기존 결과 및 개발 로그는 삭제하지 않았다.

## 모델과 초기조건에서 확정한 점

native 359×750 분자 격자 보정 결과는
`results/20261023/cavity_native_calibration_v2/calibration.json`에 있다.

- frozen-heavy ground BO는 **단일 우물**이다. 최소 R≈−2.604 a0.
- 사용자 선택대로 VSC는 실제 단일 우물 v=1, 추가 momentum kick 없음.
- R=0은 전달 계수용 관측면이지 이 모델의 barrier가 아니다.
- PEN: 기존 archive의 R0=−4, 밀도 표준편차 .3208418789467482, p0=0,
  질량 1836, bare S1, photon vacuum을 계승했다.
- PEN bare 공명 omega=.1010673388102245 Ha.
  보정 g: weak=.0010620134739017283, strong=.010620134739017285.
- VSC bare 공명 omega=.0019742532600742213 Ha.
  보정 g: weak=.0009317307926298129, strong=.00931730792629813.
- g는 PF dipole-gauge lambda이며 기존 ERF smoothing preset과 독립이다.
- VSC strong의 nominal 교환 주기는 약 770 fs다. 40 fs 결과만으로
  진동–광자 에너지 교환 주기 전체를 관찰했다고 주장하지 않는다.
- 초기 bare vacuum에서 coupling을 켜면 영구쌍극자 displacement도 나타난다.
  coherent/fluctuation photon number를 분리해 저장한다.

## 코드 위치 안내

아래는 주요 함수의 시작 줄이며, 유지보수 시 `rg -n '^def |^class '`로 확인할 수 있다.

| 파일 | 줄 / 함수 | 역할 |
|---|---|---|
| `core.py` | 44 `configuration`, 75 `frozen_potential`, 97 `build_molecular`, 130 `initial_wave` | 모델·좌표·초기조건 |
| `backend.py` | 12 `memory_estimate`, 24 `Hamiltonian` | CPU/GPU DST/FFT propagation, observable, 캐시된 potential phase |
| `calibration.py` | 48 `tracked_pair`, 83 `calibrate_strengths`, 130 `local_vibrations`, 164 `prepare` | 실제 resonance/splitting, 상태 추적, 단일 우물 준비 |
| `projections.py` | 29 `projected_vibrational_matter`, 54 `build_analysis`, 126 `project_cbo`, 149 `project`, 196 `references` | 분석 basis·누락분·CBO/POL 기준면 |
| `fields.py` | 66 `analyze`, 304 `metrics` | H-Psi 순간 미분, nested EF, bridge/source/EOM 잔차 |
| `workflow.py` | 94 `cached_prepare`, 115 `resources`, 136 `compact_fields`, 163 `run_case`, 354 `run_campaign`, 380 `main` | cache, 디스크/GPU guard, 재시작, 병렬 실행, 자동 렌더 |
| `validation.py` | 13 `validate` | 동일 격자 dt/dt2 비교 및 production gate |
| `render.py` | 164 `prepare_display`, 332 `bridge_figure`, 353 `source_figure` | 공통 축/색상, A/B diagnostics |
| `render.py` | 382 `consistency_figure`, 422 `slice_figure`, 441 `component_figure`, 479 `bridge_wavefunction_figure` | 정합성·단면·분해·bridge 동역학 |
| `render.py` | 507 `outer_force_figure`, 543 `cbo_comparison_figure`, 615 `overview_figure`, 708 `population_figure` | force·CBO·입자/광자·population |
| `render.py` | 783 `render_case`, 901 `render_comparison` | PNG/PDF 및 프레임 직접 인코딩 MP4 |
| `schematics.py` | 40 `resonance_figure`, 74 `correlation_figure`, 121 `render_schematics` | 설명용 S1/S2/S3, 실제 TDSE 결과와 구분 |
| `audit.py` | 51 `capture`, 70 `verify`, 118 `run_regressions` | 기존 파일 보존 및 회귀 로그 |
| `run_server.sh` | 13–58 | ffmpeg 사전검사, smoke/portable smoke, GPU validate/production |
| `README.md` | 75부터 | 복사해서 실행하는 서버 명령, 물리 해석·제한 |
| `tests/test_cavity_*.py` | 새 파일 5개 | 모델·전파·분해·렌더·workflow 회귀 |
| `tests/test_tdse_report.py` | 44–70 | 사용자 승인: 기존 절대밀도 1e-5 정책에 맞춘 테스트만 수정 |

기존 `core.py`, `propagate.py`, `reference.py`, 기존 renderer는 수정하지 않았다.
과거 protected manifest에는 작업 시작 전 불일치 7개가 이미 있었다.
이번 승인된 테스트 변경과 기존 불일치를 숨기거나 manifest를 갱신하지 않았다.
검증의 의미는 **기존 불일치를 제외하고 무단 변경 없음**이지 과거 manifest 전체 PASS가 아니다.

## 저장/속도 및 서버 실행

정확한 복사 실행 명령은 `README.md`의 “실행” 절을 따른다.
기본값은 full-wave 매 프레임 저장 없음, 마지막 원자적 checkpoint 1개,
압축된 표시 field, float64 검증 metrics, 스트리밍 MP4다.
BO 및 분석 basis는 cache를 재사용하고, 전파 potential phase를 재사용한다.
관측 history는 프레임별 journal에 추가하고 checkpoint/종료 시에만 합쳐
전체 누적 history를 매번 재압축하는 병목을 피한다.

`jobs=1`은 순차 실행, `jobs=3`은 동시 3개다. 3 CPU worker smoke는 검증했다.
native 한 case의 보수적 GPU 예약 추정은 약 3.90 GiB다.
11 GB GPU에서 native 3개 동시는 guard가 거부하므로 우선 `jobs=1`을 쓴다.
GPU 및 native-grid 실행시간/peak memory는 서버에서 측정해야 한다.

## 남은 제한과 다음 검증

- 로컬 GPU가 없어 CUDA 경로의 실기 검증과 native 장시간 production은 미수행.
- clean checkout의 과거 VSC 일부 테스트는 Git 미추적 과거 모듈을 요구한다.
  새 cavity 모듈의 import에는 그 의존성이 없다. `smoke`는 전체 과거 회귀를 유지하며,
  별도 `smoke-portable`은 새 cavity+기존 MCEF+기존 CLI만 검사한다고 명시한다.
  portable 결과를 과거 VSC 전체 회귀 PASS로 해석하지 않는다. 로컬에서는 전체 237개를 실제 통과했다.
- PG만 제공한다. 전역 zero-vector gauge를 구현·검증했다고 주장하지 않는다.
- `S_q+S_adv+S_rel`의 target은 **partial_t |Lambda|^2**다.
  complex Lambda 미분과 T1…T8/EOM 검증은 별도로 평가한다.
- continuum quotient 항등식의 작은 잔차와 native FFT product-rule 오차를 분리한다.
  작은 bridge residual만으로 수렴을 인증하지 않는다.
- CBO reference는 sampled full-x eigenvalues의 표시용 보간이며,
  알려진 핵 scalar/photon harmonic은 native grid에서 정확히 복원한다.
  CBO population은 retained BO 분석 부분공간 투영이므로 outside도 함께 본다.
- VSC vibration–photon LP/UP는 T_R를 포함한 global levels다.
  fixed-R electron–photon curves를 vibrational LP/UP로 개명하지 않는다.
- BO/Fock/vib 수, x/R/Q 격자와 box, dt, support 및 준비 경계 수렴은 서버에서 필요하다.
- 모든 production 상태는 완료해도 `COMPLETE_NOT_CERTIFIED`다.
  작업 실행 완료와 물리적 결론의 검증을 구분한다.

## 2026-10-23 표시 및 reference 재검토

이번 후속 변경은 renderer/어댑터/정적 reference 검사에 한정했다. cavity의 core,
calibration, propagation, projections, fields는 수정하지 않았다.

- cavity 테스트 51개 통과 (`tests_final.log`, 12.139초).
- 기존 density contour / proton-heavy movie / tdse report 회귀 40개 통과
  (`legacy_visual_tests.log`, 96.759초). 이는 이번 변경에 대한 선택 회귀이며,
  위 최초 구현의 전체 회귀 결과와 구분한다.
- VSC zero/strong의 정적 reference를 전체 96 R에서 NFock 8→16으로 비교했다.
  원본 입력 SHA 불변을 검사했다. 기존 동적 population은 NFock4 그대로이므로
  정적 reference 정제가 동역학 분석 수렴까지 개선했다고 주장하지 않는다.
- strong의 초기 밀도 peak에서 fixed-R gap은 약 0.00196335 Ha,
  별도 global vibrational LP/UP splitting은 0.00019739122 Ha였다.
  두 종류의 상태를 구분해 라벨링한다. 이 숫자는 짧은 CPU smoke 계통의 값이다.
- PG의 밀도가중 평균 GD + 총에너지 차이를 저장된 모든 시점에서 검사했다.
  최대 오차는 약 9.2e-16 Ha이며, pointwise GD가 상수라는 뜻은 아니다.
- 일반 MCEF는 기존 소형 source cache와 후처리 scalar만 읽어
  0/40.153/80.065 fs 그림을 생성했다. harmonic 포함 여부를 추적하여
  effective scalar에 한 번만 복원하고 BO에는 중복 추가하지 않았다.
  독립적인 bridge/force cache가 없는 패널은 unavailable로 명시한다.

실제 결과와 로그: `results/20261023/cavity_visual_review_v1/`.
`mcef_report/`, `vsc_strong_report/`에 PNG/PDF가 있으며, 이번 재검토는 정적 그림
재출력이다. TDSE 재전파나 새 장시간 GPU 검증을 수행한 것이 아니다.
