# 7개 로컬 약식 계산 (2026-10-25)

MCEF 1개 + PEN zero/weak/strong + shallow-double-well VSC zero/weak/strong.
기존 물리/전파 코드는 변경하지 않았다. 결과 위치:
`results/20261025/local_double_well_preview_v1/`.

## 실제 실행과 한계

- PEN/VSC: CPU real-grid (nx,nR,nQ)=(95,128,48), dt=0.2 a.u.,
  실제 종료 5.002252637 fs, 21개 저장 시각. 각 case에 자동 MP4 6개 생성.
- MCEF: 기존 CPU reference (63,128,96), dt=0.05 a.u., 약 0.5 fs.
  이 경로는 원래 FD2 전자/FD5 핵 kinetic 및 저장 시각 기반 scalar 미분을 사용한다.
  GPU spectral production이나 정밀한 TDPES 수렴 검증을 대신하지 않는다.
- VSC는 local-left-wall v=1 준비 상태이며 전파 때 준비용 벽을 제거한다.
  smoothing (left electron, right electron, moving electron)=(4,4,2.8).
  PEN은 기존 (3.1,4,5), S1 × Gaussian을 유지한다. 두 분자 모델을 혼동하지 않는다.
- 테스트: cavity 55개, 기존 MCEF 184개 및 legacy CLI smoke 통과.
  PEN/VSC 여섯 경우의 약 0.1 fs dt/2 비교도 통과했다.
- 여섯 cavity 전파의 최대 norm 오차 < 7e-13.
  최대 에너지 변동: VSC < 8.9e-8 Ha, PEN < 9.5e-7 Ha.
  경계 shell 확률 최대: VSC < 2.4e-9, PEN < 5.4e-6.
  공간 격자 수렴 및 장시간 안정성은 아직 인증하지 않았다.
- VSC의 algebraic bridge closure는 약 1e-16이지만, coarse-grid native FFT
  product-rule/outer continuity 잔차는 별개로 남는다. 동일 정의의 항등식이 닫힌다는
  사실만으로 연속체 TDPES의 정확성을 보증하지 않는다.

## 먼저 볼 결과

- `vsc_coupling_comparison.png`: VSC 비교
- `pen_5fs/pen_coupling_comparison.png`: PEN 비교
- `vsc/{zero,weak,strong}/report/`: VSC 그림/영상
- `pen_5fs/pen/{zero,weak,strong}/report/`: PEN 그림/영상
- `mcef/figures/`: 기존 MCEF 그림/영상

PEN/VSC report에는 dynamics_overview 및 A/B diagnostics, bridge consistency,
epsilon1 slices, bridge-wavefunction MP4가 있다. ffprobe로 36개 모두
21프레임 video stream을 확인했다. 별도 `pen/strong`은 먼저 실행한 0.05 fs
예비 smoke이며 위 7-case 비교에는 사용하지 않는다.

5 fs에서 photon number (zero,weak,strong):
- VSC: (약 0, 0.00003477, 0.003294)
- PEN: (약 2e-9 수치 바닥, 0.01059, 0.7355)

VSC strong의 photon number 중 coherent displacement 기여가 약 0.002691이다.
따라서 전부를 진동 들뜸의 광자 전환이라고 해석하지 않는다. VSC product 확률은
세 경우 모두 약 0.227%로 거의 같다. 5 fs만으로 반응 억제를 주장할 수 없다.
PEN product 그래프의 1e-26 수준은 의미 있는 반응량으로 해석하지 않는다.

## 재현

저장소 루트, MCEF 환경 및 ffmpeg 필요. 새 폴더만 허용한다.

```bash
conda activate MCEF
bash multi_component_exact_factorization_cavity/run_local_seven_preview.sh \
  results/20261025/local_seven_repeat_v1 3
```

마지막 숫자 1은 순차, 3은 case 3개 병렬이다. 이 스크립트는 가벼운 CPU
미리보기용이다. 서버 11GB GPU의 native 격자 3개 동시 실행을 의미하지 않는다.
매 시점 전체 파동함수는 저장하지 않고 restart와 compact fields를 저장한다.

## 서버 다음 단계

로컬의 coarse config를 그대로 장시간 production에 쓰지 않는다. 먼저 기존 README의
`run_server.sh smoke-portable` 및 native `validate` 절차를 수행한다. VSC 설정은
`configs/double_well_shallow_proposal.json`에서 만들고, PEN은 기본 분자 설정에서
만든다. 두 경우 모두 첫 native pilot은 `t_final_fs=5`, `field_every_fs=0.25`,
jobs=1로 시작한다. validation_dir와 legacy_regression_file은 서버의 실제 출력
경로로 맞춘 뒤 calibration → validate → production 순서로 실행한다.
GPU 실행 및 장시간 production은 이번 로컬 작업에서 수행하지 않았다.
