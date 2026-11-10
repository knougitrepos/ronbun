# FaceX-Zoo → ronbun 7건 개선 및 검증

작업 시작 2026-11-06 KST · 최종 검증 2026-11-07 KST · 기존 사용자 변경 보존

사용자가 요청한 추가 감사 F1–F7을 구현하고 관련 회귀 검사 및 실제 02 복구 파일의 CPU/GPU 복원을 검증했다. 정식 학습·대규모 입력 추출·실데이터 보정 실험은 실행하지 않았다. 기존 checkpoint·전처리 이미지·완료 결과를 수정하지 않았다.

## 개선 결과와 근거

| 항목 | 구현 결과 | 검증 |
|---|---|---|
| F1 체크포인트 필수 상태 | strict resume에서 backbone/head, 전달된 optimizer/scheduler, RNG, epoch/batch 및 중간 epoch 상태의 존재·형태를 로드 전에 검사. CUDA RNG 복원 오류도 정식 resume 실패 처리 | 누락·None·빈 상태·shape·RNG·batch 불일치 거부. 정상 복원과 명시적 source 전이 검사. 실제 CPU 및 GTX 1080 Ti 복원 |
| F2 부분 입력 cache | 최초 0행 progress와 배치별 배열·이미지 ledger hash chain을 저장. 배열/ledger를 flush·fsync한 뒤 progress를 확정. 재개 시 확정된 prefix를 검증하고 미확정 꼬리만 truncate·재계산 | identity·embedding·norm·quality·ledger·chain 변조 거부, 거부 시 기존 파일 hash 보존, progress 직전 오류와 잘린 UTF-8 꼬리 복구, 배치 크기 변경 재개 |
| F3 임계값 보수성 | 다음 거부 score 바로 위 float64 경계. empirical ceiling의 +1e-15 허용 제거. Global/conditional/Continuous 기록에 float64-nextafter-first-rejected-v2 명시 | 같은 false accepts 1/10에서 합성 genuine gap 수용. ties·0/1·목표 바로 아래·float32 입력·극단값·실제 덧셈 반올림 및 public Global/Continuous 경로 검사 |
| F4 중복 선택 | config 및 inspect/prepare/run 인자에서 빈 목록·알 수 없는 값·중복 dataset 거부 | 평가 작업 전 거부, 중복 config 거부 |
| F5 재사용 공간 검사 | prepare_inputs의 전체 재생성 공간 요구 제거. 완료 cache 검증 뒤 materialize 잠금 안에서 기존 파일 할당을 제외한 추가 공간·journal/metadata·품질 복사 비용 추정 | 전체 재생성보다 적은 free space에서도 public prepare_inputs 완료 cache 재사용. partial 추가량 계산 및 신규 공간 부족 시 배열 할당 전 거부 |
| F6 결과 조건 대조 | baseline/stage2 reader에서 contract hash → UID, manifest UID, 디렉터리명 대조. 자동 재사용은 현재 요청 contract도 비교 | 다른 조건 report를 요청 폴더에 복사한 경우 거부. contract 변조 거부. 정상 완료 결과·stage2 재사용 통과 |
| F7 잠금 예외 처리 | 독점 lock 획득 후 lock 내용 및 초기 manifest 쓰기부터 finally 보호. lock 획득 실패 시 타인 lock 유지 | 초기 manifest OSError 후 lock 해제, 재시도 완료, 기존 lock 보존 |

F2는 원본 이미지 불변 전제와 별개인 파생 데이터 검증이다. 새 검증 정보가 없는 과거 partial은 자동 승인하지 않고 보존한 채 새 artifact에서 재추출해야 한다. 완료 population.csv도 manifest의 population_hash와 대조한다.

F3은 오인식률의 수학적 보장을 추가하지 않는다. 합성 재현은 계산 규칙을 검증하며 실제 성능 개선량을 뜻하지 않는다. 완료된 과거 보정 결과의 reader는 저장된 결과를 그대로 읽는다. 과거 threshold를 현재 규칙으로 재현하려는 새 replay가 불일치하면 기존 검증 artifact 사용 또는 새 run 재보정 안내와 함께 중단한다.

## 기존 02 모델 생성 보존

CPU 복원 기록은 [checkpoint_restore.json](C:/ronbun/publish/support/additional_fixes_20261106_v1/checkpoint_restore.json), GPU 복원 기록은 [checkpoint_restore_gpu.json](C:/ronbun/publish/support/additional_fixes_20261106_v1/checkpoint_restore_gpu.json)에 있다.

- 현재 02가 지정한 pretrain_mobilefacenet_1024d_recovery_e12_b21162_386e280a.pt를 실제로 읽었다.
- SHA-256은 수정 전후 4acef9dcfc51b848d0354f1c00ab03dbaf12c8016c0f886125d8eeb984add182로 동일하다.
- epoch 12, 완료 batch 21,162, 다음 1-based batch 21,163이다.
- backbone/head와 optimizer 184개 상태, scheduler.last_epoch=11, LR=0.0001이 일치했다.
- CPU의 torch/Python/NumPy 및 실제 GPU의 CUDA RNG 복원이 일치했다.
- GPU는 NVIDIA GeForce GTX 1080 Ti, PyTorch 2.7.1+cu118, CUDA runtime 11.8이다.
- 3,141,890장 검증 완료 receipt 재사용. 이미지 hash 검사 및 decode 호출은 0회다.
- 02 노트북과 저장 settings JSON의 hash가 수정 전 감사 기록과 동일하다.
- 학습 DataLoader를 시작하지 않았고 gradient/optimizer update도 하지 않았다.

F1 변경으로 생기는 소스 해시 차이는 검토한 source map의 명시적 전이로 연결했다. 원래 중단 checkpoint 및 수정 직전 소스의 checkpoint에서 새 검증 코드로 오는 전이를 등록했고, 기존 역사적 전이는 보존했다. 설정·학습 계산이 다른 경우, 미등록 소스 변화, 역방향 전이는 여전히 거부한다. strict_config를 끄는 우회는 추가하지 않았다.

다음 학습 재개 때 커널을 재시작한 후 02의 설정 셀부터 순서대로 실행하면 된다. 기존 RESUME_CHECKPOINTS와 DATA_INTEGRITY_MODE=immutable_verified_reuse, FORCE_DATA_REVERIFY=False를 유지한다. 현재 실행 중인 학습을 이 수정만을 이유로 강제 종료할 필요는 없다. 00/01 재실행은 필요하지 않다. 실제로 학습을 재시작한 결과는 이번 검증에 포함하지 않는다.

## 테스트와 재실행 범위

최종 관련 검사 중복 제외 **264개 통과**.

- ronbun 입력·workflow·dataset 평가·등록·교차 저장소 검사 145개.
- 추가 public prepare_inputs 공간 회귀 1개. 해당 테스트 파일의 최종 재검사 17개 중 기존 16개는 중복 제외.
- threshold/conditional/Continuous 및 기존 FIQA 경로 71개.
- FaceX-Zoo resume-state 및 immutable 검증 47개. 기존 cropper의 skimage API 폐기 예정 경고 11건이 남는다.
- 두 저장소 git diff --check 통과. Windows LF/CRLF 안내는 별도다.

테스트 XML과 GPU 복원 스크립트는 C:/ronbun/tmp/additional_fixes_20261106, CPU 복원 스크립트는 C:/ronbun/tmp/fixes_20261106에 있다. 경량 집계·소스 hash·복구 파일 검증 결과는 [validation.json](C:/ronbun/publish/support/additional_fixes_20261106_v1/validation.json)에 기록했다. 전체 저장소의 모든 테스트를 실행한 것으로 해석하지 않는다.

ronbun에서는 새 소스 fingerprint의 입력/index/report UID를 사용한다. 기존 완료 결과와 cache를 덮어쓰거나 삭제하지 않는다. 수정된 임계값 효과를 비교하려면 같은 partition·모델·압축 조건에서 Global과 Continuous를 함께 다시 보정·평가해야 한다. 필요한 입력의 소스 계약이 달라졌다면 새 입력 추출도 필요하다. 이 재실행은 parent FR을 처음부터 다시 학습하라는 뜻이 아니다.

변경된 운영 안내는 [WORKFLOW_GUIDE.md](C:/ronbun/notebooks/WORKFLOW_GUIDE.md), 연구 결정은 [architect/20261106.md](C:/ronbun/architect/20261106.md)에 반영했다. 입력 cache·journal·상세 테스트 파일은 기존 원본 결과 ignore 규칙을 유지하며 경량 검증 기록만 Git 추적 허용 목록에 추가했다. 기존 사용자 변경은 보존했고 스테이징·커밋·푸시는 하지 않았다.

