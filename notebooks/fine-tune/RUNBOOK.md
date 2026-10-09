# MobileFaceNet 512D FR 베이스라인 재현 및 미세조정 실행 가이드 (RUNBOOK)

본 문서는 UCFace 논문(IEEE T-IFS 2024, DOI: [10.1109/TIFS.2024.3426973](https://doi.org/10.1109/TIFS.2024.3426973)) 및 저자 교신 내용을 바탕으로, **동일 백본(MobileFaceNet 512D)** 환경에서 3대 대표 얼굴인식(FR) 베이스라인 모델(**ArcFace, AdaFace, MagFace**)을 VGGFace2 사전학습 가중치로 초기화하고 QMUL-SurvFace 저화질 감시 얼굴 데이터셋으로 미세조정(Fine-Tuning)하여 평가 파이프라인에 안전하게 연계하기 위한 완전한 실행 매뉴얼입니다.

---

## 1. 관련 핵심 논문 및 기여 정리

| 모델/프로토콜 | 논문 출처 | DOI / 학술대회 | 핵심 파라미터 / 수식 정의 |
| :--- | :--- | :--- | :--- |
| **UCFace** | Z. Shen et al., *Uncertainty-Aware Contrastive Face Recognition* | IEEE T-IFS 2024, [10.1109/TIFS.2024.3426973](https://doi.org/10.1109/TIFS.2024.3426973) | 동일 백본 MobileFaceNet, Adam lr $10^{-3}$, 40 에포크, dropout 0.6 |
| **MobileFaceNets** | S. Chen et al., *MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification* | CCBR 2018, [10.1007/978-3-319-97909-0_24](https://doi.org/10.1007/978-3-319-97909-0_24) | 1.2M 파라미터, GDC(Global Depthwise Conv), 512D 임베딩 출력 |
| **ArcFace** | J. Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition* | CVPR 2019, [10.1109/CVPR.2019.00482](https://doi.org/10.1109/CVPR.2019.00482) | $m = 0.6, s = 64.0, \cos(\theta_{y_i} + m)$ 각도 마진 부여 |
| **AdaFace** | M. Kim et al., *AdaFace: Quality Adaptive Margin for Face Recognition* | CVPR 2022, [10.1109/CVPR.2022.01011](https://doi.org/10.1109/CVPR.2022.01011) | $m = 0.4, s = 64.0, h = 0.333$, 특징 노름 $\\|z_i\\|$ 기반 동적 각도/가산 마진 |
| **MagFace** | Q. Meng et al., *MagFace: A Universal Representation for High-Performance and Open-Set FR* | CVPR 2021, [10.1109/CVPR.2021.01170](https://doi.org/10.1109/CVPR.2021.01170) | $l_a=10, u_a=110, l_m=0.45, u_m=0.8, \lambda_g=20$, 크기 정규화 $\mathcal{L}_g$ |
| **QMUL-SurvFace** | L. Cheng et al., *SurvFace: Comprehensive Benchmark on Surveillance Face Recognition* | IEEE T-IFS 2020, [10.1109/TIFS.2020.2974635](https://doi.org/10.1109/TIFS.2020.2974635) | 5,319 학습 인물(220,888장), 저화질 CCTV 감시 얼굴 평가 |

---

## 2. 하드웨어 및 저장 공간 가이드

- **기준 GPU**: NVIDIA GeForce GTX 1080 Ti (11GB VRAM).
- **VRAM 최적화**: 
  - 기본 배치 크기 $64$를 그대로 유지하면서 VRAM OOM(Out of Memory)을 방지하기 위해 **Micro-batch $32$ + Gradient Accumulation $2$회** 설계를 기본 적용합니다.
  - VRAM 점유량: 약 4.2GB ~ 5.5GB 수준으로 안정적입니다.
- **저장 공간 관리 (`KEEP_RAW_RESULTS`)**:
  - `KEEP_RAW_RESULTS = True`: 40개 모든 에포크의 체크포인트(`checkpoint_epoch_*.pt`)를 보존 (약 1.5GB/모델).
  - `KEEP_RAW_RESULTS = False`: 불필요한 중간 체크포인트를 자동 정리하고 최신 가중치(`checkpoint_latest.pt`), 최고 성능 가중치(`checkpoint_best.pt`), 순수 백본(`backbone_final.pt`) 및 학습 지표 JSON만 보존 (약 80MB/모델). C 드라이브 용량 절약 권장.

---

## 3. 데이터셋 준비 및 인물 누출(Identity Leakage) 방지 프로토콜

### 3.1. VGGFace2 사전학습 데이터
- **공식 경로**: [VGGFace2 (Oxford Visual Geometry Group)](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
  - 현재 Oxford 공식 다운로드는 사용자 신청 및 승인이 필요합니다.
  - 대안 미러: 학술 연구용 공유 미러(예: Academic Torrents, HuggingFace face recognition mirrors)에서 정렬된 112x112 크롭 이미지를 `data/raw/vggface2/`에 배치합니다.
- **사전학습 가중치 및 FaceX-Zoo 소스 커밋 고정**:
  - FaceX-Zoo 표준 MobileFaceNet 구현 소스 커밋: `ba50bce7bb0811e9f13883a48e7e1ddc44566c75` ([JDAI-CV/FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/ba50bce7bb0811e9f13883a48e7e1ddc44566c75)).
  - InsightFace / FaceX-Zoo 공식 허브에서 제공하는 MobileFaceNet 사전학습 체크포인트를 `data/checkpoints/pretrained/mobilefacenet_vggface2.pt`에 직접 로드하여 미세조정 시작점으로 사용할 수 있습니다.
  - 학습-추론 전처리 계약 일치: FaceX-Zoo 표준 및 저장소 추론 규격에 맞춰 `color_order="bgr"`, $[-1.0, 1.0]$ 스케일링을 학습(`FRDataset`)과 추론(`ModelSpec`, `adapter.py`) 양쪽에 엄격히 동일하게 적용합니다.

### 3.2. SurvFace 4-Role Identity 분할 (Total: 5,319 인물)
기존 1:N 감시 얼굴 검색 및 Continuous FIQA 보정 실험과의 **인물 중복(Leakage)을 완전 차단**하기 위해 결정론적 기준 시드(`8972`)로 역할을 엄격히 분할합니다:
1. **FR Fine-Tuning Train**: 1,600 인물 (MobileFaceNet 베이스라인 분류기 학습)
2. **FR Fine-Tuning Val**: 219 인물 (조기 종료 및 손실 수렴 검증)
3. **1:N Calibration Gallery (Watchlist)**: 3,000 인물 (사후 임계값 보정 갤러리)
4. **1:N Calibration Probes (Non-mated)**: 500 인물 (사후 임계값 보정 미등록 탐색)
- **합계**: $1,600 + 219 + 3,000 + 500 = 5,319$ 인물 (SurvFace `training_set` 100% 소진, 교집합 = $\emptyset$).
- **매니페스트 파일**: `data/manifests/training_manifest_v2_finetune.csv` (기준 seed=8972)

---

## 4. 노트북 실행 순서

### Step 0: 데이터 감사 및 프로토콜 분할
- **노트북**: `notebooks/fine-tune/00_data_and_protocol.ipynb`
- **목적**: SurvFace 데이터셋 무결성을 확인하고 4-Role 분할 매니페스트를 생성합니다.
- **안전 설정**: 기본값 `EXECUTE_DATA_PREP = False`. 준비 완료 시 `True`로 전환하여 1회 생성.

### Step 1: MobileFaceNet 베이스라인 학습
- **노트북**: `notebooks/fine-tune/01_train_baselines.ipynb`
- **목적**: ArcFace, AdaFace, MagFace 손실함수로 MobileFaceNet을 순차 학습합니다.
- **안전 설정**: 기본값 `EXECUTE_TRAINING = False`. 파이프라인 검증은 `DRY_RUN_SMOKE = True`로 빠른 스모크 테스트를 우선 수행합니다.
- **출력물**:
  - `results/training/fr_baselines/<head>/backbone_final.pt`: 분류기 헤드가 제거된 순수 512D 백본 가중치.
  - `results/training/fr_baselines/<head>/metrics_history.json`: 에포크별 손실 및 정확도 이력.

### Step 2: 체크포인트 검증 및 다운스트림 등록
- **노트북**: `notebooks/fine-tune/02_validate_and_register.ipynb`
- **목적**: 내보낸 체크포인트가 `load_mobilefacenet_checkpoint` 및 공통 512D L2 정규화 계약을 만족하는지 엄격히 검증하고 다운스트림 평가 카탈로그에 등록합니다.
- **후속 평가 연계**:
  - `notebooks/lfw/00_lfw_pair_verification.ipynb` (LFW 1:1 검증)
  - `notebooks/rfw/01_rfw_continuous_calibration.ipynb` (RFW 인종별 FIQA 연속 보정)
  - `notebooks/survface/02_survface_compression.ipynb` (SurvFace 1:N 압축 검색)
  - `research/evaluation/tinyface.py` (TinyFace 초저해상도 평가)

---

## 5. 트러블슈팅 및 주의사항

1. **VRAM OOM 발생 시**:
   - `configs/training/fr_baselines/common.yaml`에서 `micro_batch_size: 16`으로 낮추고, `accumulation_steps: 4`로 증가시킵니다.
2. **CUDA 사용 불가 오류 (`CUDAExecutionProvider` / `torch.cuda`)**:
   - `AGENTS.md` 제3조에 따라 대규모 학습에서 조용한 CPU Fallback은 엄격히 금지됩니다. `torch.cuda.is_available()`이 True인지 확인하고 드라이버를 점검하십시오.
3. **체크포인트 역호환 오류**:
   - `export_backbone_checkpoint`는 헤드 가중치를 제거하고 `state_dict`와 `state_dict_backbone`을 함께 패키징하므로 공식 로더가 즉시 로드할 수 있습니다. 수동으로 `torch.save(model.state_dict())`를 수행하지 마십시오.
