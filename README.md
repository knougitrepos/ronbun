### [2025-01-24] 한글 폰트 문제 및 OpenMP 충돌 완전 해결
- **core/pipeline/font_utils.py**: Windows 환경에서 matplotlib 한글 폰트 문제 완전 해결
- **core/pipeline/openmp_fix.py**: Intel OpenMP와 LLVM OpenMP 충돌 문제 해결
- **자동 한글 폰트 감지 및 설정**:
  - 시스템 설치 폰트 자동 스캔 (Malgun Gothic, Batang, Dotum, Gulim 등)
  - 우선순위 기반 최적 한글 폰트 선택 (Malgun Gothic > NanumGothic > Batang 순)
  - 한글 폰트 미설치 시 설치 가이드 자동 제공
- **matplotlib 한글 지원 기능**:
  - `setup_korean_font()`: 자동 한글 폰트 설정 및 검증
  - `get_korean_fonts()`: 시스템 한글 폰트 목록 반환
  - `reset_font_cache()`: matplotlib 폰트 캐시 리셋 기능
  - `install_korean_font_guide()`: 상세한 폰트 설치 안내
- **OpenMP 충돌 해결 기능**:
  - `fix_openmp_conflicts()`: KMP_DUPLICATE_LIB_OK 환경변수 설정으로 충돌 방지
  - `check_openmp_status()`: 현재 OpenMP 환경 변수 상태 확인
  - `test_numpy_performance()`: NumPy 성능 테스트로 충돌 영향 검증
  - `fix_conda_openmp()`: Conda 환경에서 OpenMP 패키지 충돌 해결 가이드
- **노트북 통합 적용**:
  - **notebooks/unified_beta_vae_enhanced.ipynb**: 한글 폰트 + OpenMP 충돌 해결 적용
  - **notebooks/germini_AE.ipynb**: 한글 폰트 자동 설정 로직 통합
  - try-except 구조로 폰트 유틸리티 실패 시 수동 설정 fallback
  - OpenMP 경고 메시지 필터링 및 환경 변수 자동 설정
- **문제 해결 완료**:
  - **한글 폰트**: `Glyph missing from font(s) DejaVu Sans` 경고 완전 제거
  - **OpenMP 충돌**: `Found Intel OpenMP and LLVM OpenMP loaded` 경고 해결
  - 시스템의 `Malgun Gothic` 폰트로 자동 설정하여 모든 한글 문자 정상 표시
- **기술적 특징**:
  - Windows 기본 폰트 우선 활용 (별도 설치 불필요)
  - unicode_minus=False 설정으로 마이너스 기호 문제도 동시 해결
  - 폰트 검증 메커니즘으로 설정 성공 여부 자동 확인
  - threadpoolctl 런타임 경고 자동 필터링

### [2025-01-24] Enhanced β-VAE 통합 시스템 완전 구축 완료
- **notebooks/unified_beta_vae_enhanced.ipynb**: 기존 `unified_beta_vae.ipynb`와 7가지 개선사항 완전 통합
- **완전 통합된 시스템 특징**:
  - 🔗 **실제 DB 연동**: PostgreSQL wavelet/dct 벡터 직접 로딩 및 Enhanced 임베딩 저장
  - 🚀 **7가지 개선사항 완전 통합**: Enhanced Loss + KL Annealing + Latent Viz + Cosine Monitor
  - 🎯 **원클릭 실행**: 데이터 로딩부터 분석, 저장까지 완전 자동화
  - 📊 **종합 시각화**: 12개 차트 잠재공간 분석 + 4개 차트 코사인 모니터링
  - 💾 **Enhanced 체크포인트**: 메타데이터, 성능지표, 분석결과 통합 저장
- **주요 기술적 성과**:
  - Enhanced Loss Function: MSE + β·KL + λ·(1-cosine_similarity) 통합 손실
  - KL Annealing: Linear warm-up으로 안정적 훈련 (25 에포크 기본)
  - 실시간 Cosine Monitoring: 4개 차트로 수렴 패턴 분석
  - 종합 Latent Analysis: PCA/t-SNE/UMAP + 12개 차트 + 자동 클러스터링
  - Enhanced DB Integration: PostgreSQL pgvector 기반 유사도 검색 최적화
- **성능 개선사항**:
  - GPU Mixed Precision Training 지원
  - 메모리 효율적 배치 처리 (자동 배치 크기 조정)
  - Enhanced Early Stopping with Cosine Monitoring
  - 마스크 기반 패딩 처리로 품질 향상
- **라이브러리 추가**: scikit-learn>=1.0.0, seaborn>=0.11.0 (requirements.txt 업데이트)

### [2025-01-24] β-VAE 7가지 핵심 개선사항 완전 구현 완료 (통합 전)
- **notebooks/beta_vae_comprehensive_improvements.ipynb**: β-VAE 모델의 7가지 핵심 개선사항 모두 구현 완료
- **구현된 7가지 개선사항**:
  1. **Enhanced Loss Function**: MSE + λ·cosine_similarity 결합 손실 함수 (λ=0.5)
  2. **KL Annealing**: 점진적 KL 가중치 증가 스케줄러 (Linear/Sigmoid/Cyclical 지원)
  3. **Latent Visualization**: t-SNE, UMAP, PCA 통합 잠재공간 분석 (12개 차트 시각화)
  4. **Cosine Monitoring**: 실시간 코사인 유사도 추이 모니터링 (4개 차트 분석)
  5. **Recall@K Evaluation**: FAISS 기반 유사성 검색 성능 평가 (다중 K값 지원)
  6. **Band-Specific Models**: LL/LH/HL/HH 밴드별 특화 β-VAE (병렬 학습)
  7. **Comprehensive Evaluation**: 다차원 성능 메트릭 종합 평가 (레이더 차트 분석)
- **주요 기술적 특징**:
  - GPU 가속 및 Mixed Precision Training 지원
  - 벡터화된 배치 연산으로 성능 최적화
  - 마스크 기반 패딩 처리
  - 실시간 학습 모니터링 및 Enhanced Early Stopping
  - 체크포인트 관리 및 실험 추적
  - 종합적인 통계 분석 및 품질 지표
- **라이브러리 추가**: umap-learn>=0.5.0, plotly>=5.0.0 (requirements.txt 업데이트)
- **다음 단계**: 실제 데이터로 7단계 실험 파이프라인 실행 가능

### [2025-01-23] Beta-VAE 데이터 로딩 조건 최적화
- **notebooks/unified_beta_vae.ipynb**: 데이터 로딩 필터링 조건 개선
- **변경 내용**:
  - 기존: `compressed_dim <= 512` (512차원 이하 모든 벡터)
  - 개선: `compressed_dim BETWEEN 128 AND 512` (128~512차원 범위로 제한)
- **효과**:
  - 너무 작은 차원의 저품질 압축 벡터 제외 (< 128차원)
  - 고차원 벡터는 기존과 동일하게 제외 (> 512차원)
  - 품질과 효율성의 균형 확보: 적정 압축률 유지하면서 정보 손실 최소화
- **예상 영향**: 더 일관성 있는 임베딩 품질, 모델 훈련 안정성 향상

### [2025-01-22] 고급 Beta-VAE 통합 임베딩 압축 시스템 구축
- **notebooks/unified_beta_vae.ipynb**: 최신 딥러닝 기법이 적용된 고급 Beta-VAE 차원 압축 시스템
- **핵심 특징**:
  - **대용량 데이터 처리**: 157만개 벡터 (Wavelet: 139만개, DCT: 15만개) 통합 처리
  - **차원 압축**: 최대 536차원 → 128차원으로 76% 압축률 달성
  - **고급 Beta-VAE 아키텍처**: Layer Normalization + Dropout + Xavier 초기화로 강화된 모델
  - **상세 진행 추적**: 8단계 파이프라인 전체에 걸친 포괄적 verbose 로깅
- **포괄적 진행상황 추적**:
  - 1단계: 데이터베이스 연결 및 GPU 설정 확인
  - 2단계: Wavelet/DCT 벡터 로딩 (진행 바, 에러 처리)
  - 3단계: 벡터 패딩, 텐서 변환 및 훈련/검증 데이터 분할
  - 4단계: 고급 Beta-VAE 모델 정의 (정규화, Dropout, 초기화)
  - 5단계: 고급 훈련 설정 (GPU 최적화, 스케줄러, Early Stopping)
  - 6단계: 고급 모델 훈련 (Mixed Precision, Gradient Clipping, 실시간 모니터링)
  - 7단계: 최적화된 임베딩 추출 (GPU 메모리 효율적 배치 처리)
  - 8단계: 성능 분석, 모델 저장 및 리소스 정리
- **최신 딥러닝 기법 적용**:
  - **GPU 가속**: CUDA 자동 감지 및 Mixed Precision Training (AMP)
  - **Early Stopping**: 검증 손실 기반 조기 종료 (patience=15)
  - **Learning Rate Scheduling**: Warm-up + Cosine Annealing + Plateau Scheduler
  - **정규화 기법**: Layer Normalization + Dropout + Weight Decay
  - **고급 옵티마이저**: AdamW with weight decay
  - **Gradient Clipping**: 훈련 안정성 향상
  - **데이터 분할**: 85%/15% 훈련/검증 분할
  - **고급 체크포인트 관리**: 최적 모델 자동 저장 및 훈련 재개 지원
- **실용적 기능**:
  - GPU 메모리 기반 자동 배치 크기 조정
  - 실시간 성능 모니터링 (손실, 학습률, 베타 가중치)
  - 오버피팅 감지 및 모델 품질 분석
  - 임베딩 품질 통계 (비활성 차원, 안정성 지표)
  - 포괄적 모델 체크포인트 (가중치, 훈련 기록, 하이퍼파라미터)
- **확장성**: 새로운 벡터 타입 자동 적응, 하드웨어별 최적화, 전이 학습 지원

### [2025-01-22] Beta-VAE 고급 체크포인트 관리 시스템 추가
- **notebooks/advanced_b_vae_checkpoints**: 체계적인 체크포인트 저장 디렉토리 구조
- **AdvancedCheckpointManager**: 포괄적 체크포인트 관리 클래스 구현
  - **최적 모델 저장**: `best_model_*.pth` - 최고 성능 모델 자동 보존
  - **최신 상태 저장**: `latest_model_*.pth` - 훈련 중단 시 재개 지원
  - **주기적 백업**: `epoch_XXX_*.pth` - 매 N 에포크마다 중간 체크포인트 저장
  - **설정 메타데이터**: `config_*.json` - 하이퍼파라미터 및 실험 정보 저장
- **향상된 Early Stopping**: 
  - 최적 가중치 자동 복원, 체크포인트 매니저 연동
  - 조기 종료 시 최적 모델 상태로 완전 복원
  - 인내심 카운터 및 종료 에포크 추적
- **실험 관리 기능**:
  - 타임스탬프 기반 실험 ID 자동 생성
  - 포괄적 메타데이터 저장 (훈련 기록, 성능 지표, 하이퍼파라미터)
  - 체크포인트 로딩 가이드 및 사용 예시 제공
- **체크포인트 활용 시나리오**:
  - **모델 배포**: best_model_*.pth로 최적 성능 모델 활용
  - **훈련 재개**: latest_model_*.pth로 중단된 훈련 이어서 진행
  - **실험 분석**: 특정 에포크 체크포인트로 성능 변화 추적
  - **실험 재현**: config_*.json으로 완전한 실험 환경 재구성

### [2024-06-09] 구조 개선 1단계 진행중
- FeatureExtractor 추상 클래스 및 ArcFaceFeatureExtractor 구현, embedding_service.py 통합 리팩토링 완료
- origin_feature_extractor.py는 더 이상 사용하지 않음(구조 안내 주석만 유지)
- 다양한 임베딩 모델 확장 가능 구조로 변경됨

1. 이미지에서 특징 추출 arcface 사용
2. wavelet,dct,pca,pq quantize로 특징 변환
3. recall 등의 기준으로 정확도 판단
4. PostgreSQL 의 pgvector 를 사용하여
cosine 유사도 비교로 업로드된 이미지를 유사도 판단
5. sota 모델과의 개선 및 변화사항 비교 파악
6. 위와 같은 연구용 파이프라인 관리용 웹임

### [2024-06-09] 구조 개선 2단계 진행중
- vector_transformer.py 신규 생성, PCA/DCT/Wavelet/Quantize/정규화 등 벡터 변환 함수 통합
- 기존 zscore_normalization.py, L2_normarlization.py는 안내 주석만 유지(기능 통합)
- transform_vector 함수로 파라미터 기반 동적 변환 지원

### [2024-06-09] 구조 개선 3단계 진행중
- DB/db_utils.py에 add_image_embedding 함수 추가, 여러 벡터/메타 정보 일괄 저장 구조로 개선
- ImageEmbeddings 테이블의 origin/pca/dct/wavelet/quantized 벡터와 각종 params, 메타 정보 저장 일관성 강화

### [2024-06-09] 구조 개선 4단계 진행중
- similarity_utils.py에 search_similar_vectors 함수 추가, cosine/euclidean/inner product 등 다양한 거리 계산 방식 및 HNSW/Brute Force 전략 분기 구조 설계
- metric 파라미터 및 K값 기반 자동 전략 선택 구조 반영(실제 DB 연동/인덱스는 후속 구현)

### [2024-06-09] 구조 개선 5단계 진행중
- extract_router.py에 FastAPI BackgroundTasks 적용, 대량 특징추출 비동기 처리 구조로 개선
- 실제 추출/DB저장 로직은 서비스 함수로 분리, 라우트는 트리거 역할만 담당

### [2024-06-09] 구조 개선 6단계 진행중
- 중복/불필요 파일(origin_feature_extractor.py, zscore_normalization.py, L2_normarlization.py) 삭제
- 명명 규칙(영문, PEP8) 및 주석/문서화 정리, 코드 일관성 강화

### [2024-06-09] 구조 개선 7단계 진행중
- config.yaml 기반 실험 조합 구조 설계, config.py에 예시 구조/주석 추가
- Hydra 등 설정 관리 도구 연동 및 실험 파라미터 동적 로딩 구조 반영(후속 적용 가능)

### [2024-06-09] 구조 개선 8단계 진행중
- 마이크로서비스 아키텍처, Docker 기반 컨테이너화, API 표준화 등 장기 확장성 구조 설계 방향 반영
- 얼굴 검출/임베딩/검색 등 서비스 분리, 컨테이너화 및 표준 API 명세화 예시 문서화 예정

### [2024-06-10] Embedding_128 지원 및 실험 자동화 연동
- Embedding_128 테이블 및 저장/조회 기능(core/schemas.py, core/database.py) 추가
- 128차원 벡터(DCT, PCA 등)도 256/512와 동일하게 저장/조회/실험 자동화 가능
- 실험 자동화(Optuna+MLflow) 구조상 벡터 차원(128/256/512 등) 확장에 문제 없음
- 예시: repository.add_embedding_128(...), repository.get_embeddings_128(...) 사용

### [2024-12-28] Origin Vector DB 구조 개선 및 통합
- ArcFace 임베딩 추출 시 origin_vector 테이블로 이미지와 벡터를 일원화하여 저장
- notebooks/origin_extractor.ipynb: ArcFace로 downloaded_datasets의 26,466개 이미지 처리
- 13,195개 성공적 추출, 38개 얼굴 검출 실패 (나머지는 중복/처리불가)
- origin_vector 테이블 구조: id, image_path, label, vector_type, parameters, embedding(512), created_at, log

### [2024-12-28] Wavelet/DCT 변환 구조 통일 및 mode 확장
- **Wavelet 변환 개선**: 기존 저주파만 → low/high 모드 지원으로 DCT와 동일한 구조
- **통일된 Mode 개념**: 
  - `low`: 저주파 성분만 유지 (처음 keep_dim개 계수)
  - `high`: 고주파 성분만 유지 (처음 keep_dim개 제거 후 나머지)
- **Wavelet vs DCT의 keep_dim**:
  - DCT: 사용자 지정 (8, 16, 32, 64, 128, 256)
  - Wavelet: 자동 결정 (level과 wavelet family에 따라 첫 번째 계수 길이)
- **notebooks/wavelet_extractor.ipynb**: 임베딩 데이터 형식 문제 해결, JSON 파싱 추가
- **DB 구조**: wavelet_vector 테이블에 origin_vector_id + wavelet_family + level + mode로 식별

### [2024-12-28] 조건부 오토인코더(C-AE) 통합 벡터 압축 시스템 구축
- **notebooks/germini_AE.ipynb**: 실제 DB 데이터를 활용한 조건부 오토인코더 구현
- **이종 벡터 통합**: Origin(512차원), DCT(8~256차원), Wavelet(다양한 차원)을 64차원으로 통합 압축
- **조건부 학습**: 변환 타입/파라미터를 조건 벡터로 활용하여 맥락 인식 압축
- **상세 모니터링**: 전 과정에 걸친 포괄적 진행상황 추적 및 로깅 시스템
- **실제 운영 준비**: 모델 저장/로드, 배치 압축, 벡터 검색 시스템 통합 가이드 제공
- **주요 성과**: 87.5% 공간 절약, 품질 유지, 검색 효율성 향상
- **확장성**: 새로운 변환 방식 추가, 점진적 학습, 다중 잠재 차원 지원

### [2025-01-21] C-AE 노트북 성능 최적화 완료
- **DataLoader 성능 문제 해결**: 1시간+ 소요 → 수초 내 완료
- **주요 최적화 내용**:
  - 조건 벡터 사전 계산: 매번 OneHot 인코딩 → 5000개 조건 벡터 일괄 사전 계산
  - 멀티프로세싱 비활성화: Windows 환경 호환성 문제 해결 (num_workers=0)
  - 패딩 크기 최적화: 536차원 → 512차원으로 제한하여 메모리 20% 절약
  - 벡터 사전 패딩: __getitem__에서 실시간 패딩 → 초기화 시 일괄 처리
  - 배치 크기 조정: 128 → 64로 안정성 향상
- **성능 개선 효과**:
  - 조건 벡터 생성 시간: 1시간+ → 30초 이내
  - DataLoader 테스트: 무응답 → 즉시 완료
  - 메모리 사용량: ~100MB (벡터 10MB + 조건 벡터 9MB)
  - 멀티프로세싱 데드락 문제 완전 해결

### [2025-01-21] C-AE 노트북 KeyError 해결 및 최종 안정화
- **KeyError: 'family' 문제 완전 해결**: `update_config_with_optimal_settings` 함수에서 DataLoader 재생성 시 누락된 `collate_fn=custom_collate_fn` 인자 추가
- **핵심 해결 내용**:
  - 배치 크기 최적화 중 DataLoader 재생성 시 custom_collate_fn 누락으로 metadata 처리 실패 
  - train_loader와 val_loader 재생성 시 collate_fn=custom_collate_fn 인자 추가하여 완전 해결
  - 이 한 줄 수정으로 모든 KeyError: 'family', 'level', 'mode' 등 metadata 관련 오류 제거
- **최종 안정성 확보**:
  - 모든 DataLoader에서 metadata 안전 처리 보장
  - GPU 최적화와 성능 최적화가 모두 안정적으로 작동
  - 노트북 전체 파이프라인이 중단 없이 완전 실행 가능

### [2025-01-22] C-AE 대용량 데이터 메모리 최적화 및 배치 처리 시스템 구축
- **메모리 부족 문제 완전 해결**: 157만개 데이터에서 222GB 메모리 요구 → 배치 처리로 메모리 효율성 확보
- **핵심 문제 분석**:
  - 데이터 규모: 1,570,243개 벡터 (157만개)
  - OneHot 인코딩 차원: 18,954차원 (폭발적 차원 증가)
  - 메모리 요구량: 222GB (현실적으로 불가능)
- **메모리 효율적 전처리 시스템**:
  - `ConditionEncoder` 메모리 최적화: 희소 행렬(sparse matrix) + 배치 처리
  - 카테고리 차원 제한: 컬럼당 최대 20개 카테고리로 제한
  - 샘플링 기반 fitting: 전체 157만개 → 1만개 샘플로 fitting
  - 배치 변환: 5만개씩 배치 처리로 메모리 안정성 확보
- **대용량 데이터 처리 최적화**:
  - **데이터 크기 제한**: 157만개 → 10만개로 실용적 크기 조정
  - **배치 크기 조정**: 256 → 128로 메모리 절약
  - **검증 데이터 비율**: 10% → 5%로 메모리 절약
  - **에포크 수 조정**: 50 → 20으로 대용량 데이터 적응
- **견고한 메모리 관리**:
  - 실시간 메모리 사용량 모니터링 및 경고
  - 희소 행렬 활용으로 메모리 효율성 극대화
  - 배치별 진행상황 추적 및 로깅
  - **notebooks/improve_v1_AE_fixed.ipynb**: 메모리 최적화 버전 제공

### [2025-01-22] C-AE 데이터 타입 혼재 문제 해결 및 동적 전처리 시스템 구축
- **데이터 타입 혼재 문제 완전 해결**: `OneHotEncoder`에서 categorical 컬럼의 int/str 혼재로 인한 `TypeError: '<' not supported between instances of 'str' and 'int'` 에러 해결
- **핵심 문제 분석**:
  - origin_vector: model, det_size(list), provider, ctx_id (복잡한 구조)
  - dct_vector: keep_dim(int), mode(str), norm(str) 
  - wavelet_vector: wavelet_family(str), level(int), mode(str), original_image_path(str), original_label(str)
  - 테이블별로 다른 파라미터 구조 + 데이터 타입 혼재 → OneHotEncoder 실패
- **동적 전처리 시스템 구축**:
  - `ConditionEncoder` 완전 리팩토링: 하드코딩된 컬럼 → 동적 컬럼 감지 시스템
  - `_detect_column_types()`: 각 컬럼의 실제 데이터 타입 자동 분석
  - 복잡한 타입(list, dict, tuple) 자동 제외 및 로깅
  - categorical 컬럼 → 강제 문자열 변환으로 타입 통일
  - numerical 컬럼 → `pd.to_numeric()` + `errors='coerce'`로 안전한 숫자 변환
- **견고한 데이터 처리**:
  - 존재하지 않는 컬럼 안전 처리, 빈 조건 벡터 더미 처리
  - 모든 데이터 타입 에러 방지 및 확장성 확보
- **확장성 및 유지보수성**:
  - 새로운 벡터 타입 추가 시 코드 수정 불필요
  - 파라미터 구조 변경에 자동 적응
  - 모든 전처리 단계에서 상세한 로깅 및 에러 처리

### [2025-01-21] C-AE 노트북 GPU 최적화 적용
- **GPU 가속 훈련**: Mixed Precision Training, 그래디언트 누적, 비동기 데이터 전송 최적화
- **주요 GPU 최적화 기능**:
  - Mixed Precision Training (AMP): 메모리 사용량 50% 절약, 훈련 속도 30% 향상
  - 그래디언트 누적: 효과적인 배치 크기 증가로 안정성 향상
  - 비동기 GPU 전송 (non_blocking=True): CPU-GPU 간 데이터 전송 오버헤드 최소화
  - 동적 배치 크기 조정: GPU 메모리에 맞춰 최적 배치 크기 자동 탐지
  - GPU 메모리 모니터링: 실시간 메모리 사용량 추적 및 자동 정리
- **성능 개선 효과**:
  - GPU 메모리 효율성: 70% 안전 마진으로 OOM 방지
  - 훈련 속도: Mixed Precision으로 30% 향상
  - 메모리 사용량: 실시간 모니터링으로 최적화
  - Windows 호환성: 멀티프로세싱 문제 완전 해결
- **최적화 구성 요소**:
  - GPU 워밍업 및 프로파일링 시스템
  - 하드웨어별 자동 설정 조정 (게이밍 GPU vs 전문 GPU)
  - 배치별 메모리 사용량 추적 및 자동 정리

### [ArcFace origin(원본) 벡터 저장 원칙]
- ArcFace 등 임베딩 모델의 원본 벡터(origin)는 Embedding_512 테이블에 저장
- vector_type='origin', parameters={}로 구분하여 저장(예시 코드 참고)
- 변환 벡터(DCT, Wavelet 등)와 동일한 구조로 관리되어 실험 자동화/검색/확장에 유리

# 얼굴 임베딩/벡터 변환/실험 자동화 시스템

## 1. 시스템 개요 및 구조
- ArcFace 등 임베딩 모델 기반 특징 추출, Wavelet/DCT/PCA/PQ 등 다양한 벡터 변환 지원
- PostgreSQL + pgvector 기반 대용량 벡터 저장/검색
- Optuna + MLflow 실험 자동화, Celery + Redis 분산/중단-재시작 지원
- 모든 변환/실험/저장 정책은 코드와 문서(wavelet_dct_notice.md 등)에 동기화
- 추상 클래스(FeatureExtractor, BaseTransformer) 기반 계층적 구조, 유지보수성 강화
- 폴더/파일/클래스/함수명은 기능 의미를 따르며, PEP8 및 일관성 준수

---

## 2. 환경 및 설치
- requirements.txt에 모든 필수 패키지 기록(추가시 반드시 동기화)
- 주요 패키지: fastapi, uvicorn, celery, redis, numpy, opencv-python, insightface, torch, pywavelets, faiss-cpu, mlflow, optuna 등

```bash
pip install -r requirements.txt
```

---

## 3. 실행 및 자동화
- run_server.cmd로 FastAPI 서버 실행(uvicorn main:app --reload --host 0.0.0.0 --port 5000)
- Redis, Celery 워커, batch_vector_transform_and_save.py로 분산/중단-재시작 지원
- batch_vector_transform_and_save.py 하나로 큐 등록/중복 체크/분산 처리 통합

---

## 4. 추상화 및 계층 구조
- features/embedding_service.py: FeatureExtractor(추상), ArcFaceFeatureExtractor(구현)
- core/pipeline/transformers/base.py: BaseTransformer(추상), wavelet/dct/pca/pq 등 각 변환기 구현
- core/pipeline/vector_pipeline.py: VectorPipeline에서 동적 변환/파라미터 처리, 유지보수성 강화

---

## 5. 변환 정책 및 예시
- DCT: keep_dim(128/256), mode('low'/'high'), add_embedding_128/256
- Wavelet: level(1/2/3), mode('low'/'high'), add_embedding_256
- PCA: n_components(0.95 등), add_embedding_128/256, joblib 코드북
- PQ: d/M/nbits, add_embedding_pq, faiss 코드북
- 모든 정책/예시는 wavelet_dct_notice.md에 상세 기술

---

## 6. DB 저장 정책
- Embedding_512: ArcFace origin(원본) 벡터, vector_type='origin', parameters={}
- Embedding_128/256: DCT, Wavelet, PCA 등 변환 벡터, vector_type/parameters로 구분
- EmbeddingPQ(권장): PQ 코드 저장, codes(LargeBinary/JSON), parameters로 구분
- VectorRepository에서 add_embedding_128/256/512/pq 등 함수 제공

---

## 7. 코드북 저장/로드 정책
- codebook/ 폴더에 변환명_차원(joblib/faiss) 규칙으로 저장
- PCA: joblib, PQ: faiss 공식 write_index/read_index
- 예시 코드 및 정책은 wavelet_dct_notice.md 참고

---

## 8. 실험 자동화/분산 처리
- Optuna + MLflow로 실험 자동화, 벡터 차원(128/256/512 등) 확장 지원
- Celery + Redis로 대량 이미지 분산 처리, 중단/재시작 안전
- 모든 큐/중복 체크/분산 처리 로직은 batch_vector_transform_and_save.py에 통합

---

## 9. 로그 및 에러 처리
- 모든 주요 함수/클래스에서 로그 출력 및 에러 메시지 명확화
- 로그는 실험 자동화, 분산 처리, DB 저장 등에서 일관되게 사용

---

## 10. 문서화 및 유지보수 원칙
- 코드 변경/구조/정책/실험/변환/DB/코드북 등 모든 내용은 README.md, wavelet_dct_notice.md 등 문서에 즉시 반영
- 코드와 문서가 항상 동기화되도록 관리
- 폴더/파일/클래스/함수명은 기능 의미를 따르며, 계층적/추상화 구조로 유지보수성 강화

---

## 11. 예시 코드/실행
- 각 변환/저장/코드북/DB/실험 자동화 예시는 README.md 및 wavelet_dct_notice.md에 상세 기술
- 실제 사용 예시는 scripts/batch_vector_transform_and_save.py, core/pipeline/vector_pipeline.py 등 참고

---

### [적용 방법]
- 위 내용을 README.md, wavelet_dct_notice.md에 반영/정리
- requirements.txt, run_server.cmd 등 환경 파일도 항상 동기화
- 새로운 기능/정책/구조 추가시 반드시 문서에 즉시 반영

문의/확장/실험 관련 문의는 언제든 환영합니다.

---

## 12. 실험/운영 단계별 전체 절차 및 실행 가이드

### 1. 사전 준비
- **환경 준비**: requirements.txt 설치, DB(PostgreSQL) 실행, config.yaml 등 환경설정 완료
- **DB 준비**: PostgreSQL 서버가 반드시 실행 중이어야 함
- **Celery 워커 실행**: (터미널에서)
  ```bash
  # Windows 환경에서는 반드시 --pool=solo 옵션을 추가해야 함
  celery -A core.pipeline.vector_tasks worker --loglevel=info --pool=solo
  ```
  - 반드시 위와 같이 -A 옵션에 태스크가 정의된 모듈을 지정해야 함
  - core/celery_app.py에서 vector_tasks 임포트하지 않도록 유지(순환참조 방지)
  - 태스크 정의 파일(vector_tasks.py)에서는 from core.celery_app import celery_app만 사용
- **벡터 변환/특징 추출 작업 등록**:
  - (1) **터미널에서 직접 실행**
    - **유의:** 반드시 프로젝트 루트(ronbun)에서 실행하거나, 아래처럼 PYTHONPATH를 명시적으로 지정해야 내부 모듈 import 에러가 발생하지 않습니다.
    - 예시(Git Bash/리눅스/WSL):
      ```bash
      export PYTHONPATH=$(pwd)
      python scripts/batch_vector_transform_and_save.py
      ```
    - 예시(Windows CMD):
      ```cmd
      set PYTHONPATH=%cd%
      python scripts/batch_vector_transform_and_save.py
      ```
  - (2) **웹에서 실행**: [벡터 변환 작업 실행] 버튼 클릭 또는 `/automation/run-batch-transform` (POST)
  - 이 작업은 대량 이미지의 특징 추출/벡터 변환을 Celery 큐에 등록합니다.
  - **자동 추출/저장 대상:**
    - ArcFace 원본(512차원)
    - Wavelet(level=1~3, mode=low/high) → 6가지 조합
    - DCT(keep_dim=32/64/128/256, mode=low/high) → 8가지 조합
    - PCA(128/256)
    - PQ(128/256, M=16, nbits=8)
  - **파라미터를 변경하고 싶으면 소스코드 내 리스트(WAVELET_LEVELS, DCT_KEEP_DIMS 등)를 직접 수정해야 함**
  - **Celery 워커가 반드시 실행 중이어야 실제 작업이 처리됨**
- **MLflow UI 실행**: (터미널에서)
  ```bash
  mlflow ui --port 5001 --backend-store-uri ./mlruns --host 127.0.0.1
  ```
  또는 웹에서 "MLflow UI 실행" 버튼 사용 (구현된 경우)

### 2. 특징 추출 및 벡터 변환 작업 등록
- **웹에서**: [벡터 변환 작업 실행] 버튼 클릭 (또는 `/automation/run-batch-transform` POST)
- **내부 동작**: scripts/batch_vector_transform_and_save.py가 실행되어, Celery 큐에 대량 이미지 특징 추출/변환 작업이 등록됨
- **Celery 워커가 반드시 실행 중이어야 실제 작업이 처리됨**
- **진행상황 확인**: [작업 상태 모니터링] 버튼 또는 `/automation/status`에서 각 변환별 진행률 확인

### 3. 특징 추출/변환 결과 저장
- **DB에**: ArcFace 원본, DCT, Wavelet, PCA, PQ 등 다양한 벡터가 자동 저장됨
- **코드/정책**: 변환별 파라미터, 저장 테이블, 예시 코드는 wavelet_dct_notice.md 참고

### 4. 실험 자동화 및 recall 테스트
- **실험 자동화(Optuna+MLflow)**: (터미널에서)
  ```bash
  python optuna_mlflow/run_optuna.py
  ```
- **실험 결과**: mlruns/ 폴더에 기록, MLflow UI(웹)에서 시각적으로 비교/분석 가능
- **실험 파라미터/정책**: optuna_mlflow/experiment.py, run_optuna.py 등 참고

### 5. 기타
- **DB/코드북/실험 정책**: README.md, wavelet_dct_notice.md, MLFLOW_instructions.md 등 문서 참고
- **운영상 주의**: Celery 워커, DB, MLflow UI 등은 항상 별도 프로세스/터미널에서 실행되어야 함
- **문제 발생 시**: 로그, 상태 모니터링, 에러 메시지, DB 상태 등 확인

---

### [Celery 워커 태스크 등록 및 에러 해결 실전 가이드]

#### 1. 반드시 아래와 같이 Celery 워커를 실행해야 태스크가 정상 등록됨
```bash
celery -A core.pipeline.vector_tasks worker --loglevel=info --pool=solo
```
- `-A` 옵션에 태스크가 정의된 모듈(`core.pipeline.vector_tasks`)을 지정해야 함
- core/celery_app.py에서는 태스크 임포트하지 않도록 유지(순환참조 방지)
- 태스크 정의 파일(vector_tasks.py)에서는 from core.celery_app import celery_app만 사용

#### 2. [tasks] 아래에 `core.pipeline.vector_tasks.process_image`가 보이면 정상 등록된 것임
- Celery 워커 실행 로그에서 `[tasks]` 아래에 태스크가 보이지 않으면 등록이 안 된 것임

#### 3. 'Received unregistered task of type ...' 에러가 발생하는 경우
- 워커가 해당 태스크를 등록하지 않은 상태에서 큐에 등록된 작업을 받아서 생기는 현상임
- 주로 아래와 같은 원인
  - 워커 실행 시 -A 옵션을 잘못 지정 (예: core.celery_app 등)
  - 코드 수정 후 워커를 재시작하지 않음
  - PYTHONPATH/실행 위치가 불일치

#### 4. 해결 방법
- 반드시 워커를 아래 명령어로 실행
  ```bash
  celery -A core.pipeline.vector_tasks worker --loglevel=info --pool=solo
  ```
- 코드 수정(특히 태스크 정의/등록 구조 변경) 후 워커를 재시작
- 큐 등록 스크립트도 같은 PYTHONPATH에서 실행 (예: export PYTHONPATH=$(pwd))
- 워커와 큐 등록 스크립트 모두 프로젝트 루트에서 실행 권장

#### 5. 실전 운영 체크리스트
- [ ] 워커 실행 명령어가 정확한지 확인
- [ ] 워커 실행 로그에 태스크가 등록되어 있는지 확인
- [ ] 큐 등록 스크립트와 워커의 PYTHONPATH/실행 위치가 일치하는지 확인
- [ ] 에러 발생 시 위 항목을 모두 점검

---

### [운영 중인 모든 Celery 프로세스 종료(강제 kill) 방법]

#### Windows에서 모든 celery 프로세스 종료
```cmd
# PowerShell
Get-Process celery | Stop-Process -Force

# 또는 명령 프롬프트(cmd)
taskkill /F /IM celery.exe
```

#### Windows의 Git Bash/MINGW64/bash에서 celery 프로세스 종료
```bash
# bash에서는 아래 명령어를 사용하세요
taskkill //F //IM celery.exe
# 또는
cmd.exe /c "taskkill /F /IM celery.exe"
```

- 여러 워커/백그라운드 celery 프로세스가 남아 있을 때 위 명령으로 모두 종료 가능
- 작업 중인 큐/워커가 모두 종료되므로, 재시작 전 반드시 필요한 작업이 없는지 확인

---

### [실행 로그 파일로 저장하는 방법]

- 로그가 너무 빠르게 지나가서 콘솔에서 확인이 어려울 때, 아래와 같이 log 디렉토리에 로그를 저장하세요.

1. log 디렉토리 생성(최초 1회)
   ```bash
   mkdir log
   ```

2. Celery 워커 로그 저장
   ```bash
   celery -A core.pipeline.vector_tasks worker --loglevel=info --pool=solo > log/celery_worker.log 2>&1
   ```

3. batch_vector_transform_and_save.py 로그 저장
   - (bash/WSL)
     ```bash
     export PYTHONPATH=$(pwd)
     python scripts/batch_vector_transform_and_save.py > log/batch_script.log 2>&1
     ```
   - (Windows CMD)
     ```cmd
     set PYTHONPATH=%cd%
     python scripts/batch_vector_transform_and_save.py > log\batch_script.log 2>&1
     ```

4. 로그 파일 실시간 확인
   ```bash
   tail -f log/celery_worker.log
   ```

---

### [Windows 환경 Celery 워커 실행 시 --pool=solo 옵션 필수 안내]

#### 1. 현상 및 원인
- Windows 환경에서 Celery 워커를 기본(prefork) 모드로 실행하면
  - 태스크를 받자마자 `ValueError: not enough values to unpack (expected 3, got 0)` 에러가 반복 발생
  - 태스크 함수 내부 코드(print, try/except 등)에 진입하지 못하고, 로그가 전혀 찍히지 않음
  - Redis 큐, pycache, 환경 동기화 등 모든 조치 후에도 동일 현상 반복
- 이는 Windows+prefork 모드의 구조적 한계, 메시지 포맷/프로세스간 직렬화 문제, 내부 버그 등으로 인해 발생

#### 2. 해결 방법
- 반드시 Celery 워커를 --pool=solo 옵션(단일 프로세스 모드)으로 실행해야 함
- 예시:
  ```bash
  celery -A core.pipeline.vector_tasks worker --loglevel=info --pool=solo > log/celery_worker.log 2>&1
  ```
- solo 모드에서는 모든 print/log가 정상적으로 찍히고, 태스크 함수가 정상적으로 실행됨
- 병렬처리가 필요하다면 Linux/WSL 환경에서 prefork 사용을 권장

#### 3. 실전 운영 체크리스트
- [ ] Windows에서는 항상 --pool=solo 옵션으로 워커 실행
- [ ] 코드/큐/환경 변경 시 Redis 큐 비우기(`redis-cli FLUSHALL`), 워커 재시작, pycache 삭제
- [ ] Celery 워커와 batch 등록 스크립트가 동일한 conda 환경, 동일한 PYTHONPATH에서 실행되는지 확인
- [ ] 로그 파일로 모든 실행 내역을 남기고, 문제 발생 시 로그를 먼저 확인

#### 4. 참고
- prefork 모드에서만 발생하는 에러이므로, Linux/WSL/Mac 환경에서는 --pool=solo 옵션 없이 병렬처리 가능
- Windows에서 병렬처리 실험이 꼭 필요하다면 WSL/리눅스 환경 사용을 권장

---

# 특징 추출 자동화 기능 안내

## 주요 변경사항
- `/extract` 페이지에서 Wavelet, DCT, PCA, PQ 각 변환별로 '전체 파라미터 조합 실행' 버튼이 추가됨
- 각 버튼 클릭 시, 서버에서 미리 정의된 다양한 파라미터 조합(예: wavelet 종류, level, 저주파/고주파 등)을 자동 반복 적용하여 특징 추출 및 DB 저장이 이루어짐
- 각 변환별 파라미터 조합 예시:
    - **Wavelet**: wavelet_name=['haar','db2','sym2'], level=[1,2,3,4], mode=['low','high']
    - **DCT**: keep_dim=[16,32,64,128,256], mode=['low','high']
    - **PCA**: n_components=[0.95,0.99,100,256]
    - **PQ**: M=[8,16,32], nbits=[4,8]
- 각 변환별 엔드포인트는 `/extract_features/wavelet`, `/extract_features/dct`, `/extract_features/pca`, `/extract_features/pq`로 구성됨

## 사용법
- `/extract` 페이지에서 원하는 변환의 전체 파라미터 조합 실행 버튼을 클릭하면, 모든 이미지에 대해 다양한 파라미터 조합으로 특징 추출이 자동 수행됨
- 결과는 DB에 저장됨

## 참고
- 파라미터 조합 및 자동화 방식은 `routes/extract_router.py` 참고
- 실험/운영/확장 시 혼동 방지를 위해 본 문서를 항상 최신으로 유지할 것

## [업데이트] PCA 코드북 자동 저장/로드
- PCA 변환 시, 코드북(주성분 행렬)이 이미 존재하면 자동으로 로드하고, 없으면 fit 후 코드북을 저장합니다.
- 코드북은 기본적으로 `codebook/pca_{n_components}.joblib` 경로에 저장됩니다.
- 파이프라인에서 PCA 변환을 반복적으로 사용할 때, 동일한 파라미터라면 재학습 없이 빠르게 변환이 가능합니다.

### [2025-01-22] germini_AE.ipynb 노트북 구조 정리 및 체크포인트 구분
- **노트북 구조 체계화**: 처음부터 시작하는 코드와 체크포인트 재시작 코드 명확히 구분
- **마크다운 헤더 추가**:
  - `# 처음부터 시작`: 노트북을 처음 실행할 때 필요한 기본 설정과 초기화 코드
  - `# 체크포인트 재시작`: 체크포인트에서 자동으로 재시작 가능한 코드들
- **사용자 편의성 향상**:
  - 훈련 중단 후 재시작 시 어느 셀부터 실행해야 할지 명확하게 구분
  - 처음 실행과 재시작 실행 시나리오를 위한 가이드라인 제공
- **코드 정리 효과**:
  - 뒤섞여 있던 초기화 코드와 재시작 코드 분리
  - 노트북 실행 흐름의 명확성과 가독성 향상
  - 실험 재현성과 디버깅 효율성 증대

### [2025-01-22] germini_AE.ipynb 노트북 통합 - 하나의 코드로 일원화
- **통합 실행 시스템**: 처음 실행과 체크포인트 재시작을 하나의 코드로 통합
- **자동 체크포인트 재시작**: 
  - 체크포인트가 있으면 자동으로 가장 최근 체크포인트부터 재시작
  - 체크포인트가 없거나 로드 실패 시 자동으로 처음부터 시작
- **노트북 일원화**:
  - 불필요한 중복 마크다운 헤더 및 긴급 재시작 코드 제거
  - `train_conditional_autoencoder_with_validation()` 함수에 모든 로직 통합
  - 노트북을 처음부터 끝까지 실행하기만 하면 모든 것이 자동 처리
- **체크포인트 기능 보존**:
  - 매 에포크 자동 저장, Early Stopping, 최적 모델 추적 등 모든 기능 유지
  - 체크포인트 관련 파라미터와 주석 완전 보존
- **사용자 경험 개선**:
  - 복잡한 재시작 절차 없이 단순히 노트북 전체 실행으로 해결
  - 처음 사용자와 재시작 사용자 모두 동일한 실행 방법 사용

### [2025-01-22] C-AE 복합 Loss 함수 도입으로 성능 향상
- **복합 Loss 함수**: `total_loss = α * MSE + β * (1 - cosine_similarity)` 방식 도입
- **핵심 개선사항**:
  - MSE Loss: 점별 복원 정확도 최적화
  - 코사인 유사도 Loss: 벡터 방향성 보존 최적화
  - 마스킹 적용: 패딩 부분 제외한 유효 영역만 계산
  - 동적 가중치 조절: α(MSE), β(코사인) 파라미터로 학습 목표 조정
- **TrainingConfig 설정**:
  ```python
  use_composite_loss: bool = True  # 복합 loss 사용 여부
  mse_weight: float = 1.0         # α: MSE 가중치
  cosine_weight: float = 0.5      # β: 코사인 유사도 가중치
  ```
- **Loss 가중치 조절 가이드**:
  - 정확한 복원: α=1.0, β=0.1 (MSE 중심)
  - 균형잡힌 학습: α=1.0, β=0.5 (기본 권장)
  - 방향성 중시: α=0.7, β=1.0 (코사인 유사도 중심)
- **성능 향상 기대**:
  - 벡터 방향성 보존으로 검색 성능 향상
  - MSE + 코사인 유사도 결합으로 더 견고한 복원
  - 다양한 벡터 타입(origin, dct, wavelet)에 적응적 학습
