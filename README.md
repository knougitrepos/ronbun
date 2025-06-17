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