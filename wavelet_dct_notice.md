# Wavelet, DCT, PCA, PQ 변환 방식 및 저장 정책 안내

---

## 1. DCT(이산 코사인 변환) 벡터 추출 및 저장 정책

- **transform_method:** 'dct'
- **지원 파라미터:**
  - keep_dim: 32, 64, 128, 256 (실제 저장은 128, 256만)
  - mode: 'low', 'high' (저주파/고주파)
- **저장 차원 및 함수:**
  - 128차원: `add_embedding_128` (keep_dim=128)
  - 256차원: `add_embedding_256` (keep_dim=256)
- **설명:**
  - DCT(이산 코사인 변환)는 이미지 임베딩 벡터를 주파수 영역으로 변환하여, 저주파/고주파 성분을 선택적으로 추출할 수 있습니다.
  - 차원 축소(128, 256 등)와 주파수 모드('low', 'high')를 조합하여 다양한 실험이 가능합니다.
  - 32, 64차원도 변환은 가능하나, 현재 DB에는 128, 256차원만 저장합니다.

**예시 코드:**
```python
for keep_dim in [128, 256]:
    for mode in ['low', 'high']:
        vec_dct = pipeline.run(image, transform_method='dct', transform_params={'keep_dim': keep_dim, 'mode': mode})
        if vec_dct is not None:
            if keep_dim == 128:
                repository.add_embedding_128(
                    image_id=image_id,
                    vector_type='dct',
                    parameters={'keep_dim': keep_dim, 'mode': mode},
                    embedding=vec_dct,
                    created_at=None
                )
            elif keep_dim == 256:
                repository.add_embedding_256(
                    image_id=image_id,
                    vector_type='dct',
                    parameters={'keep_dim': keep_dim, 'mode': mode},
                    embedding=vec_dct,
                    created_at=None
                )
```

---

## 2. Wavelet 변환 벡터 추출 및 저장 정책

- **transform_method:** 'wavelet'
- **지원 파라미터:**
  - level: 1, 2, 3
  - mode: 'low', 'high'
- **저장 차원 및 함수:**
  - 256차원: `add_embedding_256`
- **설명:**
  - Wavelet 변환은 이미지 임베딩을 다중 레벨로 분해하여 저주파/고주파 성분을 추출합니다.
  - level, mode 조합별로 256차원 벡터로 저장합니다.

**예시 코드:**
```python
for level in [1, 2, 3]:
    for mode in ['low', 'high']:
        vec_wavelet = pipeline.run(image, transform_method='wavelet', transform_params={'level': level, 'mode': mode})
        if vec_wavelet is not None:
            repository.add_embedding_256(
                image_id=image_id,
                vector_type='wavelet',
                parameters={'level': level, 'mode': mode},
                embedding=vec_wavelet,
                created_at=None
            )
```

---

## 3. PCA 변환 벡터 추출 및 저장 정책

- **transform_method:** 'pca'
- **지원 파라미터:**
  - n_components: 0.95(기본값, 누적 분산 95% 설명)
- **저장 차원 및 함수:**
  - (실험/설정에 따라 동적, 예: 95% 분산 설명)
- **설명:**
  - PCA(주성분 분석)는 차원 축소용 변환입니다.
  - n_components=0.95로 지정하면 전체 분산의 95%를 설명하는 최소 차원으로 자동 축소됩니다.
  - 실험적으로 95~99% 사이를 추천합니다.

**예시 코드:**
```python
from core.pipeline.transformers.pca import PCATransformer
transformer = PCATransformer(n_components=0.95)
vecs_pca = transformer.fit_transform(vecs)  # vecs: (N, D)
# 저장 예시
repository.add_embedding_128(
    image_id=image_id,
    vector_type='pca',
    parameters={'n_components': 128},
    embedding=vecs_pca,
    created_at=None
)
```

---

## 4. PQ(Product Quantization) 변환 벡터 추출 및 저장 정책

- **transform_method:** 'pq'
- **지원 파라미터:**
  - d: 입력 벡터 차원(128, 256 등)
  - M: 서브벡터 개수(기본 16)
  - nbits: 각 서브벡터 비트 수(기본 8)
- **저장 차원 및 함수:**
  - PQ 코드(양자화된 벡터)
- **설명:**
  - PQ는 faiss의 IndexPQ를 사용하여 벡터를 여러 서브벡터로 나누고, 각 서브벡터를 양자화합니다.
  - 고속 검색 및 대용량 벡터 압축에 유리합니다.
  - M, nbits는 실험적으로 조정(예: M=16, nbits=8)

**예시 코드:**
```python
from core.pipeline.transformers.pq import PQTransformer
transformer = PQTransformer(d=128, M=16, nbits=8)
transformer.fit(vecs)  # vecs: (N, d)
codes = transformer.transform(vecs)
repository.add_embedding_pq(
    image_id=image_id,
    vector_type='pq',
    parameters={'d': 128, 'M': 16, 'nbits': 8},
    codes=codes,
    created_at=None
)
```

---

## 5. 코드북 저장/로드 정책 (PCA, PQ)

- 모든 코드북은 `codebook/` 폴더에 저장
- 파일명 규칙: 변환명_차원 (예: pca_128.joblib, pq_128.faiss)

### PCA 코드북
- 저장: joblib 사용
- 예시:
```python
transformer = PCATransformer(n_components=128)
transformer.fit(X)
transformer.save_codebook()  # codebook/pca_128.joblib
# 로드
transformer = PCATransformer.load_codebook(128)
```

### PQ 코드북
- 저장: faiss 공식 write_index/read_index 사용
- 예시:
```python
transformer = PQTransformer(d=128, M=16, nbits=8)
transformer.fit(X)
transformer.save_codebook()  # codebook/pq_128.faiss
# 로드
transformer = PQTransformer.load_codebook(128, M=16, nbits=8)
```

---

## 6. DB 저장 정책 (PCA, PQ)

### PCA
- 변환 결과: 차원 축소된 실수 벡터 (예: 128, 256차원)
- 저장 테이블: 기존 embedding_128, embedding_256
- 저장 방식: vector_type='pca', parameters={'n_components': 128} 등으로 구분
- 예시 코드:
```python
repository.add_embedding_128(
    image_id=image_id,
    vector_type='pca',
    parameters={'n_components': 128},
    embedding=vec_pca,  # (128, 실수)
    created_at=None
)
```

### PQ
- 변환 결과: 양자화된 코드(정수 배열, uint8 등)
- 저장 테이블: 별도 embedding_pq 테이블(권장) 또는 기존 embedding_128/256에 vector_type='pq'로 저장
- 예시 스키마:
```python
class EmbeddingPQ(Base):
    __tablename__ = 'embedding_pq'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    codes = Column(LargeBinary)  # 또는 JSON, depending on storage preference
    created_at = Column(TIMESTAMP)
```
- 예시 코드:
```python
repository.add_embedding_pq(
    image_id=image_id,
    vector_type='pq',
    parameters={'d': 128, 'M': 16, 'nbits': 8},
    codes=pq_codes,  # (N, M), uint8
    created_at=None
)
```
- 대규모 실험/검색용은 별도 테이블, 소규모/실험용은 기존 테이블에 저장 가능

---

## 7. 실험 자동화/분산 처리 연동

- Optuna + MLflow로 실험 자동화, Celery + Redis로 대량 이미지 분산 처리
- batch_vector_transform_and_save.py에서 모든 큐/중복 체크/분산 처리 통합
- 실험 자동화, 변환 정책, DB 구조 등은 README.md와 동기화하여 관리

---

## 8. 참고 및 유지보수 원칙

- DCT, Wavelet, PCA, PQ 등 다양한 변환은 파이프라인 구조상 확장 및 실험 자동화에 쉽게 연동할 수 있습니다.
- 코드/정책/구조 변경 시 README.md, 본 문서에 즉시 반영하여 문서와 코드가 항상 동기화되도록 관리합니다.
- 폴더/파일/클래스/함수명은 기능 의미를 따르며, 계층적/추상화 구조로 유지보수성 강화 