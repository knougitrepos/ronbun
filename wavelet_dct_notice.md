# Wavelet, DCT, PCA, PQ 변환 방식 및 저장 정책 안내

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

---

## 3. PCA 변환 벡터 추출 및 저장 정책

- **transform_method:** 'pca'
- **지원 파라미터:**
  - (예상) n_components 등

- **저장 차원 및 함수:**
  - (현재 스크립트에서는 사용/저장 코드 없음)

- **설명:**
  - PCA(주성분 분석)는 차원 축소용 변환입니다.
  - 파이프라인 구조상 확장 가능하지만, 현재 스크립트에서는 미사용입니다.

---

## 4. PQ(Product Quantization) 변환 벡터 추출 및 저장 정책

- **transform_method:** 'pq'
- **지원 파라미터:**
  - (예상) n_subvectors, n_bits 등

- **저장 차원 및 함수:**
  - (현재 스크립트에서는 사용/저장 코드 없음)

- **설명:**
  - PQ는 벡터를 여러 서브벡터로 나누어 양자화하는 고속 검색용 변환입니다.
  - 파이프라인 구조상 확장 가능하지만, 현재 스크립트에서는 미사용입니다.

---

## 5. 예시 코드 (DCT, Wavelet)

```python
# DCT 변환 및 저장 예시
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

# Wavelet 변환 및 저장 예시
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

## 6. 참고
- DCT, Wavelet, PCA, PQ 등 다양한 변환은 파이프라인 구조상 확장 및 실험 자동화에 쉽게 연동할 수 있습니다.
- 현재 스크립트에서는 DCT, Wavelet만 저장/지원하며, PCA, PQ는 추후 실험 필요 시 추가 가능합니다. 