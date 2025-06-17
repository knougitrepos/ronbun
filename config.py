import os

# 환경설정(데이터베이스 등) 관리 Config 클래스 정의
class Config:
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'postgres')
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASS = os.environ.get('DB_PASS', 'postgres')

# [구조 개선] 실험 자동화 대응을 위한 config.yaml 기반 구조 예시
# (Hydra, Dynaconf 등 도구 연동 가능, 아래는 예시 주석)
'''
# config.yaml 예시
experiment:
  feature_extractor: arcface   # arcface, efficientnet 등
  normalization: l2           # l2, zscore, none
  vector_transform:
    - pca
    - dct
    - quantize
  search_strategy: hnsw       # hnsw, brute_force
  distance_metric: cosine     # cosine, euclidean, inner_product
  pca_params:
    n_components: 256
  hnsw_params:
    ef_search: 64
    M: 16
'''
# 실제 적용 시 PyYAML 등으로 config.yaml을 읽어 실험 파라미터를 동적으로 불러올 수 있도록 구현 