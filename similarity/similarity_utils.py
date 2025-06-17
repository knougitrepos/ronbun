# 유사도 관련 함수 예시
def dummy_similarity():
    return "유사도 결과"

from typing import List, Literal
import numpy as np

# 거리 계산 함수

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def inner_product(a, b):
    return np.dot(a, b)

# 유사도 검색 함수 (DB 연동/pgvector 연산은 후속 구현)
def search_similar_vectors(
    query_vec: np.ndarray,
    db_vectors: List[np.ndarray],
    metric: Literal['cosine', 'euclidean', 'inner_product'] = 'cosine',
    top_k: int = 5,
    use_hnsw: bool = False,
    hnsw_params: dict = None
):
    """
    다양한 거리 계산 방식 및 HNSW/Brute Force 전략을 지원하는 유사도 검색 함수
    - metric: 'cosine', 'euclidean', 'inner_product' 중 선택
    - use_hnsw: True면 HNSW, False면 Brute Force (K값 등으로 자동 선택 가능)
    - hnsw_params: HNSW 파라미터(ef_search, M 등)
    """
    if metric == 'cosine':
        scores = [cosine_similarity(query_vec, v) for v in db_vectors]
        reverse = True  # 높은 값이 유사
    elif metric == 'euclidean':
        scores = [euclidean_distance(query_vec, v) for v in db_vectors]
        reverse = False  # 낮은 값이 유사
    elif metric == 'inner_product':
        scores = [inner_product(query_vec, v) for v in db_vectors]
        reverse = True
    else:
        raise ValueError(f'Unknown metric: {metric}')
    # 정렬 및 top_k 반환
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=reverse)
    top_idx = [i for i, _ in idx_scores[:top_k]]
    return top_idx, [scores[i] for i in top_idx]

# (실제 DB/pgvector 연동, HNSW 인덱스 등은 후속 구현에서 추가)
