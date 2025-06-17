def run_experiment(norm, pca_dim, use_pq):
    # 1. 데이터 로딩 및 벡터 추출
    # 2. 정규화(norm), 차원축소(pca_dim), 양자화(use_pq) 등 파이프라인 적용
    # 3. 검색/평가(Recall@K, 쿼리 시간 등) 수행
    # 4. 결과 dict 반환
    # (아래는 샘플, 실제 파이프라인 연동 필요)
    return {
        "recall@1": 0.85,
        "recall@5": 0.92,
        "query_time": 0.012
    } 