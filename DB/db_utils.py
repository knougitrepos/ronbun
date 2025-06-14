import os
import psycopg2

def connect_db():
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_USER = os.environ.get("DB_USER", "postgres")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
    DB_NAME = os.environ.get("DB_NAME", "postgres")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )

def init_db():
    conn = connect_db()
    cursor = conn.cursor()
    try:
        # 1) recall_test_results 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recall_test_results (
                id SERIAL PRIMARY KEY,
                test_name TEXT NOT NULL,
                vector_type_compared TEXT NOT NULL,
                distance_metric TEXT NOT NULL,
                k_value INTEGER NOT NULL,
                recall_value DOUBLE PRECISION NOT NULL,
                method TEXT,
                parameters JSON,
                created_at TIMESTAMP,
                cross_validation_fold INTEGER
            );
        """)
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_recall_results_metric_k
            ON recall_test_results (distance_metric, k_value);""")
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_recall_results_method
            ON recall_test_results (method);""")
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_recall_results_test_name
            ON recall_test_results (test_name);""")
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_recall_results_vector_type
            ON recall_test_results (vector_type_compared);""")
        # 2) image_embeddings 테이블 (EfficientNetB0 output 1280차원)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                image_path TEXT NOT NULL UNIQUE,
                label TEXT,
                used_feature_extract_model TEXT,
                used_distance_model TEXT,
                vec_origin VECTOR(1280) NOT NULL,
                vec_wavelet VECTOR(256),
                params_wavelet JSON,
                vec_dct VECTOR(256),
                params_dct JSON,
                vec_pca VECTOR(256),
                params_pca JSON,
                vec_quantized VECTOR(256),
                params_quantized JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_image_embeddings_label
            ON image_embeddings (label);""")
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_image_embeddings_created_at
            ON image_embeddings (created_at);""")
        # HNSW 인덱스 (L2, Cosine, Dot) 각 벡터 컬럼별 생성
        for col in ["vec_origin", "vec_wavelet", "vec_dct", "vec_pca", "vec_quantized"]:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{col}_l2_hnsw
                ON image_embeddings USING hnsw ({{col}} vector_l2_ops);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{col}_cosine_hnsw
                ON image_embeddings USING hnsw ({{col}} vector_cosine_ops);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{col}_dot_hnsw
                ON image_embeddings USING hnsw ({{col}} vector_ip_ops);
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings_summary (
                id SERIAL PRIMARY KEY,
                label TEXT NOT NULL,
                count INTEGER NOT NULL,
                updated_at TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recall_test_results_summary (
                id SERIAL PRIMARY KEY,
                label TEXT NOT NULL,
                count INTEGER NOT NULL,
                updated_at TIMESTAMP,
                explain TEXT
            );
        """)
        conn.commit()
    except psycopg2.Error as e:
        print(f"테이블 생성 에러: {e}")
    finally:
        cursor.close()
        conn.close()

def reset_db():
    conn = connect_db()
    cursor = conn.cursor()
    try:
        # 모든 테이블 드롭 (존재하면)
        cursor.execute("DROP TABLE IF EXISTS recall_test_results CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS image_embeddings CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS image_embeddings_summary CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS recall_test_results_summary CASCADE;")
        conn.commit()
        # 테이블 재생성
        init_db()
    except psycopg2.Error as e:
        print(f"DB 리셋 에러: {e}")
    finally:
        cursor.close()
        conn.close()

def update_image_embeddings_summary():
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM image_embeddings_summary;")
        cursor.execute("""
            INSERT INTO image_embeddings_summary (label, count, updated_at)
            SELECT label, COUNT(*), NOW()
            FROM image_embeddings
            GROUP BY label
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def update_recall_test_results_summary():
    conn = connect_db()
    cursor = conn.cursor()
    try:
        # explain 컬럼이 없으면 추가 (idempotent)
        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='recall_test_results_summary' AND column_name='explain') THEN
                    ALTER TABLE recall_test_results_summary ADD COLUMN explain TEXT;
                END IF;
            END$$;
        """)
        cursor.execute("DELETE FROM recall_test_results_summary;")
        # 각 vector_type_compared별로 가장 최근 parameters 값을 explain에 저장
        cursor.execute("""
            INSERT INTO recall_test_results_summary (label, count, updated_at, explain)
            SELECT vector_type_compared, COUNT(*), MAX(created_at),
                   MAX(parameters::text)
            FROM recall_test_results
            GROUP BY vector_type_compared
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()
