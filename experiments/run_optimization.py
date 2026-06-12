import mlflow
import optuna
from core.config import ConfigLoader
from core.pipeline.vector_pipeline import VectorPipeline
# from core.database import VectorRepository

def objective(trial):
    norm_type = trial.suggest_categorical("norm", ["l2", "zscore"])
    pca_dim = trial.suggest_categorical("pca_dim", [256])
    use_pq = trial.suggest_categorical("use_pq", [True, False])
    # config, pipeline, repository 등 연동 예시
    config = ConfigLoader()
    pipeline = VectorPipeline(config)
    # repository = VectorRepository(db_session)
    # 실제 실험 파이프라인 실행 및 평가 (샘플)
    # result = pipeline.run(...)
    result = {"recall_1": 0.85, "recall_5": 0.92, "query_time": 0.012}
    with mlflow.start_run():
        mlflow.log_param("norm", norm_type)
        mlflow.log_param("pca_dim", pca_dim)
        mlflow.log_param("use_pq", use_pq)
        mlflow.log_metric("recall_1", result["recall_1"])
        mlflow.log_metric("recall_5", result["recall_5"])
        mlflow.log_metric("query_time", result["query_time"])
    return result["recall_5"]

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("vector_search_optimization")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
