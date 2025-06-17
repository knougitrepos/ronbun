import mlflow
import optuna
from experiment import run_experiment

def objective(trial):
    norm_type = trial.suggest_categorical("norm", ["l2", "zscore"])
    pca_dim = trial.suggest_int("pca_dim", 64, 512, step=64)
    use_pq = trial.suggest_categorical("use_pq", [True, False])
    result = run_experiment(norm=norm_type, pca_dim=pca_dim, use_pq=use_pq)
    with mlflow.start_run():
        mlflow.log_param("norm", norm_type)
        mlflow.log_param("pca_dim", pca_dim)
        mlflow.log_param("use_pq", use_pq)
        mlflow.log_metric("recall_1", result["recall@1"])
        mlflow.log_metric("recall_5", result["recall@5"])
        mlflow.log_metric("query_time_avg", result["query_time"])
    return result["recall@5"]

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("vector_search_optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30) 