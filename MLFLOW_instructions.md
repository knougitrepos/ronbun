# MLflow 사용법 안내

## 1. MLflow UI 실행

```bash
mlflow ui --port 5001 --backend-store-uri ./mlruns --host 127.0.0.1
```
- 브라우저에서 http://localhost:5001 접속
- 실험별 파라미터, 메트릭, 아티팩트 시각적 비교 가능

---

## 2. 실험 자동화 코드(Optuna + MLflow) 실행

```bash
python optuna_mlflow/run_optuna.py
```
- 실험 파라미터/메트릭이 자동으로 MLflow에 기록됨
- 실험 결과는 mlruns/ 디렉토리에 저장

---

## 3. 실험 기록 방법 (코드 내 예시)

```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("norm", norm_type)
    mlflow.log_param("pca_dim", pca_dim)
    mlflow.log_metric("recall@5", recall5)
    mlflow.log_artifact("model.pkl")  # 파일 저장 시
```

---

## 4. 주요 명령어

| 명령어 | 설명 |
|--------|------|
| mlflow ui --port 5001 | MLflow UI 실행 (5001번 포트) |
| python optuna_mlflow/run_optuna.py | 실험 자동화 실행 |

---

## 5. 실전 팁

- 실험별 파라미터/메트릭/아티팩트 비교, 그래프 정렬, 하이라이트 가능
- 최적 조합(예: recall@5 최대) 자동 확인
- 실험 기록은 mlruns/ 폴더에 저장됨 (백업/공유 가능)
- 여러 머신에서 MLflow 서버를 원격으로 띄워 협업 가능
- 실험 자동화(Optuna 등)와 연동 시 반복 실험 결과를 한눈에 비교

---

## 6. 참고
- MLflow 공식 문서: https://mlflow.org/docs/latest/index.html
- 실험 자동화 예시 코드는 optuna_mlflow/ 디렉토리 참고 