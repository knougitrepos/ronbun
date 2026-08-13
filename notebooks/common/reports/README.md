# Reports

## 4 models × 3 open-set datasets

단일 checkpoint 상세 보고는 그대로 유지한다. 4개 checkpoint를 한 표에서 비교하려면
`MODEL_RUN_MATRIX`에 ArcFace, AdaFace, MagFace, EdgeFace 각각의 LFW, SurvFace,
RFW-Custom 완료 run directory를 모두 명시한다. 보고서는 12개 run의 completed/fallback-free
상태, model UID, v5 TPIR20 compact manifest와 SHA-256, FPIR 1%/5%/10%/20%/30% 및 CI 계약을 검증한 뒤에만
통합 표를 만든다. 최신 run을 자동 선택하지 않으며 결과 해석은 checkpoint-level로 제한한다.

완료된 LFW, SurvFace, RFW-Custom run의 CSV/manifest를 읽어 PCA/PQ,
1:N open-set, Grad-CAM/saliency, storage/latency compact 표와 그림을 생성한다.
계산의 정식 근거는 notebook 출력이 아니라 run artifact와 checksum이다.

FPIR 1%/5%/10%/20%/30%는 동일 검색 점수에 서로 다른 threshold를 적용해 보고한다.
TPIR20은 Top-20 안 정답 identity의 score가 calibration threshold를 통과한 비율로 계산하며,
Origin·compressed frozen threshold·compressed recalibrated threshold와 Origin 대비 retention을 분리한다.
각 행은 false accept 수, non-mated 분모, realized FPIR, Wilson 95% CI와
compressed-minus-origin paired CI를 보존한다. LFW/SurvFace의 두 operating point는
별도 appendix CSV로도 출력한다.

RFW-Custom 1:N 결과는 open-set 비교에 포함하지만 RFW-Official 1:1 TAR/FAR/EER는
별도 표와 artifact로 유지한다. EdgeFace와 RFW의 학습 identity overlap은
`UNKNOWN`이며 RFW 결과를 strict unseen-identity 증거로 사용하지 않는다. 네 모델
비교는 loss 또는 architecture의 인과 비교가 아닌 checkpoint-level generalization에
한정한다.
