# LFW Grad-CAM follow-up

이 폴더는 기존 LFW 압축 정량 노트북과 분리된 Step 2 후속 분석입니다.
Grad-CAM을 실행하기 전에 같은 PyTorch checkpoint의 fallback-free 정량
PCA-only/PQ-only 결과가 완료되어 있어야 합니다.

실행 순서:

1. `00_source_and_model_freeze.ipynb`
2. `01_case_selection.ipynb`
3. `02_pair_gradcam_generation.ipynb`
4. `03_saliency_feature_analysis.ipynb`
5. `04_faithfulness_and_report.ipynb`

핵심 경계:

- 설명 target은 원본 query와 detached 원본 gallery template의 cosine입니다.
- hard PQ code나 양자화 연산을 미분하지 않습니다.
- 압축 profile은 정량 결과에서 사례를 선택하고 민감도를 연결하는 조건입니다.
- 기존 Step 1/Step 2 정량 artifact를 수정하거나 원본 fallback으로 대체하지
  않습니다.
- heatmap 예시만으로 원인을 주장하지 않고 random/low-saliency occlusion
  control보다 faithful한지 확인합니다.

각 노트북은 새 커널에서 위에서 아래로 실행합니다. 입력 hash가 달라지면
기존 결과를 덮어쓰지 말고 새 Grad-CAM run을 만듭니다.
