> 보관된 수동 실행 계약이다. 현재 주실험 순서는 [활성 노트북 안내](C:/ronbun/notebooks/README.md)를 따른다.
> LFW/SurvFace 최초 manifest 준비는 활성 경로에 있다. 아래의 과거 설정/단계는 현재 코드와 입력을 확인한 뒤 사용한다.

# BalancedFace 노트북

먼저 `../rfw/00_data_preparation/00_data_preparation.ipynb`를 실행한 뒤
`00_data_preparation/00_data_preparation.ipynb`를 실행한다.

BalancedFace는 RFW와 겹치는 identity를 제거한 development/calibration 후보이며
최종 test가 아니다. RecordIO image materialization이 구현되기 전에는 source
index를 실제 image manifest로 해석하지 않는다.
