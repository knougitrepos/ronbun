# BalancedFace 노트북

먼저 `../rfw/00_data_preparation/00_data_preparation.ipynb`를 실행한 뒤
`00_data_preparation/00_data_preparation.ipynb`를 실행한다.

BalancedFace는 RFW와 겹치는 identity를 제거한 development/calibration 후보이며
최종 test가 아니다. RecordIO image materialization이 구현되기 전에는 source
index를 실제 image manifest로 해석하지 않는다.
