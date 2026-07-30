# RFW 노트북

현재 구현된 단계는 `00_data_preparation/00_data_preparation.ipynb`이다. RFW의
공식 4개 그룹 10-fold 1:1 verification protocol을 준비한다.

RFW는 PCA/PQ 학습 데이터가 아니며 현재 open-set headline 데이터셋으로 자동
승격하지 않는다. 이 단계가 생성한 source identity 목록은 BalancedFace 중복
제거 단계의 입력이다.
