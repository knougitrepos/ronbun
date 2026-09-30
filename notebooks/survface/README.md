# SurvFace 최초 준비

[준비 노트북](C:/ronbun/notebooks/survface/00_data_preparation/00_data_preparation.ipynb)은
training 및 official manifest를 준비한다. 공통 runner는 이 입력 파일이 있어야 진행한다.
신규 환경 또는 원본 프로토콜 변경 때 실행하고 기존 완료 입력을 사용하는 보정에서는 재사용한다.

이후 source run은 공통 batch를 사용한다.
과거 pgvector exact/HNSW 평가와 수동 Step 4 경로는
[보관 안내](C:/ronbun/notebooks/_archive/README.md)에 있다.
현재 PQ ADC 실험과 과거 DB 평가의 검색·점수·프로토콜 조건을 구분한다.
현재 실행 순서는 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
