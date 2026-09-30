# 공통 실행·보고·관리

- orchestration/: source run 생성/재사용, 전체 PQ 보정, PQ 결과 집계, 별도 전이 실험.
- reports/: 여러 완료 run의 보고. 실행기가 notebook을 직접 호출하므로 경로를 유지한다.
- maintenance/: 특정 DB/run 폐기·격리. 일반 실험 실행 순서에 포함하지 않는다.

현재 실행 메뉴는 [노트북 안내](C:/ronbun/notebooks/README.md)를 따른다.
독립 checkpoint 등록·smoke 노트북은 _archive/common/model_preparation/에 보관한다.
일반 source 생성에서는 공통 runner가 모델 준비 함수를 호출한다.
