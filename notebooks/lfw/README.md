# LFW 최초 준비

[준비 노트북](C:/ronbun/notebooks/lfw/00_data_preparation/00_data_preparation.ipynb)은
원본에서 face_manifest.csv와 identity 분할 입력을 생성한다.
신규 환경 또는 원본/분할 변경 때 필요하며 기존 입력으로 보정할 때마다 재실행하지 않는다.

이후 source run은 공통 batch를 사용한다.
aligned crop·landmark·Grad-CAM 수동 단계와 과거 DB 경로는
[보관 안내](C:/ronbun/notebooks/_archive/README.md)에 있다.
현재 실행 순서는 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
