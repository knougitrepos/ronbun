# LFW 최초 준비

BLUFR 공개 목록을 사용할 때는 준비 노트북 첫 코드 셀에서
`LFW_PROTOCOL_MODE="blufr_benchmark"`를 선택한다. 처음에만
`DOWNLOAD_BLUFR_CONFIG=True`로 hash가 고정된 MAT를 받는다.
원본 manifest에 공개 이미지가 모두 대응하는지 확인한 뒤 공통 batch 00의 같은 모드로 이동한다.
이 단계는 원본 manifest나 기존 identity split을 덮어쓰지 않는다.

공개 목록과 원본 13,233장은 대응하지만, 현재 완료 FR 임베딩은 네 모델 모두 13,195장이므로
38장 복구 전에는 BLUFR 정식 실행이 차단된다. 공통 batch가 누락 파일명을 표시한다.
별도 calibration 개선은 `calibration/03`의 `matched_calibration` 모드이며 BLUFR protocol 자체와 다르다.

[준비 노트북](C:/ronbun/notebooks/lfw/00_data_preparation/00_data_preparation.ipynb)은
원본에서 face_manifest.csv와 identity 분할 입력을 생성한다.
신규 환경 또는 원본/분할 변경 때 필요하며 기존 입력으로 보정할 때마다 재실행하지 않는다.

이후 source run은 공통 batch를 사용한다.
aligned crop·landmark·Grad-CAM 수동 단계와 과거 DB 경로는
[보관 안내](C:/ronbun/notebooks/_archive/README.md)에 있다.
현재 실행 순서는 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
