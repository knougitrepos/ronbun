# LFW 준비와 BLUFR 분할 기반 보정

공개 목록이 없다면 [준비 노트북](C:/ronbun/notebooks/lfw/00_data_preparation/00_data_preparation.ipynb)의
`LFW_PROTOCOL_MODE="blufr_lists"`, `DOWNLOAD_BLUFR_CONFIG=True`로 Kernel Restart → Run All한다.
SHA가 고정된 MAT를 받고 원본 manifest와 대응을 확인한다. 기존 분할 파일은 덮어쓰지 않는다.

목록이 준비되어 있다면 calibration 03의 `blufr_calibration`, `EXECUTE=False`로 직접 점검한다.
현재 공개 13,233장 중 고정된 네 모델 source 임베딩/FIQA가 38장 부족하여 복구 전 정식 실행은 차단된다.
준비 노트북의 목록 다운로드가 얼굴 검출/임베딩 누락까지 복구하는 것은 아니다.

공개 train/test·gallery/probe를 유지하고 train 내부 7가지 압축 학습:보정 비율을 비교한다.
실행 순서와 설정은 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
별도 benchmark는 노트북에서 실행하지 않으며 Python 파일만 보존한다.

원본 face_manifest.csv 자체가 없는 신규 환경에서만 준비 노트북의 `legacy` 모드가 필요하다.
기존 source 생성은 공통 batch의 `legacy` 경로를 사용한다. 수동 단계와 DB 실험은
[보관 안내](C:/ronbun/notebooks/_archive/README.md)에 있다.
