# 보관 노트북

2026-09-30 정리 후 40개 노트북을 보관한다. 현재 주실험은
[노트북 실행 안내](C:/ronbun/notebooks/README.md)를 따른다.
보관은 기능 폐기나 계산 오류 판정이 아니다.
개별 단계 디버깅·별도 DB 실험·선택 ablation을 다시 할 때 사용할 수 있다.

## 분류와 사용 조건

| 구분 | 수 | 역할 |
|---|---:|---|
| 공통 함수의 수동 단계 실행 창 | 18 | LFW/SurvFace 각각 aligned crop·landmark 2개와 Grad-CAM 7개. 공통 runner가 같은 단계 함수를 호출한다. |
| 과거 임베딩·압축·DB 평가 | 14 | LFW/SurvFace 각각 embeddings 2개, compression 3개, open_set 2개. 별도 프로토콜·DB 경로로 보존한다. |
| 단일 조건 보정·진단 | 2 | calibration 00의 S/L·bin 비교, 02의 saliency 점검. 전체 행렬은 활성 batch 사용. |
| 독립 checkpoint 준비·smoke | 2 | common/model_preparation. 일반 source 생성에서는 공통 runner가 등록/검증을 호출한다. |
| RFW-Official 수동 단계 | 3 | protocol 준비, origin 추출, frozen codec 평가. 일반 보조 평가에는 활성 all-in-one 사용. |
| BalancedFace 개발 후보 준비 | 1 | source index·overlap 제거. 실제 이미지 materialization 완료나 현재 최종 test를 뜻하지 않는다. |
| 합계 | **40** | 원래의 상대 디렉터리 구조를 유지한다. |

LFW/SurvFace의 00_data_preparation/00_data_preparation.ipynb 두 개는 원시 manifest 생성에
여전히 필요하므로 활성 경로로 복원했다. 보관된 aligned crop/landmark 단계와 구분한다.

SurvFace의 과거 pgvector 경로는 원본/PCA의 exact/HNSW 평가를 수행한다.
현재 파일 기반 PQ ADC 실험과 검색·평가 조건이 다르므로 완전히 대체되었다고 해석하지 않는다.
PQ code는 이 경로에서도 pgvector HNSW vector가 아니다.
DB 실험 재사용 시 점수 공간·프로토콜·학습/평가 분할을 별도로 확인한다.

calibration 00의 S/L·2/5-bin 개별 진단도 batch의 한 설정과 완전히 같은 메뉴는 아니다.
보관 노트북 전체를 주실험의 필수 선행 단계로 실행하지 않는다.
RFW-Official 준비 출력의 source_identities.txt와 _SUCCESS는
BalancedFace overlap 제거를 시험할 때 필요하다.

## 보관 경로에서 실행하기

프로젝트 루트 또는 해당 notebook 디렉터리에서 커널을 재시작해 실행한다.
노트북은 research/와 configs/를 기준으로 프로젝트 루트를 찾는다.
첫 설정 셀·현재 환경·source artifact·완료/재사용 정책을 확인한다.
위치 변경을 이유로 과거 manifest나 완료 결과의 출처를 수정하지 않는다.

특정 노트북을 다시 활성 메뉴로 승격할 때에는 해당 파일의 호출자·설정·테스트·안내도 갱신한다.
여러 폴더를 wildcard로 일괄 복원하면 활성 폴더와 충돌할 수 있으므로 전체 복원 명령은 제공하지 않는다.
dataset별 보관 README는 과거 수동 실행 계약이다.
저장 출력 유무와 관계없이 실제 결과의 근거는 run manifest와 검증된 artifact다.
