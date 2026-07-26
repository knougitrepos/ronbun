# RFW 및 BUPT-BalancedFace 로컬 데이터 기록

## 문서 상태

- 기록일: 2026-07-23 KST
- 저장소: `C:\ronbun`
- 원본 위치:
  - `data/raw/RFW`
  - `data/raw/RFW-balancedface`
- 원칙: 원본·변환 이미지·생체 임베딩은 Git에 포함하거나 재배포하지 않는다.
- 상세 고정값: `configs/datasets/rfw_balancedface.yaml`

## 1. 데이터셋 역할

| 데이터셋 | 현재 연구 역할 | 허용 | 금지 |
| --- | --- | --- | --- |
| RFW | 인종 그룹별 압축 민감도에 대한 공식 1:1 verification 평가 | frozen PCA/PQ 적용, pair score drift, threshold crossing, ROC/EER/TAR@FAR, 그룹 격차 | RFW test에서 PCA/PQ fit, 전체 test에서 threshold fit 후 같은 test 평가, DIR/FPIR open-set 주장 |
| BUPT-BalancedFace(Equalizedface) | PCA/PQ development-fit 및 threshold calibration 후보 | RFW 중복 제거 후 그룹별 identity-disjoint development/calibration | FR 모델 재학습, 최종 test 또는 headline 성능 보고 |

RFW 원 논문은 4개 그룹으로 구성된 face verification benchmark를 정의한다. 저장소에서는 이를 기존 SurvFace 1:N open-set protocol과 분리한다. [RFW ICCV 2019 공식 논문](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Racial_Faces_in_the_Wild_Reducing_Racial_Bias_by_Information_ICCV_2019_paper.pdf)

BUPT-BalancedFace는 모델을 새로 학습하기 위한 것이 아니라 frozen FR checkpoint에서 압축기를 fit하고 threshold를 보정하기 위한 외부 development source로만 사용한다. 데이터셋 규모·구성의 출처는 저자 공개 연구를 참고하되, 이 저장소의 숫자는 다운로드된 로컬 파일을 직접 감사한 값이다. [RL-RBN 저자 공개 프리프린트](https://arxiv.org/abs/1911.10692)

## 2. 로컬 RFW 식별 결과

| 물리 파일 | 크기 | SHA-256 | 상태 |
| --- | ---: | --- | --- |
| `images/test.tar.gz` | 1,349,393,437 B | `2EC1875599D01888431BE83FE51ED9614188A026539FB970653081CA0FBF3E39` | EOF·경로 안전성·protocol 검증 통과 |
| `bin_for_mxnet/RFW_test.tar.gz` | 231,211,429 B | `8259E53AB1B542A9747335F34B2AE48E22A675BCE472E391D6B94BC900FDE572` | EOF·경로 안전성 검증 통과 |
| `readme.txt` | 2,233 B | `FBFA5FE0759B76EBB6E5455A93BA965C41F647BD56101DF6F85EB1FE776C0BE2` | 확인됨 |

JPG와 BIN은 같은 RFW test의 대체 표현이므로 수량을 더하지 않는다.

공식 JPG/protocol 감사 결과:

| RFW 그룹 | 이미지 | 그룹 내부 identity | pair |
| --- | ---: | ---: | ---: |
| African | 10,415 | 2,995 | 6,000 |
| Asian | 9,688 | 2,492 | 6,000 |
| Caucasian | 10,196 | 2,958 | 6,000 |
| Indian | 10,308 | 2,984 | 6,000 |
| 합계 | 40,607 | 11,429 | 24,000 |

- 각 그룹은 10 folds이며 fold마다 genuine 300개, impostor 300개다.
- pair가 참조하는 이미지, landmark, image list 및 archive member가 모두 일치한다.
- JPG는 400×400 loose crop이고 BIN은 pair 순서의 112×112 정렬 이미지다.
- 실제 이미지가 있는 고유 source identity는 11,416개다.
- source identity 13개는 실제 이미지 기준 둘 이상의 RFW 그룹에 나타난다. 그룹은 데이터셋이 제공한 평가 그룹이며 개인의 보편적인 인종 ground truth로 일반화하지 않는다.
- `Caucasian_people.txt`에는 이미지 수가 0인 행이 1개 존재한다. pair와 image list에는 나타나지 않으므로 경고로 보존한다.

## 3. 로컬 BalancedFace 식별 결과

| 물리 파일 | 크기 | SHA-256 | 상태 |
| --- | ---: | --- | --- |
| `images/Equalizedface.tar.gz` | 33,377,022,995 B | `B9630A4A9E7C67CB12EB286B32DD06E31992725C79F953222AF7DBEDE496AE42` | 교체 완료, EOF·경로 안전성 검증 통과 |
| `rec_for_mxnet/Equalizedface.tar.gz` | 6,213,545,547 B | `7B47F25492858C87F91F79C745A5C8A92D555212F2FC2AF38DD375E3AD81CF66` | EOF 검증 통과 |
| `rec_for_mxnet/train_balancedface.lst` | 46,369,652 B | `E43B0A820400F201DF534588E49DEA06D78FF5E38EBDE7A87A3F559D0E5CD1AF` | 1,251,416행 검증 통과 |

2026-07-23에 기존 16,740,773,888 B 절단 파일을 다시 다운로드한 33,377,022,995 B 파일로 교체했다. 새 파일은 1,279,435개 tar member를 EOF까지 읽었으며 unsafe path, duplicate member 및 unexpected regular file이 모두 0건이다.

JPG archive와 RecordIO `.lst`는 같은 논리적 BalancedFace의 대체 배포 형식이지만 완전히 동일한 행 집합은 아니다.

| 그룹 | JPG 이미지/identity | RecordIO 목록 이미지/identity | JPG−RecordIO |
| --- | ---: | ---: | ---: |
| African | 324,376 / 7,000 | 324,376 / 7,000 | 0 / 0 |
| Asian | 325,475 / 7,000 | 325,493 / 7,000 | -18 / 0 |
| Caucasian | 326,484 / 7,000 | 326,484 / 7,000 | 0 / 0 |
| Indian | 275,095 / 7,000 | 275,063 / 6,999 | +32 / +1 |
| 합계 | 1,251,430 / 28,000 | 1,251,416 / 27,999 | +14 / +1 |

따라서 두 형식을 행 번호로 결합하거나 수량을 더하지 않는다. 실험 run마다 JPG 또는 RecordIO 중 하나만 선택하고 그 source hash를 고정한다.

JPG 표본 검사에서는 African/Caucasian이 400×400 RGB였지만 Asian/Indian에는 작은 가변 해상도 RGB 이미지가 포함됐다. 담당자 안내의 정렬 문제와 일치하므로 JPG 사용 시 얼굴 검출·정렬과 그룹별 성공률 검증이 필수다.

정상 RecordIO archive:

- `Equalizedface/property`: `27999,112,112`
- `Equalizedface/train.idx`
- `Equalizedface/train.rec`
- `.lst`: 1,251,416 이미지, 27,999 identity, 중복 경로 0

| race label | 데이터셋 그룹 | identity | 이미지 |
| ---: | --- | ---: | ---: |
| 0 | Caucasian | 7,000 | 326,484 |
| 1 | Indian | 6,999 | 275,063 |
| 2 | Asian | 7,000 | 325,493 |
| 3 | African | 7,000 | 324,376 |

label 0~2는 JPG archive의 named folder와 대조했고, label 3은 남은 유일 그룹인 African으로 판정했다. 완전한 새 JPG archive에서도 네 named group과 identity 수를 확인했다.

## 4. RFW–BalancedFace 누수 감사

prefix 이전의 provider Freebase identity ID로 교집합을 검사한 결과:

- 겹치는 source identity: 3개
- 제거할 BalancedFace 이미지: 177장
- RFW 쪽 관련 이미지: 11장
- 한 identity는 두 데이터셋의 그룹 표기가 서로 다르다.

정확한 ID는 공개 문서가 아니라 실행 artifact인 `excluded_rfw_overlap_identities.csv`에만 기록한다. `build_balancedface_index_bundle()`은 RFW identity 목록 없이 실행되지 않으며, overlap identity의 모든 행을 제거한 뒤에만 development/calibration을 나눈다.

이 제거는 BalancedFace와 RFW 사이의 직접 누수를 막을 뿐이다. 현재 Step 2 후보 checkpoint는 모두 `training_dataset: ms1mv2`로 선언되어 있으며, RFW 제공자는 MS-Celeb-1M 계열로 학습한 모델의 RFW 평가를 경고한다. 따라서 현재 checkpoint는 RFW headline 평가가 차단된다. 허용하려면 다음 중 하나에 대한 검증 가능한 evidence가 필요하다.

1. 공식 `MS1M_wo_RFW` 학습 checkpoint
2. RFW identity 제거가 검증된 checkpoint
3. RFW와 비중복임이 문서화된 다른 학습 source

## 5. 코드와 실행 순서

관련 코드:

- `research/datasets/sources.py`: 소스 식별, SHA-256, tar 안전성·EOF 검사, 안전 추출
- `research/datasets/rfw.py`: 공식 image/pair/landmark protocol과 dev scope
- `research/datasets/balancedface.py`: `.lst` 검증, overlap 제거, 그룹별 identity split
- `research/datasets/compatibility.py`: 데이터 역할과 checkpoint 자격 fail-closed gate

실행 순서:

1. `notebooks/rfw/00_data_preparation/00_data_preparation.ipynb`
2. `notebooks/balancedface/00_data_preparation/00_data_preparation.ipynb`

두 노트북 모두 기본값은 `DATA_FRACTION=1.0`, `EXECUTE_STAGE=True`, `WRITE_OUTPUTS=True`, `OVERWRITE=True`다. RFW 출력의 `_SUCCESS`와 `source_identities.txt`가 있어야 BalancedFace 준비를 시작할 수 있다.

현재 중단점:

- RFW aligned BIN의 제한적 안전 parser/materializer는 미구현
- BalancedFace RecordIO decoder/materializer는 미구현
- BalancedFace JPG의 공통 정렬·그룹별 coverage 검증은 미구현
- 공통 112×112 crop hash와 그룹별 decode/alignment coverage는 미검증
- 적격 checkpoint가 없어 RFW headline 모델 평가는 차단됨

따라서 현재 구현은 **데이터 식별·공식 protocol·누수 없는 source index까지**이며 실제 RFW/BalancedFace 임베딩 실험 완료를 뜻하지 않는다.

## 6. 이용·윤리 기록

- 제공자 접근 메일은 다운로드 경위의 증거로 보존하되 공개 저장소에 발신자 개인정보를 복사하지 않는다.
- 명시적 재배포 라이선스가 확인되지 않았으므로 원본, 정렬 이미지, RecordIO, 파생 생체 임베딩을 공개하지 않는다.
- 논문에는 생성 절차, checksum, 집계 통계와 재현 코드만 공개한다.
- 생체정보 처리와 그룹별 공정성 해석은 소속 기관의 윤리·보안 기준을 별도로 확인한다.
