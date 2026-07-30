# 연관 논문 읽기 우선순위

이 문서는 현재 논문 방향인 **PostgreSQL/pgvector 기반 압축 얼굴 임베딩 검색에서 유사도 분포 보정을 통한 미등록 인물 거부 성능 개선**에 맞춰 읽을 논문을 우선순위대로 정리한다.

우선순위 기준은 다음과 같다.

1. open-set 미등록 거부와 직접 연결되는가
2. known unknown / unknown unknown 분리, DIR@FPIR, FPIR 같은 실험 프로토콜을 설계하는 데 필요한가
3. 압축으로 인한 유사도 분포 변화, score calibration, reconstruction error feature 설계에 필요한가
4. ArcFace, FIQA, template aggregation, PQ, HNSW 같은 구현 사전지식을 제공하는가

## 1. 최우선 필수 논문

| 순위 | 논문 | 먼저 읽어야 하는 이유 | 본 연구에서 쓰는 위치 |
| --- | --- | --- | --- |
| 1 | [Toward Open-Set Face Recognition, Günther et al., CVPRW 2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Gunther_Toward_Open-Set_Face_CVPR_2017_paper.pdf) | 현재 연구의 open-set 논리를 가장 직접적으로 지지한다. 단순 cosine threshold가 closed-set에서는 괜찮아도 open-set에서는 약해질 수 있고, known unknown과 unknown unknown을 분리해야 한다는 근거가 된다. | 문제 정의, unknown probe 분리, 단순 threshold baseline 비판 |
| 2 | [IARPA Janus Benchmark-C: Face Dataset and Protocol, Maze et al., ICB 2018](https://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf) | template 기반 face recognition, verification, identification, open-set 평가 프로토콜의 기준점이다. 본 연구의 등록 템플릿, probe 구성, DIR/FPIR 계열 지표를 정당화하는 데 필요하다. | 데이터셋/평가 프로토콜, template 단위 평가, 지표 정의 |
| 3 | [ArcFace: Additive Angular Margin Loss for Deep Face Recognition, Deng et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) | 현재 시스템의 기본 임베딩이 ArcFace/InsightFace이므로, cosine similarity, L2-normalized embedding, angular margin의 의미를 이해해야 한다. | 임베딩 모델 배경, cosine score 해석, baseline feature extractor |
| 4 | [Tutorial on Logistic-Regression Calibration and Fusion, Morrison, 2021](https://arxiv.org/pdf/2104.08846) | proposed calibration의 기본 모델을 logistic regression으로 둘 때 필요한 사전지식이다. score를 확률 또는 likelihood-ratio-like 값으로 보정하는 관점을 제공한다. | BCE 기반 calibration, logistic regression baseline, Brier/ECE 설명 |
| 5 | [Product Quantization for Nearest Neighbor Search, Jégou et al., TPAMI 2011](https://inria.hal.science/inria-00514462v2/document) | PQ의 핵심 원리인 sub-vector quantization을 이해해야 PCA reconstruction error와 PQ reconstruction error를 같은 raw 변수로 섞으면 안 되는 이유를 설명할 수 있다. | PQ 보조 실험, 저장량 계산, reconstruction error 해석 |
| 6 | [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs, Malkov and Yashunin, TPAMI 2020](https://arxiv.org/abs/1603.09320) | pgvector HNSW의 검색 성능, recall-latency trade-off를 논문에서 설명하려면 HNSW 기본 원리를 알아야 한다. | pgvector/HNSW 실험, latency/recall 분석 |

## 2. 방법론 강화 논문

| 순위 | 논문 | 먼저 읽어야 하는 이유 | 본 연구에서 쓰는 위치 |
| --- | --- | --- | --- |
| 7 | [SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness, Terhörst et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_CVPR_2020_paper.pdf) | 사전학습 face recognition model의 embedding robustness로 품질을 측정한다는 점이 현재 연구의 FIQA 후보와 잘 맞는다. 별도 품질 라벨 없이 사용할 수 있다는 장점도 있다. | FIQA feature, 품질 기반 template ablation |
| 8 | [MagFace: A Universal Representation for Face Recognition and Quality Assessment, Meng et al., CVPR 2021](https://arxiv.org/abs/2103.06627) | 얼굴 품질과 recognition embedding을 함께 다루는 대표 논문이다. 품질 점수가 recognition 성능과 연결된다는 배경 설명에 유용하다. | quality feature의 의미, FIQA 관련 연구 |
| 9 | [Probabilistic Face Embeddings, Shi and Jain, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf) | embedding uncertainty와 feature fusion을 다룬다. 본 연구가 PFE를 구현하지 않더라도, 템플릿 분산과 품질을 uncertainty proxy로 쓰는 논리를 보강한다. | template dispersion feature, uncertainty 관련 discussion |
| 10 | [FaceQnet: Quality Assessment for Face Recognition Based on Deep Learning, Hernández-Ortega et al., 2019](https://arxiv.org/abs/1904.01740) | 성능 기반 face image quality estimator의 대표 baseline이다. SER-FIQ와 비교하여 지도학습 FIQA 계열을 설명할 수 있다. | FIQA 후보, 관련 연구 비교 |
| 11 | [Billion-Scale Similarity Search with GPUs, Johnson et al., 2017](https://arxiv.org/abs/1702.08734) | Faiss, PQ, compressed-domain search의 실용적 배경을 제공한다. PQ 보조 실험과 저장공간/검색속도 논의를 연결할 수 있다. | Faiss PQ 보조 실험, compressed retrieval 배경 |

## 3. 배경 및 방어용 논문

| 순위 | 논문 | 먼저 읽어야 하는 이유 | 본 연구에서 쓰는 위치 |
| --- | --- | --- | --- |
| 12 | [Towards Open Set Deep Networks, Bendale and Boult, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) | open-set recognition의 일반 이론과 open space risk를 이해하는 데 필요하다. 얼굴 특화 논문은 아니지만 논문 서론/관련연구의 이론적 배경으로 좋다. | open-set recognition 배경, closed-set softmax/threshold 한계 |
| 13 | [Toward Open Set Recognition, Scheirer et al., TPAMI 2013](https://pubmed.ncbi.nlm.nih.gov/23682001/) | open-set recognition의 고전적 정의와 평가 관점을 제공한다. 얼굴 인식 외 일반 open-set 문제 정의에 필요하다. | open-set 용어 정의, unknown class 논리 |
| 14 | [Face Image Quality Assessment: A Literature Survey, Schlett et al., ACM CSUR 2022](https://arxiv.org/abs/2009.01103) | FIQA 문헌을 넓게 정리한 survey다. 품질 기반 템플릿이 독창적 핵심 기여가 아니라는 점을 인정하고, 본 연구의 novelty를 calibration 쪽으로 옮기는 데 도움이 된다. | 관련연구 정리, 품질 기반 기법의 기존성 설명 |
| 15 | [Calibrated Confidence Scoring for Biometric Identification, Gorodnichy, NIST](https://www.nist.gov/document/gorodnichy2dmitrycalibratedconfidencescoringforbiometricidpdf) | biometric identification에서 여러 matching score를 confidence로 보정하는 관점이다. 본 연구의 top-1, margin, 품질, 압축 feature 기반 보정과 방향이 가깝다. | 보정 score의 실용적 의미, confidence reporting |
| 16 | [Score Normalization in Multimodal Biometric Systems, Jain and Ross, Pattern Recognition 2005](https://www.cse.msu.edu/~rossarun/pubs/RossScoreNormalization_PR05.pdf) | 서로 다른 score/source를 공통 domain으로 맞추는 고전적 biometric score normalization 논문이다. PCA/PQ profile별 score와 reconstruction error를 정규화해야 하는 이유를 보강한다. | score normalization 배경, profile별 정규화 논리 |

## 4. 읽는 순서

1. `Toward Open-Set Face Recognition`
2. `IJB-C: Face Dataset and Protocol`
3. `ArcFace`
4. `Tutorial on Logistic-Regression Calibration and Fusion`
5. `Product Quantization for Nearest Neighbor Search`
6. `HNSW`
7. `SER-FIQ`
8. `MagFace`
9. `Probabilistic Face Embeddings`
10. `FaceQnet`

위 10편을 먼저 읽으면 현재 논문의 문제 정의, 실험 설계, 핵심 feature, baseline, 압축 실험의 대부분을 방어할 수 있다. 나머지 논문은 관련연구 장과 질의응답 방어용으로 사용한다.

## 5. 논문별 반영 우선순위

- 반드시 본문에 인용: 1, 2, 3, 4, 5, 6
- 방법론/ablation 설명에 인용: 7, 8, 9, 10, 11
- 관련연구와 방어 논리에 인용: 12, 13, 14, 15, 16
