# RFW 1:1 Continuous FIQA Threshold Calibration Summary

## Executive Summary
- **Execution Mode**: Real RFW Data & Live Model Inference
- **Evaluation Scope**: Quick Smoke Slice (African 2 folds, 1,200 pairs)
- **Model**: arcface (arcface-7972a704552df378345f)
- **Evaluated Demographic Groups**: African
- **Evaluated Folds**: 2 folds per group
- **Total Evaluated Pairs**: 1,200 pairs
- **Quality Estimator**: cr_fiqa
- **Profiles Evaluated**: origin, pq_512_m128_b8, pq_512_m64_b8, pq_512_m32_b8
- **Methods**: global_safe, fiqa_5bin, continuous_fiqa
- **Target FMRs**: 0.001, 0.01, 0.05, 0.1
- **Pair Quality Definition**: symmetric_min (min(q_left, q_right))
- **Formal FMR Guarantee**: None (formal_fmr_guarantee = False; empirical safety thresholding only)
- **Fairness Guarantee**: None (demographic differences are reported as empirical observations)

## 1. Storage Efficiency vs Compression Payload
| Profile | Payload Bytes | Codebook Bytes | Total Storage (Bytes) | Compression Ratio |
|---|---:|---:|---:|---:|
| origin | 2048 | 0 | 4915200 | 1.00x |
| pq_512_m128_b8 | 128 | 524288 | 831488 | 5.91x |
| pq_512_m64_b8 | 64 | 524288 | 677888 | 7.25x |
| pq_512_m32_b8 | 32 | 524288 | 601088 | 8.18x |

## 2. Target FMR vs Realized FMR, TAR, and TAR Gain by Demographic Group
| Group | Profile | Method | Target FMR | Realized FMR | Mean TAR | TAR Gain vs Global | Target Met Rate | Pooled FA / Impostors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| African | origin | continuous_fiqa | 0.001 | 0.0033 | 0.9817 | +0.0050 | 0.50 | 2 / 600 |
| African | origin | continuous_fiqa | 0.010 | 0.0033 | 0.9900 | +0.0133 | 1.00 | 2 / 600 |
| African | origin | continuous_fiqa | 0.050 | 0.0467 | 0.9933 | -0.0050 | 0.50 | 28 / 600 |
| African | origin | continuous_fiqa | 0.100 | 0.0933 | 0.9983 | +0.0000 | 0.50 | 56 / 600 |
| African | origin | fiqa_5bin | 0.001 | 0.0050 | 0.9767 | +0.0000 | 0.50 | 3 / 600 |
| African | origin | fiqa_5bin | 0.010 | 0.0050 | 0.9767 | +0.0000 | 1.00 | 3 / 600 |
| African | origin | fiqa_5bin | 0.050 | 0.0500 | 0.9983 | +0.0000 | 0.50 | 30 / 600 |
| African | origin | fiqa_5bin | 0.100 | 0.0767 | 0.9983 | +0.0000 | 0.50 | 46 / 600 |
| African | origin | global_safe | 0.001 | 0.0050 | 0.9767 | +0.0000 | 0.50 | 3 / 600 |
| African | origin | global_safe | 0.010 | 0.0050 | 0.9767 | +0.0000 | 1.00 | 3 / 600 |
| African | origin | global_safe | 0.050 | 0.0500 | 0.9983 | +0.0000 | 0.50 | 30 / 600 |
| African | origin | global_safe | 0.100 | 0.0767 | 0.9983 | +0.0000 | 0.50 | 46 / 600 |
| African | pq_512_m128_b8 | continuous_fiqa | 0.001 | 0.0033 | 0.9800 | +0.0033 | 0.50 | 2 / 600 |
| African | pq_512_m128_b8 | continuous_fiqa | 0.010 | 0.0033 | 0.9883 | +0.0117 | 1.00 | 2 / 600 |
| African | pq_512_m128_b8 | continuous_fiqa | 0.050 | 0.0400 | 0.9967 | -0.0017 | 0.50 | 24 / 600 |
| African | pq_512_m128_b8 | continuous_fiqa | 0.100 | 0.0867 | 0.9983 | +0.0000 | 0.50 | 52 / 600 |
| African | pq_512_m128_b8 | fiqa_5bin | 0.001 | 0.0050 | 0.9767 | +0.0000 | 0.50 | 3 / 600 |
| African | pq_512_m128_b8 | fiqa_5bin | 0.010 | 0.0050 | 0.9767 | +0.0000 | 1.00 | 3 / 600 |
| African | pq_512_m128_b8 | fiqa_5bin | 0.050 | 0.0467 | 0.9983 | +0.0000 | 0.50 | 28 / 600 |
| African | pq_512_m128_b8 | fiqa_5bin | 0.100 | 0.0767 | 0.9983 | +0.0000 | 0.50 | 46 / 600 |
| African | pq_512_m128_b8 | global_safe | 0.001 | 0.0050 | 0.9767 | +0.0000 | 0.50 | 3 / 600 |
| African | pq_512_m128_b8 | global_safe | 0.010 | 0.0050 | 0.9767 | +0.0000 | 1.00 | 3 / 600 |
| African | pq_512_m128_b8 | global_safe | 0.050 | 0.0467 | 0.9983 | +0.0000 | 0.50 | 28 / 600 |
| African | pq_512_m128_b8 | global_safe | 0.100 | 0.0767 | 0.9983 | +0.0000 | 0.50 | 46 / 600 |
| African | pq_512_m64_b8 | continuous_fiqa | 0.001 | 0.0050 | 0.9567 | +0.0133 | 0.50 | 3 / 600 |
| African | pq_512_m64_b8 | continuous_fiqa | 0.010 | 0.0100 | 0.9900 | +0.0467 | 0.50 | 6 / 600 |
| African | pq_512_m64_b8 | continuous_fiqa | 0.050 | 0.0267 | 0.9850 | -0.0117 | 1.00 | 16 / 600 |
| African | pq_512_m64_b8 | continuous_fiqa | 0.100 | 0.0717 | 0.9967 | -0.0017 | 1.00 | 43 / 600 |
| African | pq_512_m64_b8 | fiqa_5bin | 0.001 | 0.0050 | 0.9433 | +0.0000 | 0.50 | 3 / 600 |
| African | pq_512_m64_b8 | fiqa_5bin | 0.010 | 0.0050 | 0.9433 | +0.0000 | 1.00 | 3 / 600 |
| African | pq_512_m64_b8 | fiqa_5bin | 0.050 | 0.0400 | 0.9967 | +0.0000 | 1.00 | 24 / 600 |
| African | pq_512_m64_b8 | fiqa_5bin | 0.100 | 0.0567 | 0.9983 | +0.0000 | 1.00 | 34 / 600 |
| African | pq_512_m64_b8 | global_safe | 0.001 | 0.0050 | 0.9433 | +0.0000 | 0.50 | 3 / 600 |
| African | pq_512_m64_b8 | global_safe | 0.010 | 0.0050 | 0.9433 | +0.0000 | 1.00 | 3 / 600 |
| African | pq_512_m64_b8 | global_safe | 0.050 | 0.0400 | 0.9967 | +0.0000 | 1.00 | 24 / 600 |
| African | pq_512_m64_b8 | global_safe | 0.100 | 0.0567 | 0.9983 | +0.0000 | 1.00 | 34 / 600 |
| African | pq_512_m32_b8 | continuous_fiqa | 0.001 | 0.0033 | 0.9217 | +0.0433 | 0.50 | 2 / 600 |
| African | pq_512_m32_b8 | continuous_fiqa | 0.010 | 0.0067 | 0.8883 | +0.0100 | 1.00 | 4 / 600 |
| African | pq_512_m32_b8 | continuous_fiqa | 0.050 | 0.0183 | 0.9583 | -0.0200 | 1.00 | 11 / 600 |
| African | pq_512_m32_b8 | continuous_fiqa | 0.100 | 0.0783 | 0.9817 | -0.0083 | 1.00 | 47 / 600 |
| African | pq_512_m32_b8 | fiqa_5bin | 0.001 | 0.0033 | 0.8783 | +0.0000 | 0.50 | 2 / 600 |
| African | pq_512_m32_b8 | fiqa_5bin | 0.010 | 0.0033 | 0.8783 | +0.0000 | 1.00 | 2 / 600 |
| African | pq_512_m32_b8 | fiqa_5bin | 0.050 | 0.0283 | 0.9783 | +0.0000 | 1.00 | 17 / 600 |
| African | pq_512_m32_b8 | fiqa_5bin | 0.100 | 0.0700 | 0.9900 | +0.0000 | 1.00 | 42 / 600 |
| African | pq_512_m32_b8 | global_safe | 0.001 | 0.0033 | 0.8783 | +0.0000 | 0.50 | 2 / 600 |
| African | pq_512_m32_b8 | global_safe | 0.010 | 0.0033 | 0.8783 | +0.0000 | 1.00 | 2 / 600 |
| African | pq_512_m32_b8 | global_safe | 0.050 | 0.0283 | 0.9783 | +0.0000 | 1.00 | 17 / 600 |
| African | pq_512_m32_b8 | global_safe | 0.100 | 0.0700 | 0.9900 | +0.0000 | 1.00 | 42 / 600 |

## 3. Key Observations & Cautions
1. **Continuous FIQA vs Global Safe**: Continuous FIQA adapts thresholds dynamically based on symmetric pair quality min(q_left, q_right), preserving genuine matches on higher-quality pairs under compression.
2. **Score Space Separation**: Cosine similarity [-1, 1] and PQ ADC negative squared L2 [-4, 0] operate in distinct score spaces; thresholds are calibrated independently within each space.
3. **Two-Endpoint Identity Disjointness**: Calibration candidate pairs crossing fit and safety roles are excluded to ensure zero identity leakage between model fitting and safety offset calibration.
4. **No Mathematical Guarantees**: Target met rates reflect held-out test fold realization. They do not constitute mathematical FMR or fairness guarantees.
