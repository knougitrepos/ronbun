# Certified Compression-Bound Runner

This document describes the `thesis3` experiment runner path for the
compression-error-bound open-set face search method. Generated files are kept
inside a dated, immutable run directory:

```text
runs/YYYY/MM/DD/YYYYMMDD-RNNN-<config-hash>_<experiment-name>/
```

## Dry run

```powershell
py experiments/run_face_search_study.py --config configs/experiments/lfw_face_search.yaml --phase all --dry-run
```

이 문서의 generic runner 예시는 LFW 흐름을 기준으로 합니다. SurvFace 공식
평가는 `notebooks/survface/03_open_set/`와
`configs/experiments/survface_face_search.yaml`을 사용하며, 공식 test에서
압축기나 calibration 모델을 학습하지 않습니다.

The expected phase order is:

```text
protocol,templates,compression,search,certification,calibration
```

## Search inputs

To generate `artifacts/search/certified_features.csv` inside the dated run,
set these keys in the config:

```yaml
search:
  probes_path: data/derived/face_search/probes.csv
  templates_path: data/derived/face_search/templates.csv
  compression_profile: pca_256
  top_k: 2
  candidate_scope: exhaustive
  # gallery_size must be a positive integer. It is optional for exhaustive
  # inputs and must equal the number of supplied templates if set. Set it
  # explicitly when candidate_scope is candidate_set; the runner rejects
  # candidate_set without it.
  # gallery_size: 100000
```

`candidate_scope` controls what the certificate is allowed to claim:

- `exhaustive`: `templates_path` is the complete gallery for the evaluated
  protocol. Certified rank and threshold decisions can be interpreted globally
  for that gallery. If `gallery_size` is set in this mode, it must equal the
  number of supplied templates.
- `candidate_set`: `templates_path` is only a candidate subset, for example
  rows returned by pgvector/HNSW approximate search. Certified rank and
  threshold decisions are valid only within that supplied candidate set, not
  over the full gallery. `gallery_size` is required in this mode so the full
  gallery size is recorded explicitly.

Probe CSV columns:

- `image_id`
- `identity_id`
- `probe_type`: `registered`, `known_unknown`, or `unknown_unknown`
- `embedding`: JSON array string, for example `"[1.0, 0.0]"`
- optional `fallback_embedding`: full-precision JSON array string used only
  when the bound decision is `defer`. If this is omitted, exact fallback uses
  the probe `embedding` only when the probe `angular_error` is zero.
- optional: `quality`, `reconstruction_error_norm`, `angular_error`

Template CSV columns:

- `identity_id`
- `embedding`: JSON array string
- optional `fallback_embedding`: full-precision JSON array string used only
  when the bound decision is `defer`; it is required before any
  `exact_fallback` resolution can be reported.
- `quality`
- `variance`
- `enrollment_count`
- `angular_error`: angular compression error in radians

As a temporary bridge, the runner also accepts a precomputed feature table:

```yaml
search:
  input_certified_features_path: data/derived/face_search/certified_features.csv
```

When this is set, the file is copied to the run artifact directory as
`artifacts/search/certified_features.csv`. If the table already contains
`certification_candidate_scope`, `certification_candidate_count`,
`certification_gallery_size`, and `certification_global_claim`, those fields are
also summarized in `artifacts/search/phase_metadata.json`. If
`certification_candidate_scope` contains `candidate_set`, the precomputed table
must include `certification_candidate_count`, `certification_gallery_size`, and
`certification_global_claim`; otherwise the runner rejects the table. The only
valid scope values are `exhaustive` and `candidate_set`. For `candidate_set`
rows, `certification_global_claim` must be a boolean value (`true`, `false`,
`1`, `0`, `yes`, or `no`) and must be false. Also,
`certification_candidate_count` and `certification_gallery_size` must be
positive integers, with `certification_gallery_size` greater than or equal to
`certification_candidate_count`. For `exhaustive` rows that include these
metadata fields, `certification_gallery_size` must equal
`certification_candidate_count` and `certification_global_claim` must be true.

## Certification outputs

The certification phase reads `artifacts/search/certified_features.csv` from
the same run by default when `certification.input_features_path` is not set.

If `certification.input_features_path` points directly to a precomputed feature
table, the certification phase preserves the same candidate-scope metadata in
`artifacts/certification/phase_metadata.json` and applies the same scope-value,
required-column, boolean, gallery-size, and global-claim consistency checks
described above.

It writes:

- `artifacts/certification/certification_config.json`
- `artifacts/certification/certification_method.json`
- `artifacts/certification/certification_summary.json`
- `artifacts/certification/phase_metadata.json`

`certification_method.json` is the method card for the bound-based decision
layer. It records the cosine score type, angular-error unit, unit-vector
assumptions, lower/upper bound formulas, certified accept/reject/defer rules,
fallback rule, and candidate-set caveat. Use this file when writing the thesis
method section or checking that an experiment used the intended mathematical
decision rule.

Phase metadata records reproducibility fields for feature artifacts:

- `certified_features_rows`
- `certified_features_sha256`
- `input_features_rows`
- `input_features_sha256`
- `certification_candidate_scope`
- `certification_candidate_scope_counts`
- `certification_candidate_count`
- `certification_gallery_size`
- `certification_global_claim`

Key result columns:

- `certified_decision`
- `certified_identity`: populated only for certified `accept`
- `certified_fallback_required`
- `certified_rank`
- `certified_query_angular_error`
- `certified_top1_template_angular_error`
- `certified_top1_total_angular_error`
- `certified_top1_approximate_angle`
- `certified_top1_lower_bound`
- `certified_top1_upper_bound`
- `certified_top1_bound_width`
- `certified_max_upper_bound`
- `certified_max_other_upper_bound`
- `certified_top1_threshold_margin`
- `certified_rank_margin`
- `certified_reject_margin`
- `certified_decision_margin`
- `certification_candidate_scope`
- `certification_candidate_count`
- `certification_gallery_size`
- `certification_global_claim`
- `fallback_used`
- `fallback_query_source`: `fallback_embedding`, `embedding`, or empty
- `fallback_template_source`: `fallback_embedding` or empty
- `fallback_decision`
- `fallback_identity`
- `fallback_top1_score`
- `final_decision`
- `final_identity`
- `final_decision_source`: `certified_bound`, `exact_fallback`, or
  `defer_unresolved`

Key summary fields:

- `decision_counts`
- `certification_coverage`
- `defer_rate`
- `fallback_rate`
- `final_decision_counts`
- `exact_fallback_rate`
- `fallback_resolution_rate`
- `mean_top1_bound_width`
- `max_top1_bound_width`
- `mean_certified_decision_margin`
- `mean_query_angular_error`
- `mean_top1_template_angular_error`
- `mean_top1_total_angular_error`
- `candidate_scope_counts`
- `by_probe_type`

Interpretation:

- `certified_decision` reports what can be proven from compressed vectors and
  angular-error bounds.
- `final_decision` reports the operational decision after exact fallback, when
  fallback embeddings are available.
- `exact_fallback_rate` is the fraction of all rows resolved by exact fallback.
- `fallback_resolution_rate` is the fraction of fallback-required rows resolved
  by exact fallback; it is not normalized by all rows.
- `fallback_query_source=embedding` means the original query embedding was
  treated as full precision because its row-level `angular_error` was zero.
- A row with `certified_decision=defer` and
  `final_decision_source=exact_fallback` is not a certified compressed-vector
  decision; it is an exact fallback resolution.
- A row with `certification_candidate_scope=candidate_set` is not a global
  gallery certificate. It only proves the decision against the supplied
  candidate vectors. Approximate pgvector/HNSW candidate recall must be reported
  separately if this mode is used.
