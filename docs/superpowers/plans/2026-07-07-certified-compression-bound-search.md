# Certified Compression-Bound Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` where available, or
> `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Add a mathematically explicit angular-error-bound decision layer for
compressed ArcFace-style open-set face search.

**Architecture:** Keep the existing calibration pipeline intact. Add angular
distortion metadata to compression results and implement certified `accept` /
`reject` / `defer` decisions in a reusable search module. The certificate uses
unit-vector angular bounds, not raw reconstruction MSE. Deferred decisions can
optionally be resolved by exact full-precision fallback embeddings.

**Tech Stack:** Python, NumPy, pandas, pytest, existing `research/compression`,
`research/search`, and `experiments` modules.

---

### Task 1: Store angular compression error beside reconstruction error

**Files:**

- `research/compression/profiles.py`
- `tests/research/test_compression.py`

- [x] Add `CompressionResult.angular_error`.
- [x] Compute row-wise angular error between original and reconstructed unit
  vectors.
- [x] Return zero angular error for `origin_512`.
- [x] Verify with focused compression tests.

### Task 2: Add certified open-set decisions from angular bounds

**Files:**

- `research/search/certification.py`
- `tests/research/test_certification.py`

- [x] Add similarity-bound dataclasses.
- [x] Implement spherical triangle-inequality cosine lower/upper bounds.
- [x] Implement certified `accept`, `reject`, and `defer` rules.
- [x] Verify bound containment, accept, reject, defer, and summary behavior.

### Task 3: Connect certified decisions to search feature outputs

**Files:**

- `research/search/open_set.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_face_search_pipeline.py`

- [x] Preserve existing baseline search/calibration feature columns.
- [x] Add certified decision, identity, fallback-required, rank, and bound
  columns.
- [x] Keep `registered`, `known_unknown`, and `unknown_unknown` probe types
  available for split reporting.
- [x] Verify with research and synthetic pipeline tests.

### Task 4: Add certification phase to the experiment runner

**Files:**

- `experiments/run_face_search_study.py`
- `experiments/configs/face_search.yaml`
- `tests/integration/test_cli_dry_run.py`

- [x] Add `certification` to phase expansion.
- [x] Require and validate `certification.threshold`.
- [x] Print certification threshold during dry-run.
- [x] Write certification config artifacts in non-dry-run mode.

### Task 5: Summarize certified feature frames for result tables

**Files:**

- `research/search/open_set.py`
- `tests/research/test_open_set.py`

- [x] Summarize total rows, certified decision counts, coverage, defer rate,
  fallback rate, and per-probe-type metrics.
- [x] Reject empty frames and frames missing certified columns.

### Task 6: Write certification summaries from feature CSVs

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`

- [x] Read `certification.input_features_path`.
- [x] Write `certification/certification_summary.json`.
- [x] Record feature input path in phase metadata.

### Task 7: Hand search certified features to certification by default

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`

- [x] Copy `search.input_certified_features_path` to
  `search/certified_features.csv`.
- [x] Use `search/certified_features.csv` as the certification default when no
  explicit certification feature path is configured.

### Task 8: Generate certified search features from probe/template CSV inputs

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`

- [x] Parse JSON-array `embedding` values from probe and template CSVs.
- [x] Generate `search/certified_features.csv`.
- [x] Summarize generated features in the certification phase.

### Task 9: Document runner input/output contract

**Files:**

- `experiments/configs/face_search.yaml`
- `docs/certified_compression_bound_runner.md`

- [x] Document dry-run command.
- [x] Document probe/template CSV schema.
- [x] Document certified feature columns and summary fields.
- [x] Keep default local file paths commented.

### Task 10: Record certified feature artifact hashes and row counts

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add CSV row-count and SHA-256 helpers.
- [x] Record generated/copied `certified_features.csv` row counts and hashes.
- [x] Record consumed certification feature row counts and hashes.
- [x] Document metadata fields.

### Task 11: Resolve deferred certified decisions with exact fallback embeddings

**Files:**

- `research/search/certification.py`
- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `tests/research/test_certification.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add exact full-precision open-set decision helper.
- [x] When certified decision is `defer` and `fallback_embedding` columns are
  present, compute exact fallback `accept`/`reject`.
- [x] Add `fallback_*`, `final_decision`, `final_identity`, and
  `final_decision_source` columns.
- [x] Parse optional `fallback_embedding` JSON arrays in runner CSV inputs.
- [x] Summarize final decision counts and fallback resolution rate, including
  per-probe-type summaries.
- [x] Document the distinction between certified compressed decisions and exact
  fallback resolutions.

### Task 12: Mark candidate-set certificates separately from global gallery certificates

**Files:**

- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `experiments/configs/face_search.yaml`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add `candidate_scope` and `gallery_size` inputs to certified search
  feature generation.
- [x] Add `certification_candidate_scope`, `certification_candidate_count`,
  `certification_gallery_size`, and `certification_global_claim` columns.
- [x] Record candidate-scope metadata in generated search phase artifacts.
- [x] Include candidate-scope counts in certification summaries.
- [x] Document that pgvector/HNSW candidate-subset certificates are not global
  gallery certificates unless candidate recall is handled separately.

### Task 13: Preserve candidate-scope metadata for precomputed certified feature CSVs

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] When `search.input_certified_features_path` is copied, summarize existing
  candidate-scope columns in `search/phase_metadata.json`.
- [x] When `certification.input_features_path` is consumed directly, summarize
  existing candidate-scope columns in `certification/phase_metadata.json`.
- [x] Record `certification_candidate_scope_counts` in phase metadata when the
  feature table contains scope labels.
- [x] Document the precomputed feature-table scope metadata contract.

### Task 14: Write a certification method artifact for thesis-method traceability

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Write `certification/certification_method.json` for every certification
  phase.
- [x] Record method name, cosine score type, angular-error unit, assumptions,
  bound formulas, decision rules, fallback rule, and candidate-set caveat.
- [x] Include the method artifact in `certification/phase_metadata.json`
  outputs.
- [x] Document how to use the method artifact for the thesis method section and
  experiment traceability.

### Task 15: Require gallery size for candidate-set certification runs

**Files:**

- `research/search/open_set.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject `candidate_scope="candidate_set"` when `gallery_size` is missing.
- [x] Verify the failure at both reusable search-feature and CLI runner levels.
- [x] Document that `gallery_size` is mandatory for candidate-set inputs so the
  non-global certificate scope is explicit in artifacts.

### Task 16: Reject incomplete precomputed candidate-set feature tables

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject precomputed feature CSVs whose `certification_candidate_scope`
  contains `candidate_set` without `certification_candidate_count`,
  `certification_gallery_size`, and `certification_global_claim`.
- [x] Verify this for both `certification.input_features_path` and
  `search.input_certified_features_path` handoff paths.
- [x] Document the precomputed feature-table required-column contract.

### Task 17: Validate precomputed candidate-scope consistency

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject unknown `certification_candidate_scope` values in precomputed
  feature tables.
- [x] Reject `candidate_set` rows that incorrectly set
  `certification_global_claim=true`.
- [x] Reject `candidate_set` rows whose `certification_gallery_size` is smaller
  than `certification_candidate_count`.
- [x] Verify the three validation failures with CLI integration tests.
- [x] Document the scope-value, non-global-claim, and gallery-size consistency
  contract.

### Task 18: Expose certificate interval-width and margin diagnostics

**Files:**

- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add row-level top-1 bound width, top-1 threshold margin, rank margin,
  reject margin, and certified decision margin columns.
- [x] Add summary fields for mean/max top-1 bound width and mean certified
  decision margin.
- [x] Include the new bound/margin fields in `certification_config.json`.
- [x] Verify row-level and summary behavior with research tests.
- [x] Verify artifact metadata exposure with a CLI integration test.
- [x] Document the new result and summary fields.

### Task 19: Validate candidate-set scope-size values in precomputed tables

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject `candidate_set` rows whose `certification_candidate_count` or
  `certification_gallery_size` values are not positive integers.
- [x] Verify zero and fractional candidate-count failures with CLI integration
  tests.
- [x] Document that precomputed candidate-set scope sizes must be positive
  integers before the gallery-size comparison is meaningful.

### Task 20: Record angular-error traceability columns for certificates

**Files:**

- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add row-level query angular error, top-1 template angular error, top-1
  total angular error, and top-1 approximate angle columns to certified search
  features.
- [x] Add summary means for query, top-1 template, and total angular error.
- [x] Expose the traceability columns in `certification_config.json` as
  `angular_error_columns`.
- [x] Verify row-level traceability and summary behavior with research tests.
- [x] Verify artifact metadata exposure with a CLI integration test.
- [x] Document the new result and summary fields.

### Task 21: Guard exact fallback claims with full-precision query evidence

**Files:**

- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add `fallback_query_source` and `fallback_template_source` output columns.
- [x] Resolve `defer` rows with `exact_fallback` only when template
  `fallback_embedding` is available and the query is either supplied as
  `fallback_embedding` or has zero `query_angular_error`.
- [x] Keep rows as `defer_unresolved` when the query is compressed and no
  full-precision query fallback is available.
- [x] Expose fallback source columns in `certification_config.json`.
- [x] Update the method-card fallback rule and runner documentation.
- [x] Verify the exact-fallback and missing-query-fallback cases with research
  tests, plus metadata exposure with a CLI integration test.

### Task 22: Separate exact fallback rate from defer resolution rate

**Files:**

- `research/search/open_set.py`
- `tests/research/test_open_set.py`
- `docs/certified_compression_bound_runner.md`

- [x] Add `exact_fallback_rate` as the fraction of all rows resolved by exact
  fallback.
- [x] Redefine `fallback_resolution_rate` as the fraction of
  fallback-required rows resolved by exact fallback.
- [x] Return `fallback_resolution_rate=None` when no rows required fallback.
- [x] Verify the distinction with a mixed resolved/unresolved defer summary
  test.
- [x] Document the different denominators for both fields.

### Task 23: Strictly parse precomputed global-claim booleans

**Files:**

- `experiments/run_face_search_study.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject invalid `certification_global_claim` literals in precomputed
  feature tables instead of treating arbitrary non-empty strings as truthy.
- [x] Preserve accepted boolean forms for normal CSV handoff paths.
- [x] Verify an invalid `maybe` value is rejected with a boolean-specific error.
- [x] Document the accepted boolean literals for `certification_global_claim`.

### Task 24: Enforce exhaustive-scope gallery consistency

**Files:**

- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject generated certified features when `candidate_scope="exhaustive"`
  but `gallery_size` differs from the supplied template count.
- [x] Reject precomputed `exhaustive` rows whose
  `certification_gallery_size` differs from
  `certification_candidate_count`.
- [x] Require precomputed `exhaustive` rows with complete scope metadata to set
  `certification_global_claim=true`.
- [x] Verify reusable search and CLI precomputed failure paths.
- [x] Document that `exhaustive` means the supplied templates are the full
  gallery.

### Task 25: Validate generated gallery-size values before certificate claims

**Files:**

- `research/search/open_set.py`
- `experiments/run_face_search_study.py`
- `tests/research/test_open_set.py`
- `tests/integration/test_cli_dry_run.py`
- `docs/certified_compression_bound_runner.md`

- [x] Reject fractional, non-finite, non-positive, or boolean `gallery_size`
  values in generated certified feature paths.
- [x] Stop the experiment runner from truncating configured `gallery_size`
  values before reusable search validation.
- [x] Verify both the reusable search API and CLI config path reject fractional
  gallery sizes.
- [x] Document that generated `gallery_size` must be a positive integer.

### Verification commands used during implementation

```powershell
py -m pytest tests/research/test_certification.py::test_exact_open_set_decision_accepts_or_rejects_with_full_precision_scores tests/research/test_open_set.py::test_build_certified_search_features_adds_bound_decisions_without_replacing_baseline_scores tests/research/test_open_set.py::test_build_certified_search_features_resolves_defer_with_exact_fallback_embeddings tests/research/test_open_set.py::test_summarizes_certified_search_feature_frame_for_result_tables -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_generates_final_decisions_with_fallback_embeddings -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_writes_certification_phase_artifacts -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_marks_candidate_set_scope_without_global_claim tests/research/test_open_set.py::test_summarizes_certified_search_feature_frame_for_result_tables -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_generates_certified_features_from_probe_and_template_csv -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_writes_certification_summary_from_feature_csv -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_hands_search_certified_features_to_certification_by_default -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_writes_certification_phase_artifacts -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_requires_gallery_size_for_candidate_set_scope -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_requires_gallery_size_for_candidate_set_generation -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_candidate_set_without_gallery_size -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_search_handoff_candidate_set_without_gallery_size -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_unknown_candidate_scope tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_candidate_set_global_claim tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_gallery_smaller_than_candidate_count -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_adds_bound_decisions_without_replacing_baseline_scores tests/research/test_open_set.py::test_summarizes_certified_search_feature_frame_for_result_tables -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_writes_certification_phase_artifacts -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_non_positive_or_fractional_candidate_counts -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_adds_bound_decisions_without_replacing_baseline_scores tests/research/test_open_set.py::test_summarizes_certified_search_feature_frame_for_result_tables -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_writes_certification_phase_artifacts -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_resolves_defer_with_exact_fallback_embeddings tests/research/test_open_set.py::test_build_certified_search_features_does_not_call_fallback_exact_without_full_precision_query -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_writes_certification_phase_artifacts -v
py -m pytest tests/research/test_open_set.py::test_summarizes_certified_search_feature_frame_for_result_tables -v
py -m pytest tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_invalid_global_claim_value tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_candidate_set_global_claim tests/integration/test_cli_dry_run.py::test_face_search_cli_hands_search_certified_features_to_certification_by_default -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_rejects_exhaustive_scope_when_gallery_size_differs tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_precomputed_exhaustive_scope_with_partial_gallery -v
py -m pytest tests/research/test_open_set.py::test_build_certified_search_features_rejects_fractional_gallery_size tests/integration/test_cli_dry_run.py::test_face_search_cli_rejects_fractional_gallery_size_for_candidate_set_generation -v
```
