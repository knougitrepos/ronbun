"""Small, immutable chat exports of origin/PQ calibration campaign tables.

The checkpoint remains the authoritative seed-level evidence. These exports keep
every observed condition while describing variation across shared-test seeds;
neither seed counts nor ranges of CI endpoints are independent replications.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd

from research.runtime.hashing import canonical_sha256, sha256_file


CONTEXT = ["dataset_id", "model", "source_run_id"]
METHOD_KEYS = [*CONTEXT, "compression_profile", "method", "target_fpir"]
MAX_PAGE_ROWS = 125
PAGE_BYTES_GUIDANCE = 64 * 1024
SCHEMA_VERSION = 1


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported provenance type: {type(value).__name__}")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False, default=_json_default) + "\n").encode("utf8")


def _csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _bool_values(values, name):
    normalized = values.astype(str).str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"invalid boolean column: {name}")
    return normalized.isin(["true", "1"])


def _coverage(expected_jobs, completed_jobs):
    def describe(value):
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if value < 0:
                raise ValueError("job counts must be non-negative")
            return int(value), None
        if isinstance(value, (str, bytes, dict)):
            raise ValueError("jobs must be counts or sequences of job keys")
        entries = list(value)
        hashes = [canonical_sha256(entry) for entry in entries]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate job keys")
        return len(entries), dict(zip(hashes, entries))

    expected, all_keys = describe(expected_jobs)
    completed, done_keys = describe(completed_jobs)
    if completed > expected:
        raise ValueError("completed job count exceeds expected jobs")
    missing = None
    if all_keys is not None and done_keys is not None:
        if set(done_keys) - set(all_keys):
            raise ValueError("unexpected completed jobs")
        missing = [all_keys[key] for key in sorted(set(all_keys) - set(done_keys))]
    return dict(expected_jobs=expected, completed_jobs=completed,
                missing_job_count=expected - completed,
                experiment_status="completed" if completed == expected else "partial",
                expected_job_keys=list(all_keys.values()) if all_keys is not None else None,
                completed_job_keys=list(done_keys.values()) if done_keys is not None else None,
                missing_job_keys=missing,
                individual_missing_jobs_known=missing is not None)


def _aggregate(frame, keys, *, numeric=(), constants=(), flags=()):
    """One observed scientific condition per row; never sum shared-test counts."""
    if frame.empty:
        return pd.DataFrame()
    required = [*keys, "partition_seed"]
    if set(required) - set(frame):
        raise ValueError(f"missing compact export keys: {sorted(set(required) - set(frame))}")
    if frame.duplicated(required).any():
        raise ValueError("duplicate seed/condition in compact export")
    rows = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        seeds = sorted(group.partition_seed.tolist())
        row = dict(zip(keys, values))
        row.update(seed_count=len(seeds), partition_seeds=";".join(str(s) for s in seeds))
        for column in constants:
            if column not in group:
                continue
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"shared-test invariant differs across seeds: {column}")
            row[column] = group[column].iloc[0]
        for column in numeric:
            if column not in group:
                continue
            data = pd.to_numeric(group[column], errors="raise")
            if not np.isfinite(data).all():
                raise ValueError(f"non-finite compact metric: {column}")
            row.update({f"{column}_min": data.min(), f"{column}_median": data.median(),
                        f"{column}_max": data.max()})
        for column in flags:
            if column in group:
                count = int(_bool_values(group[column], column).sum())
                row[f"{column}_seed_count"] = count
                row[f"{column}_seed_fraction"] = count / len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_for_chat(tables):
    """Aggregate all observed conditions, retaining counts and separate diagnostics."""
    result = {}
    methods = tables.get("method_summary", pd.DataFrame())
    if not methods.empty:
        essentials = {"test_mated_count", "test_non_mated_count", "false_accept_count",
                      "true_identification_at_rank_k_count", "realized_fpir", "tpir_at_rank_k",
                      "rank_failure_count", "threshold_failure_count", "target_met_on_test"}
        if essentials - set(methods):
            raise ValueError(f"missing essential method metrics: {sorted(essentials - set(methods))}")
        # A count/rate export must not silently turn malformed records into summaries.
        mated = methods.test_mated_count.to_numpy(dtype=float)
        nonmated = methods.test_non_mated_count.to_numpy(dtype=float)
        counts = methods[["false_accept_count", "true_identification_at_rank_k_count",
                          "rank_failure_count", "threshold_failure_count"]].to_numpy(dtype=float)
        if (not np.isfinite(np.c_[mated, nonmated, counts]).all() or (mated <= 0).any()
                or (nonmated <= 0).any() or (counts < 0).any()
                or not np.equal(counts, np.floor(counts)).all()
                or not np.equal(counts[:, 1:].sum(axis=1), mated).all()
                or (counts[:, 0] > nonmated).any()
                or not np.allclose(methods.realized_fpir, counts[:, 0] / nonmated, atol=1e-15, rtol=0)
                or not np.allclose(methods.tpir_at_rank_k, counts[:, 1] / mated, atol=1e-15, rtol=0)):
            raise ValueError("invalid compact method counts/rates")
        if not np.array_equal(_bool_values(methods.target_met_on_test, "target_met_on_test"),
                              counts[:, 0] / nonmated <= methods.target_fpir):
            raise ValueError("invalid unrounded target attainment flag")
        constants = ("rank_k", "score_space", "test_probe_count", "test_mated_count", "test_non_mated_count")
        result["operating_performance"] = _aggregate(methods, METHOD_KEYS,
            numeric=("realized_fpir", "tpir_at_rank_k", "false_accept_count", "true_identification_at_rank_k_count"),
            constants=constants, flags=("target_met_on_test", "target_met_by_wilson_upper"))
        result["operating_failures"] = _aggregate(methods, METHOD_KEYS,
            numeric=("rank_failure_count", "threshold_failure_count", "rank_failure_rate",
                     "threshold_failure_rate", "rank_k_ceiling"), constants=constants)
        intervals = [c for c in methods if "95_low" in c or "95_high" in c]
        if intervals:
            result["operating_ci_endpoints"] = _aggregate(methods, METHOD_KEYS, numeric=intervals)

    paired = tables.get("paired_comparisons", pd.DataFrame())
    if not paired.empty:
        result["paired_effects"] = _aggregate(paired,
            [*CONTEXT, "target_fpir", "comparison", "reference_profile", "reference_method",
             "candidate_profile", "candidate_method", "metric"],
            numeric=("reference_successes", "candidate_successes", "both_successes", "candidate_minus_reference",
                     "reference_realized_fpir", "candidate_realized_fpir", "paired_bootstrap95_low", "paired_bootstrap95_high"),
            constants=("total", "resamples", "resampling_unit", "comparison_basis"), flags=("both_target_met",))
    interactions = tables.get("interactions", pd.DataFrame())
    if not interactions.empty:
        result["operating_interactions"] = _aggregate(interactions, METHOD_KEYS,
            numeric=("pq_gain", "origin_gain", "interaction", "pq_method_fpir", "pq_global_fpir",
                     "origin_method_fpir", "origin_global_fpir", "paired_bootstrap95_low", "paired_bootstrap95_high"),
            constants=("comparison_basis",), flags=("all_four_target_met",))
    diagnostic_keys = [*CONTEXT, "compression_profile", "method", "fitted_target_fpir", "diagnostic_fpir"]
    for source, metrics in (("diagnostic_curves", ("diagnostic_tpir",)),
                            ("diagnostic_interactions", ("pq_gain", "origin_gain", "interaction"))):
        frame = tables.get(source, pd.DataFrame())
        if not frame.empty:
            result[source] = _aggregate(frame, diagnostic_keys, numeric=metrics,
                constants=("diagnostic_only", "interpolated_test_curve", "deployment_threshold_selected", "diagnostic_ci_available"))
    partitions = tables.get("partition_inventory", pd.DataFrame())
    if not partitions.empty:
        result["partition_inventory"] = _aggregate(partitions,
            [*CONTEXT, "compression_profile", "partition"], numeric=("query_count", "non_mated_count", "identity_count"))
    return result


def _safe_component(value):
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip(".")
    if not safe or safe != raw:
        safe = (safe or "condition") + "-" + canonical_sha256(raw)[:8]
    return safe


def _pages(frame, max_rows, byte_guidance):
    """Page by logical rows; an unusually wide single row is kept intact."""
    start = 0
    while start < len(frame):
        stop = min(start + max_rows, len(frame))
        data = _csv_bytes(frame.iloc[start:stop])
        while len(data) > byte_guidance and stop - start > 1:
            stop = start + max(1, (stop - start) // 2)
            data = _csv_bytes(frame.iloc[start:stop])
        yield start, stop, data
        start = stop


def validate_chat_bundle(directory, expected_spec=None):
    """Verify a published bundle and every archived entry before reuse."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    if (manifest.get("status") != "completed" or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("artifact_type") != "origin_pq_chat_bundle"
            or (expected_spec is not None and manifest.get("spec") != expected_spec)):
        raise ValueError("chat bundle manifest/spec mismatch")
    if manifest.get("result_uid") != "origin-pq-chat-" + canonical_sha256(manifest["spec"])[:24]:
        raise ValueError("chat bundle content identity mismatch")
    if set(manifest.get("files", {})) != {"START_HERE.md", "analysis.zip"}:
        raise ValueError("unexpected chat bundle file inventory")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"chat bundle file hash mismatch: {name}")
    with zipfile.ZipFile(directory / "analysis.zip") as archive:
        expected = manifest["archive_entries"]
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError("chat archive entry inventory mismatch")
        for name, entry in expected.items():
            data = archive.read(name)
            if len(data) != entry["bytes"] or _digest(data) != entry["sha256"]:
                raise ValueError(f"chat archive entry hash mismatch: {name}")
    return manifest


def write_chat_bundle(root, tables, provenance, expected_jobs, completed_jobs, *,
                      max_page_rows=MAX_PAGE_ROWS, page_bytes_guidance=PAGE_BYTES_GUIDANCE):
    """Publish exactly three physical files; ZIP contains all condition pages.

    Job arguments are non-negative counts or sequences of unique JSON-compatible
    keys. Explicit sequences let the archive identify individual missing jobs.
    Full seed-level models/results stay in the campaign checkpoint, not chat CSVs.
    """
    if (isinstance(max_page_rows, bool) or not isinstance(max_page_rows, int) or max_page_rows < 1
            or isinstance(page_bytes_guidance, bool) or not isinstance(page_bytes_guidance, int)
            or page_bytes_guidance < 1):
        raise ValueError("positive integer page sizes required")
    coverage = json.loads(_json_bytes(_coverage(expected_jobs, completed_jobs)))
    normalized_provenance = json.loads(_json_bytes(provenance))
    summarized = summarize_for_chat(tables)
    method_rows = tables.get("method_summary", pd.DataFrame())
    observed_jobs = (len(method_rows[[*CONTEXT, "partition_seed"]].drop_duplicates())
                     if not method_rows.empty else 0)
    if observed_jobs != coverage["completed_jobs"]:
        raise ValueError("completed job coverage differs from method table")
    source_tables = {name: dict(rows=len(frame), sha256=_digest(_csv_bytes(frame)))
                     for name, frame in sorted(tables.items())}
    spec = dict(schema_version=SCHEMA_VERSION, provenance_sha256=canonical_sha256(normalized_provenance),
                source_tables=source_tables, coverage=coverage,
                implementation_sha256=sha256_file(__file__),
                page_settings=dict(max_rows=max_page_rows, bytes_guidance=page_bytes_guidance))
    uid = "origin-pq-chat-" + canonical_sha256(spec)[:24]
    destination = Path(root) / uid
    if destination.exists():
        validate_chat_bundle(destination, spec)
        return destination
    entries = {"provenance.json": _json_bytes(normalized_provenance), "coverage.json": _json_bytes(coverage)}
    page_index = []
    for name, frame in sorted(summarized.items()):
        for (dataset, model), group in frame.groupby(["dataset_id", "model"], sort=True):
            for number, (start, stop, payload) in enumerate(_pages(group, max_page_rows, page_bytes_guidance), 1):
                path = f"{_safe_component(dataset)}/{_safe_component(model)}/{name}_{number:03d}.csv"
                if path in entries:
                    raise ValueError("archive path collision")
                entries[path] = payload
                page_index.append(dict(entry=path, table=name, dataset_id=dataset, model=model,
                                       rows=stop-start, bytes=len(payload),
                                       exceeds_byte_guidance=len(payload) > page_bytes_guidance))
    entries["INDEX.csv"] = _csv_bytes(pd.DataFrame(page_index, columns=[
        "entry", "table", "dataset_id", "model", "rows", "bytes", "exceeds_byte_guidance"]))
    readme = f"""# 원본/PQ FIQA 분석 자료

실험 상태: **{coverage['experiment_status']}** — 예정 {coverage['expected_jobs']}개 중 {coverage['completed_jobs']}개 job 완료.
누락 job 수: {coverage['missing_job_count']}. 완료된 조건만 집계했으며 미완료 결과를 0으로 채우지 않았습니다.

ChatGPT에 `START_HERE.md`와 `analysis.zip`을 함께 첨부하십시오. ZIP 처리가 지원되지 않으면
압축을 풀고 `INDEX.csv`에서 원하는 데이터셋/FR 모델의 CSV를 골라 첨부하십시오.
각 CSV는 기본 {max_page_rows}행 이하, 약 {page_bytes_guidance // 1024} KiB 기준으로 나눕니다.
하나의 긴 행은 잘라 버리지 않습니다. 모든 관측된 target/method/profile 조건과 모든 페이지를 보존합니다.

- `operating_performance`: FPIR/TPIR, 오류·성공 수와 분모, 목표 충족 seed 수/비율.
- `operating_failures`: Top-K 밖 순위 실패와 Top-K 안 threshold 실패, rank 상한.
- `operating_ci_endpoints`: 개별 seed CI의 하한/상한이 seed에 따라 어떻게 달라지는지.
- `paired_effects`: 같은 cohort의 candidate-minus-reference 비교와 각 실제 FPIR.
- `operating_interactions`: (PQ FIQA 이득) − (원본 FIQA 이득), 네 조건의 실제 FPIR.
- `diagnostic_curves`, `diagnostic_interactions`: 고정 보정 모델의 test 곡선 보간 진단.
- `partition_inventory`: calibration fit/safety 인원·query 수의 seed별 범위.
- `coverage.json`: 전체 job 완료 범위. `provenance.json`: 입력·설정·원본 checkpoint 출처.

`min/median/max`는 같은 test cohort를 공유하는 calibration seed들의 기술 통계입니다.
seed별 분모·오류 수를 합치지 마십시오. 개별 seed CI endpoint의 범위는 통합 CI가 아닙니다.
목표 충족은 반올림 전 FPIR로 판정합니다. 같은 목표 FPIR은 같은 실제 FPIR을 뜻하지 않습니다.
TPIR@K는 genuine identity의 Top-K 포함과 genuine score의 threshold 통과를 모두 요구합니다.
진단 곡선은 test 보간이며 deployment threshold를 선택하지 않고 CI·수학적 FPIR 보장을 제공하지 않습니다.
원본 cosine과 PQ ADC는 서로 다른 score space입니다. checkpoint 학습 데이터 overlap은 별도 확인이 필요합니다.

이 묶음은 분석용 요약이며 seed별 모델 JSON·전체 seed 결과를 중복 저장하지 않습니다.
그 원자료는 provenance에 연결된 checkpoint/보고서에 남고 source table SHA-256으로 연결됩니다.
해석 요청 예: “모든 target/method/profile을 확인하고, 목표 FPIR 충족과 실제 FPIR을 함께 비교하며,
원본 대비 PQ의 추가 FIQA 이득 및 순위/threshold 실패를 구분해 분석해 주세요.”
"""
    entries["README.md"] = readme.encode("utf8")
    staging = Path(root) / (".staging-chat-" + uuid4().hex)
    staging.mkdir(parents=True)
    (staging / "START_HERE.md").write_text(readme, encoding="utf8")
    with zipfile.ZipFile(staging / "analysis.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    manifest = dict(artifact_type="origin_pq_chat_bundle", schema_version=SCHEMA_VERSION,
                    status="completed", result_uid=uid, spec=spec,
                    experiment_status=coverage["experiment_status"],
                    formal_fpir_guarantee=False, same_test_seeds_are_independent=False,
                    ci_endpoint_ranges_are_merged_intervals=False,
                    summarized_table_rows={name: len(frame) for name, frame in summarized.items()},
                    files={name: sha256_file(staging / name) for name in ("START_HERE.md", "analysis.zip")},
                    archive_entries={name: dict(bytes=len(payload), sha256=_digest(payload))
                                     for name, payload in sorted(entries.items())})
    (staging / "manifest.json").write_bytes(_json_bytes(manifest))
    validate_chat_bundle(staging, spec)
    # Same-volume rename publishes only a fully validated bundle. Retain staging
    # after a Windows lock error so no completed source needs to be overwritten.
    for attempt in range(6):
        if destination.exists():
            validate_chat_bundle(destination, spec)
            return destination
        try:
            staging.rename(destination)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)
    return destination
