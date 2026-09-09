"""Read-only integration of common TPIR CI and FIQA split evidence.

Never fit thresholds, replay search, choose a best seed, or silently select latest.
Notebook/Markdown tables are deliberately separate comparison families.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from research.calibration.conditional import _boolean_values
from research.experiments.fiqa_split_stability import summarize_split_stability
from research.runtime.hashing import canonical_sha256, sha256_file

PINNED_RUN_ID = "20260902-R001-61915edf"
PINNED_MODEL_UID = "arcface-7972a704552df378345f"


def pinned_report_sources(root):
    root = Path(root)
    return {
        "common_ci": (root / "results/evaluation/common_ci" / PINNED_RUN_ID /
                      "common-ci-889aededfcf9b089a2c25c88"),
        "split_stability": (root / "results/calibration/fiqa_split_stability" / PINNED_RUN_ID /
                            "fiqa-split-f473f8c1a2384db105604e61"),
        "condition": (root / "results/calibration/condition_scores" / PINNED_RUN_ID /
                      "pq_512_m128_b8__pq_adc_exhaustive/genuine-score-topk-v2"),
        "expected_hashes": {
            "common_ci": "3b1895810a2d273fa774331dc7223bea20e373e0717f6841c0ac9a2a6bfb0e10",
            "split_stability": "c094ec2ef00311ad298bfadf2d098fc429c8dfd0383857ab4368faa8025c10a6",
            "condition": "db77ec98b3fabb07f54a0b34d28ffb92dae8d57a84e79ce2a7f5c6a0198fb46a",
        },
    }


def _manifest(directory, expected_hash, artifact_type):
    path = Path(directory).resolve() / "manifest.json"
    if sha256_file(path) != expected_hash:
        raise ValueError(f"report source manifest hash mismatch: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("artifact_type") != artifact_type:
        raise ValueError("report requires completed compatible artifacts")
    return manifest


def _verified_file(directory, name, digest):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != digest:
        raise ValueError(f"report source file hash/path mismatch: {name}")
    return path


def _same(a, b, message):
    if a != b:
        raise ValueError(message)


def _rate_check(frame, numerator, denominator, rate):
    counts = frame[[numerator, denominator]].to_numpy(float)
    if (not np.isfinite(counts).all() or not np.equal(counts, np.floor(counts)).all()
            or (counts[:, 1] <= 0).any() or (counts[:, 0] < 0).any()
            or (counts[:, 0] > counts[:, 1]).any()):
        raise ValueError("invalid report counts")
    if not np.allclose(frame[rate], counts[:, 0] / counts[:, 1], rtol=1e-10, atol=1e-12):
        raise ValueError(f"report rate/count mismatch: {rate}")


def _interval_check(frame, low, high, *, difference=False):
    values = frame[[low, high]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, 0] > values[:, 1]).any()
            or (values < (-1 if difference else 0)).any() or (values > 1).any()):
        raise ValueError(f"invalid report interval: {low}")


def load_calibration_evidence_report(
    *, common_ci, split_stability, condition, expected_hashes,
    expected_run_id, expected_model_uid, reference_seed=8972,
):
    cm = _manifest(common_ci, expected_hashes["common_ci"], "common_tpir_identity_cluster_ci")
    sm = _manifest(split_stability, expected_hashes["split_stability"], "fiqa_split_stability")
    fm = _manifest(condition, expected_hashes["condition"], "compressed_calibration_test_score_tables")
    _same(cm.get("contract"), "query-weighted-mated-identity-percentile-v1", "unsupported common CI contract")
    _same(fm.get("metric_contract"), "genuine-score-topk-v2", "legacy genuine-score contract")
    _same(sm.get("condition_uid"), fm.get("condition_uid"), "condition UID mismatch")
    _same(fm.get("source_run_id"), expected_run_id, "selected run mismatch")
    _same(fm.get("model_uid"), expected_model_uid, "selected model mismatch")
    if isinstance(reference_seed, bool) or reference_seed not in sm["partition_seeds"]:
        raise ValueError("explicit reference seed is absent")
    for key in ("dataset_id", "model_uid", "extraction_uid", "origin_embedding_artifact_uid",
                "protocol_uid", "compression_profile", "search_mode"):
        _same(cm["condition"].get(key), fm.get(key), f"report lineage mismatch: {key}")
    _same(fm["dataset_id"], "survface", "FIQA report coverage is currently SurvFace only")
    for seed_manifest in sm["seed_manifests"]:
        _same(seed_manifest["condition_manifest_sha256"], canonical_sha256(fm), "seed condition hash mismatch")
        for field in ("metric_contract", "target_fpirs", "safety_fraction", "bin_count",
                      "shrinkage_strength", "minimum_group_non_mated", "resamples", "bootstrap_seed",
                      "fiqa_s_manifest_sha256", "fiqa_l_manifest_sha256"):
            _same(seed_manifest.get(field), sm["seed_manifests"][0].get(field), f"mixed seed contract: {field}")
    _same([m["partition_seed"] for m in sm["seed_manifests"]], sm["partition_seeds"], "seed manifest panel mismatch")
    if sm["uncertainty"].get("test_based_seed_selection") is not False:
        raise ValueError("test-selected seeds are not supported")
    # Source score tables are verified without loading/replaying embeddings.
    for name, entry in fm["files"].items():
        _verified_file(condition, name, entry["sha256"])
    ledger_path = Path(cm["source_manifest"])
    if sha256_file(ledger_path) != cm["source_manifest_sha256"]:
        raise ValueError("common CI source ledger changed")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    matches = [c for c in ledger["conditions"] if c["condition_id"] == cm["condition_id"]]
    if len(matches) != 1:
        raise ValueError("ambiguous source ledger condition")
    _same(matches[0]["core"]["sha256"], fm["persisted_test_core_sha256"], "different test score core")
    _verified_file(ledger_path.parent, matches[0]["core"]["path"], matches[0]["core"]["sha256"])
    common = pd.read_csv(_verified_file(common_ci, "retrieval_summary.csv", cm["summary_sha256"]))
    frames = {name.removesuffix(".csv"): pd.read_csv(_verified_file(split_stability, name, digest))
              for name, digest in sm["files"].items()}
    seeds = frames["seed_metrics"]
    if not seeds.rank_k.eq(20).all() or not seeds.score_space.eq(fm["score_space"]).all():
        raise ValueError("FIQA rank/score-space mismatch")
    if _boolean_values(seeds.threshold_fit_on_test, column="threshold_fit_on_test").any():
        raise ValueError("test-fitted thresholds cannot enter this report")
    for flag in ("target_met_on_test", "target_met_by_wilson_upper"):
        seeds[flag] = _boolean_values(seeds[flag], column=flag)
    targets = sm["target_fpirs"]
    _same(sorted(common.target_fpir.tolist()), targets, "common/FIQA target grid mismatch or duplicates")
    _same(set(seeds.method), {"global_empirical", "global_safe", "fiqa_s", "fiqa_l"}, "method coverage mismatch")
    for name, group in seeds.groupby("method"):
        _same(sorted(group.target_fpir.unique().tolist()), targets, f"missing targets: {name}")
    if not common.tpir_cluster_ci_status.eq("ok").all() or not common.tpir_rank.eq(20).all():
        raise ValueError("common TPIR20 identity CI unavailable")
    if not common.threshold_policy.eq("recalibrated_compressed").all():
        raise ValueError("report requires recalibrated compressed thresholds")
    _same(set(common.compressed_score_space), {fm["score_space"]}, "score space mismatch")
    for key in ("model_uid", "protocol_uid", "compression_profile", "search_mode"):
        if key in common and common[key].notna().any():
            _same(set(common[key].dropna()), {fm[key]}, f"common CSV lineage mismatch: {key}")
    for prefix in ("origin", "compressed"):
        _rate_check(common, f"{prefix}_tpir20_count", f"{prefix}_tpir20_denominator", f"{prefix}_tpir20")
        _rate_check(common, f"{prefix}_false_accept_count", f"{prefix}_fpir_denominator", f"{prefix}_fpir")
        _interval_check(common, f"{prefix}_tpir_at_rank_k_identity_cluster95_low", f"{prefix}_tpir_at_rank_k_identity_cluster95_high")
    _rate_check(seeds, "true_identification_at_rank_k_count", "test_mated_count", "tpir_at_rank_k")
    _rate_check(seeds, "false_accept_count", "test_non_mated_count", "realized_fpir")
    _interval_check(seeds, "tpir_cluster95_low", "tpir_cluster95_high")
    _interval_check(seeds, "fpir_wilson95_low", "fpir_wilson95_high")
    _interval_check(common, "compressed_minus_origin_tpir_at_rank_k_identity_cluster95_low",
                    "compressed_minus_origin_tpir_at_rank_k_identity_cluster95_high", difference=True)
    _interval_check(frames["seed_paired_comparisons"], "paired_bootstrap95_low", "paired_bootstrap95_high", difference=True)
    if not np.array_equal(seeds.target_met_on_test, seeds.realized_fpir <= seeds.target_fpir):
        raise ValueError("target attainment flags disagree with rates")
    if not np.array_equal(seeds.target_met_by_wilson_upper, seeds.fpir_wilson95_high <= seeds.target_fpir):
        raise ValueError("Wilson attainment flags disagree with bounds")
    if not np.allclose(common.compressed_minus_origin_tpir_at_rank_k,
                       common.compressed_tpir20 - common.origin_tpir20, rtol=1e-10, atol=1e-12):
        raise ValueError("compression paired delta disagrees with rates")
    all_pairs = frames["seed_paired_comparisons"]
    expected_pairs = {("global_empirical", "fiqa_s"), ("global_safe", "fiqa_s"),
                      ("global_empirical", "fiqa_l"), ("global_safe", "fiqa_l"), ("fiqa_s", "fiqa_l")}
    _same(set(zip(all_pairs.reference_method, all_pairs.candidate_method)), expected_pairs, "paired method coverage mismatch")
    _same(set(all_pairs.metric), {"fpir", "tpir_at_rank_k"}, "paired metric coverage mismatch")
    for metric, count, total in (("fpir", "false_accept_count", "test_non_mated_count"),
                                 ("tpir_at_rank_k", "true_identification_at_rank_k_count", "test_mated_count")):
        selected = all_pairs.loc[all_pairs.metric.eq(metric)]
        for role in ("reference", "candidate"):
            lookup = seeds[["method", "partition_seed", "target_fpir", count, total]].rename(
                columns={"method": f"{role}_method", count: "expected_count", total: "expected_total"})
            joined = selected.merge(lookup, on=[f"{role}_method", "partition_seed", "target_fpir"],
                                    validate="many_to_one", how="left")
            if (not np.array_equal(joined[f"{role}_successes"], joined.expected_count)
                    or not np.array_equal(joined.total, joined.expected_total)):
                raise ValueError("paired rows disagree with seed counts")
    if not np.allclose(all_pairs.candidate_minus_reference,
                       (all_pairs.candidate_successes - all_pairs.reference_successes) / all_pairs.total,
                       rtol=1e-10, atol=1e-12):
        raise ValueError("FIQA paired delta disagrees with counts")
    # Recompute descriptive summaries, never thresholds or bootstrap draws.
    recomputed = summarize_split_stability(seeds, frames["seed_paired_comparisons"], frames["seed_thresholds"], sm["partition_seeds"])
    for name, rebuilt in zip(("stability_summary", "paired_stability_summary", "threshold_stability_summary"), recomputed):
        pd.testing.assert_frame_equal(frames[name], rebuilt, check_dtype=False, rtol=1e-10, atol=1e-12)
    # The compressed baseline must reproduce the all-calibration global control.
    control = seeds.query("method == 'global_empirical'").merge(common, on="target_fpir", validate="many_to_one")
    for left, right in (("false_accept_count", "compressed_false_accept_count"),
                        ("true_identification_at_rank_k_count", "compressed_tpir20_count"),
                        ("test_mated_count", "compressed_tpir20_denominator"),
                        ("test_non_mated_count", "compressed_fpir_denominator"),
                        ("mated_identity_count", "tpir_cluster_ci_identity_count")):
        if not np.array_equal(control[left], control[right]):
            raise ValueError(f"baseline population/count mismatch: {left}")
    fixed = seeds.loc[seeds.partition_seed.eq(reference_seed)].copy()
    paired = frames["seed_paired_comparisons"].loc[lambda x: x.partition_seed.eq(reference_seed)].copy()
    paired["comparison_family"] = "fiqa_same_seed"
    records = []
    for row in common.to_dict("records"):
        for prefix, label in (("origin", "Origin cosine"), ("compressed", "PQ ADC global")):
            records.append({"comparison_family": "origin_vs_compressed", "method": label,
                "target_fpir": row["target_fpir"], "realized_fpir": row[f"{prefix}_fpir"],
                "tpir20": row[f"{prefix}_tpir20"],
                "tpir_cluster95_low": row[f"{prefix}_tpir_at_rank_k_identity_cluster95_low"],
                "tpir_cluster95_high": row[f"{prefix}_tpir_at_rank_k_identity_cluster95_high"],
                "target_met": row[f"{prefix}_fpir"] <= row["target_fpir"],
                "tpir_ci_unit": "mated_identity_cluster"})
    tables = {"compression_operating_points": pd.DataFrame(records),
              "compression_paired_ci": common[["target_fpir",
                  "compressed_minus_origin_tpir_at_rank_k",
                  "compressed_minus_origin_tpir_at_rank_k_identity_cluster95_low",
                  "compressed_minus_origin_tpir_at_rank_k_identity_cluster95_high"]].copy(),
              "fiqa_fixed_split": fixed, "fiqa_paired_ci": paired,
              "split_stability": frames["stability_summary"],
              "paired_split_stability": frames["paired_stability_summary"]}
    manifest = {
        "artifact_type": "calibration_evidence_report", "schema_version": 1,
        "source_run_id": expected_run_id, "model_uid": expected_model_uid,
        "condition_uid": fm["condition_uid"], "reference_seed": int(reference_seed),
        "source_manifest_sha256": expected_hashes,
        "source_directories": {"common_ci": str(Path(common_ci).resolve()),
                               "split_stability": str(Path(split_stability).resolve()),
                               "condition": str(Path(condition).resolve())},
        "target_fpirs": targets, "partition_seeds": sm["partition_seeds"],
        "coverage": cm["condition"], "cross_model_generalization": False,
        "claim_status": "exploratory_fixed_cohort_not_final_paper_evidence",
        "between_split_statistics_are_ci": False, "same_realized_fpir_comparison": False,
        "implementation_sha256": sha256_file(__file__),
        "summary_implementation_sha256": sha256_file(Path(__file__).with_name("fiqa_split_stability.py")),
        "presentation": {"audience": "technical", "surface": "existing_notebooks_and_markdown",
                         "tables_reason": "exact operating-point lookup with CI units and distinct baselines"},
    }
    manifest["report_uid"] = "calibration-report-" + canonical_sha256(manifest)[:24]
    return {"tables": tables, "manifest": manifest}


def _percent(values):
    return [f"{100 * x:.4f}" for x in values]


def render_calibration_evidence_markdown(report):
    t, m = report["tables"], report["manifest"]
    comp = t["compression_operating_points"].copy()
    comp["TPIR20 [인물 CI 95%], %"] = [f"{100*r.tpir20:.4f} [{100*r.tpir_cluster95_low:.4f}, {100*r.tpir_cluster95_high:.4f}]" for r in comp.itertuples()]
    comp["목표 FPIR, %"] = _percent(comp.target_fpir)
    comp["실제 FPIR, %"] = _percent(comp.realized_fpir)
    fixed = t["fiqa_fixed_split"].copy()
    fixed["목표 FPIR, %"] = _percent(fixed.target_fpir)
    fixed["실제 FPIR [query CI 95%], %"] = [f"{100*r.realized_fpir:.4f} [{100*r.fpir_wilson95_low:.4f}, {100*r.fpir_wilson95_high:.4f}]" for r in fixed.itertuples()]
    fixed["TPIR20 [인물 CI 95%], %"] = [f"{100*r.tpir_at_rank_k:.4f} [{100*r.tpir_cluster95_low:.4f}, {100*r.tpir_cluster95_high:.4f}]" for r in fixed.itertuples()]
    stability = t["split_stability"].copy()
    stability["목표 FPIR, %"] = _percent(stability.target_fpir)
    stability["목표 충족 분할"] = [f"{r.target_met_split_count}/{r.split_count}" for r in stability.itertuples()]
    stability["실제 FPIR 최소 / 중앙 / 최대, %"] = [f"{100*r.fpir_min:.4f} / {100*r.fpir_median:.4f} / {100*r.fpir_max:.4f}" for r in stability.itertuples()]
    method_names = {"global_empirical": "Global empirical", "global_safe": "Global safe",
                    "fiqa_s": "FIQA-S", "fiqa_l": "FIQA-L"}
    for frame in (comp, fixed, stability):
        frame["방법"] = frame.method.replace(method_names)
    comp["목표 충족"] = comp.target_met.map({True: "충족", False: "초과"})
    fixed["목표 충족"] = fixed.target_met_on_test.map({True: "충족", False: "초과"})
    low = stability[stability.target_fpir.eq(min(m["target_fpirs"])) & stability.method.ne("global_empirical")]
    answer = "; ".join(f"{r.method}: {r.target_met_split_count}/{r.split_count}" for r in low.itertuples())
    pairs = t["fiqa_paired_ci"].query("reference_method == 'global_safe' and candidate_method == 'fiqa_l'").copy()
    pairs["목표 FPIR, %"] = _percent(pairs.target_fpir)
    pairs["L−Global-safe [paired CI 95%], %p"] = [f"{100*r.candidate_minus_reference:+.4f} [{100*r.paired_bootstrap95_low:+.4f}, {100*r.paired_bootstrap95_high:+.4f}]" for r in pairs.itertuples()]
    def table(frame, columns):
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        return "\n".join([
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame[columns].itertuples(index=False, name=None)],
        ])
    delta = t["compression_paired_ci"].copy()
    delta["목표 FPIR, %"] = _percent(delta.target_fpir)
    delta["PQ−원본 TPIR20 [paired 인물 CI 95%], %p"] = [
        f"{100*r[1]:+.4f} [{100*r[2]:+.4f}, {100*r[3]:+.4f}]"
        for r in delta.iloc[:, :4].itertuples(index=False, name=None)]
    return "\n\n".join([
        "# Calibration CI and Split Stability — 통합 보고",
        f"## 기술 요약\n\n최저 목표 FPIR {100*min(m['target_fpirs']):g}%의 관측 분할 충족 수는 {answer}입니다. "
        "압축 비교와 FIQA 보정 비교를 분리하고, 각 seed의 CI와 seed 간 기술통계를 구분했습니다. "
        "목표 초과를 해결한 보고서가 아니라 현재 증거의 통합입니다.",
        f"## 범위와 지표\n\n{m['model_uid']} × {m['coverage']['dataset_id']} × "
        f"{m['coverage']['compression_profile']} / {m['coverage']['search_mode']}의 명시적 완료 조건만 포함합니다. "
        f"mated {int(fixed.test_mated_count.iloc[0]):,} queries / {int(fixed.mated_identity_count.iloc[0]):,} identities, "
        f"non-mated {int(fixed.test_non_mated_count.iloc[0]):,} queries입니다. "
        "TPIR20(정답 인물이 Top-20 안에 있고 정답 점수가 threshold 이상인 비율)의 분모는 mated queries입니다. "
        "FPIR(미등록 query가 잘못 수락되는 비율)의 분모는 non-mated queries입니다. "
        "CI는 TPIR 인물 단위·FPIR query 단위이며 threshold/gallery 고정 조건입니다. 비율은 %, 차이는 %p입니다. "
        "원본 cosine과 ADC의 threshold는 각각의 score space에서 정하며 서로 재사용하지 않습니다.",
        "## 1. 원본과 PQ — 공통 인물 단위 CI\n\n동일 cohort의 원본과 압축을 비교합니다. "
        "실제 FPIR가 목표를 초과한 행을 목표 달성 성능으로 인용하지 않습니다. FIQA 방법 순위표가 아닙니다.",
        table(comp, ["목표 FPIR, %", "방법", "실제 FPIR, %", "TPIR20 [인물 CI 95%], %", "목표 충족"]),
        "### 압축에 따른 TPIR20 변화\n\n같은 query/인물을 공동 재표집한 PQ−원본 차이입니다. "
        "원본·압축 각각의 CI가 겹치는지만으로 차이의 유의성을 판단하지 않습니다.",
        table(delta, ["목표 FPIR, %", "PQ−원본 TPIR20 [paired 인물 CI 95%], %p"]),
        f"## 2. FIQA 보정 — 고정 기준 seed {m['reference_seed']}\n\n기준 seed는 명시적으로 선택한 기존 기준이며 test 최고 seed가 아닙니다. "
        "Global empirical은 전체 calibration을 사용하고, Global-safe/S/L은 동일 fit/safety 분할을 사용합니다. "
        "TPIR 증가가 실제 FPIR 증가와 함께 나타나는지도 확인해야 합니다.",
        table(fixed, ["목표 FPIR, %", "방법", "실제 FPIR [query CI 95%], %", "TPIR20 [인물 CI 95%], %", "목표 충족"]),
        "### FIQA-L과 Global-safe의 직접 차이\n\nTPIR 양수는 증가, FPIR 양수는 오수락 증가입니다. "
        "서로 다른 목표 FPIR 행을 합치거나 동일 realized FPIR의 순수 개선으로 해석하지 않습니다.",
        table(pairs, ["목표 FPIR, %", "metric", "L−Global-safe [paired CI 95%], %p", "resampling_unit"]),
        "## 3. 분할 안정성 — CI가 아닌 관측 변동 범위\n\n같은 calibration 집합을 재분할한 결과입니다. "
        "목표 충족 횟수는 독립 반복 성공 확률이 아니며, Global empirical의 반복 행은 seed 불변 대조군입니다. "
        "상세한 seed별 CI와 threshold는 입력 artifact에 보존됩니다.",
        table(stability, ["목표 FPIR, %", "방법", "목표 충족 분할", "실제 FPIR 최소 / 중앙 / 최대, %"]),
        "## 검증·불확실성·다음 질문\n\nmanifest/CSV hash, 동일 model·protocol·test core·목표 grid, "
        "비율/성공 수와 Global empirical의 PQ baseline 재현을 검사했습니다. 분할 요약은 seed 행으로 재계산해 대조했습니다. "
        "calibration 재적합과 test 표본의 결합 CI, 다중 비교 보정, 새로운 gallery/codec/데이터셋 일반화는 포함하지 않습니다. "
        "낮은 FPIR의 지속 초과가 어떤 집단·gallery 차이에서 생기는지는 아직 분리되지 않았습니다. "
        "다음 검증은 사전에 고정한 외부 조건과 동일 realized FPIR 비교이며, 현재 test를 보고 좋은 seed를 고르지 않습니다. "
        "FIQA+Saliency 추가가치와 4×3 전체의 새 CI가 검증됐다고 주장하지 않습니다.",
    ]) + "\n"


def write_calibration_evidence_report(root, report):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / report["manifest"]["report_uid"]
    if destination.exists():
        raise FileExistsError(destination)
    staging = root / (".staging-" + uuid4().hex)
    staging.mkdir()
    files = {}
    for name, table in report["tables"].items():
        path = staging / f"{name}.csv"
        table.to_csv(path, index=False)
        files[path.name] = sha256_file(path)
    path = staging / "REPORT.md"
    path.write_text(render_calibration_evidence_markdown(report), encoding="utf-8")
    files[path.name] = sha256_file(path)
    manifest = {**report["manifest"], "status": "completed", "files": files}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.rename(staging, destination)
    return destination
