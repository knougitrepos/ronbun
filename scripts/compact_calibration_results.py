"""Read-only matrix validation and bounded, loss-of-detail-explicit chat summaries."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from research.runtime.hashing import canonical_sha256, sha256_file

CONDITION = ["dataset_id", "model", "compression_profile"]
KEY = CONDITION + ["target_fpir", "partition_seed"]
METHODS = ("global_safe", "fiqa_2bin", "fiqa_5bin", "continuous_fiqa", "plus_outside", "plus_entropy", "plus_both")
PAIRS = [(a, b) for i, a in enumerate(METHODS[:4]) for b in METHODS[:4][i + 1:]] + [
    ("baseline", m) for m in METHODS[4:]]
DEFAULT_REPORTS = (
    "matrix-report-2e2ab02ddb1d039ef1084383",
    "matrix-report-d690567c6f4f7e7ad44ef19e",
)


def _bool(series):
    converted = series.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise ValueError(f"invalid Boolean: {series.name}")
    return converted.astype(bool)


def _close(left, right, label):
    if not np.allclose(left, right, rtol=0, atol=1e-10, equal_nan=False):
        raise ValueError(f"inconsistent {label}")


def load_report(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("status") != "completed" or manifest["observed_jobs"] != manifest["expected_jobs"]:
        raise ValueError("only complete reports can be summarized")
    hashes = {"manifest.json": sha256_file(directory / "manifest.json")}
    frames = {}
    for name in ("method_summary.csv", "paired_comparisons.csv", "split_summary.csv"):
        hashes[name] = sha256_file(directory / name)
        if hashes[name] != manifest["files"][name]:
            raise ValueError(f"source hash mismatch: {name}")
        frames[name] = pd.read_csv(directory / name, low_memory=False)
    metrics, pairs, saved = (frames[n] for n in ("method_summary.csv", "paired_comparisons.csv", "split_summary.csv"))
    sources = pd.DataFrame(manifest["sources"])
    seeds = sorted(sources.partition_seed.unique().tolist())
    expected_conditions = pd.MultiIndex.from_product(
        [["lfw", "rfw_custom", "survface"], ["arcface", "adaface", "magface", "edgeface"],
         ["pq_512_m128_b8", "pq_512_m64_b8", "pq_512_m32_b8"]], names=CONDITION)
    if set(map(tuple, sources[CONDITION].drop_duplicates().values)) != set(expected_conditions):
        raise ValueError("report does not cover the full 36-condition matrix")
    targets = sorted(metrics.target_fpir.unique())
    if targets != [.01, .05, .1, .2, .3]:
        raise ValueError("unexpected target FPIR grid")
    source_key = CONDITION + ["partition_seed", "family"]
    if sources.duplicated(source_key).any() or len(sources) != 36 * len(seeds) * 2:
        raise ValueError("missing/duplicate source jobs")
    if set(sources.family) != {"fiqa", "saliency"}:
        raise ValueError("invalid method families")
    for frame in (metrics, pairs):
        lineage = frame.merge(sources[source_key + ["model_uid", "source_run_id", "result_dir"]],
                              on=source_key, how="left", validate="many_to_one", suffixes=("", "_source"))
        if any(not lineage[k].equals(lineage[k + "_source"]) for k in ("model_uid", "source_run_id", "result_dir")):
            raise ValueError("table/source lineage mismatch")
        if set(frame.partition_seed) != set(seeds):
            raise ValueError("table seed inventory mismatch")
    if metrics.duplicated(KEY + ["method"]).any() or len(metrics) != 36 * len(seeds) * 5 * 7:
        raise ValueError("missing/duplicate metric rows")
    counts = metrics.groupby(CONDITION + ["target_fpir", "method"]).partition_seed.nunique()
    if len(counts) != 36 * 5 * 7 or not counts.eq(len(seeds)).all() or set(metrics.method) != set(METHODS):
        raise ValueError("incomplete method grid")
    metrics["target_met_on_test"] = _bool(metrics.target_met_on_test)
    if _bool(metrics.threshold_fit_on_test).any() or not metrics.rank_k.eq(20).all():
        raise ValueError("test-fitted threshold or non-TPIR20 contract")
    for count, total, rate in (("false_accept_count", "test_non_mated_count", "realized_fpir"),
                               ("true_identification_at_rank_k_count", "test_mated_count", "tpir_at_rank_k")):
        if (metrics[total].le(0).any() or metrics[count].lt(0).any() or metrics[count].gt(metrics[total]).any()
                or not np.equal(metrics[[count, total]], np.floor(metrics[[count, total]])).all().all()):
            raise ValueError("invalid counts/denominators")
        _close(metrics[count] / metrics[total], metrics[rate], rate)
        if not metrics.groupby(CONDITION)[total].nunique().eq(1).all():
            raise ValueError("test denominator changes across methods/seeds")
    _close(metrics.target_met_on_test.astype(int), (metrics.realized_fpir <= metrics.target_fpir).astype(int), "target flags")
    for low, high in (("fpir_wilson95_low", "fpir_wilson95_high"), ("tpir_cluster95_low", "tpir_cluster95_high")):
        if metrics[[low, high]].isna().any().any() or (metrics[low] > metrics[high]).any():
            raise ValueError("invalid reported CI")
    group = CONDITION + ["target_fpir", "method"]
    recomputed = metrics.groupby(group).agg(split_count=("partition_seed", "nunique"),
        target_met_split_count=("target_met_on_test", "sum"), fpir_min=("realized_fpir", "min"),
        fpir_median=("realized_fpir", "median"), fpir_max=("realized_fpir", "max"),
        tpir_min=("tpir_at_rank_k", "min"), tpir_median=("tpir_at_rank_k", "median"),
        tpir_max=("tpir_at_rank_k", "max")).sort_index()
    actual = saved.set_index(group).sort_index()
    if not actual.index.equals(recomputed.index):
        raise ValueError("split summary keys mismatch")
    _close(actual[recomputed.columns], recomputed, "split summary")
    pair_key = CONDITION + ["target_fpir", "reference_method", "candidate_method", "metric"]
    if pairs.duplicated(pair_key + ["partition_seed"]).any() or len(pairs) != 36 * 5 * len(seeds) * 18:
        raise ValueError("missing/duplicate paired rows")
    if (set(zip(pairs.reference_method, pairs.candidate_method)) != set(PAIRS)
            or set(pairs.metric) != {"fpir", "tpir_at_rank_k"}
            or not pairs.groupby(pair_key).partition_seed.nunique().eq(len(seeds)).all()):
        raise ValueError("incomplete pair grid")
    for side, column in (("reference", "reference_method"), ("candidate", "candidate_method")):
        selected = pairs[KEY].copy()
        selected["method"] = pairs[column].replace({"baseline": "continuous_fiqa"})
        selected = selected.merge(metrics[KEY + ["method", "false_accept_count", "test_non_mated_count",
                                                 "true_identification_at_rank_k_count", "test_mated_count",
                                                 "realized_fpir", "target_met_on_test"]],
                                  on=KEY + ["method"], how="left", validate="many_to_one")
        is_fpir = pairs.metric.eq("fpir")
        _close(pairs[side + "_successes"], np.where(is_fpir, selected.false_accept_count,
               selected.true_identification_at_rank_k_count), side + " paired counts")
        _close(pairs.total, np.where(is_fpir, selected.test_non_mated_count, selected.test_mated_count), "pair denominator")
        pairs[side + "_actual_fpir"] = selected.realized_fpir.to_numpy()
        pairs[side + "_met"] = selected.target_met_on_test.to_numpy()
    _close(pairs.candidate_minus_reference, (pairs.candidate_successes - pairs.reference_successes) / pairs.total, "paired delta")
    if (pairs.paired_bootstrap95_low > pairs.paired_bootstrap95_high).any() or pairs[
            ["paired_bootstrap95_low", "paired_bootstrap95_high"]].isna().any().any():
        raise ValueError("invalid paired CI")
    pairs["both_met"] = pairs.reference_met & pairs.candidate_met
    pairs["positive_ci"] = pairs.paired_bootstrap95_low > 0
    pairs["negative_ci"] = pairs.paired_bootstrap95_high < 0
    pairs["both_met_positive_ci"] = pairs.both_met & pairs.positive_ci
    return dict(directory=directory, hashes=hashes, seeds=seeds, metrics=metrics, pairs=pairs,
                summary=recomputed.reset_index(), source_jobs=len(sources))


def aggregate(report):
    metrics, pairs = report["metrics"], report["pairs"]
    group = CONDITION + ["target_fpir", "method"]
    extra = metrics.groupby(group).agg(non_mated=("test_non_mated_count", "first"),
        mated=("test_mated_count", "first"), false_accept_min=("false_accept_count", "min"),
        false_accept_max=("false_accept_count", "max"), true_identification_min=("true_identification_at_rank_k_count", "min"),
        true_identification_max=("true_identification_at_rank_k_count", "max"),
        fpir_wilson_low_min=("fpir_wilson95_low", "min"), fpir_wilson_high_max=("fpir_wilson95_high", "max"),
        tpir_cluster_low_min=("tpir_cluster95_low", "min"), tpir_cluster_high_max=("tpir_cluster95_high", "max"),
        fallback_count_max=("fallback_query_count", "max"), fallback_fraction_max=("fallback_query_fraction", "max"),
        faithfulness_status=("faithfulness_status", lambda s: "|".join(sorted(s.dropna().unique())))).reset_index()
    summary = report["summary"].merge(extra, on=group, validate="one_to_one")
    summary["all_seeds_met"] = summary.target_met_split_count == summary.split_count
    overview = summary.groupby(["dataset_id", "target_fpir", "method"]).agg(
        conditions=("model", "size"), all_seeds_met_conditions=("all_seeds_met", "sum"),
        met_condition_seeds=("target_met_split_count", "sum"), condition_seeds=("split_count", "sum"),
        fpir_min=("fpir_min", "min"), fpir_max=("fpir_max", "max")).reset_index()
    paired = pairs.groupby(CONDITION + ["target_fpir", "reference_method", "candidate_method", "metric"]).agg(
        seeds=("partition_seed", "nunique"), delta_min=("candidate_minus_reference", "min"),
        delta_median=("candidate_minus_reference", "median"), delta_max=("candidate_minus_reference", "max"),
        positive_ci_seeds=("positive_ci", "sum"), negative_ci_seeds=("negative_ci", "sum"),
        ci_low_min=("paired_bootstrap95_low", "min"), ci_high_max=("paired_bootstrap95_high", "max"),
        both_target_met_seeds=("both_met", "sum"), both_met_positive_ci_seeds=("both_met_positive_ci", "sum"),
        reference_fpir_min=("reference_actual_fpir", "min"), reference_fpir_max=("reference_actual_fpir", "max"),
        candidate_fpir_min=("candidate_actual_fpir", "min"), candidate_fpir_max=("candidate_actual_fpir", "max")).reset_index()
    return summary, overview, paired


def csv_pages(frame, *, max_rows):
    """Keep logical condition tables together; paginate by rows, never truncate."""
    pages, start = [], 0
    while start < len(frame):
        end = min(start + max_rows, len(frame))
        payload = frame.iloc[start:end].to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf8")
        pages.append(payload)
        start = end
    return pages


def verify_compact(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["inventory_pages"].items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError("compact inventory hash mismatch")
        for row in pd.read_csv(directory / name).itertuples(index=False):
            path = (directory / row.path).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256_file(path) != row.sha256 or path.stat().st_size != row.bytes:
                raise ValueError("compact content hash mismatch")
    return manifest


def build_compact(report_directories, output_root, *, preferred_file_bytes=32000, max_rows=150):
    """Summarize each explicit report separately; never combine overlapping seeds."""
    if preferred_file_bytes < 1000 or not 1 <= max_rows <= 1000:
        raise ValueError("positive size guidance and 1–1000 rows per page required")
    directories = [Path(p).resolve() for p in report_directories]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("explicit unique source reports required")
    output_root = Path(output_root).resolve()
    if any(output_root.is_relative_to(p.parent.parent) or p.is_relative_to(output_root) for p in directories):
        raise ValueError("compact output must be separate from the source matrix tree")
    reports = [load_report(p) for p in directories]
    sources = [dict(report=p.name, hashes=r["hashes"], seeds=r["seeds"], jobs=r["source_jobs"],
                    metric_rows=len(r["metrics"]), paired_rows=len(r["pairs"])) for p, r in zip(directories, reports)]
    spec = dict(schema_version=1, sources=sources, preferred_file_bytes=preferred_file_bytes, max_rows=max_rows,
                implementation_sha256=sha256_file(Path(__file__)), pandas=pd.__version__, numpy=np.__version__)
    destination = output_root / ("compact-" + canonical_sha256(spec)[:24])
    if destination.exists():
        manifest = verify_compact(destination)
        if manifest["spec"] != spec:
            raise ValueError("existing compact specification differs")
        return destination, manifest
    files = {}

    def add(path, payload):
        payload = payload.encode("utf8") if isinstance(payload, str) else payload
        files[path] = payload

    def table(prefix, frame):
        paths = []
        for i, payload in enumerate(csv_pages(frame, max_rows=max_rows), 1):
            path = f"{prefix}-{i:02d}.csv"
            add(path, payload)
            paths.append(path)
        return paths

    guide = """# 읽는 방법과 지표

한 번에 START_HERE.md 또는 CSV 한 페이지만 요청하세요. 전체 폴더/ZIP/원본 CSV를 한꺼번에 채팅으로 읽지 마세요.
CSV의 모든 비율·차이는 0–1 단위이며 0.1은 10%입니다. 대상 FPIR은 설정값이고 실제 FPIR은 측정된 오수락률입니다.

- split_count / seeds: 같은 calibration 집합을 나눈 seed 수. 독립 실험 반복 횟수가 아닙니다.
- target_met_split_count: 실제 FPIR가 목표 이하인 seed 수. 미래 FPIR 보장이 아닙니다.
- fpir_min/median/max, tpir_min/median/max: seed별 관측 범위/중앙값. TPIR은 genuine-score 기준 Rank-20입니다.
- non_mated/mated: test 분모. false_accept / true_identification 최소·최대는 seed별 성공 수입니다.
- CI low_min/high_max: 기존 seed별 95% CI 끝점들의 범위입니다. 20-seed 통합 CI가 아니며 CI를 새로 계산하지 않았습니다.
- paired delta: candidate − reference. TPIR 양수는 증가, FPIR 양수는 오수락 증가입니다.
- positive_ci_seeds / negative_ci_seeds: 기존 paired CI가 각각 0보다 완전히 큰/작은 seed 수입니다. seed 간 검정이나 다중 비교 보정이 아닙니다.
- both_target_met_seeds: 두 방법 모두 목표 FPIR를 충족한 seed 수. 두 방법의 실제 FPIR가 같다는 의미는 아닙니다.
- both_met_positive_ci_seeds: 위 조건을 만족하면서 paired CI가 양수인 seed 수. FPIR metric에서 양수는 이득이 아닙니다.
- saliency paired reference의 baseline은 continuous_fiqa입니다. FIQA 방법 간 비교도 모두 보존했습니다.
- faithfulness_status=failed는 진단 실패를 그대로 표시한 것입니다. threshold 성능 개선이나 인과성을 입증하지 않습니다.
- fallback_*는 무효 saliency에서 FIQA-only로 처리한 query의 최대 수/비율입니다. FIQA-only 행의 빈 칸은 해당 없음입니다.

단일 seed 8972 보고서와 20-seed 보고서는 중복 seed가 있으므로 서로 합쳐 평균내지 않습니다.
데이터셋/모델/PQ를 pooling한 FPIR·TPIR를 만들지 않았습니다. 전체 현황의 condition-seed 수는 기술적 집계입니다.
이 compact는 모든 조건·방법·목표·seed를 집계하지만 개별 seed의 수치, fitted 계수, query 기록을 무손실 복제하지는 않습니다.
정확한 개별 수치·추가 분석은 원본 matrix reports 또는 기존 analysis_archive ZIP에서 확인하세요. 원본은 수정하지 않습니다.
파일 크기는 참고값이며 크기를 맞추기 위해 내용을 삭제하지 않았습니다. 조건별 표를 함께 유지하고 행 수가 많을 때만 페이지를 나눕니다.
연결 도구의 추가 출력/기존 대화 길이에 따른 토큰 한도까지 보장하지는 않습니다. 여러 파일을 한꺼번에 출력하지 마세요.
"""
    add("METRICS.md", guide)
    root_lines = ["# FIQA + saliency compact", "", "로컬 Python으로 원본 전체를 검증·집계했습니다. 아래 보고서를 하나씩 읽으세요.",
                  "먼저 [지표 설명](METRICS.md)을 확인하세요. FPIR 목표 실패를 숨기거나 TPIR 개선만으로 방법을 선정하지 않았습니다.", ""]
    for report, src in zip(reports, sources):
        name = src["report"]
        summary, overview, pairs = aggregate(report)
        headline = overview.pivot(index=["dataset_id", "target_fpir"], columns="method",
                                  values="all_seeds_met_conditions").reindex(columns=METHODS).reset_index()
        quick = [f"# 핵심 요약 — {len(src['seeds'])}개 seed", "",
                 "아래 표의 수치는 각 데이터셋의 12조건(4 FR × 3 PQ) 중 해당 조건의 수입니다. FPIR·TPIR를 pooling한 값이 아닙니다.",
                 "", "## 모든 seed에서 목표 FPIR를 충족한 조건 수 / 12", "",
                 "```csv", headline.to_csv(index=False, float_format="%.3g").strip(), "```", "",
                 "## FIQA 대비 saliency 효과를 읽는 순서", "",
                 "목표 충족 횟수만으로 효과를 판정하지 않습니다. conditions/의 paired 표에서 reference_method=baseline, metric=tpir_at_rank_k를 확인하세요.",
                 "먼저 both_target_met_seeds와 두 방법의 실제 FPIR 범위를 확인하고, delta_min/median/max 및 positive/negative_ci_seeds를 함께 읽습니다.",
                 "두 방법이 모두 목표를 충족해도 동일 실제 FPIR는 아닙니다. seed들은 독립 반복이 아니며 다중 비교 보정도 없습니다.",
                 "일괄적인 효과 있음/없음 판정이나 좋은 조건만의 선정 없이, 모든 조건의 크기·손실·CI·fallback을 상세 표에 보존했습니다."]
        add(f"{name}/QUICK_SUMMARY.md", "\n".join(quick) + "\n")
        rows = []
        for condition, frame in summary.groupby(CONDITION, sort=True):
            dataset, model, profile = condition
            base = f"{name}/conditions/{dataset}_{model}_{profile}"
            paths = table(base + "_metrics", frame.drop(columns=CONDITION))
            subset = pairs.loc[(pairs[CONDITION] == pd.Series(condition, index=CONDITION)).all(axis=1)]
            pair_paths = table(base + "_paired", subset.drop(columns=CONDITION))
            rows.append(dict(dataset=dataset, model=model, pq=profile,
                             metrics="|".join(Path(p).name for p in paths), paired="|".join(Path(p).name for p in pair_paths)))
        indexes = table(f"{name}/conditions/index", pd.DataFrame(rows))
        overviews = []
        for dataset, frame in overview.groupby("dataset_id", sort=True):
            overviews.extend(table(f"{name}/overview_{dataset}", frame.drop(columns="dataset_id")))
        lines = [f"# {name}", "", f"조건 36 · seed {len(src['seeds'])}개 · 방법 7 · 목표 5 · 원본 metric {src['metric_rows']:,}행 / paired {src['paired_rows']:,}행.",
                 "", "먼저 [15행 핵심 요약](QUICK_SUMMARY.md)을 읽으세요.",
                 "", "## 목표 FPIR 충족 현황", "", *[f"- [{Path(p).name}]({Path(p).name})" for p in overviews],
                 "", "## 모든 조건의 상세 파일 찾기", "", *[f"- [{Path(p).name}](conditions/{Path(p).name})" for p in indexes],
                 "", "색인의 파일명은 conditions/ 기준입니다. metrics와 paired 파일을 각각 한 페이지씩 읽으세요.",
                 "모든 seed를 반영한 범위·중앙값·충족 횟수이며, 개별 seed CI를 평균내어 통합 CI로 만들지 않았습니다."]
        add(f"{name}/README.md", "\n".join(lines) + "\n")
        root_lines.append(f"- [{len(src['seeds'])}개 seed 보고서]({name}/README.md): metric {src['metric_rows']:,}행, paired {src['paired_rows']:,}행 → 조건별 compact.")
    add("START_HERE.md", "\n".join(root_lines) + "\n")
    inventory = pd.DataFrame([dict(path=name, bytes=len(data), above_preferred_size=len(data) > preferred_file_bytes,
                                   sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(files.items())])
    inventory_paths = table("inventory", inventory)
    manifest = dict(status="completed", spec=spec, preferred_file_bytes=preferred_file_bytes,
                    above_preferred_files=int(inventory.above_preferred_size.sum()),
                    file_count=len(files) + 1, inventory_pages={p: hashlib.sha256(files[p]).hexdigest() for p in inventory_paths})
    add("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Recheck source bytes before publication; no source file is written.
    for report in reports:
        if any(sha256_file(report["directory"] / n) != h for n, h in report["hashes"].items()):
            raise ValueError("source changed during compaction")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / (".staging-" + uuid4().hex)
    staging.mkdir()
    for name, payload in files.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    verify_compact(staging)
    from research.explainability.gradcam.artifacts import _publish_atomic_directory
    _publish_atomic_directory(staging, destination, overwrite=False)
    return destination, manifest


def preview_file(directory, relative="START_HERE.md", *, max_bytes=6000):
    path = (Path(directory) / relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError("preview escapes compact directory")
    if path.stat().st_size > max_bytes:
        return f"미리보기 생략: {path.name} ({path.stat().st_size:,} bytes). 파일 하나를 별도로 읽으세요."
    return path.read_text(encoding="utf8")
