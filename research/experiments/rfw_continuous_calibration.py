"""Origin vs PQ(m=128, 64, 32) x (Global, 5-bin, Continuous FIQA) evaluation for RFW."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd
import yaml

from research.compression import PQCompressor
from research.evaluation.rfw_verification import (
    compute_rfw_pair_qualities,
    compute_rfw_pair_scores,
    evaluate_rfw_continuous_fiqa_10fold,
)
from research.runtime.hashing import canonical_sha256, sha256_file

DEFAULT_PQ_PROFILES = ("pq_512_m128_b8", "pq_512_m64_b8", "pq_512_m32_b8")
DEFAULT_METHODS = ("global_safe", "fiqa_5bin", "continuous_fiqa")
DEFAULT_TARGET_FMRS = (0.001, 0.01, 0.05, 0.1)
DEFAULT_CONFIG_PATH = "configs/experiments/rfw_continuous_calibration.yaml"


@dataclass(frozen=True)
class RFWExperimentMatrixResult:
    """Result of Origin vs PQ across calibration methods on RFW."""

    fold_metrics: pd.DataFrame
    group_summary: pd.DataFrame
    comparison_table: pd.DataFrame
    summary: dict[str, Any]
    models: dict[tuple[str, str, int, str, float], Any] | None = None
    raw_pair_evaluations: pd.DataFrame | None = None


def _parse_pq_m(profile: str) -> int:
    """Extract subquantizer count m from profile string like 'pq_512_m128_b8'."""
    match = re.search(r"_m(\d+)", profile)
    if not match:
        raise ValueError(
            f"cannot determine PQ subquantizers m from profile {profile!r}; "
            "expected format: pq_512_m<M>_b8"
        )
    m = int(match.group(1))
    if m not in (128, 64, 32):
        raise ValueError(f"expected PQ m in (128, 64, 32), got {m} from {profile!r}")
    return m


def compute_storage_accounting(
    profile: str,
    vector_count: int,
    source_dim: int = 512,
) -> dict[str, Any]:
    """Compute per-vector payload, shared codebook size, and total storage cost."""
    if profile == "origin":
        payload_bytes = source_dim * 4  # float32
        codebook_bytes = 0
    else:
        m = _parse_pq_m(profile)
        payload_bytes = m  # 8-bit per subvector = 1 byte * m
        codebook_bytes = source_dim * 256 * 4  # 512 * 256 * float32 = 524,288 bytes
    total_bytes = vector_count * payload_bytes + codebook_bytes
    origin_total = vector_count * (source_dim * 4)
    compression_ratio = float(origin_total / total_bytes) if total_bytes > 0 else 1.0
    return {
        "code_payload_bytes": payload_bytes,
        "codebook_bytes": codebook_bytes,
        "total_storage_bytes": total_bytes,
        "compression_ratio": compression_ratio,
    }


def load_lfw_disjoint_development_embeddings(
    project_root: str | Path,
    selected_model: str,
    *,
    sources_root: str | Path = "results/lfw_deepfunneled_resize_v1/sources",
    pairs_path: str | Path = "data/external/lfw/pairs.txt",
    min_samples: int = 256,
    expected_model_uid: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    source_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load verified LFW non-test development embeddings disjoint from LFW View-2 test pairs.

    Note on identity overlap:
    While this dataset is rigorously disjoint from LFW View-2 1:1 test pairs, identity disjointness
    against external evaluation benchmarks (such as RFW) has not been formally verified.
    """
    from research.datasets.lfw_pairs import load_pairs

    root = Path(project_root).resolve()
    resolved_pairs = (root / pairs_path).resolve()
    if not resolved_pairs.is_file():
        raise FileNotFoundError(f"LFW pairs protocol file missing: {resolved_pairs}")

    sources_dir = (root / sources_root).resolve()
    if not sources_dir.is_dir():
        raise FileNotFoundError(f"LFW sources root directory missing: {sources_dir}")

    # Determine candidate source directory
    selected_dir: Path | None = None
    if source_dir is not None:
        cand = (root / source_dir).resolve() if not Path(source_dir).is_absolute() else Path(source_dir).resolve()
        if not cand.is_dir():
            raise FileNotFoundError(f"Explicitly specified source directory missing: {cand}")
        selected_dir = cand
    else:
        candidates = [
            d for d in sources_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith(f"{selected_model.lower()}-")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No source directory found for model {selected_model!r} under {sources_dir}. "
                "Model-specific development embeddings are required."
            )
        # Match against expected_model_uid or expected_checkpoint_sha256 if provided
        if expected_model_uid or expected_checkpoint_sha256:
            for cand in candidates:
                mf_path = cand / "manifest.json"
                if mf_path.is_file():
                    try:
                        with mf_path.open("r", encoding="utf-8") as f:
                            m_data = json.load(f)
                        m_uid = m_data.get("model_uid") or m_data.get("contract", {}).get("model_uid")
                        m_ckpt = m_data.get("checkpoint_sha256") or m_data.get("contract", {}).get("checkpoint_sha256")
                        uid_ok = (expected_model_uid is None) or (m_uid == expected_model_uid)
                        ckpt_ok = (expected_checkpoint_sha256 is None) or (str(m_ckpt).lower() == str(expected_checkpoint_sha256).lower())
                        if uid_ok and ckpt_ok:
                            selected_dir = cand
                            break
                    except Exception:
                        pass
        if selected_dir is None:
            selected_dir = candidates[0]

    manifest_file = selected_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Source manifest.json missing in {selected_dir}")

    # 1. Verify source completion marker (_SUCCESS)
    success_marker = selected_dir / "_SUCCESS"
    if not success_marker.is_file():
        raise FileNotFoundError(
            f"Complete LFW resize source marker '_SUCCESS' missing in {selected_dir}. "
            "Incomplete or uncommitted source artifacts cannot be reused."
        )

    with manifest_file.open("r", encoding="utf-8") as f:
        src_manifest = json.load(f)

    # 2. Verify status is completed
    src_status = src_manifest.get("status")
    if src_status not in ("completed", "SUCCESS"):
        raise ValueError(
            f"Development embedding source {selected_dir.name} status is {src_status!r}; "
            "must be 'completed' before reuse."
        )

    # 3. Verify model_uid if expected
    actual_uid = src_manifest.get("model_uid") or src_manifest.get("contract", {}).get("model_uid")
    if expected_model_uid and actual_uid != expected_model_uid:
        raise ValueError(
            f"Development embedding model UID mismatch: source manifest has {actual_uid!r}, "
            f"evaluation requires {expected_model_uid!r}."
        )

    # 4. Verify checkpoint SHA-256 if expected
    actual_ckpt = src_manifest.get("checkpoint_sha256") or src_manifest.get("contract", {}).get("checkpoint_sha256")
    if expected_checkpoint_sha256 and str(actual_ckpt).lower() != str(expected_checkpoint_sha256).lower():
        raise ValueError(
            f"Development embedding checkpoint hash mismatch: source manifest has {actual_ckpt!r}, "
            f"evaluation requires {expected_checkpoint_sha256!r}."
        )

    pop_path = selected_dir / "population.csv"
    emb_path = selected_dir / "normalized_embeddings.npy"
    if not pop_path.is_file():
        raise FileNotFoundError(f"population.csv missing in {selected_dir}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"normalized_embeddings.npy missing in {selected_dir}")

    # 5. Verify mandatory files section and expected hashes recorded in manifest
    files_meta = src_manifest.get("files")
    if not isinstance(files_meta, dict):
        raise ValueError(
            f"Mandatory 'files' section missing in manifest of {selected_dir}. "
            "Source integrity cannot be verified."
        )
    pop_entry = files_meta.get("population.csv")
    emb_entry = files_meta.get("normalized_embeddings.npy")
    if not isinstance(pop_entry, dict) or "sha256" not in pop_entry:
        raise ValueError(
            f"Mandatory files['population.csv']['sha256'] entry missing in manifest of {selected_dir}."
        )
    if not isinstance(emb_entry, dict) or "sha256" not in emb_entry:
        raise ValueError(
            f"Mandatory files['normalized_embeddings.npy']['sha256'] entry missing in manifest of {selected_dir}."
        )

    expected_pop_sha = str(pop_entry["sha256"]).strip()
    expected_emb_sha = str(emb_entry["sha256"]).strip()
    if not expected_pop_sha or len(expected_pop_sha) != 64:
        raise ValueError(f"Invalid population.csv sha256 in manifest of {selected_dir}: {expected_pop_sha!r}")
    if not expected_emb_sha or len(expected_emb_sha) != 64:
        raise ValueError(f"Invalid normalized_embeddings.npy sha256 in manifest of {selected_dir}: {expected_emb_sha!r}")

    actual_pop_sha = sha256_file(pop_path)
    if actual_pop_sha.lower() != expected_pop_sha.lower():
        raise ValueError(
            f"Corrupted population.csv in {selected_dir}: expected {expected_pop_sha}, got {actual_pop_sha}"
        )

    actual_emb_sha = sha256_file(emb_path)
    if actual_emb_sha.lower() != expected_emb_sha.lower():
        raise ValueError(
            f"Corrupted normalized_embeddings.npy in {selected_dir}: expected {expected_emb_sha}, got {actual_emb_sha}"
        )

    population = pd.read_csv(pop_path)
    if "row_count" in src_manifest and len(population) != int(src_manifest["row_count"]):
        raise ValueError(
            f"population.csv row count mismatch in {selected_dir}: manifest says {src_manifest['row_count']}, file has {len(population)}"
        )

    all_embs = np.load(emb_path)
    if all_embs.ndim != 2 or all_embs.shape[1] != 512 or all_embs.shape[0] != len(population):
        raise ValueError(
            f"normalized_embeddings.npy dimension mismatch in {selected_dir}: expected ({len(population)}, 512), got {all_embs.shape}"
        )
    if not np.isfinite(all_embs).all():
        raise ValueError(f"normalized_embeddings.npy in {selected_dir} contains NaN or infinite values")

    pairs, dev_df = load_pairs(resolved_pairs, population)
    if len(dev_df) < min_samples:
        raise ValueError(
            f"Insufficient disjoint development samples for {selected_model}: {len(dev_df)} < {min_samples}"
        )

    dev_vectors = all_embs[dev_df.index].astype(np.float32)

    metadata = {
        "development_dataset_id": "lfw-disjoint-non-test",
        "development_source_dir": str(
            selected_dir.relative_to(root) if selected_dir.is_relative_to(root) else selected_dir
        ),
        "development_sample_count": len(dev_vectors),
        "source_model_uid": actual_uid,
        "source_checkpoint_sha256": actual_ckpt,
        "source_status": src_status,
        "source_manifest_sha256": sha256_file(manifest_file),
        "population_sha256": actual_pop_sha,
        "embeddings_sha256": actual_emb_sha,
        "pairs_sha256": sha256_file(resolved_pairs),
        "identity_overlap_verified": False,
        "identity_overlap_note": (
            "LFW View-2 non-test development split guarantees disjointness from LFW test pairs, "
            "but cross-dataset identity overlap against RFW evaluation subjects has not been formally verified."
        ),
    }
    return dev_vectors, metadata


def run_rfw_continuous_calibration(
    pairs: pd.DataFrame,
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    fiqa_scores: Mapping[str, float] | pd.Series | np.ndarray,
    development_embeddings: np.ndarray | None = None,
    pq_profiles: Sequence[str] = DEFAULT_PQ_PROFILES,
    methods: Sequence[str] = DEFAULT_METHODS,
    target_fmrs: Sequence[float] = DEFAULT_TARGET_FMRS,
    quality_mode: str = "symmetric_min",
    strict_official: bool = True,
    bootstrap_seed: int = 8972,
    bootstrap_repeats: int = 2000,
    safety_fraction: float = 0.3,
    knot_quantiles: Sequence[float] = (1 / 3, 2 / 3),
    smoothing: float = 0.01,
    ridge: float = 0.001,
    max_iterations: int = 2000,
    margin_slope_cap: float = 0.95,
    keep_raw_pairs: bool = False,
) -> RFWExperimentMatrixResult:
    """Run full experimental matrix on RFW: Origin vs PQ x (Global, 5-bin, Continuous FIQA)."""
    has_pq = any(p != "origin" for p in pq_profiles)
    if has_pq:
        if development_embeddings is None:
            raise ValueError(
                "development_embeddings must be explicitly provided for PQ codec training. "
                "Fitting PQ codecs on evaluation/test embeddings is strictly prohibited to prevent data leakage."
            )
        if development_embeddings is embeddings:
            raise ValueError(
                "development_embeddings cannot be identical to evaluation embeddings."
            )
        dev_arr = np.asarray(development_embeddings, dtype=np.float32)
        eval_arr = np.asarray(embeddings, dtype=np.float32)
        if dev_arr.shape == eval_arr.shape and np.array_equal(dev_arr, eval_arr):
            raise ValueError(
                "development_embeddings cannot have identical content to evaluation embeddings."
            )
        # Vector byte hash intersection check: catches subset slices, permutations, reordered rows
        eval_row_hashes = {hashlib.sha256(row.tobytes()).digest() for row in eval_arr}
        dev_row_hashes = {hashlib.sha256(row.tobytes()).digest() for row in dev_arr}
        intersection = eval_row_hashes & dev_row_hashes
        if intersection:
            raise ValueError(
                f"Data leakage detected: {len(intersection)} development vector(s) "
                "overlap with evaluation/test embeddings."
            )

    pair_qualities = compute_rfw_pair_qualities(
        pairs, fiqa_scores, quality_mode=quality_mode
    )
    all_profiles = ("origin", *pq_profiles)
    num_vectors = len(image_ids)

    all_fold_metrics: list[pd.DataFrame] = []
    all_group_summaries: list[pd.DataFrame] = []
    all_raw_pairs: list[pd.DataFrame] = []
    all_models: dict[tuple[str, str, int, str, float], Any] = {}

    for profile in all_profiles:
        storage = compute_storage_accounting(profile, num_vectors)
        if profile == "origin":
            codec = None
            score_space = "cosine"
        else:
            m = _parse_pq_m(profile)
            codec = PQCompressor(source_dim=512, m=m, nbits=8, random_state=bootstrap_seed)
            codec.fit(np.asarray(development_embeddings, dtype=np.float32))
            score_space = "negative_squared_l2_adc"

        scores = compute_rfw_pair_scores(
            pairs,
            image_ids=image_ids,
            embeddings=embeddings,
            codec=codec,
        )

        result = evaluate_rfw_continuous_fiqa_10fold(
            pairs,
            scores=scores,
            pair_qualities=pair_qualities,
            target_fmrs=target_fmrs,
            methods=methods,
            score_space=score_space,
            compression_profile=profile,
            strict_official=strict_official,
            bootstrap_seed=bootstrap_seed,
            bootstrap_repeats=bootstrap_repeats,
            safety_fraction=safety_fraction,
            knot_quantiles=tuple(knot_quantiles),
            smoothing=smoothing,
            ridge=ridge,
            max_iterations=max_iterations,
            margin_slope_cap=margin_slope_cap,
            return_raw_pairs=keep_raw_pairs,
        )

        if keep_raw_pairs and result.raw_pair_evaluations is not None:
            all_raw_pairs.append(result.raw_pair_evaluations)
        if result.models:
            for (grp, fld, mth, tfmr), mdl in result.models.items():
                all_models[(profile, grp, fld, mth, tfmr)] = mdl

        fold_df = result.fold_metrics.copy()
        for k, v in storage.items():
            fold_df[k] = v
        all_fold_metrics.append(fold_df)

        group_df = result.group_summary.copy()
        for k, v in storage.items():
            group_df[k] = v
        all_group_summaries.append(group_df)

    combined_folds = pd.concat(all_fold_metrics, ignore_index=True)
    combined_groups = pd.concat(all_group_summaries, ignore_index=True)

    # Compute tar_gain_vs_global: difference in mean_tar relative to global_safe under identical group, profile, target_fmr
    globals_subset = combined_groups.loc[
        combined_groups["method"] == "global_safe",
        ["rfw_group", "compression_profile", "target_fmr", "mean_tar"]
    ].rename(columns={"mean_tar": "global_mean_tar"})

    merged_groups = combined_groups.merge(
        globals_subset,
        on=["rfw_group", "compression_profile", "target_fmr"],
        how="left",
    )
    merged_groups["tar_gain_vs_global"] = (
        merged_groups["mean_tar"] - merged_groups["global_mean_tar"]
    ).fillna(0.0)

    # Assemble comprehensive comparison table preserving audit counts and gains
    cols_to_include = [
        "compression_profile",
        "rfw_group",
        "method",
        "target_fmr",
        "mean_accuracy",
        "mean_tar",
        "tar_gain_vs_global",
        "mean_realized_fmr",
        "target_met_rate",
        "pooled_true_accepts",
        "pooled_false_accepts",
        "pooled_genuine_pairs",
        "pooled_impostor_pairs",
        "pooled_realized_fmr",
        "pooled_tar",
        "mean_safety_pairs_retained",
        "mean_cross_partition_pairs_excluded",
        "code_payload_bytes",
        "codebook_bytes",
        "total_storage_bytes",
        "compression_ratio",
    ]
    comparison_table = merged_groups[[
        c for c in cols_to_include if c in merged_groups.columns
    ]].copy()

    summary: dict[str, Any] = {
        "protocol": "rfw_official_groupwise_10fold_continuous_calibration",
        "profiles": list(all_profiles),
        "methods": list(methods),
        "target_fmrs": [float(t) for t in target_fmrs],
        "quality_mode": quality_mode,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "safety_fraction": float(safety_fraction),
        "total_pairs": len(pairs),
        "total_images": num_vectors,
        "formal_fmr_guarantee": False,
        "fairness_guarantee": False,
    }

    raw_pairs_df = (
        pd.concat(all_raw_pairs, ignore_index=True)
        if keep_raw_pairs and all_raw_pairs
        else None
    )

    return RFWExperimentMatrixResult(
        fold_metrics=combined_folds,
        group_summary=combined_groups,
        comparison_table=comparison_table,
        summary=summary,
        models=all_models if keep_raw_pairs else None,
        raw_pair_evaluations=raw_pairs_df,
    )


def generate_chatgpt_summary(
    result: RFWExperimentMatrixResult,
    manifest: dict[str, Any],
) -> str:
    """Generate compact Markdown summary optimized for ChatGPT analysis."""
    is_real = manifest.get("real_data", False)
    is_smoke = manifest.get("quick_smoke", False)
    model_alias = manifest.get("model_alias", "unknown")
    model_uid = manifest.get("model_uid", "unknown")
    groups_eval = sorted(result.group_summary["rfw_group"].unique().tolist())
    total_pairs = manifest.get("total_pairs", len(result.comparison_table))
    quality_src = manifest.get("quality_source", "cr_fiqa")
    fold_count = (
        int(result.group_summary["fold_count"].iloc[0])
        if "fold_count" in result.group_summary.columns and not result.group_summary.empty
        else (2 if is_smoke else 10)
    )

    lines = [
        "# RFW 1:1 Continuous FIQA Threshold Calibration Summary",
        "",
        "## Executive Summary",
        f"- **Execution Mode**: {'Real RFW Data & Live Model Inference' if is_real else 'Synthetic Fixture (Fast Test Mode)'}",
        f"- **Evaluation Scope**: {'Quick Smoke Slice (African 2 folds, 1,200 pairs)' if is_smoke else f'Full Evaluation ({len(groups_eval)} groups, {fold_count} folds/group)'}",
        f"- **Model**: {model_alias} ({model_uid})",
        f"- **Evaluated Demographic Groups**: {', '.join(groups_eval)}",
        f"- **Evaluated Folds**: {fold_count} folds per group",
        f"- **Total Evaluated Pairs**: {total_pairs:,} pairs",
        f"- **Quality Estimator**: {quality_src}",
        f"- **Profiles Evaluated**: {', '.join(result.summary['profiles'])}",
        f"- **Methods**: {', '.join(result.summary['methods'])}",
        f"- **Target FMRs**: {', '.join(str(x) for x in result.summary['target_fmrs'])}",
        f"- **Pair Quality Definition**: {result.summary.get('quality_mode', 'symmetric_min')} (min(q_left, q_right))",
        "- **Cross-Dataset Identity Disjointness**: Unverified (identity_overlap_verified = False; disjoint from LFW test pairs, but RFW identity overlap not formally verified)",
        "- **Formal FMR Guarantee**: None (formal_fmr_guarantee = False; empirical safety thresholding only)",
        "- **Fairness Guarantee**: None (demographic differences are reported as empirical observations)",
        "",
        "## 1. Storage Efficiency vs Compression Payload",
        "| Profile | Payload Bytes | Codebook Bytes | Total Storage (Bytes) | Compression Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    profiles = result.comparison_table["compression_profile"].unique()
    for p in profiles:
        sub = result.comparison_table[result.comparison_table["compression_profile"] == p].iloc[0]
        lines.append(
            f"| {p} | {sub['code_payload_bytes']} | {sub['codebook_bytes']} | "
            f"{sub['total_storage_bytes']} | {sub['compression_ratio']:.2f}x |"
        )

    lines.extend([
        "",
        "## 2. Target FMR vs Realized FMR, TAR, and TAR Gain by Demographic Group",
        "| Group | Profile | Method | Target FMR | Realized FMR | Mean TAR | TAR Gain vs Global | Target Met Rate | Pooled FA / Impostors |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in result.comparison_table.iterrows():
        gain_str = f"{row['tar_gain_vs_global']:+.4f}" if "tar_gain_vs_global" in row else "N/A"
        fa_str = (
            f"{int(row['pooled_false_accepts'])} / {int(row['pooled_impostor_pairs'])}"
            if "pooled_false_accepts" in row and pd.notna(row["pooled_false_accepts"])
            else "N/A"
        )
        lines.append(
            f"| {row['rfw_group']} | {row['compression_profile']} | {row['method']} | "
            f"{row['target_fmr']:.3f} | {row['mean_realized_fmr']:.4f} | "
            f"{row['mean_tar']:.4f} | {gain_str} | {row['target_met_rate']:.2f} | {fa_str} |"
        )

    lines.extend([
        "",
        "## 3. Key Observations & Cautions",
        "1. **Continuous FIQA vs Global Safe**: Continuous FIQA adapts thresholds dynamically based on symmetric pair quality min(q_left, q_right), preserving genuine matches on higher-quality pairs under compression.",
        "2. **Score Space Separation**: Cosine similarity [-1, 1] and PQ ADC negative squared L2 [-4, 0] operate in distinct score spaces; thresholds are calibrated independently within each space.",
        "3. **Two-Endpoint Identity Disjointness**: Calibration candidate pairs crossing fit and safety roles are excluded to ensure zero identity leakage between model fitting and safety offset calibration.",
        "4. **Identity Overlap Limitation**: PQ development embeddings are strictly disjoint from LFW test pairs, but disjointness from RFW evaluation identities has not been verified (`identity_overlap_verified = False`).",
        "5. **No Mathematical Guarantees**: Target met rates reflect held-out test fold realization. They do not constitute mathematical FMR or fairness guarantees.",
    ])
    return "\n".join(lines) + "\n"


def generate_interpretation_guide() -> str:
    """Generate Markdown guide explaining how to read and interpret RFW calibration results."""
    return """# RFW 1:1 Continuous FIQA Threshold Calibration 안내서

## 1. 연구 질문과 비교 조건
본 연구는 사전학습된 얼굴 임베딩 및 fine-tuned 모델을 대상으로,
Product Quantization(PQ m=128, 64, 32) 압축 시 발생하는 점수 분포 변화에 대해
Continuous FIQA(구간별 선형 basis를 사용하는 스플라인 분위수 회귀와 safety offset)가
Global Safe 및 FIQA 5-bin 대비 어떤 차이를 만드는지
RFW 공식 4개 인종 그룹(African, Asian, Caucasian, Indian) 10-fold 1:1 검증 프로토콜에서 평가합니다.

- **비교 조건**:
  - 압축: Origin 512D float32, PQ m128 b8, PQ m64 b8, PQ m32 b8
  - 보정: Global Safe, FIQA 5-bin, Continuous FIQA
  - 목표 FMR: 0.001, 0.01, 0.05, 0.10

## 2. 데이터 분할과 역할 분리
1. **공식 10-fold 분할 보존**:
   각 fold를 held-out test로 두고, 나머지 9개 fold를 calibration에 사용합니다.
2. **양쪽 endpoint identity 역할 일치 계약 (Two-Endpoint Identity Disjointness)**:
   1:1 pair 검증에서 impostor pair는 서로 다른 identity를 가집니다.
   임의의 impostor pair가 fit과 safety에 걸치는 경우 발생하는 identity 누수를 차단하기 위해,
   좌/우 identity를 동일한 hash 시드로 'fit'/'safety'로 분할하고,
   양쪽 모두 동일한 역할에 배정된 pair만 calibration에 유지합니다.
   불일치하는 cross-partition pair는 calibration에서 제외하며 제외 수량과 분모를 기록합니다.
3. **PQ Development 분리 및 한계 명시**:
   PQ 코덱은 LFW View-2 test pair와 분리된 모델별 development 임베딩(LFW disjoint non-test 1,549장)으로 학습합니다.
   평가 임베딩 벡터와의 byte hash 교집합 검사를 통해 동일 임베딩 벡터의 유출을 차단합니다.
   **한계 명시**: LFW 개발 데이터의 인물과 RFW 평가 데이터 인물 간의 cross-dataset identity 분리 여부는 공식적으로 검증되지 않았습니다 (`identity_overlap_verified = False`).

## 3. 품질 정의: LFW와의 차이
- **LFW**: 1:1 pair 평가에 query/reference 비대칭 압축을 적용(비대칭 DB 검색 모사), query-only FIQA($q_\\text{pair} = q_\\text{query}$).
- **RFW**: 순수 1:1 대칭 검증(symmetric min FIQA, $q_\\text{pair} = \\min(q_\\text{left}, q_\\text{right})$).
두 데이터셋의 품질 정의는 프로토콜 목적이 다르므로 동일한 설정으로 취급하지 않습니다.

## 4. 점수 공간과 임계값
- Origin: Cosine similarity [-1, 1]
- PQ ADC: Asymmetric distance computation, negative squared L2 [-4, 0]
Cosine과 PQ ADC는 별도의 점수 공간이며, 임계값 범위도 독립적으로 적합됩니다.

## 5. 지표 해석 및 원본 보존 정책
- **realized_fmr**: test fold에서 실제로 관측된 False Match Rate.
- **tar**: True Accept Rate (genuine pair 중 임계값을 통과한 비율).
- **tar_gain_vs_global**: 동일 조건에서 Continuous FIQA가 Global Safe 대비 달성한 TAR 차이 ($TAR_\\text{continuous} - TAR_\\text{global}$).
- **target_met_on_test**: realized_fmr <= target_fmr 여부.
- **pooled_false_accepts / pooled_impostor_pairs**: 10개 fold에 걸친 풀링된 오인식 수 및 분모.
- **fold_metrics.csv**: 요약 모드(`KEEP_RAW_RESULTS=False`)에서도 fold별 오류 수·분모·신뢰구간 감사 추적을 위해 필수 보존됩니다.
- **KEEP_RAW_RESULTS 옵션**:
  - `True`: pair별 점수·품질·임계값·판정 상세 테이블(`raw_pair_evaluations.csv.gz`) 및 훈련된 calibration 모델 파라미터(`fitted_calibration_models.pkl`)를 실제 보존.
  - `False`: 상세 pair 테이블 생성을 생략하여 저장 용량을 절약하고 요약 CSV와 해석 안내만 보존.
- **주의사항**:
  - realized_fmr <= target_fmr는 경험적 표본에서의 달성이며 수학적/통계적 FMR 보장이 아닙니다 (`formal_fmr_guarantee = False`).
  - 그룹 간 성능 차이를 보고하되, 이를 공정성 개선이나 보장으로 해석하지 않습니다 (`fairness_guarantee = False`).

## 6. 실행 방법 및 CLI
```powershell
# 1. 계획 및 설정 검증 (Dry run)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml

# 2. 정식 실행 (실제 RFW 데이터 및 실모델 가중치 기본 적용)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute

# 3. 고속 축소 검증 (Quick smoke: African 2-fold 축소 검증, 1,200쌍)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --quick-smoke

# 4. 상세 원본 결과 보존 실행
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --quick-smoke --keep-raw-results

# 5. CI/회귀 테스트용 합성 데이터 실행
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --synthetic
```
"""


def _extract_rfw_embeddings(
    archive_path: Path,
    pairs_slice: pd.DataFrame,
    spec_path: Path,
    *,
    expected_archive_sha256: str | None = None,
    quality_source: str = "cr_fiqa",
    cr_fiqa_checkpoint: Path | None = None,
    expected_fiqa_sha256: str | None = None,
    fiqa_variant: str = "L",
    device: str = "cuda",
    strict_official: bool = True,
) -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, float], str, dict[str, Any]]:
    """Extract real embeddings and quality scores for RFW pairs using verified PyTorch FR adapter and CR-FIQA."""
    import torch
    from research.datasets.rfw_aligned_bin import iter_rfw_aligned_pair_batches
    from research.embeddings.manifests import read_model_spec
    from research.embeddings.registry import create_pytorch_adapter_from_spec
    from research.fiqa.cr_fiqa import infer_cr_fiqa_scores, load_cr_fiqa

    requested_device = str(device).strip().lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA execution requested but torch.cuda is unavailable; silent CPU fallback is prohibited."
        )

    use_device = "cuda" if (requested_device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_device == "cuda" else "CPU"

    spec = read_model_spec(spec_path)
    adapter = create_pytorch_adapter_from_spec(spec, device=use_device)

    cr_fiqa_meta: dict[str, Any] = {}
    cr_model = None
    if quality_source == "cr_fiqa":
        if cr_fiqa_checkpoint is None or not Path(cr_fiqa_checkpoint).is_file():
            raise FileNotFoundError(
                f"CR-FIQA checkpoint missing: {cr_fiqa_checkpoint}. "
                "CR-FIQA is required as independent quality estimator."
            )
        actual_fiqa_sha = sha256_file(cr_fiqa_checkpoint)
        if expected_fiqa_sha256 and actual_fiqa_sha.lower() != str(expected_fiqa_sha256).lower():
            raise ValueError(
                f"CR-FIQA checkpoint hash mismatch: file has {actual_fiqa_sha}, "
                f"config expects {expected_fiqa_sha256}"
            )
        cr_model, cr_spec = load_cr_fiqa(
            cr_fiqa_checkpoint, variant=fiqa_variant, device=use_device, verify_official_hash=True
        )
        cr_fiqa_meta = {
            "quality_source": "cr_fiqa",
            "cr_fiqa_variant": fiqa_variant,
            "cr_fiqa_model_uid": cr_spec.model_uid,
            "cr_fiqa_checkpoint_sha256": cr_model.cr_fiqa_checkpoint_sha256,
            "cr_fiqa_expected_sha256": expected_fiqa_sha256,
        }
    elif quality_source == "embedding_norm":
        cr_fiqa_meta = {
            "quality_source": "embedding_norm",
            "ablation_description": "Face recognition backbone L2 raw norm prior to normalization",
        }
    else:
        raise ValueError(
            f"Unsupported quality_source: {quality_source!r}; expected 'cr_fiqa' or 'embedding_norm'"
        )

    image_ids: list[str] = []
    vectors: list[np.ndarray] = []
    fiqa_dict: dict[str, float] = {}
    occurrence_frames: list[pd.DataFrame] = []

    for batch in iter_rfw_aligned_pair_batches(
        archive_path,
        pairs_slice,
        batch_size=128,
        strict_official=strict_official,
        expected_sha256=expected_archive_sha256,
    ):
        output = adapter.embed(batch.faces)
        norm_emb = output.normalized_embedding
        vectors.append(norm_emb)
        occurrence_frames.append(batch.occurrences)

        if quality_source == "cr_fiqa":
            assert cr_model is not None
            cr_scores = infer_cr_fiqa_scores(cr_model, batch.faces, device=use_device)
            for idx, row in batch.occurrences.iterrows():
                occ_id = str(row["occurrence_id"])
                image_ids.append(occ_id)
                fiqa_dict[occ_id] = float(cr_scores[idx])
        else:
            for idx, row in batch.occurrences.iterrows():
                occ_id = str(row["occurrence_id"])
                image_ids.append(occ_id)
                fiqa_dict[occ_id] = float(output.raw_norm[idx])

    embeddings = np.vstack(vectors)
    all_occurrences = pd.concat(occurrence_frames, ignore_index=True)
    sides = all_occurrences.pivot(index="pair_id", columns="side", values="occurrence_id")

    pairs_with_occ = pairs_slice.copy().reset_index(drop=True)
    pairs_with_occ["left_image_id"] = pairs_with_occ["pair_id"].map(sides["left"])
    pairs_with_occ["right_image_id"] = pairs_with_occ["pair_id"].map(sides["right"])

    return pairs_with_occ, image_ids, embeddings, fiqa_dict, device_name, cr_fiqa_meta


def run_rfw_calibration_workflow(
    project_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_root: str | Path = "results/calibration/rfw_continuous_calibration",
    execute: bool = False,
    keep_raw_results: bool | None = None,
    synthetic: bool = False,
    real_data: bool | None = None,
    quick_smoke: bool = False,
    selected_model: str = "arcface",
    quality_source: str | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Execute the complete RFW continuous calibration workflow with manifest and storage policies."""
    root = Path(project_root).resolve()
    cfg_file = (root / config_path).resolve()
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exec_cfg = cfg.get("execution", {})
    fiqa_cfg = cfg.get("datasets", {}).get("fiqa", {})

    # Priority hierarchy: Explicit argument > YAML config > Default
    eff_seed = seed if seed is not None else int(exec_cfg.get("bootstrap_seed", 8972))
    eff_device = device if device is not None else str(exec_cfg.get("device", "cuda"))
    eff_quality_source = (
        quality_source
        if quality_source is not None
        else str(fiqa_cfg.get("quality_source", "cr_fiqa"))
    )
    fiqa_variant = str(fiqa_cfg.get("variant", "L"))
    expected_fiqa_sha = fiqa_cfg.get("expected_sha256")

    # Resolve whether to run real data or synthetic fixture (default: real data)
    is_real = bool(real_data) if real_data is not None else not bool(synthetic)

    keep = (
        keep_raw_results
        if keep_raw_results is not None
        else exec_cfg.get("keep_raw_results", False)
    )

    run_id = f"rfw-cal-{uuid.uuid4().hex[:12]}"
    out_dir = (root / output_root / run_id).resolve()

    if not execute:
        return {
            "status": "planned",
            "run_id": run_id,
            "output_dir": str(out_dir),
            "execute": False,
            "config": cfg,
            "keep_raw_results": keep,
            "real_data": is_real,
            "synthetic": not is_real,
            "quick_smoke": quick_smoke,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(eff_seed)

    device_name = "Synthetic/CPU"
    model_uid = None
    checkpoint_sha = None
    archive_sha = None
    dev_meta: dict[str, Any] = {}
    quality_meta: dict[str, Any] = {}

    if is_real:
        # Real data execution path
        rfw_archive = root / cfg.get("datasets", {}).get("rfw", {}).get(
            "archive_path", "data/raw/RFW/bin_for_mxnet/RFW_test.tar.gz"
        )
        protocol_path = root / cfg.get("datasets", {}).get("rfw", {}).get(
            "protocol_path", "data/interim/rfw/pair_protocol.csv"
        )
        if not protocol_path.is_file():
            raise FileNotFoundError(f"RFW protocol file missing: {protocol_path}")
        if not rfw_archive.is_file():
            raise FileNotFoundError(f"RFW archive missing: {rfw_archive}")

        expected_archive_sha = cfg.get("datasets", {}).get("rfw", {}).get("archive_sha256")
        archive_sha = sha256_file(rfw_archive)
        if expected_archive_sha and archive_sha.lower() != str(expected_archive_sha).lower():
            raise ValueError(
                f"RFW archive hash mismatch: expected {expected_archive_sha}, got {archive_sha}"
            )

        pairs_all = pd.read_csv(protocol_path)

        # Select model spec and strictly verify checkpoint hash
        model_entry = cfg.get("models", {}).get(selected_model)
        if not model_entry:
            raise ValueError(
                f"Model {selected_model!r} not configured in models section of {cfg_file}"
            )
        spec_path = root / model_entry.get(
            "spec_path", f"runs/step2/model_registry/{selected_model}-7972a704552df378345f.json"
        )
        if not spec_path.is_file():
            raise FileNotFoundError(f"Model spec file missing for {selected_model}: {spec_path}")

        with spec_path.open("r", encoding="utf-8") as sf:
            spec_dict = json.load(sf)
            model_uid = spec_dict.get("model_uid", selected_model)
            checkpoint_sha = spec_dict.get("checkpoint", {}).get("sha256")

        expected_ckpt_sha = model_entry.get("expected_sha256")
        if expected_ckpt_sha and checkpoint_sha.lower() != str(expected_ckpt_sha).lower():
            raise ValueError(
                f"Checkpoint hash mismatch for {selected_model}: spec has {checkpoint_sha}, "
                f"config expects {expected_ckpt_sha}"
            )

        ckpt_path = root / model_entry.get("checkpoint", spec_dict.get("checkpoint", {}).get("path", ""))
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file missing: {ckpt_path}")
        actual_ckpt_sha = sha256_file(ckpt_path)
        if actual_ckpt_sha.lower() != str(checkpoint_sha).lower():
            raise ValueError(
                f"Checkpoint file {ckpt_path} hash mismatch: file has {actual_ckpt_sha}, spec has {checkpoint_sha}"
            )

        if quick_smoke:
            # African group, 2 complete official folds (fold 0 and 1, 600 pairs each = 1,200 pairs)
            pairs_slice = pairs_all.loc[
                (pairs_all["rfw_group"] == "African") & (pairs_all["fold_index"].isin([0, 1]))
            ].copy().reset_index(drop=True)
            strict_official = False
            bootstrap_repeats = 100
        else:
            pairs_slice = pairs_all.copy()
            strict_official = True
            bootstrap_repeats = int(cfg.get("execution", {}).get("bootstrap_repeats", 500))

        # Extract real embeddings and quality scores
        cr_ckpt_path = root / fiqa_cfg.get("checkpoint_path", "models/fiqa/CR-FIQA(L).pth")

        pairs_df, image_ids, eval_embeddings, fiqa_dict, device_name, quality_meta = _extract_rfw_embeddings(
            rfw_archive,
            pairs_slice,
            spec_path,
            expected_archive_sha256=expected_archive_sha,
            quality_source=eff_quality_source,
            cr_fiqa_checkpoint=cr_ckpt_path,
            expected_fiqa_sha256=expected_fiqa_sha,
            fiqa_variant=fiqa_variant,
            device=eff_device,
            strict_official=strict_official,
        )

        # Load model-specific disjoint development embeddings with strict manifest integrity validation
        dev_cfg = cfg.get("datasets", {}).get("development", {})
        dev_source_dir = model_entry.get("development_source_dir") or dev_cfg.get("source_dir")
        dev_vectors, dev_meta = load_lfw_disjoint_development_embeddings(
            root,
            selected_model,
            sources_root=dev_cfg.get("sources_root", "results/lfw_deepfunneled_resize_v1/sources"),
            pairs_path=dev_cfg.get("pairs_path", "data/external/lfw/pairs.txt"),
            min_samples=int(dev_cfg.get("min_samples", 256)),
            expected_model_uid=model_uid,
            expected_checkpoint_sha256=checkpoint_sha,
            source_dir=dev_source_dir,
        )
    else:
        # Synthetic fixture execution path (for fast offline CI / regression verification)
        dev_vectors = rng.normal(size=(512, 512)).astype(np.float32)
        dev_norms = np.linalg.norm(dev_vectors, axis=1, keepdims=True)
        dev_vectors /= np.where(dev_norms > 0, dev_norms, 1.0)
        dev_meta = {
            "development_dataset_id": "synthetic-gaussian",
            "development_sample_count": 512,
        }

        groups = ("African", "Asian", "Caucasian", "Indian")
        rows: list[dict[str, Any]] = []
        image_ids: list[str] = []
        vectors: list[np.ndarray] = []
        total_pairs = 0

        pairs_per_fold = 250  # 125 genuine, 125 impostor
        for group in groups:
            for fold in range(4):  # 4 folds for workflow verification
                for i in range(pairs_per_fold):
                    is_genuine = (i % 2 == 0)
                    left_id = f"rfw:{group.lower()}:f{fold}:p{i}:L"
                    right_id = f"rfw:{group.lower()}:f{fold}:p{i}:R"
                    image_ids.extend([left_id, right_id])

                    v_l = rng.normal(size=512).astype(np.float32)
                    v_l /= np.linalg.norm(v_l)
                    if is_genuine:
                        v_r = v_l + rng.normal(scale=0.15, size=512).astype(np.float32)
                    else:
                        v_r = rng.normal(size=512).astype(np.float32)
                    v_r /= np.linalg.norm(v_r)
                    vectors.extend([v_l, v_r])

                    rows.append({
                        "pair_id": f"rfw:{group.lower()}:fold{fold:02d}:pair{i:03d}",
                        "rfw_group": group,
                        "fold_index": fold,
                        "left_image_id": left_id,
                        "right_image_id": right_id,
                        "left_identity_id": f"id-{group}-{fold}-L-{i // 2 if is_genuine else i}",
                        "right_identity_id": f"id-{group}-{fold}-R-{i // 2 if is_genuine else i + 1000}",
                        "is_genuine": is_genuine,
                    })
                    total_pairs += 1

        pairs_df = pd.DataFrame(rows)
        eval_embeddings = np.vstack(vectors)
        fiqa_dict = {img: float(rng.uniform(25.0, 75.0)) for img in image_ids}
        strict_official = False
        bootstrap_repeats = 100
        model_uid = f"synthetic-{selected_model}"
        quality_meta = {"quality_source": "synthetic_uniform"}

    pq_profiles = tuple(cfg.get("compression", {}).get("pq_profiles", DEFAULT_PQ_PROFILES))
    methods = tuple(cfg.get("calibration", {}).get("methods", DEFAULT_METHODS))
    target_fmrs = tuple(cfg.get("calibration", {}).get("target_fmrs", DEFAULT_TARGET_FMRS))

    # Read calibration hyperparameters
    cal_cfg = cfg.get("calibration", {})
    knot_quantiles = tuple(cal_cfg.get("knot_quantiles", [1 / 3, 2 / 3]))
    smoothing = float(cal_cfg.get("smoothing", 0.01))
    ridge = float(cal_cfg.get("ridge", 0.001))
    max_iterations = int(cal_cfg.get("max_iterations", 2000))
    margin_slope_cap = float(cal_cfg.get("margin_slope_cap", 0.95))

    result = run_rfw_continuous_calibration(
        pairs_df,
        image_ids=image_ids,
        embeddings=eval_embeddings,
        fiqa_scores=fiqa_dict,
        development_embeddings=dev_vectors,
        pq_profiles=pq_profiles,
        methods=methods,
        target_fmrs=target_fmrs,
        quality_mode=cfg.get("datasets", {}).get("rfw", {}).get("quality_mode", "symmetric_min"),
        strict_official=strict_official,
        bootstrap_seed=eff_seed,
        bootstrap_repeats=bootstrap_repeats,
        safety_fraction=float(cfg.get("execution", {}).get("safety_fraction", 0.3)),
        knot_quantiles=knot_quantiles,
        smoothing=smoothing,
        ridge=ridge,
        max_iterations=max_iterations,
        margin_slope_cap=margin_slope_cap,
        keep_raw_pairs=keep,
    )

    # Save summary tables
    group_summary_path = out_dir / "group_summary.csv"
    comparison_table_path = out_dir / "comparison_table.csv"
    fold_metrics_path = out_dir / "fold_metrics.csv"

    result.group_summary.to_csv(group_summary_path, index=False)
    result.comparison_table.to_csv(comparison_table_path, index=False)
    # fold_metrics.csv is ALWAYS preserved as essential aggregate audit table
    result.fold_metrics.to_csv(fold_metrics_path, index=False)

    # Detailed raw artifacts retention (only when keep_raw_results is True)
    raw_artifacts_meta: dict[str, Any] = {}
    raw_pairs_path: Path | None = None
    models_path: Path | None = None
    if keep:
        import pickle

        if result.raw_pair_evaluations is not None:
            raw_pairs_path = out_dir / "raw_pair_evaluations.csv.gz"
            result.raw_pair_evaluations.to_csv(raw_pairs_path, index=False, compression="gzip")
            raw_artifacts_meta["raw_pair_evaluations.csv.gz"] = {
                "sha256": sha256_file(raw_pairs_path),
                "row_count": len(result.raw_pair_evaluations),
                "format": "csv.gz",
            }

        if result.models:
            models_path = out_dir / "fitted_calibration_models.pkl"
            with models_path.open("wb") as mf:
                pickle.dump(result.models, mf)
            raw_artifacts_meta["fitted_calibration_models.pkl"] = {
                "sha256": sha256_file(models_path),
                "model_count": len(result.models),
                "format": "pickle",
            }

    # Manifest creation
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "completed",
        "dataset": "rfw",
        "real_data": is_real,
        "synthetic": not is_real,
        "quick_smoke": quick_smoke,
        "device_name": device_name,
        "model_alias": selected_model,
        "model_uid": model_uid,
        "checkpoint_sha256": checkpoint_sha,
        "archive_sha256": archive_sha,
        "config_sha256": sha256_file(cfg_file),
        "keep_raw_results": keep,
        "raw_results_preserved": keep,
        "raw_artifacts": raw_artifacts_meta,
        "seed": eff_seed,
        "quality_source": quality_meta.get("quality_source", eff_quality_source),
        "quality_meta": quality_meta,
        "development_meta": dev_meta,
        "profiles": list(result.summary["profiles"]),
        "methods": list(methods),
        "target_fmrs": list(target_fmrs),
        "quality_mode": result.summary.get("quality_mode", "symmetric_min"),
        "total_pairs": len(pairs_df),
        "formal_fmr_guarantee": False,
        "fairness_guarantee": False,
        "identity_overlap_verified": False,
        "identity_overlap_note": (
            "LFW View-2 non-test development split guarantees disjointness from LFW test pairs, "
            "but cross-dataset identity overlap against RFW evaluation subjects has not been formally verified."
        ),
        "fold_metrics_sha256": sha256_file(fold_metrics_path),
        "group_summary_sha256": sha256_file(group_summary_path),
        "comparison_table_sha256": sha256_file(comparison_table_path),
    }

    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ChatGPT Summary
    chatgpt_summary_text = generate_chatgpt_summary(result, manifest)
    chatgpt_path = out_dir / "CHATGPT_SUMMARY.md"
    chatgpt_path.write_text(chatgpt_summary_text, encoding="utf-8")

    # Interpretation Guide
    guide_text = generate_interpretation_guide()
    guide_path = out_dir / "RFW_CONTINUOUS_CALIBRATION.md"
    guide_path.write_text(guide_text, encoding="utf-8")

    return {
        "status": "completed",
        "run_id": run_id,
        "output_dir": str(out_dir),
        "manifest": manifest,
        "fold_metrics_path": str(fold_metrics_path),
        "group_summary_path": str(group_summary_path),
        "comparison_table_path": str(comparison_table_path),
        "chatgpt_summary_path": str(chatgpt_path),
        "interpretation_guide_path": str(guide_path),
        "raw_pair_evaluations_path": str(raw_pairs_path) if raw_pairs_path else None,
        "fitted_models_path": str(models_path) if models_path else None,
        "keep_raw_results": keep,
        "real_data": is_real,
        "synthetic": not is_real,
        "quick_smoke": quick_smoke,
        "device_name": device_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", default="results/calibration/rfw_continuous_calibration")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--keep-raw-results",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to retain detailed per-fold/per-pair raw outputs",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run fast synthetic test fixture (for offline testing/CI only; default is real data)",
    )
    parser.add_argument(
        "--real-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly toggle real data vs synthetic fixture",
    )
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run fast smoke test slice of real data/model (African 2 folds, 1,200 pairs)",
    )
    parser.add_argument(
        "--quality-source",
        choices=["cr_fiqa", "embedding_norm"],
        default=None,
        help="Quality estimator source: 'cr_fiqa' or 'embedding_norm' (defaults to YAML setting)",
    )
    parser.add_argument("--model", default="arcface", help="Selected model alias")
    parser.add_argument("--device", default=None, help="Execution device (cuda or cpu; defaults to YAML setting)")
    parser.add_argument("--seed", type=int, default=None, help="Bootstrap seed (defaults to YAML setting)")
    args = parser.parse_args()

    # If --synthetic flag was passed and --real-data not explicitly given, synthetic wins
    real_data_flag = args.real_data
    if real_data_flag is None and args.synthetic:
        real_data_flag = False

    result = run_rfw_calibration_workflow(
        project_root=args.project_root,
        config_path=args.config,
        output_root=args.output_root,
        execute=args.execute,
        keep_raw_results=args.keep_raw_results,
        synthetic=args.synthetic,
        real_data=real_data_flag,
        quick_smoke=args.quick_smoke,
        selected_model=args.model,
        quality_source=args.quality_source,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
