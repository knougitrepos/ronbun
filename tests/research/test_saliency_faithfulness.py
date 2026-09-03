import numpy as np
import pandas as pd
import pytest

from research.evaluation import (
    assess_saliency_faithfulness_reliability,
    select_stratified_faithfulness_sample,
    summarize_faithfulness,
)


def _candidate_frame() -> pd.DataFrame:
    rows = []
    for role in ("registered_probe", "unknown_unknown_probe"):
        for index in range(32):
            rows.append(
                {
                    "sample_id": f"{role}-{index:02d}",
                    "identity_id": f"identity-{index // 2:02d}",
                    "protocol_role": role,
                    "raw_embedding_norm": float(index + 1),
                    "gradcam_target_score": float((index % 4) * 8 + index // 4),
                }
            )
    return pd.DataFrame.from_records(rows)


def test_faithfulness_selection_is_deterministic_and_stratum_balanced():
    candidates = _candidate_frame()
    first = select_stratified_faithfulness_sample(
        candidates,
        maximum_samples=32,
        seed=42,
    )
    repeated = select_stratified_faithfulness_sample(
        candidates.sample(frac=1.0, random_state=7),
        maximum_samples=32,
        seed=42,
    )

    assert first["sample_id"].tolist() == repeated["sample_id"].tolist()
    assert len(first) == 32
    assert first["faithfulness_stratum"].nunique() == 32
    assert first.groupby("faithfulness_stratum").size().eq(1).all()
    assert set(first["raw_norm_bin"]) == {"q1", "q2", "q3", "q4"}
    assert set(first["target_score_bin"]) == {"q1", "q2", "q3", "q4"}


def test_faithfulness_summary_reports_paired_effect_and_cluster_ci():
    rows = _candidate_frame().iloc[:12].copy()
    rows["high_saliency_occlusion_score_drop"] = np.linspace(0.3, 0.5, len(rows))
    rows["low_saliency_occlusion_score_drop"] = 0.1
    rows["random_occlusion_score_drop"] = 0.2
    rows["faithfulness_gain_over_low_saliency"] = (
        rows["high_saliency_occlusion_score_drop"]
        - rows["low_saliency_occlusion_score_drop"]
    )
    rows["faithfulness_gain_over_random"] = (
        rows["high_saliency_occlusion_score_drop"]
        - rows["random_occlusion_score_drop"]
    )

    summary = summarize_faithfulness(rows, bootstrap_repeats=100, seed=9)
    effect = summary.loc[
        (summary["group"] == "all")
        & (summary["metric"] == "faithfulness_gain_over_random")
    ].iloc[0]
    assert effect["sample_count"] == 12
    assert effect["identity_count"] == 6
    assert effect["mean"] == pytest.approx(0.2)
    assert effect["positive_fraction"] == pytest.approx(1.0)
    assert effect["mean_ci_lower"] > 0.0
    assert effect["mean_ci_upper"] > effect["mean_ci_lower"]


def _reliability_summary(
    *,
    high: float = 0.30,
    low: float = 0.10,
    random: float = 0.20,
    high_low_ci: tuple[float, float] = (0.15, 0.25),
    high_random_ci: tuple[float, float] = (0.05, 0.15),
) -> pd.DataFrame:
    metrics = {
        "high_saliency_occlusion_score_drop": (high, high - 0.01, high + 0.01),
        "low_saliency_occlusion_score_drop": (low, low - 0.01, low + 0.01),
        "random_occlusion_score_drop": (random, random - 0.01, random + 0.01),
        "faithfulness_gain_over_low_saliency": (
            high - low,
            high_low_ci[0],
            high_low_ci[1],
        ),
        "faithfulness_gain_over_random": (
            high - random,
            high_random_ci[0],
            high_random_ci[1],
        ),
    }
    return pd.DataFrame.from_records(
        [
            {
                "group": "all",
                "metric": metric,
                "mean": values[0],
                "mean_ci_lower": values[1],
                "mean_ci_upper": values[2],
            }
            for metric, values in metrics.items()
        ]
    )


def test_saliency_reliability_requires_high_to_beat_low_and_random():
    result = assess_saliency_faithfulness_reliability(_reliability_summary())

    assert result.status == "passed"
    assert result.strong_faithfulness_pass is True
    assert result.gated_high_low_contrast == pytest.approx(0.20)
    assert result.as_dict()["random_control_role"] == (
        "faithfulness_negative_control_only"
    )
    assert result.as_dict()["random_is_threshold_feature"] is False


def test_saliency_reliability_blocks_when_random_beats_high():
    summary = _reliability_summary(
        high=0.09,
        low=0.03,
        random=0.115,
        high_low_ci=(0.05, 0.07),
        high_random_ci=(-0.03, -0.02),
    )

    result = assess_saliency_faithfulness_reliability(summary)

    assert result.status == "blocked"
    assert result.strong_faithfulness_pass is False
    assert result.gated_high_low_contrast == 0.0
    assert any("Random" in reason for reason in result.reasons)


def test_saliency_reliability_fails_closed_on_missing_random_control():
    summary = _reliability_summary().loc[
        lambda frame: frame["metric"] != "faithfulness_gain_over_random"
    ]

    with pytest.raises(ValueError, match="exactly one row"):
        assess_saliency_faithfulness_reliability(summary)
