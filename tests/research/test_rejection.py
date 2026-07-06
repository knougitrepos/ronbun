import pandas as pd

from research.calibration.rejection import (
    CALIBRATION_FEATURE_COLUMNS,
    GlobalThresholdCalibrator,
    LogisticRegressionCalibrator,
    PerCompressionThresholdCalibrator,
    ShallowMLPCalibrator,
)


def _features():
    rows = []
    for score, margin, label, profile in [
        (0.95, 0.40, 1, "origin_512"),
        (0.90, 0.30, 1, "pca_2"),
        (0.55, 0.05, 0, "origin_512"),
        (0.50, 0.04, 0, "pca_2"),
        (0.85, 0.20, 1, "origin_512"),
        (0.45, 0.03, 0, "pca_2"),
    ]:
        rows.append(
            {
                "top1_score": score,
                "score_margin": margin,
                "probe_quality": 0.8,
                "template_quality": 0.7,
                "template_variance": 0.01,
                "enrollment_count": 2,
                "reconstruction_error_norm": 0.0 if profile == "origin_512" else 1.0,
                "compression_profile": profile,
                "y_true_accept": label,
            }
        )
    return pd.DataFrame(rows)


def test_threshold_calibrators_fit_on_calibration_rows_and_predict_acceptance():
    features = _features()

    global_model = GlobalThresholdCalibrator(target_fpir=0.0).fit(features)
    per_profile = PerCompressionThresholdCalibrator(target_fpir=0.0).fit(features)

    assert global_model.threshold > 0.55
    assert set(per_profile.thresholds) == {"origin_512", "pca_2"}
    assert global_model.predict(features).tolist().count(1) == 3
    assert per_profile.predict(features).tolist().count(1) == 3


def test_learned_calibrators_use_the_stable_feature_schema():
    features = _features()

    logistic = LogisticRegressionCalibrator().fit(features)
    mlp = ShallowMLPCalibrator(random_state=0).fit(features)

    assert logistic.feature_columns == CALIBRATION_FEATURE_COLUMNS
    assert mlp.feature_columns == CALIBRATION_FEATURE_COLUMNS
    assert len(logistic.predict_proba(features)) == len(features)
    assert len(mlp.predict_proba(features)) == len(features)
