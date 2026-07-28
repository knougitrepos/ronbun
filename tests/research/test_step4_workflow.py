from __future__ import annotations

import pytest

from research.experiments.step4_workflow import freeze_step4_source_and_model


def test_step4_execution_requires_explicit_runtime_acknowledgement(tmp_path):
    config = tmp_path / "step4.yaml"
    config.write_text("execution: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit local acknowledgement"):
        freeze_step4_source_and_model(
            config,
            project_root=tmp_path,
            dataset_id="lfw",
        )
