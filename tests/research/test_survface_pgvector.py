from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np
import pandas as pd

from research.experiments.survface_pgvector import (
    build_survface_official_templates,
    run_survface_official_search,
    run_survface_official_search_matrix,
)


def _manifest() -> pd.DataFrame:
    records = []
    index = 0
    for identity, vector in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        for copy in range(2):
            records.append(
                {
                    "image_id": f"g-{identity}-{copy}",
                    "identity_id": identity,
                    "image_path": f"gallery/{identity}-{copy}.jpg",
                    "protocol_role": "gallery",
                    "protocol_index": index,
                    "_vector": vector,
                }
            )
            index += 1
    records.extend(
        [
            {
                "image_id": "p-a",
                "identity_id": "a",
                "image_path": "probe/a.jpg",
                "protocol_role": "registered_probe",
                "protocol_index": 0,
                "_vector": [1.0, 0.0],
            },
            {
                "image_id": "p-b",
                "identity_id": "b",
                "image_path": "probe/b.jpg",
                "protocol_role": "registered_probe",
                "protocol_index": 1,
                "_vector": [0.0, 1.0],
            },
            {
                "image_id": "p-u",
                "identity_id": "unknown",
                "image_path": "probe/u.jpg",
                "protocol_role": "unknown_unknown_probe",
                "protocol_index": 0,
                "_vector": [-1.0, 0.0],
            },
        ]
    )
    frame = pd.DataFrame.from_records(records)
    frame["split"] = "test"
    frame["dataset"] = "qmul-survface-v1"
    frame["probe_type"] = frame["protocol_role"].map(
        {
            "gallery": "gallery",
            "registered_probe": "registered",
            "unknown_unknown_probe": "unknown_unknown",
        }
    )
    frame["official_identity_id"] = pd.NA
    frame["source_protocol_index"] = frame["protocol_index"]
    return frame


class _FakeRepository:
    template_calls: list[dict[str, object]] = []

    def __init__(self, session):
        self.session = session

    def upsert_template_512(self, **values):
        self.template_calls.append(values)
        return object(), "inserted"

    def find_similar_templates_512(self, query, **kwargs):
        ranked = [
            {
                "identity_id": "a",
                "distance": 0.1,
                "query_elapsed_ms": 0.5,
            },
            {
                "identity_id": "b",
                "distance": 0.5,
                "query_elapsed_ms": 0.5,
            },
        ]
        return ranked[: kwargs["top_k"]]


@contextmanager
def _fake_session_scope(engine):
    yield object()


def test_official_templates_average_every_gallery_image():
    manifest = _manifest()
    gallery = manifest.loc[manifest["protocol_role"].eq("gallery")].copy()
    gallery["embedding"] = gallery["_vector"].map(
        lambda value: np.asarray(value, dtype=np.float32)
    )

    templates = build_survface_official_templates(gallery)

    assert templates["identity_id"].tolist() == ["a", "b"]
    assert templates["enrollment_count"].tolist() == [2, 2]
    assert templates.iloc[0]["source_image_ids"] == ["g-a-0", "g-a-1"]


def test_runner_preserves_roles_order_and_top_k(monkeypatch, tmp_path):
    import research.experiments.survface_pgvector as module

    manifest = _manifest()
    embeddings = pd.DataFrame(
        {
            "image_path_key": manifest["image_path"].map(module._canonical_path),
            "embedding": manifest["_vector"].map(
                lambda value: np.asarray(value, dtype=np.float32)
            ),
            "parameters": [{"model": "buffalo_l"}] * len(manifest),
        }
    )
    _FakeRepository.template_calls = []
    monkeypatch.setattr(module, "_load_db_embeddings", lambda *args, **kwargs: embeddings)
    monkeypatch.setattr(module, "session_scope", _fake_session_scope)
    monkeypatch.setattr(module, "VectorRepository", _FakeRepository)
    monkeypatch.setattr(module, "ensure_vector_indexes", lambda engine: None)
    output = tmp_path / "official.csv"

    summary = run_survface_official_search(
        object(),
        run_uid="survface-run",
        manifest=manifest.drop(columns=["_vector"]),
        compression_profile="origin_512",
        search_mode="hnsw",
        top_k=2,
        enrollment_policy="official_all",
        enrollment_target=0,
        output_path=output,
        batch_size=1,
    )

    result = pd.read_csv(output)
    assert result["probe_type"].tolist() == [
        "registered",
        "registered",
        "unknown_unknown",
    ]
    assert result["protocol_index"].tolist() == [0, 1, 0]
    assert result["ranked_identities"].map(json.loads).map(len).tolist() == [2, 2, 2]
    assert summary["gallery_image_count"] == 4
    assert summary["gallery_identity_count"] == 2
    assert summary["known_unknown_count"] == 0
    assert len(_FakeRepository.template_calls) == 2


def test_search_matrix_combines_profiles_modes_and_progress(
    monkeypatch,
    tmp_path,
):
    import research.experiments.survface_pgvector as module

    manifest = _manifest().drop(columns=["_vector"])
    progress_events = []

    def fake_search(*args, **kwargs):
        frame = pd.DataFrame(
            {
                "probe_type": [
                    "registered",
                    "registered",
                    "unknown_unknown",
                ],
                "protocol_index": [0, 1, 0],
                "query_id": ["p-a", "p-b", "p-u"],
                "query_identity_id": ["a", "b", "unknown"],
                "ranked_identities": ['["a","b"]'] * 3,
                "ranked_distances": ["[0.1,0.2]"] * 3,
                "query_elapsed_ms": [0.5] * 3,
                "compression_profile": [kwargs["compression_profile"]] * 3,
                "search_mode": [kwargs["search_mode"]] * 3,
                "model_uid": ["model"] * 3,
            }
        )
        frame.to_csv(kwargs["output_path"], index=False)
        if kwargs["progress"] is not None:
            kwargs["progress"](
                "SurvFace official probes searched",
                {"processed": 3, "total": 3},
            )
        return {
            "rows": 3,
            "compression_profile": kwargs["compression_profile"],
            "search_mode": kwargs["search_mode"],
        }

    monkeypatch.setattr(module, "run_survface_official_search", fake_search)
    output = tmp_path / "matrix.csv"
    summary = run_survface_official_search_matrix(
        object(),
        run_uid="survface-run",
        manifest=manifest,
        compression_profiles=["origin_512", "pca_256"],
        search_modes=["exact", "hnsw"],
        top_k=2,
        enrollment_policy="official_all",
        enrollment_target=0,
        output_path=output,
        progress=lambda message, details: progress_events.append(
            (message, details)
        ),
    )

    result = pd.read_csv(output)
    assert len(result) == 12
    assert set(zip(result["compression_profile"], result["search_mode"])) == {
        ("origin_512", "exact"),
        ("origin_512", "hnsw"),
        ("pca_256", "exact"),
        ("pca_256", "hnsw"),
    }
    assert summary["combination_count"] == 4
    assert summary["all_official_complete"] is True
    assert [details["processed"] for _, details in progress_events] == [
        3,
        6,
        9,
        12,
    ]
    assert all(details["total"] == 12 for _, details in progress_events)
