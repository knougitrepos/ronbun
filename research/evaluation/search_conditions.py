from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class SearchCondition:
    search_mode: str
    compression_family: str
    query_representation: str
    gallery_representation: str
    distance_function: str
    compressed_score_space: str
    score_spaces_comparable: bool
    frozen_origin_threshold_applicable: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_CONDITIONS = {
    "pca_direct_cosine": SearchCondition(
        search_mode="pca_direct_cosine",
        compression_family="pca",
        query_representation="pca_projected_float32",
        gallery_representation="pca_projected_float32",
        distance_function="cosine_similarity",
        compressed_score_space="cosine_similarity",
        score_spaces_comparable=True,
        frozen_origin_threshold_applicable=True,
    ),
    "pca_reconstruction_cosine": SearchCondition(
        search_mode="pca_reconstruction_cosine",
        compression_family="pca",
        query_representation="pca_reconstructed_float32",
        gallery_representation="pca_reconstructed_float32",
        distance_function="cosine_similarity",
        compressed_score_space="cosine_similarity",
        score_spaces_comparable=True,
        frozen_origin_threshold_applicable=True,
    ),
    "pq_reconstruction_cosine": SearchCondition(
        search_mode="pq_reconstruction_cosine",
        compression_family="pq",
        query_representation="pq_reconstructed_float32",
        gallery_representation="pq_reconstructed_float32",
        distance_function="cosine_similarity",
        compressed_score_space="cosine_similarity",
        score_spaces_comparable=True,
        frozen_origin_threshold_applicable=True,
    ),
    "pq_one_sided_cosine": SearchCondition(
        search_mode="pq_one_sided_cosine",
        compression_family="pq",
        query_representation="origin_float32",
        gallery_representation="pq_reconstructed_float32",
        distance_function="cosine_similarity",
        compressed_score_space="cosine_similarity",
        score_spaces_comparable=True,
        frozen_origin_threshold_applicable=True,
    ),
    "pq_adc_exhaustive": SearchCondition(
        search_mode="pq_adc_exhaustive",
        compression_family="pq",
        query_representation="origin_float32",
        gallery_representation="pq_code",
        distance_function="squared_l2_asymmetric",
        compressed_score_space="negative_squared_l2_adc",
        score_spaces_comparable=False,
        frozen_origin_threshold_applicable=False,
    ),
    "pq_sdc_exhaustive": SearchCondition(
        search_mode="pq_sdc_exhaustive",
        compression_family="pq",
        query_representation="pq_code",
        gallery_representation="pq_code",
        distance_function="squared_l2_symmetric",
        compressed_score_space="negative_squared_l2_sdc",
        score_spaces_comparable=False,
        frozen_origin_threshold_applicable=False,
    ),
}

SEARCH_CONDITIONS: Mapping[str, SearchCondition] = MappingProxyType(_CONDITIONS)
PCA_SEARCH_MODES = tuple(
    mode for mode, condition in _CONDITIONS.items()
    if condition.compression_family == "pca"
)
PQ_SEARCH_MODES = tuple(
    mode for mode, condition in _CONDITIONS.items()
    if condition.compression_family == "pq"
)
ALL_SEARCH_MODES = (*PCA_SEARCH_MODES, *PQ_SEARCH_MODES)
CROSS_SCORE_SPACE_SEARCH_MODES = tuple(
    mode for mode, condition in _CONDITIONS.items()
    if not condition.score_spaces_comparable
)


def search_condition(search_mode: str) -> SearchCondition:
    mode = str(search_mode).strip()
    try:
        return SEARCH_CONDITIONS[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported search_mode: {mode!r}") from exc


def search_condition_metadata(search_mode: str) -> dict[str, object]:
    condition = search_condition(search_mode)
    return {
        "query_representation": condition.query_representation,
        "gallery_representation": condition.gallery_representation,
        "distance_function": condition.distance_function,
        "compressed_score_space": condition.compressed_score_space,
        "score_spaces_comparable": condition.score_spaces_comparable,
        "frozen_origin_threshold_applicable": (
            condition.frozen_origin_threshold_applicable
        ),
    }


def threshold_policies_for_search_mode(search_mode: str) -> tuple[str, ...]:
    condition = search_condition(search_mode)
    if condition.frozen_origin_threshold_applicable:
        return ("frozen_origin", "recalibrated_compressed")
    return ("recalibrated_compressed",)
