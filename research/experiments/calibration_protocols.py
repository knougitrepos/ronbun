"""Reconstruct the frozen Step-4 protocols without resampling test cohorts."""

from pathlib import Path

from research.datasets import adapt_rfw_custom_manifest_to_open_set_protocol
from research.protocols.open_set import (
    build_calibration_protocol,
    build_group_matched_calibration_protocol,
    build_open_set_protocol,
    build_survface_matched_calibration_protocol,
    build_survface_official_protocol,
)

OPEN_SET_DATASETS = ("lfw", "rfw_custom", "survface")


def calibration_protocol(run, population, split, seed):
    """Use the original dataset-specific enrollment and group-matching rules."""
    dataset = run["config"]["dataset_id"]
    config = run["config"]["step4"]
    evaluation = config["evaluation"]
    if dataset not in OPEN_SET_DATASETS or split not in ("calibration", "test"):
        raise ValueError("unsupported open-set dataset/split")
    if dataset == "survface":
        if split == "calibration":
            return build_survface_matched_calibration_protocol(
                population, seed=seed,
                gallery_identity_count=int(evaluation["survface_calibration_gallery_identities"]))
        return build_survface_official_protocol(population.loc[population.protocol_role.isin(
            {"gallery", "registered_probe", "unknown_unknown_probe"})].copy())
    if dataset == "rfw_custom":
        test = adapt_rfw_custom_manifest_to_open_set_protocol(population)
        if split == "test":
            return test
        counts = test.gallery.groupby("rfw_group").identity_id.nunique().to_dict()
        if sum(counts.values()) != len(test.gallery):
            raise ValueError("RFW-Custom requires one enrollment image per identity")
        return build_group_matched_calibration_protocol(
            population, split_name="calibration", gallery_identity_count_by_group=counts,
            enrollment_count=1, seed=seed, group_column="rfw_group")
    if split == "calibration":
        sizes = population.loc[population.split.eq("calibration")].groupby("identity_id").image_id.nunique()
        count = min(int(evaluation["calibration_gallery_identities"]), max(1, int((sizes > 1).sum())))
        return build_calibration_protocol(population, split_name="calibration",
                                          gallery_identity_count=count, enrollment_count=1, seed=seed)
    project = Path(__file__).resolve().parents[2]

    def read_ids(key):
        return tuple(line.strip() for line in (project / config["datasets"]["lfw"][key])
                     .read_text(encoding="utf8").splitlines() if line.strip())

    return build_open_set_protocol(
        population, read_ids("gallery_identities_path"), read_ids("unknown_unknown_identities_path"),
        enrollment_count=int(evaluation["lfw_enrollment_count"]), seed=seed)
