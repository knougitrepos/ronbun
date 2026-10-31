"""Malformed RFW pairs must fail before embedding extraction or calibration."""
import pandas as pd
import pytest

from research.protocols.facex_handoff import load_pair_dataset


@pytest.fixture
def protocol(tmp_path):
    images, pairs = [], []
    for group in ("African", "Asian", "Caucasian", "Indian"):
        for name, identity in (("a", "person1"), ("b", "person1"), ("c", "person2")):
            images.append(dict(image_id=f"{group}:{name}", identity_id=f"{group}:{identity}", image_path=f"{group}/{name}.jpg"))
        for fold in range(10):
            for i in range(600):
                genuine = i < 300
                pairs.append(dict(pair_id=f"{group}:{fold}:{i}", fold_index=fold, rfw_group=group,
                                  left_image_id=f"{group}:a", right_image_id=f"{group}:{'b' if genuine else 'c'}",
                                  left_identity_id=f"{group}:person1", right_identity_id=f"{group}:person{1 if genuine else 2}",
                                  is_genuine=genuine))
    pop, frame = pd.DataFrame(images), pd.DataFrame(pairs)
    pop.to_csv(tmp_path / "images.csv", index=False)
    frame.to_csv(tmp_path / "pairs.csv", index=False)
    return tmp_path, dict(image_manifest="images.csv", pairs_file="pairs.csv"), frame


def test_complete_four_group_ten_fold_protocol_is_accepted(protocol):
    root, config, _ = protocol
    _, pairs = load_pair_dataset(root, config, "rfw")
    assert len(pairs) == 24000 and pairs.groupby(["group", "fold"]).ngroups == 40
    assert pairs.is_genuine.sum() == 12000


@pytest.mark.parametrize("damage,match", [
    ("swapped_labels", "contradicts"), ("missing_fold", "all ten"),
    ("missing_group", "four official"), ("empty_identity", "Empty"),
    ("identity_mismatch", "image/identity mismatch"), ("empty_image", "Empty"),
    ("self_pair", "Self-pair"), ("missing_column", "required columns"),
    ("duplicate_pair", "Duplicate"), ("bad_count", "300 genuine"),
])
def test_invalid_protocol_is_rejected_at_load(protocol, damage, match):
    root, config, pairs = protocol
    if damage == "swapped_labels":
        pairs.loc[[0, 300], "is_genuine"] = [False, True]
    elif damage == "missing_fold":
        pairs = pairs.iloc[600:]
    elif damage == "missing_group":
        pairs = pairs.loc[pairs.rfw_group.ne("African")]
    elif damage == "empty_identity":
        pairs.loc[0, "left_identity_id"] = " "
    elif damage == "empty_image":
        pairs.loc[0, "left_image_id"] = ""
    elif damage == "identity_mismatch":
        pairs.loc[0, ["left_identity_id", "right_identity_id"]] = "unknown"
    elif damage == "self_pair":
        pairs.loc[0, "right_image_id"] = pairs.loc[0, "left_image_id"]
    elif damage == "missing_column":
        pairs = pairs.drop(columns="fold_index")
    elif damage == "duplicate_pair":
        pairs.loc[0, "pair_id"] = pairs.loc[1, "pair_id"]
    else:
        pairs = pairs.iloc[1:]
    pairs.to_csv(root / config["pairs_file"], index=False)
    with pytest.raises(ValueError, match=match):
        load_pair_dataset(root, config, "rfw")


@pytest.mark.parametrize("value", ["", "1.5", "nan", "-1", "10", "True"])
def test_noncanonical_fold_is_rejected(protocol, value):
    root, config, pairs = protocol
    pairs["fold_index"] = pairs.fold_index.astype(str)
    pairs.loc[0, "fold_index"] = value
    pairs.to_csv(root / config["pairs_file"], index=False)
    with pytest.raises(ValueError, match="fold_index"):
        load_pair_dataset(root, config, "rfw")
