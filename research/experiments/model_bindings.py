"""Explicit checkpoint identity and dimension bindings for configurable workflows."""

from research.experiments.calibration_matrix import MODEL_UIDS


def model_bindings(config):
    """Keep historical 512D configs readable; new models require pinned UIDs."""
    dimension = config.get("embedding_dim", 512)
    if type(dimension) is not int or dimension not in (512, 1024):
        raise ValueError("embedding_dim must be 512 or 1024")
    aliases = config["source_runs"]
    bindings = config.get("model_uids")
    if bindings is None:
        if dimension != 512 or not set(aliases) <= set(MODEL_UIDS):
            raise ValueError("new checkpoints require explicit model_uids")
        bindings = {name: MODEL_UIDS[name] for name in aliases}
    if (not isinstance(bindings, dict) or set(bindings) != set(aliases)
            or any(not isinstance(uid, str) or not uid.strip() for uid in bindings.values())
            or len(set(bindings.values())) != len(bindings)):
        raise ValueError("one unique pinned model_uid is required per source alias")
    return dimension, bindings
