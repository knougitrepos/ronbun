"""Complete LFW-deepfunneled population inputs without face re-detection.

All public images use whole-image bilinear resize. No sample fallback and no
test-dependent preprocessing selection. Completed sources are immutable.
"""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import gc
import json
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import __version__ as pillow_version
from threadpoolctl import threadpool_limits

from research.datasets.lfw_blufr import bind_image_manifest, load_blufr_lists
from research.embeddings.manifests import read_model_spec
from research.embeddings.registry import create_pytorch_adapter_from_spec
from research.experiments.calibration_matrix import MODEL_UIDS
from research.experiments.origin_pq_resources import check_resources
from research.experiments.origin_pq_storage import SplitStore
from research.fiqa import (
    CRFIQA_VARIANTS,
    load_cr_fiqa,
    load_fiqa_score_artifact,
    materialize_aligned_bundle_score_artifact,
)
from research.preprocessing.aligned_crops import (
    LFW_DEEPFUNNELED_RESIZE,
    materialize_aligned_crops,
    validate_aligned_crop_bundle,
)
from research.runtime.hashing import canonical_sha256, sha256_file

CONFIG_PATH = "configs/experiments/lfw_blufr_calibration_resize.yaml"
SOURCE_TYPE = "lfw_full_population_resize_embeddings"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _implementation():
    root = Path(__file__).resolve().parents[1]
    return {
        name: sha256_file(root / name)
        for name in (
            "experiments/lfw_resize_inputs.py",
            "preprocessing/aligned_crops.py",
            "embeddings/pytorch/adapter.py",
            "embeddings/pytorch/official_loaders.py",
            "embeddings/pytorch/official_backbones.py",
            "fiqa/artifacts.py",
            "fiqa/cr_fiqa.py",
        )
    }


def _bundle(root, config, manifest, progress):
    destination = root / config["resize_inputs"]["bundle_dir"]
    contract = dict(
        source_manifest_sha256=sha256_file(root / config["image_manifest"]),
        protocol_sha256=config["protocol_sha256"],
        preprocessing_mode=LFW_DEEPFUNNELED_RESIZE,
        pillow_version=pillow_version,
        implementation_sha256=sha256_file(
            Path(__file__).resolve().parents[1] / "preprocessing/aligned_crops.py"
        ),
    )
    if destination.exists():
        metadata = validate_aligned_crop_bundle(
            destination,
            dataset_id="lfw",
            expected_source_count=len(manifest),
            preprocessing_mode=LFW_DEEPFUNNELED_RESIZE,
            require_full_coverage=True,
        )
        if metadata.get("source_contract") != contract:
            raise ValueError(
                "existing resize bundle contract differs; select a new output path"
            )
        for entry in metadata["outputs"].values():
            if sha256_file(destination / entry["path"]) != entry["sha256"]:
                raise ValueError("resize bundle member hash mismatch")
        return destination
    staging = destination.with_name("." + destination.name + "-" + uuid4().hex)
    try:
        result = materialize_aligned_crops(
            manifest,
            project_root=root,
            output_dir=staging,
            dataset_id="lfw",
            preprocessing_mode=LFW_DEEPFUNNELED_RESIZE,
            require_full_coverage=True,
            overwrite=False,
        )
        metadata = {**result.bundle_manifest, "source_contract": contract}
        (staging / "bundle_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf8"
        )
        mmap = result.aligned_faces
        if getattr(mmap, "_mmap", None) is not None:
            mmap._mmap.close()
        staging.rename(destination)
        progress(f"LFW whole-image resize complete: {len(manifest)} / {len(manifest)}")
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def load_resize_source(project, directory):
    """Validate a dedicated source artifact; never impersonate a RunStore run."""
    root = (Path(project) / directory).resolve()
    if not (root / "_SUCCESS").is_file():
        raise FileNotFoundError(f"complete LFW resize inputs required: {root}")
    meta = _json(root / "manifest.json")
    if (
        meta.get("artifact_type") != SOURCE_TYPE
        or meta.get("status") != "completed"
        or meta.get("preprocessing_mode") != LFW_DEEPFUNNELED_RESIZE
    ):
        raise ValueError("expected a completed no-detection LFW source")
    for name, entry in meta["files"].items():
        if sha256_file(root / name) != entry["sha256"]:
            raise ValueError("source member hash mismatch: " + name)
    index = pd.read_csv(root / "population.csv")
    vectors = np.load(
        root / "normalized_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    norms = np.load(root / "raw_norms.npy", mmap_mode="r", allow_pickle=False)
    if (
        index.image_id.duplicated().any()
        or len(index) != meta["row_count"]
        or vectors.shape != (len(index), 512)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
        or norms.shape != (len(index),)
        or not np.isfinite(norms).all()
        or (norms <= 0).any()
    ):
        raise ValueError("invalid source population/embedding contract")
    lineage = meta["lineage"]
    prepared = SimpleNamespace(
        sample_ids=index.image_id.to_numpy(),
        identity_ids=index.identity_id.to_numpy(),
        normalized_embeddings=vectors,
        raw_norms=norms,
        model_uid=meta["model_uid"],
        extraction_uid=meta["source_uid"],
        origin_embedding_artifact_uid=meta["source_uid"],
        checkpoint_sha256=meta["checkpoint_sha256"],
        preprocess_hash=meta["preprocess_hash"],
    )
    lineage = dict(
        **lineage,
        source_run_id=meta["source_uid"],
        source_kind=SOURCE_TYPE,
        source_run_manifest_sha256=sha256_file(root / "manifest.json"),
        source_freeze_manifest_sha256=sha256_file(root / "manifest.json"),
        selected_manifest_sha256=sha256_file(root / "population.csv"),
        prepared_population_manifest_sha256=sha256_file(root / "manifest.json"),
        embedding_reused=True,
    )
    return root, meta, root, prepared, index, lineage


def _write_source(destination, index, vectors, norms, contract, lineage):
    staging = destination.with_name("." + destination.name + "-" + uuid4().hex)
    staging.mkdir(parents=True)
    try:
        index.to_csv(staging / "population.csv", index=False)
        np.save(staging / "normalized_embeddings.npy", vectors, allow_pickle=False)
        np.save(staging / "raw_norms.npy", norms, allow_pickle=False)
        metadata = dict(
            artifact_type=SOURCE_TYPE,
            status="completed",
            schema_version=1,
            contract=contract,
            source_uid="lfw-resize-" + canonical_sha256(contract)[:24],
            dataset_id="lfw",
            preprocessing_mode=LFW_DEEPFUNNELED_RESIZE,
            model_uid=contract["model_uid"],
            checkpoint_sha256=contract["checkpoint_sha256"],
            preprocess_hash=contract["preprocess_hash"],
            row_count=len(index),
            lineage=lineage,
            files={
                name: dict(sha256=sha256_file(staging / name))
                for name in (
                    "population.csv",
                    "normalized_embeddings.npy",
                    "raw_norms.npy",
                )
            },
        )
        (staging / "manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf8"
        )
        (staging / "_SUCCESS").write_text("complete\n", encoding="utf8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _extract_source(
    root, config, model, spec, bundle, device, batch_size, progress, resource_check
):
    import torch

    metadata = _json(bundle / "bundle_manifest.json")
    bundle_hash = sha256_file(bundle / "bundle_manifest.json")
    contract = dict(
        model_uid=spec.model_uid,
        checkpoint_sha256=spec.checkpoint.sha256,
        preprocess_hash=spec.preprocessing.preprocess_hash,
        model_spec=spec.to_manifest(),
        bundle_manifest_sha256=bundle_hash,
        implementation=_implementation(),
        python=sys.version,
        torch=torch.__version__,
        numpy=np.__version__,
        device=device,
        device_name=torch.cuda.get_device_name(),
        batch_size=batch_size,
    )
    destination = root / config["source_runs"][model]
    if destination.exists():
        _, meta, _, _, _, _ = load_resize_source(root, destination)
        if meta["contract"] != contract:
            raise ValueError(
                "completed source contract differs; use a new explicit path"
            )
        progress(f"{model}: completed resize source reused")
        return
    index = pd.read_csv(bundle / metadata["outputs"]["aligned_index"]["path"])
    population = index.rename(columns={"sample_id": "image_id"})
    faces = np.load(
        bundle / metadata["outputs"]["aligned_faces"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    lineage = dict(
        model_uid=spec.model_uid,
        aligned_bundle_manifest_sha256=bundle_hash,
        aligned_faces_sha256=metadata["outputs"]["aligned_faces"]["sha256"],
        checkpoint_sha256=spec.checkpoint.sha256,
        preprocess_hash=spec.preprocessing.preprocess_hash,
        input_preprocessing_mode=LFW_DEEPFUNNELED_RESIZE,
        checkpoint_training_overlap_verified=False,
    )
    shard_size = config["resize_inputs"]["embedding_shard_size"]
    parts = []
    adapter = None
    try:
        with SplitStore(root / config["resize_inputs"]["checkpoint_file"]) as store:
            for start in range(0, len(index), shard_size):
                stop = min(start + shard_size, len(index))
                key = dict(contract=contract, start=start, stop=stop)
                saved = store.get(key)
                if saved is not None:
                    part = saved[0]["embeddings"]
                else:
                    resource_check()
                    if adapter is None:
                        adapter = create_pytorch_adapter_from_spec(spec, device=device)
                    vectors, norms = [], []
                    for offset in range(start, stop, batch_size):
                        output = adapter.embed(
                            np.asarray(faces[offset : min(offset + batch_size, stop)])
                        )
                        vectors.append(output.normalized_embedding)
                        norms.append(output.raw_norm)
                    part = pd.DataFrame(
                        dict(
                            sample_id=index.iloc[start:stop].sample_id.to_numpy(),
                            vector=list(np.concatenate(vectors)),
                            raw_norm=np.concatenate(norms),
                        )
                    )
                    store.put(key, {"embeddings": part})
                if part.sample_id.tolist() != index.iloc[start:stop].sample_id.tolist():
                    raise ValueError("checkpoint sample order mismatch")
                parts.append(part)
                progress(f"{model}: {stop} / {len(index)} images")
        all_rows = pd.concat(parts, ignore_index=True)
        _write_source(
            destination,
            population,
            np.stack(all_rows.vector).astype("float32"),
            all_rows.raw_norm.to_numpy(dtype="float32"),
            contract,
            lineage,
        )
    finally:
        if adapter is not None:
            del adapter
        if getattr(faces, "_mmap", None) is not None:
            faces._mmap.close()
        gc.collect()
        torch.cuda.empty_cache()


def prepare_resize_inputs(
    project_root,
    *,
    config_path=CONFIG_PATH,
    models=None,
    execute=False,
    resize_only=False,
    device="cuda",
    batch_size=32,
    progress=print,
):
    """One full-population resize, four FR extractions, one CR-FIQA extraction."""
    import torch

    root = Path(project_root).resolve()
    config = yaml.safe_load((root / config_path).read_text(encoding="utf8"))
    if config.get("source_kind") != SOURCE_TYPE:
        raise ValueError("dedicated resize source config required")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive batch_size required")
    selected = tuple(config["source_runs"] if models is None else models)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(config["source_runs"])
    ):
        raise ValueError("unique configured models required")
    lists = load_blufr_lists(
        root / config["protocol_file"], expected_sha256=config["protocol_sha256"]
    )
    raw = pd.read_csv(root / config["image_manifest"])
    bound = bind_image_manifest(lists, raw)
    manifest = raw.set_index("image_id").loc[bound.image_id].reset_index()
    if (
        len(manifest) != 13233
        or not manifest.image_path.str.contains("lfw-deepfunneled", regex=False).all()
    ):
        raise ValueError("complete LFW-deepfunneled source required")
    manifest["split"] = (
        "population"  # Trial roles are assigned later; no historical split is used.
    )
    specs = {
        model: read_model_spec(root / config["resize_inputs"]["model_specs"][model])
        for model in selected
    }
    for model, spec in specs.items():
        if spec.model_uid != MODEL_UIDS[model]:
            raise ValueError("model checkpoint UID mismatch")
    variant = CRFIQA_VARIANTS[config["fiqa_variant"]]
    fiqa_checkpoint = root / config["resize_inputs"]["fiqa_checkpoint"]
    if (
        not fiqa_checkpoint.is_file()
        or sha256_file(fiqa_checkpoint) != variant.expected_sha256
    ):
        raise ValueError("official FIQA checkpoint missing or hash mismatch")
    cuda = bool(torch.cuda.is_available())
    state = dict(
        required_images=len(manifest),
        preprocessing_mode=LFW_DEEPFUNNELED_RESIZE,
        cuda_available=cuda,
        device_name=torch.cuda.get_device_name() if cuda else None,
        selected_models=selected,
        source_paths={m: config["source_runs"][m] for m in selected},
        execute=execute,
        resize_only=resize_only,
    )
    if not execute:
        return state
    if device != "cuda" or not cuda:
        raise RuntimeError(
            "full LFW extraction requires CUDA; no automatic CPU fallback"
        )

    def resource_check():
        return check_resources(minimum_available_gb=8.0, maximum_process_gb=16.0)

    resource_check()
    with threadpool_limits(limits=2):
        bundle = _bundle(root, config, manifest, progress)
        if resize_only:
            return dict(**state, bundle_dir=str(bundle), resize_completed=True)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            for model, spec in specs.items():
                resource_check()
                _extract_source(
                    root,
                    config,
                    model,
                    spec,
                    bundle,
                    device,
                    batch_size,
                    progress,
                    resource_check,
                )
            destination = root / config["fiqa_root"] / "lfw" / variant.model_uid
            if destination.exists():
                quality = load_fiqa_score_artifact(destination)
                if (
                    quality.manifest["aligned_bundle_manifest_sha256"]
                    != sha256_file(bundle / "bundle_manifest.json")
                    or quality.manifest["checkpoint_sha256"] != variant.expected_sha256
                    or len(quality.scores) != len(manifest)
                ):
                    raise ValueError(
                        "existing FIQA artifact does not match full resized population"
                    )
                progress("FIQA: completed resize scores reused")
            else:
                resource_check()
                fiqa_model, loaded = load_cr_fiqa(
                    fiqa_checkpoint, variant=variant.variant, device=device
                )
                try:
                    materialize_aligned_bundle_score_artifact(
                        bundle,
                        destination,
                        model=fiqa_model,
                        model_uid=loaded.model_uid,
                        checkpoint_sha256=loaded.expected_sha256,
                        variant=loaded.variant,
                        device=device,
                        batch_size=batch_size,
                        shard_size=4096,
                        overwrite=False,
                    )
                    progress(f"FIQA: {len(manifest)} / {len(manifest)} images complete")
                finally:
                    del fiqa_model
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            torch.set_num_threads(old_threads)
    return dict(**state, completed=True, bundle_dir=str(bundle))
