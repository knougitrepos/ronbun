"""Cold-start LFW inputs from raw images and checkpoints, without legacy splits.

The resize/extraction primitives are shared with the older population extractor;
its BLUFR preparation entry point and model registry are never used here.
"""

from pathlib import Path
import gc
import json
import shutil

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from research.datasets.lfw_pairs import load_pairs, PAIRS_SHA256
from research.embeddings.manifests import write_model_spec
from research.experiments.lfw_baselines import model_specs
from research.experiments.lfw_resize_inputs import _bundle, _extract_source, load_resize_source
from research.experiments.origin_pq_resources import check_resources
from research.fiqa import (CRFIQA_VARIANTS, load_cr_fiqa, load_fiqa_score_artifact,
                           materialize_aligned_bundle_score_artifact)
from research.runtime.hashing import sha256_file

CONFIG_PATH = "configs/experiments/lfw_pair_verification.yaml"


def population_from_raw(root, config):
    """Hash the actual full population, assigning no training/test roles yet."""
    root = Path(root).resolve()
    folder = (root / config["inputs"]["raw_image_root"]).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Extract LFW-deepfunneled images to {folder}")
    records = []
    for path in sorted(folder.glob("*/*.jpg")):
        person = path.parent.name
        suffix = path.stem.removeprefix(person + "_")
        if path.stem != f"{person}_{suffix}" or len(suffix) != 4 or not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(f"invalid LFW image name: {path}")
        try:
            image_path = path.relative_to(root).as_posix()
        except ValueError:
            image_path = path.as_posix()
        records.append(dict(image_id=f"lfw:{person}:{path.stem}", identity_id=f"lfw:{person}",
            split="population", image_path=image_path, source_content_sha256=sha256_file(path)))
    frame = pd.DataFrame(records)
    expected = config["inputs"]
    if len(frame) != expected["expected_images"] or frame.identity_id.nunique() != expected["expected_identities"]:
        raise ValueError("full LFW population count mismatch; no implicit sample exclusion")
    if frame.image_id.duplicated().any():
        raise ValueError("duplicate LFW source image")
    return frame


def _publish_population(path, population):
    payload = population.to_csv(index=False, lineterminator="\n").encode("utf8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("raw population changed; choose a new image_manifest and input output paths")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)


def prepare_pair_inputs(project_root, *, config_path=CONFIG_PATH, models=None, execute=False,
                        download_pairs=False, resize_only=False, device="cuda", batch_size=32,
                        keep_raw_results=None, progress=print):
    """Inspect or build all FR/FIQA inputs. No old manifest/registry/MAT is read.

    Core features are retained as reusable evaluation inputs. keep_raw_results
    controls the redundant FR shard checkpoint, removed only after verification.
    """
    import torch
    root = Path(project_root).resolve()
    config = yaml.safe_load((root / config_path).read_text(encoding="utf8"))
    selected = tuple(config["source_runs"] if models is None else models)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(config["source_runs"]):
        raise ValueError("unique configured models required")
    keep = config["keep_raw_results"] if keep_raw_results is None else keep_raw_results
    if type(batch_size) is not int or batch_size < 1 or type(keep) is not bool:
        raise ValueError("invalid batch size or retention setting")
    progress = progress or (lambda message: None)
    population = population_from_raw(root, config)
    pairs, development = load_pairs(root / config["pairs_file"], population, download=download_pairs)
    specs = model_specs(root, config, selected)
    manifest_path = root / config["image_manifest"]
    if manifest_path.exists():
        _publish_population(manifest_path, population)  # read-only consistency check
    variant = CRFIQA_VARIANTS[config["fiqa_variant"]]
    checkpoint = root / config["resize_inputs"]["fiqa_checkpoint"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != variant.expected_sha256:
        raise ValueError("official FIQA checkpoint missing or hash mismatch")
    cuda = bool(torch.cuda.is_available())
    coverage = pd.DataFrame([dict(model=m, embeddings_present=(root / config["source_runs"][m] / "_SUCCESS").is_file()) for m in selected])
    destination = root / config["fiqa_root"] / "lfw" / variant.model_uid
    new_extraction = not bool(coverage.embeddings_present.all() and (destination / "manifest.json").is_file())
    ready = bool(coverage.embeddings_present.all() and (destination / "manifest.json").is_file()
                 and (root / config["image_manifest"]).is_file())
    state = dict(ready=ready, coverage=coverage, required_images=len(population),
        identities=int(population.identity_id.nunique()), development_images=len(development),
        development_identities=int(development.identity_id.nunique()), pairs=len(pairs),
        cuda_available=cuda, device_name=torch.cuda.get_device_name() if cuda else None,
        source_paths={m: config["source_runs"][m] for m in selected},
        keep_raw_results=keep, reusable_inputs_retained=True, execute=execute)
    if not execute:
        if ready:
            quality = load_fiqa_score_artifact(destination)
            if len(quality.scores) != len(population):
                raise ValueError("FIQA does not cover the full population")
            for alias in selected:
                _, metadata, _, _, _, lineage = load_resize_source(root, config["source_runs"][alias])
                if (metadata["model_uid"] != specs[alias].model_uid or metadata["row_count"] != len(population)
                        or lineage["aligned_bundle_manifest_sha256"] != quality.manifest["aligned_bundle_manifest_sha256"]):
                    raise ValueError("existing input checkpoint/coverage/FIQA lineage mismatch")
        return state
    if device != "cuda" or not cuda:
        raise RuntimeError("full LFW extraction requires CUDA; no automatic CPU fallback")
    for output in [config["resize_inputs"]["bundle_dir"], config["fiqa_root"], *state["source_paths"].values()]:
        ancestor = root / output
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < 4 * 2**30:
            raise RuntimeError(f"at least 4 GiB free disk required for LFW input preparation: {ancestor}")
    shard = (root / config["resize_inputs"]["checkpoint_file"]).resolve()
    input_root = (root / config["inputs"]["registry_root"]).resolve().parent
    if not shard.is_relative_to(input_root) or shard.suffix != ".sqlite3":
        raise ValueError("extraction checkpoint must be a .sqlite3 inside the configured input root")
    _publish_population(manifest_path, population)
    for spec in specs.values():
        write_model_spec(root / config["inputs"]["registry_root"] / f"{spec.model_uid}.json", spec)
    extraction_config = dict(config, protocol_sha256=PAIRS_SHA256)
    resource_check = lambda: check_resources(minimum_available_gb=8., maximum_process_gb=16.)
    resource_check()
    with threadpool_limits(limits=2):
        bundle = _bundle(root, extraction_config, population, progress)
        if resize_only:
            return dict(state, bundle_dir=str(bundle), resize_completed=True)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            for model, spec in specs.items():
                _extract_source(root, extraction_config, model, spec, bundle, device, batch_size, progress, resource_check)
            if destination.exists():
                quality = load_fiqa_score_artifact(destination)
                if (quality.manifest["aligned_bundle_manifest_sha256"] != sha256_file(bundle / "bundle_manifest.json")
                        or quality.manifest["checkpoint_sha256"] != variant.expected_sha256
                        or len(quality.scores) != len(population)):
                    raise ValueError("FIQA input lineage differs from resized population")
                progress("FIQA: completed pair inputs reused")
            else:
                resource_check()
                model, loaded = load_cr_fiqa(checkpoint, variant=variant.variant, device=device)
                try:
                    materialize_aligned_bundle_score_artifact(bundle, destination, model=model,
                        model_uid=loaded.model_uid, checkpoint_sha256=loaded.expected_sha256,
                        variant=loaded.variant, device=device, batch_size=batch_size, shard_size=4096, overwrite=False)
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                load_fiqa_score_artifact(destination)
                progress(f"FIQA: {len(population)} images complete")
        finally:
            torch.set_num_threads(old_threads)
    # Verify published inputs before declaring completion or pruning resume data.
    source_hashes = {}
    for model in selected:
        _, metadata, _, _, _, _ = load_resize_source(root, config["source_runs"][model])
        if metadata["row_count"] != len(population):
            raise ValueError("published FR source is not the full population")
        source_hashes[model] = sha256_file(root / config["source_runs"][model] / "manifest.json")
    coverage = coverage.assign(embeddings_present=True)
    all_sources_complete = all((root / path / "_SUCCESS").is_file() for path in config["source_runs"].values())
    prune_shard = not keep and new_extraction and all_sources_complete
    record = dict(state, ready=True, completed=True, status="completed", coverage=coverage, bundle_dir=str(bundle),
        input_policy="retain reusable population, resized images, embeddings, FIQA and registry; prune redundant FR shards only",
        source_manifest_sha256=source_hashes, fiqa_manifest_sha256=sha256_file(destination / "manifest.json"),
        expected_models=len(selected), completed_models=len(selected), failed_images=0,
        torch_version=torch.__version__, device=device, batch_size=batch_size,
        extraction_checkpoint_cleanup_eligible=prune_shard,
        extraction_checkpoint_retained=bool(shard.is_file() and not prune_shard),
        pairs_sha256=PAIRS_SHA256, image_manifest_sha256=sha256_file(root / config["image_manifest"]),
        config_sha256=sha256_file(root / config_path), implementation_sha256=sha256_file(Path(__file__)))
    # Completion metadata belongs to this preparation request, not the frozen source.
    from research.runtime.hashing import canonical_sha256
    serializable = {k: v for k, v in record.items() if k != "coverage"}
    audit = root / config["inputs"]["registry_root"] / ("preparation-" + canonical_sha256(serializable)[:24] + ".json")
    if not audit.exists():
        audit.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf8")
    if json.loads(audit.read_text(encoding="utf8")) != serializable:
        raise ValueError("input summary verification failed")
    guide = Path(__file__).resolve().parents[2] / "notebooks/calibration/LFW_PAIR_VERIFICATION.md"
    guide_copy = audit.with_suffix(".md")
    if not guide_copy.exists():
        guide_copy.write_text("# LFW input preparation\n\n" + f"Summary: `{audit.name}`\n\n" + guide.read_text(encoding="utf8"), encoding="utf8")
    # Only this explicitly configured redundant checkpoint is eligible for cleanup.
    if prune_shard and shard.is_file():
        if shard in (checkpoint.resolve(), manifest_path.resolve()):
            raise ValueError("unsafe shard cleanup path")
        shard.unlink()
    return record


def run_pair_workflow(project_root, *, config_path=CONFIG_PATH, execute=False, models=None,
                      download_pairs=False, device="cuda", batch_size=32, keep_raw_results=None,
                      progress=None, **evaluation):
    """Single notebook/CLI entry: raw inputs -> pairs -> PQ -> calibration -> report."""
    from research.experiments.lfw_pair_verification import run_verification
    # Validate the requested experiment before any costly input preparation.
    from research.experiments.lfw_pair_verification import validate_selection
    config = yaml.safe_load((Path(project_root) / config_path).read_text(encoding="utf8"))
    validate_selection(config, models=models, fold_ids=evaluation.get("fold_ids"), partition_seeds=evaluation.get("partition_seeds"))
    if evaluation.get("max_new_jobs") is not None and (type(evaluation["max_new_jobs"]) is not int or evaluation["max_new_jobs"] < 1):
        raise ValueError("max_new_jobs must be a positive integer or None")
    inputs = prepare_pair_inputs(project_root, config_path=config_path, models=models, execute=execute,
        download_pairs=download_pairs, device=device, batch_size=batch_size,
        keep_raw_results=keep_raw_results, progress=progress)
    if execute or inputs["ready"]:
        result = run_verification(project_root, config_path=config_path, models=models, execute=execute,
            download_pairs=download_pairs, keep_raw_results=keep_raw_results, progress=progress, **evaluation)
        return dict(result, input_preparation={k:v for k,v in inputs.items() if k != "coverage"})
    config = yaml.safe_load((Path(project_root) / config_path).read_text(encoding="utf8"))
    folds = evaluation.get("fold_ids") or config["fold_ids"]
    seeds = evaluation.get("partition_seeds") or config["partition_seeds"]
    return dict(inputs, inventory=pd.DataFrame(), expected_jobs=len(inputs["source_paths"])*len(folds)*len(seeds),
                next_step="execute=True builds missing inputs before evaluation; no legacy preparation required")
