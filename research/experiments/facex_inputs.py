"""CUDA inputs shared by the five-dataset FaceX workflow."""
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from research.runtime.hashing import canonical_sha256, sha256_file
from research.runtime.source_fingerprints import source_fingerprints


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def population_hash(population):
    return canonical_sha256(population.fillna("").astype(str).to_dict("records"))


def input_transform_from_metadata(metadata):
    """Consume a declared export policy; never infer a missing resize recipe."""
    policy = metadata.get("preprocessing", {}).get("alignment_policy")
    if (not isinstance(policy, dict) or policy.get("method") != "resize_only"
            or policy.get("interpolation") not in ("bicubic", "bilinear")
            or policy.get("input_size") != [112, 112] or policy.get("color") != "BGR"
            or policy.get("geometry") != "full_crop_stretch" or policy.get("face_detection") != "none"):
        raise ValueError("Export must declare supported resize_only bicubic/bilinear 112x112 BGR preprocessing")
    return dict(version="facex_export_resize_v2", method="resize_only",
                interpolation=policy["interpolation"], input_size=[112, 112],
                source_color_order="rgb", model_color_order="bgr",
                geometry="full_crop_stretch", face_detection="none")


def resolve_input_transform(spec):
    # Only files are consumed. No FaceX-Zoo checkout/import is needed at runtime.
    from research.embeddings.facex_zoo import _read_validated_export, _spec_from_metadata
    path, metadata, _ = _read_validated_export(spec.checkpoint.path)
    if spec.model_uid != _spec_from_metadata(path, metadata, spec.preprocessing.source_color_order).model_uid:
        raise ValueError("Input model spec differs from its export")
    return input_transform_from_metadata(metadata)


def input_path(root, uid, dataset, population_digest, quality, transform):
    fingerprint = canonical_sha256(dict(population=population_digest, quality=quality,
                                       code=input_code_hashes(), transform=transform))
    return Path(root) / "inputs" / uid / dataset / fingerprint[:20]


def input_code_hashes():
    return source_fingerprints("research.experiments.facex_inputs", "research.embeddings.pytorch.facex_zoo")


def archive_order(population, project_root):
    """Read gzip-tar members monotonically, avoiding repeated decompression.

    Pair IDs/folds remain unchanged; evaluation joins inputs by image ID.
    """
    offsets = {}
    for value in population.image_path:
        if not str(value).startswith("tar://"):
            continue
        archive, _ = str(value)[6:].split("#", 1)
        if archive not in offsets:
            with tarfile.open(Path(project_root) / archive, "r:*") as handle:
                offsets[archive] = {m.name: m.offset_data for m in handle if m.isfile()}
    if not offsets:
        return population
    def order(value):
        if not str(value).startswith("tar://"):
            return str(value), -1
        archive, member = str(value)[6:].split("#", 1)
        return archive, offsets[archive][member]
    indices = sorted(range(len(population)), key=lambda i: order(population.iloc[i].image_path))
    return population.iloc[indices].reset_index(drop=True)


class ImageReader:
    def __init__(self, project_root, *, interpolation="bicubic"):
        self.root = Path(project_root)
        self.archives = {}
        if interpolation not in ("bicubic", "bilinear"):
            raise ValueError("Unsupported resize interpolation")
        self.interpolation = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR}[interpolation]

    def read_bytes(self, path):
        """Consume ordered tar members once, with explicit reopen for backward reads.

        Passing TarInfo to extractfile avoids getmember's full-list scan. RFW's
        archive_order gives a forward-only pass even for gzip-compressed input.
        """
        if str(path).startswith("tar://"):
            archive, member = str(path)[6:].split("#", 1)
            archive = (self.root / archive).resolve()
            if archive not in self.archives:
                self.archives[archive] = tarfile.open(archive, "r|*")
            for attempt in range(2):
                handle = self.archives[archive]
                while (info := handle.next()) is not None:
                    if info.name == member:
                        if not info.isfile():
                            raise ValueError(f"Archive member is not a regular image: {path}")
                        with handle.extractfile(info) as stream:
                            return stream.read()
                if attempt == 0:
                    handle.close()
                    self.archives[archive] = tarfile.open(archive, "r|*")
            raise FileNotFoundError(f"Archive member not found: {path}")
        else:
            return (self.root / path).read_bytes()

    def read(self, path):
        payload = self.read_bytes(path)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image decode failed: {path}")
        image = cv2.cvtColor(cv2.resize(image, (112, 112), interpolation=self.interpolation), cv2.COLOR_BGR2RGB)
        return image, hashlib.sha256(payload).hexdigest()

    def close(self):
        for archive in self.archives.values():
            archive.close()


def verify_raw_inputs(project, population, ledger, *, reader=None):
    """Strict byte validation on cache reuse; does not decode or run a model."""
    owned = reader is None
    reader = reader or ImageReader(project)
    try:
        lines = Path(ledger).read_text(encoding="utf-8").splitlines()
        if len(lines) != len(population):
            raise ValueError("Raw input hash ledger coverage differs")
        for row, line in zip(population.itertuples(index=False), lines):
            record = json.loads(line)
            if str(record["image_id"]) != str(row.image_id):
                raise ValueError("Raw input hash ledger ordering differs")
            if hashlib.sha256(reader.read_bytes(row.image_path)).hexdigest() != record["sha256"]:
                raise ValueError("Raw input changed; preserve the old artifact and use a new output root for intentional replacements")
    finally:
        if owned:
            reader.close()


def quality_cache_path(root, dataset, population_digest, quality, transform):
    contract = dict(population_hash=population_digest, quality=quality, transform=transform,
                    code=source_fingerprints("research.experiments.facex_inputs", "research.fiqa.cr_fiqa"))
    return Path(root) / "quality" / dataset / canonical_sha256(contract)[:24], contract


def read_quality_cache(path, contract, count):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not (path / "COMPLETED").is_file() or manifest["contract"] != contract:
        raise ValueError("Quality cache is incomplete or has different provenance")
    for name in ("quality.npy", "image_hashes.jsonl"):
        if sha256_file(path / name) != manifest["files"][name]:
            raise ValueError(f"Quality cache changed: {name}")
    quality = np.load(path / "quality.npy", allow_pickle=False)
    hashes = [json.loads(line) for line in (path / "image_hashes.jsonl").read_text(encoding="utf-8").splitlines()]
    if quality.shape != (count,) or not np.isfinite(quality).all() or len(hashes) != count:
        raise ValueError("Quality cache coverage/values differ")
    return quality, hashes


def publish_quality_cache(path, contract, inputs_dir, count):
    """Publish only after a complete extraction; keep shared quality independent of FR weights."""
    if path.exists():
        read_quality_cache(path, contract, count)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=path.name + ".partial-", dir=path.parent))
    try:
        files = ("quality.npy", "image_hashes.jsonl")
        for name in files:
            shutil.copyfile(inputs_dir / name, work / name)
        write_json(work / "manifest.json", dict(status="completed", contract=contract,
                   source_input=str(inputs_dir), source_manifest_sha256=sha256_file(inputs_dir / "manifest.json"),
                   files={name: sha256_file(work / name) for name in files}))
        (work / "COMPLETED").write_text("completed\n", encoding="utf-8")
        try:
            work.rename(path)
        except FileExistsError:
            read_quality_cache(path, contract, count)
    finally:
        if work.exists():
            # Only this function's fresh temporary directory is eligible.
            work.resolve().relative_to(path.parent.resolve())
            shutil.rmtree(work)


def read_inputs(path, *, expected_uid=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not (path / "COMPLETED").is_file():
        raise ValueError("Input artifact is incomplete")
    if expected_uid and manifest["model_uid"] != expected_uid:
        raise ValueError("Input checkpoint mismatch")
    for name, digest in manifest["files"].items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"Input file changed: {name}")
    population = pd.read_csv(path / "population.csv", keep_default_na=False,
                             dtype={"image_id": str, "identity_id": str, "source_identity_id": str})
    vectors = np.load(path / "embeddings.npy", mmap_mode="r")
    norms = np.load(path / "raw_norm.npy", mmap_mode="r")
    quality = np.load(path / "quality.npy", mmap_mode="r") if "quality.npy" in manifest["files"] else None
    if (vectors.shape != (len(population), 1024) or norms.shape != (len(population),)
            or quality is not None and quality.shape != norms.shape
            or quality is None and manifest.get("quality", {}).get("source") != "not_required"):
        raise ValueError("Input shape/coverage mismatch")
    return manifest, population, vectors, norms, quality


def materialize_inputs(project, population, spec, dataset, root, quality_config, *, batch_size=16, quality_required=True):
    from research.embeddings.registry import create_pytorch_adapter_from_spec
    import torch

    if spec.embedding_dim != 1024 or spec.preprocessing.source_color_order != "rgb":
        raise ValueError("New workflow requires a registered 1024D RGB-source spec")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal input extraction requires CUDA; no silent CPU fallback")
    if batch_size < 1:
        raise ValueError("Positive batch size required")
    transform = resolve_input_transform(spec)
    population = population.reset_index(drop=True)
    population_digest = population_hash(population)
    quality_config = dict(quality_config) if quality_required else dict(source="not_required")
    source = quality_config.get("source", "cr_fiqa")
    fiqa_model = None
    if source == "cr_fiqa":
        from research.fiqa.cr_fiqa import load_cr_fiqa
        checkpoint = Path(project) / quality_config["checkpoint"]
        quality_config["checkpoint_sha256"] = sha256_file(checkpoint)
    elif source != "raw_norm" and quality_required:
        raise ValueError("Choose cr_fiqa or an explicitly declared raw_norm experiment")
    destination = input_path(root, spec.model_uid, dataset, population_digest, quality_config, transform)
    if destination.exists():
        read_inputs(destination, expected_uid=spec.model_uid)
        verify_raw_inputs(project, population, destination / "image_hashes.jsonl")
        return destination
    shared_quality = shared_hashes = quality_path = quality_contract = None
    if source == "cr_fiqa":
        quality_path, quality_contract = quality_cache_path(root, dataset, population_digest, quality_config, transform)
        if quality_path.exists():
            shared_quality, shared_hashes = read_quality_cache(quality_path, quality_contract, len(population))
    work = destination.with_name(destination.name + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    progress_path = work / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress["population_hash"] != population_digest or progress["model_uid"] != spec.model_uid
                or progress.get("transform") != transform):
            raise ValueError("Interrupted input lineage changed")
        start, mode = progress["completed_images"], "r+"
    else:
        start, mode = 0, "w+"
        population.to_csv(work / "population.csv", index=False)
    arrays = {
        "embeddings.npy": np.lib.format.open_memmap(work / "embeddings.npy", mode=mode, dtype=np.float32, shape=(len(population), 1024)),
        "raw_norm.npy": np.lib.format.open_memmap(work / "raw_norm.npy", mode=mode, dtype=np.float32, shape=(len(population),)),
    }
    if quality_required:
        arrays["quality.npy"] = np.lib.format.open_memmap(work / "quality.npy", mode=mode, dtype=np.float32, shape=(len(population),))
    reader = ImageReader(project, interpolation=transform["interpolation"])
    try:
        if start:
            # Verify the raw bytes of previously checkpointed images before resume.
            lines = (work / "image_hashes.jsonl").read_text(encoding="utf-8").splitlines()
            if len(lines) < start:
                raise ValueError("Interrupted input hash ledger is incomplete")
            for i, line in enumerate(lines[:start]):
                record = json.loads(line)
                if (str(record["image_id"]) != str(population.iloc[i].image_id)
                        or hashlib.sha256(reader.read_bytes(population.iloc[i].image_path)).hexdigest() != record["sha256"]):
                    raise ValueError("Raw input changed during resume")
                if shared_hashes is not None and record != shared_hashes[i]:
                    raise ValueError("Interrupted input differs from shared quality")
            if shared_quality is not None:
                arrays["quality.npy"][:start] = shared_quality[:start]
            (work / "image_hashes.jsonl").write_text("\n".join(lines[:start]) + "\n", encoding="utf-8")
        adapter = create_pytorch_adapter_from_spec(spec, device="cuda")
        if source == "cr_fiqa" and shared_quality is None:
            from research.fiqa.cr_fiqa import load_cr_fiqa, infer_cr_fiqa_scores
            fiqa_model, _ = load_cr_fiqa(Path(project) / quality_config["checkpoint"],
                                      variant=quality_config.get("variant", "L"), device="cuda")
        with (work / "image_hashes.jsonl").open("a" if start else "w", encoding="utf-8") as ledger:
            for offset in range(start, len(population), batch_size):
                batch = population.iloc[offset:offset + batch_size]
                decoded = [reader.read(path) for path in batch.image_path]
                stop = offset + len(batch)
                if shared_hashes is not None:
                    for image_id, (_, digest), record in zip(batch.image_id, decoded, shared_hashes[offset:stop]):
                        if str(image_id) != str(record["image_id"]) or digest != record["sha256"]:
                            raise ValueError("Raw input changed relative to shared quality; use a new output root")
                images = np.stack([item[0] for item in decoded])
                result = adapter.embed(images)
                stop = offset + len(batch)
                arrays["embeddings.npy"][offset:stop] = result.normalized_embedding
                arrays["raw_norm.npy"][offset:stop] = result.raw_norm.reshape(-1)
                if quality_required:
                    q = (shared_quality[offset:stop] if shared_quality is not None else
                         infer_cr_fiqa_scores(fiqa_model, images, batch_size=batch_size, device="cuda")
                         if source == "cr_fiqa" else result.raw_norm.reshape(-1))
                    if not np.isfinite(q).all():
                        raise ValueError("Nonfinite quality scores")
                    arrays["quality.npy"][offset:stop] = q
                for image_id, (_, digest) in zip(batch.image_id, decoded):
                    ledger.write(json.dumps(dict(image_id=image_id, sha256=digest)) + "\n")
                ledger.flush()
                for array in arrays.values():
                    array.flush()
                write_json(progress_path, dict(completed_images=stop, model_uid=spec.model_uid,
                                              population_hash=population_digest, transform=transform))
        del adapter, fiqa_model
        files = [*arrays, "population.csv", "image_hashes.jsonl"]
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays.clear()
        manifest = dict(status="completed", model_uid=spec.model_uid, embedding_dim=1024,
                        dataset=dataset, quality=quality_config, population_hash=population_digest,
                        transform=transform,
                        input_code=input_code_hashes(),
                        quality_cache=str(quality_path) if quality_path else None,
                        quality_cache_reused=shared_quality is not None,
                        torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                        device=torch.cuda.get_device_name(0), batch_size=batch_size,
                        files={name: sha256_file(work / name) for name in files})
        write_json(work / "manifest.json", manifest)
        (work / "COMPLETED").write_text("completed\n", encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work), str(destination))
    finally:
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        reader.close()
    read_inputs(destination, expected_uid=spec.model_uid)
    if quality_path is not None:
        publish_quality_cache(quality_path, quality_contract, destination, len(population))
    return destination
