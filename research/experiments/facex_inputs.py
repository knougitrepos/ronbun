"""CUDA inputs shared by the five-dataset FaceX workflow."""
import hashlib
import json
import os
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
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def population_hash(population):
    return canonical_sha256(population.fillna("").astype(str).to_dict("records"))


def input_transform_from_metadata(metadata):
    """Consume a declared export policy; never infer a missing resize recipe."""
    policy = metadata.get("preprocessing", {}).get("alignment_policy")
    if isinstance(policy, dict) and policy.get("method") == "direct_similarity_transform":
        versions = {"scface": "scface_two_eye_arc112_v1", "tinyface": "tinyface_pfld_five_point_arc112_v1"}
        dataset = metadata["dataset"]
        package = metadata.get("input_geometry", {})
        expected = dict(input_size=[112, 112], color="BGR", interpolation="bicubic", border="zero",
                        face_detection="none", opencv=cv2.__version__)
        if (dataset not in versions or policy.get("version") != versions[dataset]
                or any(policy.get(k) != v for k, v in expected.items())
                or package.get("schema") != "raw_to_arc112_affine_v1" or package.get("file") != "affine.csv"
                or not isinstance(package.get("rows"), int) or package["rows"] <= 0
                or len(package.get("sha256", "")) != 64):
            raise ValueError("Unsupported alignment control or geometry package")
        return dict(version=versions[dataset], dataset=dataset, method=policy["method"],
                    source_color_order="rgb", model_color_order="bgr", geometry_package=package, **expected)
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
    transform = input_transform_from_metadata(metadata)
    if package := transform.get("geometry_package"):
        if sha256_file(path.parent / package["file"]) != package["sha256"]:
            raise ValueError("Alignment package checksum mismatch")
    return transform


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

    def read(self, path, *, affine=None):
        payload = self.read_bytes(path)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image decode failed: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        if affine is None:
            image = cv2.resize(image, (112, 112), interpolation=self.interpolation)
        else:
            expected, matrix = affine
            if digest != expected:
                raise ValueError(f"Raw image differs from alignment source: {path}")
            image = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), digest

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


def _read_population(path):
    # Keep IDs, leading zeroes and optional numeric-looking metadata byte-stable.
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def _close_arrays(arrays):
    for array in arrays:
        if array is not None and getattr(array, "_mmap", None) is not None:
            array._mmap.close()


def read_inputs(path, *, expected_uid=None, expected_population_hash=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not (path / "COMPLETED").is_file():
        raise ValueError("Input artifact is incomplete")
    if expected_uid and manifest["model_uid"] != expected_uid:
        raise ValueError("Input checkpoint mismatch")
    required = {"population.csv", "embeddings.npy", "raw_norm.npy"}
    if not required.issubset(manifest.get("files", {})) or not manifest.get("population_hash"):
        raise ValueError("Input manifest lacks population/content integrity metadata")
    for name, digest in manifest["files"].items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"Input file changed: {name}")
    population = _read_population(path / "population.csv")
    if (population_hash(population) != manifest["population_hash"]
            or expected_population_hash is not None and manifest["population_hash"] != expected_population_hash):
        raise ValueError("Input population hash differs from its pinned contract")
    vectors = np.load(path / "embeddings.npy", mmap_mode="r")
    norms = np.load(path / "raw_norm.npy", mmap_mode="r")
    quality = np.load(path / "quality.npy", mmap_mode="r") if "quality.npy" in manifest["files"] else None
    if (vectors.shape != (len(population), 1024) or norms.shape != (len(population),)
            or quality is not None and quality.shape != norms.shape
            or quality is None and manifest.get("quality", {}).get("source") != "not_required"):
        _close_arrays((vectors, norms, quality))
        raise ValueError("Input shape/coverage mismatch")
    return manifest, population, vectors, norms, quality


RESUME_SCHEMA = "facex-input-batch-integrity-v2"


def _batch_receipt(arrays, start, stop, records, previous):
    """Hash each newly committed row once, never rehash the growing prefix."""
    return dict(start=start, stop=stop, previous=previous,
                arrays={name: hashlib.sha256(np.ascontiguousarray(array[start:stop]).tobytes()).hexdigest()
                        for name, array in arrays.items()},
                image_hashes=canonical_sha256(records))


def _sync_file(path):
    with Path(path).open("rb+") as stream:
        os.fsync(stream.fileno())


def _verify_committed_batches(work, progress, arrays, population):
    """Audit the committed prefix; uncommitted crash tails are not trusted."""
    start = progress["completed_images"]
    count = progress.get("committed_batches")
    if (progress.get("schema") != RESUME_SCHEMA or type(start) is not int
            or not 0 <= start <= len(population) or type(count) is not int or count < 0):
        raise ValueError("Unverified/invalid partial input; preserve it and use a new output root")
    if (sha256_file(work / "population.csv") != progress.get("population_csv_sha256")
            or population_hash(_read_population(work / "population.csv")) != progress["population_hash"]):
        raise ValueError("Interrupted input population changed")
    # Binary lines permit recovery even when an uncommitted tail ends mid-UTF8.
    raw_lines = (work / "image_hashes.jsonl").read_bytes().splitlines(keepends=True)
    batch_lines = (work / "batch_integrity.jsonl").read_bytes().splitlines(keepends=True)
    if len(raw_lines) < start or len(batch_lines) < count:
        raise ValueError("Interrupted input integrity ledger is incomplete")
    previous, cursor = None, 0
    try:
        records = [json.loads(line) for line in raw_lines[:start]]
        for line in batch_lines[:count]:
            entry = json.loads(line)
            stop = entry["stop"]
            if (type(stop) is not int or not cursor < stop <= start
                    or entry.get("start") != cursor or entry.get("previous") != previous):
                raise ValueError("Interrupted input batch coverage changed")
            expected = _batch_receipt(arrays, cursor, stop, records[cursor:stop], previous)
            if entry != expected:
                raise ValueError("Interrupted input committed batch content changed")
            previous, cursor = canonical_sha256(entry), stop
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Interrupted input integrity ledger is invalid") from exc
    if cursor != start or previous != progress.get("batch_chain_sha256"):
        raise ValueError("Interrupted input batch commit differs from progress")
    for i, record in enumerate(records):
        if str(record.get("image_id")) != str(population.iloc[i].image_id):
            raise ValueError("Interrupted input ledger ordering changed")
    return records, raw_lines[:start], batch_lines[:count]


def _check_input_space(work, count, start, names, batch_size, *, population_bytes=0,
                       publish_quality=False):
    """Additional allocation estimate; completed artifacts bypass this check."""
    required = sum(max(0, count * (1024 if name == "embeddings.npy" else 1) * 4 + 128
                       - ((work / name).stat().st_size if (work / name).exists() else 0)) for name in names)
    remaining = count - start
    # Append-only raw/batch journals, atomic progress replacement and final manifest.
    required += population_bytes + remaining * 200 + ((remaining + batch_size - 1) // batch_size) * 1024 + 10240
    if publish_quality:
        required += count * 204 + 128  # shared quality array plus image-hash ledger copy
    if shutil.disk_usage(work).free < required * 1.1:
        raise RuntimeError(f"Insufficient free space for additional input cache data; estimate={required} bytes")
    return required


def load_affines(path, package, population):
    table = pd.read_csv(path, dtype={"image_id": str, "source_sha256": str}, float_precision="round_trip")
    columns = ["a", "b", "tx", "c", "d", "ty"]
    matrices = table[columns].to_numpy(dtype=np.float64).reshape(-1, 2, 3)
    if (len(table) != package["rows"] or not table.image_id.is_unique or not population.image_id.is_unique
            or set(table.image_id) != set(population.image_id) or not np.isfinite(matrices).all()):
        raise ValueError("Affine map and requested cohort differ")
    return dict(zip(table.image_id, zip(table.source_sha256, matrices)))


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
    geometry = transform.get("geometry_package")
    if geometry and dataset != transform["dataset"]:
        raise ValueError("Alignment control applies only to its declared dataset")
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
    from research.runtime.storage import artifact_lock
    with artifact_lock(destination):
        if destination.exists():
            _, _, *cached_arrays = read_inputs(destination, expected_uid=spec.model_uid,
                                              expected_population_hash=population_digest)
            _close_arrays(cached_arrays)
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
        names = ["embeddings.npy", "raw_norm.npy"] + (["quality.npy"] if quality_required else [])
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if (progress["population_hash"] != population_digest or progress["model_uid"] != spec.model_uid
                    or progress.get("transform") != transform):
                raise ValueError("Interrupted input lineage changed")
            if progress.get("schema") != RESUME_SCHEMA or progress.get("quality") != quality_config:
                raise ValueError("Unverified partial input; preserve it and use a new output root")
            start, mode = progress["completed_images"], "r+"
            if type(start) is not int or not 0 <= start <= len(population):
                raise ValueError("Invalid completed image count")
            population_csv = None
        else:
            if any(work.iterdir()):
                raise ValueError("Unverified partial input without progress; preserve it and use a new output root")
            start, mode = 0, "w+"
            population_csv = population.fillna("").astype(str).to_csv(index=False).encode("utf-8")
        _check_input_space(work, len(population), start, names, batch_size,
                           population_bytes=len(population_csv) if population_csv else 0,
                           publish_quality=quality_path is not None and not quality_path.exists())
        arrays = {}
        reader = ImageReader(project, interpolation=transform["interpolation"])
        affines = (load_affines(Path(spec.checkpoint.path).parent / geometry["file"], geometry, population)
                   if geometry else None)
        try:
            for name in names:
                shape = (len(population), 1024) if name == "embeddings.npy" else (len(population),)
                array = np.lib.format.open_memmap(work / name, mode=mode, dtype=np.float32, shape=shape)
                arrays[name] = array
                if array.shape != shape or array.dtype != np.float32:
                    raise ValueError("Interrupted input array shape/dtype changed")
            if population_csv is not None:
                (work / "population.csv").write_bytes(population_csv)
                _sync_file(work / "population.csv")
                for name, array in arrays.items():
                    array.flush()
                    _sync_file(work / name)
                for name in ("image_hashes.jsonl", "batch_integrity.jsonl"):
                    (work / name).write_bytes(b"")
                    _sync_file(work / name)
                progress = dict(schema=RESUME_SCHEMA, completed_images=0, committed_batches=0,
                                batch_chain_sha256=None, model_uid=spec.model_uid,
                                population_hash=population_digest, transform=transform, quality=quality_config,
                                population_csv_sha256=sha256_file(work / "population.csv"))
                write_json(progress_path, progress)
            else:
                records, raw_lines, batch_lines = _verify_committed_batches(work, progress, arrays, population)
                # Verify the raw bytes of previously checkpointed images before resume.
                for i, record in enumerate(records):
                    if (str(record["image_id"]) != str(population.iloc[i].image_id)
                            or hashlib.sha256(reader.read_bytes(population.iloc[i].image_path)).hexdigest() != record["sha256"]):
                        raise ValueError("Raw input changed during resume")
                    if shared_hashes is not None and record != shared_hashes[i]:
                        raise ValueError("Interrupted input differs from shared quality")
                if shared_quality is not None:
                    if not np.array_equal(arrays["quality.npy"][:start], shared_quality[:start]):
                        raise ValueError("Interrupted quality differs from completed shared quality")
                # Drop only uncommitted journal tails after checking every committed byte.
                for name, lines in (("image_hashes.jsonl", raw_lines), ("batch_integrity.jsonl", batch_lines)):
                    with (work / name).open("rb+") as stream:
                        stream.truncate(sum(map(len, lines)))
                        stream.flush()
                        os.fsync(stream.fileno())
            adapter = create_pytorch_adapter_from_spec(spec, device="cuda")
            if source == "cr_fiqa" and shared_quality is None:
                from research.fiqa.cr_fiqa import load_cr_fiqa, infer_cr_fiqa_scores
                fiqa_model, _ = load_cr_fiqa(Path(project) / quality_config["checkpoint"],
                                          variant=quality_config.get("variant", "L"), device="cuda")
            with (work / "image_hashes.jsonl").open("a", encoding="utf-8") as ledger, \
                    (work / "batch_integrity.jsonl").open("a", encoding="utf-8") as receipts:
                for offset in range(start, len(population), batch_size):
                    batch = population.iloc[offset:offset + batch_size]
                    decoded = ([reader.read(row.image_path, affine=affines[row.image_id]) for row in batch.itertuples()]
                               if affines is not None else [reader.read(path) for path in batch.image_path])
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
                    if not all(np.isfinite(array[offset:stop]).all() for array in arrays.values()
                               if array is not arrays.get("quality.npy")):
                        raise ValueError("Nonfinite embedding/norm output")
                    if quality_required:
                        q = (shared_quality[offset:stop] if shared_quality is not None else
                             infer_cr_fiqa_scores(fiqa_model, images, batch_size=batch_size, device="cuda")
                             if source == "cr_fiqa" else result.raw_norm.reshape(-1))
                        if not np.isfinite(q).all():
                            raise ValueError("Nonfinite quality scores")
                        arrays["quality.npy"][offset:stop] = q
                    records = [dict(image_id=str(image_id), sha256=digest)
                               for image_id, (_, digest) in zip(batch.image_id, decoded)]
                    for record in records:
                        ledger.write(json.dumps(record) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    for name, array in arrays.items():
                        array.flush()
                        _sync_file(work / name)
                    receipt = _batch_receipt(arrays, offset, stop, records, progress["batch_chain_sha256"])
                    receipts.write(json.dumps(receipt) + "\n")
                    receipts.flush()
                    os.fsync(receipts.fileno())
                    progress = {**progress, "completed_images": stop,
                                "committed_batches": progress["committed_batches"] + 1,
                                "batch_chain_sha256": canonical_sha256(receipt)}
                    write_json(progress_path, progress)
            del adapter, fiqa_model
            files = [*arrays, "population.csv", "image_hashes.jsonl", "batch_integrity.jsonl"]
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
