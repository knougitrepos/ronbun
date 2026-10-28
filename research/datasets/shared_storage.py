"""Publish external, portable dataset references without moving raw data.

Run once with explicit source repositories. This is a migration of data records,
not a runtime dependency on FaceX-Zoo's Python implementation.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import uuid

from research.runtime.hashing import sha256_file, canonical_sha256
from research.runtime.storage import artifact_lock, roots, write_provenance

FOLDERS = {"survface": "QMUL-SurvFace", "tinyface": "tinyface", "scface": "SCface_database"}


def verify_bundle(path):
    path = Path(path)
    manifest = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    if not (path / "COMPLETED").is_file() or manifest.get("status") != "completed":
        raise ValueError(f"Incomplete shared bundle: {path}")
    for name, digest in manifest["files"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path.resolve()) or sha256_file(file) != digest:
            raise ValueError(f"Changed shared bundle: {name}")
    return manifest


def publish_bundle(path, recipe, build):
    path = Path(path)
    with artifact_lock(path):
        if path.exists():
            record = verify_bundle(path)
            if record["recipe"] != recipe:
                raise FileExistsError(f"Shared bundle recipe changed: {path}")
            return path
        work = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
        work.mkdir()
        try:
            counts = build(work)
            write_provenance(work, operation=recipe["operation"], sources=recipe["sources"],
                             parameters=recipe["parameters"], counts=counts)
            files = {p.relative_to(work).as_posix(): sha256_file(p) for p in work.rglob("*") if p.is_file()}
            record = dict(schema="shared-dataset-bundle-v1", status="completed", recipe=recipe, files=files, counts=counts)
            (work / "bundle.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (work / "COMPLETED").touch()
            os.rename(work, path)
        except BaseException:
            # Preserve failed work and its inputs for diagnosis, never publish it.
            raise
    verify_bundle(path)
    return path


def _source(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(path))


def migrate_shared_storage(project_root, facex_root, dataset_root=None, model_root=None):
    project, facex = Path(project_root).resolve(), Path(facex_root).resolve()
    locations = roots(dataset_root, model_root)
    data, models = Path(locations["DATASET_ROOT"]), Path(locations["MODEL_ROOT"])
    if not data.is_dir() or not models.is_dir():
        raise FileNotFoundError("Verify the external dataset and model directories before migration")
    original = facex / "fine-tune/data/protocols/research_v1/handoff.json"
    index = json.loads(original.read_text(encoding="utf-8"))
    handoff = {}
    for name, item in index["protocols"].items():
        if name not in FOLDERS or sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Unexpected or changed source handoff")
        old = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        root = data / FOLDERS[name]
        sources = {"protocol": _source(item["path"]), "roles": _source(old["roles_path"]),
                   "train_list": _source(old["train_list"])}
        if sources["roles"]["sha256"] != old["roles_sha256"] or sources["train_list"]["sha256"] != old["train_sha256"]:
            raise ValueError("Changed source roles or train list")
        for key, source in old["sources"].items():
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError("Changed protocol source")
            sources[key] = _source(source["path"])
        recipe = dict(operation="portable_protocol_migration_v1", sources=sources,
                      parameters=dict(dataset=name, raw_folder=FOLDERS[name], roles_sha256=old["roles_sha256"],
                                      command="python -m research.datasets.shared_storage --project-root ... --facex-root ..."))
        out = root / "protocols/shared_v1" / canonical_sha256(recipe)[:20]

        def build(work, old=old, sources=sources, root=root):
            with Path(old["roles_path"]).open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            # Directory listings avoid an expensive per-image stat on Windows.
            by_dir = {}
            for row in rows:
                rel = Path(row["image_path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Unsafe image path")
                directory = root / rel.parent
                if directory not in by_dir:
                    by_dir[directory] = set(os.listdir(directory))
                if rel.name not in by_dir[directory]:
                    raise FileNotFoundError(root / rel)
            shutil.copyfile(old["roles_path"], work / "roles.csv")
            shutil.copyfile(old["train_list"], work / "train_list.txt")
            (work / "sources").mkdir()
            record = dict(old, schema="facex-ronbun-roles-v2", raw_root=os.path.relpath(root, work).replace("\\", "/"),
                          roles_path="roles.csv", train_list="train_list.txt", sources={},
                          migrated_from=sources["protocol"], image_validation="reference_existence_only; historical byte audit separate")
            for key, source in old["sources"].items():
                rel = f"sources/{key}{Path(source['path']).suffix}"
                shutil.copyfile(source["path"], work / rel)
                record["sources"][key] = dict(path=rel, sha256=source["sha256"])
            (work / "protocol.json").write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            return dict(rows=len(rows), missing=0, roles_unchanged=True, role_counts=old["role_counts"], all_images_decode="not_checked")

        publish_bundle(out, recipe, build)
        handoff[name] = dict(path=os.path.relpath(out / "protocol.json", data / "protocols/shared_v1").replace("\\", "/"), sha256=sha256_file(out / "protocol.json"))

    for name in ("lfw", "rfw"):
        folder = data / name.upper()
        originals = {"image_manifest": project / f"data/interim/{name}/{'face' if name == 'lfw' else 'image'}_manifest.csv",
                     "pairs_file": project / ("data/external/lfw/pairs.txt" if name == "lfw" else "data/interim/rfw/pair_protocol.csv")}
        recipe = dict(operation="portable_pair_migration_v1", sources={k: _source(v) for k,v in originals.items()},
                      parameters=dict(dataset=name, image_root="dataset_root", preserve_pairs=True))
        def build_pair(work, name=name, originals=originals):
            with originals["image_manifest"].open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f); columns = reader.fieldnames; rows = list(reader)
            for row in rows:
                for key in ("image_path", "source_archive_path"):
                    if key in row:
                        row[key] = row[key].replace("data/raw/", "", 1)
            with (work / "image_manifest.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, columns); writer.writeheader(); writer.writerows(rows)
            shutil.copyfile(originals["pairs_file"], work / ("pairs.txt" if name=="lfw" else "pair_protocol.csv"))
            return dict(images=len(rows), pairs_unchanged=True, image_path_prefix_only=True, all_images_decode="not_checked")
        publish_bundle(folder / "protocols/shared_v1", recipe, build_pair)

    destination = data / "protocols/shared_v1"
    recipe = dict(operation="portable_handoff_v1", sources={"handoff": _source(original)}, parameters=dict(protocols=handoff))
    def build_index(work):
        (work / "handoff.json").write_text(json.dumps(dict(schema="facex-ronbun-handoff-v2", protocols=handoff), sort_keys=True, indent=2)+"\n", encoding="utf-8")
        return dict(datasets=len(handoff))
    publish_bundle(destination, recipe, build_index)
    # Quality models are a separate dependency, not FR checkpoints.
    for source in sorted((project / "models/fiqa").glob("*.pth")):
        dest = models / "fiqa" / source.name
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != sha256_file(source):
                raise FileExistsError(f"Different quality checkpoint: {dest}")
        else:
            with source.open("rb") as src, dest.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256_file(dest) != sha256_file(source):
                raise ValueError("Quality checkpoint copy mismatch")
    return dict(handoff=str(destination / "handoff.json"), dataset_root=str(data), model_root=str(models))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--facex-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_shared_storage(**vars(args)), ensure_ascii=False, indent=2))
