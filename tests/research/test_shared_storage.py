from pathlib import Path
import csv
import io
import json
import shutil
import tarfile

import pytest

from research.datasets.shared_storage import migrate_shared_storage, publish_bundle, verify_bundle
from research.runtime.hashing import sha256_file
from research.runtime.storage import artifact_lock, roots, expand_paths
from research.protocols.facex_handoff import load_handoff


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)


def source_fixture(tmp_path):
    project, facex, data, models = [tmp_path / name for name in ("project","producer","external","models")]
    for p in (project,facex,data,models):p.mkdir()
    image = data / "tinyface/Training_Set/a/a_1.jpg"
    image.parent.mkdir(parents=True);image.write_bytes(b"fixture image; no decode claimed")
    source=facex / "fine-tune/data/protocols/research_v1/tinyface/fixture"
    rows=[dict(image_id="tinyface:train:a:a_1",identity_id="tinyface:train:a",source_identity_id="a",
               image_path="Training_Set/a/a_1.jpg",role="pq_development",fr_train=1,group="",distance="")]
    write_csv(source / "roles.csv",rows)
    (source / "train_list.txt").write_text("Training_Set/a/a_1.jpg 0\n")
    record=dict(schema="facex-ronbun-roles-v1",name="tinyface",raw_root=str(facex / "tinyface"),
                roles_path=str(source / "roles.csv"),roles_sha256=sha256_file(source / "roles.csv"),
                train_list=str(source / "train_list.txt"),train_sha256=sha256_file(source / "train_list.txt"),
                sources={},num_classes=1,images=1,split_seed=8972,training_identity_sha256="fixture",
                role_counts=dict(pq_development=dict(identities=1,images=1)))
    (source / "protocol.json").write_text(json.dumps(record))
    index=source.parents[1] / "handoff.json"
    index.write_text(json.dumps(dict(schema="facex-ronbun-handoff-v1",protocols=dict(tinyface=dict(path=str(source / "protocol.json"),sha256=sha256_file(source / "protocol.json"))))))
    for name in ("lfw","rfw"):
        (data / name.upper()).mkdir()
        write_csv(project / f"data/interim/{name}/{'face' if name=='lfw' else 'image'}_manifest.csv",
                  [dict(image_id="fixture",identity_id="a",image_path=f"data/raw/{name.upper()}/a.jpg")])
        pair=project / ("data/external/lfw/pairs.txt" if name=="lfw" else "data/interim/rfw/pair_protocol.csv")
        pair.parent.mkdir(parents=True,exist_ok=True);pair.write_text("fixture pairs\n")
    return project,facex,data,models,index


def test_migration_preserves_roles_and_runs_after_relocation(tmp_path):
    project,facex,data,models,old_index=source_fixture(tmp_path)
    before=sha256_file(old_index)
    result=migrate_shared_storage(project,facex,data,models)
    record,pop=load_handoff(result["handoff"])["tinyface"]
    assert Path(pop.image_path.iloc[0]).read_bytes().startswith(b"fixture")
    assert sha256_file(old_index)==before
    assert sha256_file(record["roles_path"])==record["roles_sha256"]
    portable=json.loads(Path(result["handoff"]).read_text())
    assert not Path(portable["protocols"]["tinyface"]["path"]).is_absolute()
    files={p:sha256_file(p) for p in data.rglob("*") if p.is_file()}
    assert migrate_shared_storage(project,facex,data,models)==result
    assert all(sha256_file(p)==h for p,h in files.items())
    relocated=tmp_path / "mounted/linux-style/data"
    shutil.copytree(data,relocated)
    # Original source paths are historical provenance, never runtime dependencies.
    shutil.rmtree(data);shutil.rmtree(project);shutil.rmtree(facex)
    rec,pop=load_handoff(relocated / "protocols/shared_v1/handoff.json")["tinyface"]
    assert Path(pop.image_path.iloc[0]).is_relative_to(relocated)
    assert Path(pop.image_path.iloc[0]).read_bytes().startswith(b"fixture")
    assert rec["num_classes"]==1


def test_lock_and_partial_publication_and_tamper(tmp_path):
    dest=tmp_path / "bundle";recipe=dict(operation="test",sources={},parameters={})
    with artifact_lock(dest):
        with pytest.raises(FileExistsError):
            publish_bundle(dest,recipe,lambda p: {})
    def fail(p):
        (p / "incomplete").write_text("partial")
        raise RuntimeError("injected interruption")
    with pytest.raises(RuntimeError):publish_bundle(dest,recipe,fail)
    assert not dest.exists()
    assert list(tmp_path.glob("*.partial"))
    publish_bundle(dest,recipe,lambda p:dict(images=0))
    verify_bundle(dest)
    (dest / "README.md").write_text("changed")
    with pytest.raises(ValueError,match="Changed"):verify_bundle(dest)


def test_external_root_env_and_unknown_tokens(tmp_path,monkeypatch):
    monkeypatch.setenv("FACE_DATASET_ROOT",str(tmp_path / "data"))
    monkeypatch.setenv("FACE_MODEL_ROOT",str(tmp_path / "weights"))
    v=roots()
    assert expand_paths("${DATASET_ROOT}/LFW",v)==str(tmp_path / "data")+"/LFW"
    with pytest.raises(ValueError,match="Unresolved"):expand_paths("${MISSPELLED_ROOT}",v)


def test_extraction_provenance_and_lock(tmp_path):
    from research.datasets.sources import safe_extract_tar_archive
    archive=tmp_path / "source.tar.gz"
    with tarfile.open(archive,"w:gz") as t:
        m=tarfile.TarInfo("images/a.jpg");m.size=3;t.addfile(m,io.BytesIO(b"abc"))
    out=tmp_path / "external/extracted/v1"
    with artifact_lock(out):
        with pytest.raises(FileExistsError):safe_extract_tar_archive(archive,out)
    safe_extract_tar_archive(archive,out,expected_prefixes=["images"])
    assert (out / "images/a.jpg").read_bytes()==b"abc"
    d=json.loads((out / "derivation.json").read_text())
    assert d["sources"]["archive"]["sha256"]==sha256_file(archive)
    assert d["counts"]["decode"]=="not_checked"
    assert (out / "README.md").is_file() and (out / "COMPLETED").exists()


def test_pair_root_escape_is_rejected_before_protocol_read(tmp_path):
    from research.protocols.facex_handoff import load_pair_dataset
    root=tmp_path / "external";root.mkdir()
    manifest=tmp_path / "images.csv"
    write_csv(manifest,[dict(image_id="a",identity_id="a",image_path="../outside.jpg")])
    with pytest.raises(ValueError,match="escapes"):
        load_pair_dataset(tmp_path,dict(image_root=str(root),image_manifest=str(manifest),pairs_file="unused"),"lfw")


def test_manifest_builder_accepts_external_data_root(tmp_path):
    from research.datasets.manifests import build_survface_training_manifest
    repo=tmp_path / "repo";repo.mkdir()
    data=tmp_path / "external";dataset=data / "SurvFace"
    for identity in range(10):
        path=dataset / "training_set" / str(identity) / f"{identity}_1.jpg"
        path.parent.mkdir(parents=True);path.write_bytes(b"fixture")
    result=build_survface_training_manifest(dataset,repo,data_root=data)
    assert all(p.startswith("SurvFace/training_set/") for p in result.manifest.image_path)
    with pytest.raises(ValueError,match="inside project_root"):
        build_survface_training_manifest(dataset,repo)
