"""Package checked matrix job evidence for Git while keeping expanded jobs local."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def package(root):
    root = Path(root).resolve()
    jobs = root / "jobs"
    inventory = {}
    results = {}
    reports = []

    def checked(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path escapes matrix root: {path}")
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"hash mismatch: {path}")
        return value

    for path in sorted((root / "reports").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"unfinished report: {path}")
        for name, sha in manifest["files"].items():
            checked(path.parent / name, sha)
        reports.append(dict(path=path.relative_to(root).as_posix(), sha256=checked(path),
                            expected_jobs=manifest["expected_jobs"], observed_jobs=manifest["observed_jobs"]))
        for source in manifest["sources"]:
            directory = Path(source["result_dir"]).resolve()
            if not directory.is_relative_to(jobs):
                raise ValueError("report result is outside jobs")
            if directory in results:
                if results[directory] != source["result_manifest_sha256"]:
                    raise ValueError("conflicting report result hashes")
                continue
            data = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
            if data.get("status") != "completed" or canonical(data) != source["result_manifest_sha256"]:
                raise ValueError("report source result manifest mismatch")
            results[directory] = source["result_manifest_sha256"]
            for name, sha in {**data["files"], "manifest.json": checked(directory / "manifest.json")}.items():
                file = directory / name
                inventory[file.relative_to(root).as_posix()] = dict(sha256=checked(file, sha), bytes=file.stat().st_size)
    if not results:
        raise ValueError("no completed report sources to package")
    for path in sorted(jobs.glob("*/receipt.json")):
        data = json.loads(path.read_text(encoding="utf8"))
        directory = Path(data["result_dir"]).resolve()
        if directory not in results or data["result_manifest_sha256"] != results[directory]:
            raise ValueError("unreported job receipt; retain it separately before ignoring jobs")
        inventory[path.relative_to(root).as_posix()] = dict(sha256=checked(path), bytes=path.stat().st_size)
    # Do not silently omit partial/new files when making the full jobs tree local.
    actual = {p.relative_to(root).as_posix() for p in jobs.rglob("*") if p.is_file()}
    if actual != set(inventory):
        raise ValueError("unpackaged job files remain")
    spec = dict(schema_version=1, artifact_type="calibration_matrix_analysis_archive", reports=reports,
                result_count=len(results), files=inventory)
    uid = "matrix-evidence-" + canonical(spec)[:24]
    output = root / "analysis_archive"
    output.mkdir(exist_ok=True)
    archive = output / f"{uid}.zip"
    manifest_path = output / f"{uid}.json"
    if archive.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest["spec_sha256"] != canonical(spec) or digest(archive) != manifest["archive_sha256"]:
            raise ValueError("existing evidence archive mismatch")
    else:
        with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            bundle.writestr("inventory.json", json.dumps(spec, ensure_ascii=False, indent=2))
            for name in sorted(inventory):
                bundle.write(root / name, name)
        # Check archived payload hashes, not only ZIP CRCs, before changing Git visibility.
        with zipfile.ZipFile(temporary) as bundle:
            for name, receipt in inventory.items():
                with bundle.open(name) as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != receipt["sha256"]:
                        raise ValueError(f"archive content mismatch: {name}")
        temporary.rename(archive)
        manifest = dict(artifact_type=spec["artifact_type"], schema_version=1, status="completed",
                        archive=archive.name, archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                        spec_sha256=canonical(spec), result_count=len(results), file_count=len(inventory),
                        unpacked_bytes=sum(value["bytes"] for value in inventory.values()), reports=reports,
                        producer_script_sha256=digest(Path(__file__)))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/calibration/matrix")
    package(parser.parse_args().matrix_root)
