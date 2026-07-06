from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PHASES = ("protocol", "templates", "compression", "search", "calibration")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def hash_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expand_phases(phase: str) -> list[str]:
    if phase == "all":
        return list(PHASES)
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    return [phase]


def artifact_dir(config: dict[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    root = Path(run_cfg.get("artifact_root", "artifacts/research_runs"))
    name = run_cfg.get("name", "face_search_run")
    return root / name


def validate_config(config: dict[str, Any], *, dry_run: bool) -> None:
    required_sections = {"run", "dataset", "protocol", "compression", "calibration"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if not dry_run:
        manifest = Path(config["dataset"]["manifest_path"])
        if not manifest.exists():
            raise FileNotFoundError(f"manifest file not found: {manifest}")


def write_run_artifacts(config: dict[str, Any], config_hash: str) -> Path:
    out_dir = artifact_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config_hash": config_hash, "config": config}
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compressed face-search research phases.")
    parser.add_argument("--config", required=True, help="Path to face search YAML config.")
    parser.add_argument(
        "--phase",
        choices=("all",) + PHASES,
        default="all",
        help="Phase to run. Use all for the full v1 pipeline.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned execution only.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config, dry_run=args.dry_run)
    phases = expand_phases(args.phase)
    config_hash = hash_config(config)

    if not args.dry_run:
        out_dir = write_run_artifacts(config, config_hash)
    else:
        out_dir = artifact_dir(config)

    print(f"dry_run={args.dry_run}")
    print(f"phase={args.phase}")
    print(f"phases={','.join(phases)}")
    print(f"config_hash={config_hash}")
    print(f"artifact_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
