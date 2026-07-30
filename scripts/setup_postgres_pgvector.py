from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.database import (  # noqa: E402
    check_database_health,
    create_database_engine,
    init_database,
    load_database_settings,
)


def print_health(
    health: dict[str, object],
    *,
    dry_run: bool,
    connection: dict[str, object],
) -> None:
    print(f"dry_run={dry_run}")
    print(f"host={connection['host']}")
    print(f"port={connection['port']}")
    print(f"database={health['database']}")
    print(f"user={health['user']}")
    print(f"password_source={connection['password_source']}")
    print(f"vector_extension_version={health['vector_extension_version']}")
    print(f"missing_tables={','.join(str(item) for item in health['missing_tables'])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare PostgreSQL/pgvector for the thesis3 face-search experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Base DB YAML. Defaults to configs/database.yaml.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        help="Optional secret-bearing local override YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report current database health; do not create extension, tables, or indexes.",
    )
    args = parser.parse_args()

    settings = load_database_settings(
        config_path=args.config,
        local_config_path=args.local_config,
    )
    engine = create_database_engine(settings)
    try:
        if not args.dry_run:
            init_database(engine)
        health = check_database_health(engine)
        print_health(health, dry_run=args.dry_run, connection=settings.redacted())
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
