from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.database import check_database_health, init_database


def print_health(health: dict, *, dry_run: bool) -> None:
    print(f"dry_run={dry_run}")
    print(f"database={health['database']}")
    print(f"user={health['user']}")
    print(f"vector_extension_version={health['vector_extension_version']}")
    print(f"missing_tables={','.join(health['missing_tables'])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare PostgreSQL/pgvector for the thesis face-search application."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report current database health; do not create extension or tables.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        init_database()
    print_health(check_database_health(), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
