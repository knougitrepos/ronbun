from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

import yaml
from sqlalchemy import URL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "database.yaml"
DEFAULT_LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "database.local.yaml"


class DatabaseConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DatabaseSettings:
    driver: str
    host: str
    port: int
    database: str
    user: str
    password: str
    password_source: str
    connect_timeout_seconds: int = 10
    pool_pre_ping: bool = True
    echo_sql: bool = False

    @property
    def sqlalchemy_url(self) -> URL:
        return URL.create(
            drivername=self.driver,
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    def redacted(self) -> dict[str, object]:
        return {
            "driver": self.driver,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": "***",
            "password_source": self.password_source,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "pool_pre_ping": self.pool_pre_ping,
            "echo_sql": self.echo_sql,
        }


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise DatabaseConfigurationError(f"database config must be a mapping: {path}")
    section = payload.get("database", payload.get("postgres", payload))
    if not isinstance(section, dict):
        raise DatabaseConfigurationError(f"database section must be a mapping: {path}")
    return dict(section)


def _overlay(base: dict, override: dict) -> dict:
    merged = dict(base)
    merged.update({key: value for key, value in override.items() if value is not None})
    return merged


def load_database_settings(
    config_path: str | Path | None = None,
    *,
    local_config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    require_password: bool = True,
) -> DatabaseSettings:
    env = os.environ if environ is None else environ
    base_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if local_config_path is not None:
        local_path = Path(local_config_path)
    elif config_path is None:
        local_path = DEFAULT_LOCAL_CONFIG_PATH
    else:
        local_path = None

    values = _read_yaml(base_path)
    if local_path is not None and local_path.exists():
        values = _overlay(values, _read_yaml(local_path))

    env_overrides = {
        "host": env.get("RONBUN_DB_HOST"),
        "port": env.get("RONBUN_DB_PORT"),
        "database": env.get("RONBUN_DB_NAME"),
        "user": env.get("RONBUN_DB_USER"),
    }
    values = _overlay(values, env_overrides)

    password_env = str(values.get("password_env", "RONBUN_DB_PASSWORD"))
    if env.get(password_env) is not None:
        password = str(env[password_env])
        password_source = f"environment:{password_env}"
    elif values.get("password") is not None:
        password = str(values["password"])
        password_source = "local_config"
    else:
        password = ""
        password_source = "missing"

    if require_password and not password:
        raise DatabaseConfigurationError(
            f"database password is missing; set {password_env} or create "
            f"{DEFAULT_LOCAL_CONFIG_PATH.name}"
        )

    database_name = values.get("database", values.get("dbname", "postgres"))
    try:
        port = int(values.get("port", 5432))
        timeout = int(values.get("connect_timeout_seconds", 10))
    except (TypeError, ValueError) as exc:
        raise DatabaseConfigurationError("database port and timeout must be integers") from exc
    if not 1 <= port <= 65535:
        raise DatabaseConfigurationError("database port must be between 1 and 65535")
    if timeout < 1:
        raise DatabaseConfigurationError("connect_timeout_seconds must be positive")

    return DatabaseSettings(
        driver=str(values.get("driver", "postgresql+psycopg2")),
        host=str(values.get("host", "localhost")),
        port=port,
        database=str(database_name),
        user=str(values.get("user", "postgres")),
        password=password,
        password_source=password_source,
        connect_timeout_seconds=timeout,
        pool_pre_ping=bool(values.get("pool_pre_ping", True)),
        echo_sql=bool(values.get("echo_sql", False)),
    )
