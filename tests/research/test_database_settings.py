from research.database.connection import create_database_engine
from research.database.settings import (
    DatabaseConfigurationError,
    load_database_settings,
)


def test_database_settings_load_password_from_environment_and_redact_it(tmp_path):
    config_path = tmp_path / "database.yaml"
    config_path.write_text(
        "\n".join(
            [
                "database:",
                "  driver: postgresql+psycopg2",
                "  host: db.example.test",
                "  port: 5432",
                "  database: face_search",
                "  user: researcher",
                "  password_env: TEST_DATABASE_PASSWORD",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_database_settings(
        config_path,
        environ={"TEST_DATABASE_PASSWORD": "do-not-log-this"},
    )

    assert settings.database == "face_search"
    assert settings.password_source == "environment:TEST_DATABASE_PASSWORD"
    assert settings.redacted()["password"] == "***"
    assert "do-not-log-this" not in str(settings.redacted())

    engine = create_database_engine(settings)
    try:
        assert engine.url.database == "face_search"
        assert engine.url.host == "db.example.test"
    finally:
        engine.dispose()


def test_database_settings_require_an_explicit_password_source(tmp_path):
    config_path = tmp_path / "database.yaml"
    config_path.write_text(
        "database:\n  host: localhost\n  password_env: TEST_DATABASE_PASSWORD\n",
        encoding="utf-8",
    )

    try:
        load_database_settings(config_path, environ={})
    except DatabaseConfigurationError as exc:
        assert "TEST_DATABASE_PASSWORD" in str(exc)
    else:  # pragma: no cover - protects the no-secret default contract
        raise AssertionError("missing database password was accepted")
