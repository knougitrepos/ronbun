from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


SENSITIVE_KEY_PARTS = ("password", "passwd", "token", "secret", "api_key", "dsn")
_URL_CREDENTIAL = re.compile(
    r"(?P<prefix>[a-z][a-z0-9+.-]*://[^:/@\s]+:)(?P<secret>[^@\s/]+)(?=@)",
    flags=re.IGNORECASE,
)
_KEY_VALUE_SECRET = re.compile(
    r"(?P<key>password|passwd|token|secret|api[_-]?key)"
    r"(?P<separator>\s*[=:]\s*)(?P<secret>[^\s,;]+)",
    flags=re.IGNORECASE,
)


def is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def redact_text(value: str) -> str:
    value = _URL_CREDENTIAL.sub(r"\g<prefix>***", value)
    return _KEY_VALUE_SECRET.sub(r"\g<key>\g<separator>***", value)


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "***" if is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
