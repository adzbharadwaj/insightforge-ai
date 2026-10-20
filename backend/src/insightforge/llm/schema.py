"""Helpers for schemas that models fill in.

Provider-native JSON schema treats `maxLength` / `maxItems` as guidance, not a guarantee:
a model can return a 320-character scope for a 300-character field, and a hard Pydantic
error would fail the whole run over formatting. Fields a model writes therefore clip to
their limit before validation (the limit stays in the schema, so the model still sees it).
"""

from typing import Any

from pydantic import BeforeValidator


def _clip(n: int):
    def clip(v: Any) -> Any:
        return v[:n] if isinstance(v, str | list) else v
    return clip


def clip(n: int) -> BeforeValidator:
    return BeforeValidator(_clip(n))
