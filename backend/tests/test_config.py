import pytest
from pydantic import ValidationError

from insightforge.config import Settings


def test_defaults_match_plan():
    s = Settings(_env_file=None)
    assert s.max_subtasks == 6
    assert s.evidence_bundle_token_budget == 6_000
    assert s.factcheck_similarity_threshold == 0.85
    assert s.memory_summaries_loaded == 3
    assert s.embedding_dim == 384


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("EVIDENCE_BUNDLE_TOKEN_BUDGET", "8000")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.evidence_bundle_token_budget == 8000
    assert s.openrouter_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(s)  # secrets never print


def test_subtask_cap_cannot_exceed_six(monkeypatch):
    monkeypatch.setenv("MAX_SUBTASKS", "7")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
