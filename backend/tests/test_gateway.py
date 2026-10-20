import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from insightforge.config import Settings, get_settings
from insightforge.llm.gateway import ROLE_SPECS, ModelGateway, Role, Tier, TokenBudgetExceeded, TokenMeter


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, openrouter_api_key="sk-test", **overrides)


def _reply(text: str, tokens_in: int, tokens_out: int) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
    )


class _Broken(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise ConnectionError("primary is down")


def test_every_role_has_a_spec():
    assert set(ROLE_SPECS) == set(Role)


def test_chat_model_is_configured_for_openrouter():
    gw = ModelGateway(_settings())
    m = gw.chat_model(Role.WRITER)
    assert m.model_name == "anthropic/claude-sonnet-5.5"
    assert m.openai_api_base == "https://openrouter.ai/api/v1"
    assert m.max_tokens == ROLE_SPECS[Role.WRITER].max_tokens
    assert m.request_timeout == ROLE_SPECS[Role.WRITER].timeout_s == 240  # sized for long reports
    assert m.default_headers["X-Title"] == "InsightForge AI"


def test_reasoning_is_bounded_for_narrow_roles_only():
    gw = ModelGateway(_settings())  # Claude
    assert gw.chat_model(Role.GROUNDING).extra_body == {"reasoning": {"effort": "low"}}
    assert gw.fallback_model(Role.GROUNDING).extra_body == {"reasoning": {"effort": "low"}}  # Sonnet accepts "low"
    assert gw.chat_model(Role.WRITER).extra_body is None  # judgment-heavy: provider default


def test_open_models_route_only_to_hosts_that_honour_every_parameter():
    gw = ModelGateway(_settings(model_primary="deepseek/deepseek-v4.1-flash", model_secondary="z-ai/glm-5.3-flash"))
    route = {"require_parameters": True}
    assert gw.chat_model(Role.GROUNDING).extra_body == {"provider": route, "reasoning": {"effort": "low"}}
    assert gw.chat_model(Role.WRITER).extra_body == {"provider": route}
    # Claude never gets the filter: with zero-data-retention settings it leaves no Claude host (live 404).
    assert "provider" not in (ModelGateway(_settings()).chat_model(Role.WRITER).extra_body or {})


def test_agent_output_strategy_follows_the_model():
    assert ModelGateway(_settings()).agent_output_strategy(Role.RESEARCHER) == "provider"  # Claude rejects forced tool_choice
    open_models = _settings(model_primary="deepseek/deepseek-v4.1-flash", model_secondary="z-ai/glm-5.3-flash")
    assert ModelGateway(open_models).agent_output_strategy(Role.RESEARCHER) == "tool"
    forced = _settings(agent_output_strategy="provider")
    assert ModelGateway(forced).agent_output_strategy(Role.RESEARCHER) == "provider"


def test_roles_map_to_tiers_and_fallback_flips_tier():
    gw = ModelGateway(_settings())
    assert gw.chat_model(Role.EXTRACTOR).model_name == "anthropic/claude-haiku-5.5"
    assert gw.fallback_model(Role.EXTRACTOR).model_name == "anthropic/claude-sonnet-5.5"
    assert gw.fallback_model(Role.PLANNER).model_name == gw.model_name(Tier.SECONDARY)


def test_missing_key_fails_loudly():
    gw = ModelGateway(Settings(_env_file=None, openrouter_api_key=None))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        gw.chat_model(Role.PLANNER)


def test_meter_counts_tokens_and_stops_the_run():
    meter = TokenMeter(budget=100)
    model = GenericFakeChatModel(messages=iter([_reply("a", 40, 20), _reply("b", 30, 20), _reply("c", 1, 1)]))
    model = model.with_config(callbacks=[meter])

    model.invoke("first")
    assert (meter.input_tokens, meter.output_tokens, meter.calls) == (40, 20, 1)
    model.invoke("second")  # starts at 60 < 100, so allowed; ends at 110
    assert meter.total == 110 and meter.remaining == 0
    with pytest.raises(TokenBudgetExceeded):
        model.invoke("third")


def test_fallback_serves_when_primary_fails():
    meter = TokenMeter(budget=1_000)
    primary = _Broken(messages=iter([])).with_config(callbacks=[meter])
    fallback = GenericFakeChatModel(messages=iter([_reply("from fallback", 5, 5)])).with_config(callbacks=[meter])
    assert primary.with_fallbacks([fallback]).invoke("hi").content == "from fallback"


def test_budget_breach_is_not_rescued_by_fallback():
    meter = TokenMeter(budget=10)
    meter.input_tokens = 10
    primary = GenericFakeChatModel(messages=iter([_reply("x", 1, 1)])).with_config(callbacks=[meter])
    fallback = GenericFakeChatModel(messages=iter([_reply("y", 1, 1)])).with_config(callbacks=[meter])
    with pytest.raises(TokenBudgetExceeded):
        primary.with_fallbacks([fallback]).invoke("hi")
    assert meter.calls == 0


@pytest.mark.skipif(get_settings().openrouter_api_key is None, reason="live OpenRouter call; set OPENROUTER_API_KEY in .env")
def test_live_openrouter_smoke():
    gw = ModelGateway.for_run()
    reply = gw.runnable(Role.GUARDRAIL).invoke("Reply with the single word: ok")
    assert "ok" in reply.content.lower()
    assert gw.meter.total > 0
