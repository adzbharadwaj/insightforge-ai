"""Model gateway (component 13).

Agents ask for a model by *role*, never by name. The gateway decides which
model serves the role, applies timeouts, retries and an output cap, wires a
fallback model, and meters tokens against the per-run budget. Swapping the
provider or a model is a config change here, not a change in any agent.

All models are reached through OpenRouter's OpenAI-compatible API.
"""

import enum
import threading
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from insightforge.config import Settings, get_settings


class Role(enum.StrEnum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    EXTRACTOR = "extractor"
    SYNTHESIS = "synthesis"
    WRITER = "writer"
    GROUNDING = "grounding"
    FACT_CHECK = "fact_check"
    DIFF = "diff"
    SUMMARIZER = "summarizer"
    GUARDRAIL = "guardrail"


class Tier(enum.StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


@dataclass(frozen=True)
class RoleSpec:
    tier: Tier
    max_tokens: int
    temperature: float = 0.0


# Judgement-heavy roles get the primary model; high-volume, narrow roles get the cheaper one.
# Each role falls back to the other tier.
ROLE_SPECS: dict[Role, RoleSpec] = {
    Role.PLANNER: RoleSpec(Tier.PRIMARY, 4_000),
    Role.RESEARCHER: RoleSpec(Tier.SECONDARY, 2_000),
    Role.EXTRACTOR: RoleSpec(Tier.SECONDARY, 4_000),
    Role.SYNTHESIS: RoleSpec(Tier.PRIMARY, 8_000),
    Role.WRITER: RoleSpec(Tier.PRIMARY, 8_000, temperature=0.2),
    Role.GROUNDING: RoleSpec(Tier.SECONDARY, 2_000),
    Role.FACT_CHECK: RoleSpec(Tier.SECONDARY, 1_000),
    Role.DIFF: RoleSpec(Tier.SECONDARY, 2_000),
    Role.SUMMARIZER: RoleSpec(Tier.SECONDARY, 1_500),
    Role.GUARDRAIL: RoleSpec(Tier.SECONDARY, 500),
}


class TokenBudgetExceeded(RuntimeError):
    """Raised before a model call once the run has used its token budget."""


class TokenMeter(BaseCallbackHandler):
    """Counts tokens across every model call in one run and enforces the run budget.

    Shared by all agents in a run (including parallel subagents), so it is thread-safe.
    The check happens *before* each call: a call already in flight is allowed to finish.
    """

    raise_error = True  # let TokenBudgetExceeded propagate instead of being logged and swallowed

    def __init__(self, budget: int):
        self.budget = budget
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.total)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, **kwargs: Any) -> None:
        if self.total >= self.budget:
            raise TokenBudgetExceeded(f"run used {self.total:,} of {self.budget:,} tokens")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage_in = usage_out = 0
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                usage_in += usage.get("input_tokens", 0)
                usage_out += usage.get("output_tokens", 0)
        with self._lock:
            self.input_tokens += usage_in
            self.output_tokens += usage_out
            self.calls += 1


class ModelGateway:
    def __init__(self, settings: Settings | None = None, meter: TokenMeter | None = None):
        self.settings = settings or get_settings()
        self.meter = meter

    @classmethod
    def for_run(cls, settings: Settings | None = None) -> "ModelGateway":
        settings = settings or get_settings()
        return cls(settings, TokenMeter(settings.run_token_budget))

    def model_name(self, tier: Tier) -> str:
        return self.settings.model_primary if tier is Tier.PRIMARY else self.settings.model_secondary

    def chat_model(self, role: Role, *, tier: Tier | None = None) -> BaseChatModel:
        """A single configured model for a role. Use this where a framework needs a real chat model
        (e.g. deepagents `create_deep_agent`), and attach fallback there."""
        spec = ROLE_SPECS[role]
        s = self.settings
        if s.openrouter_api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
        return ChatOpenAI(
            model=self.model_name(tier or spec.tier),
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key,
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            default_headers={"HTTP-Referer": s.openrouter_app_url, "X-Title": s.openrouter_app_name},
            callbacks=[self.meter] if self.meter else None,
            metadata={"role": role.value},
        )

    def fallback_model(self, role: Role) -> BaseChatModel:
        spec = ROLE_SPECS[role]
        other = Tier.SECONDARY if spec.tier is Tier.PRIMARY else Tier.PRIMARY
        return self.chat_model(role, tier=other)

    def runnable(self, role: Role) -> Runnable:
        """The role's model with automatic fallback to the other tier.

        A budget breach still stops the run: the fallback shares the same meter, so it
        raises TokenBudgetExceeded before making any call.
        """
        return self.chat_model(role).with_fallbacks([self.fallback_model(role)])

    def structured(self, role: Role, schema: type) -> Runnable:
        """The role's model constrained to return `schema`, with the same fallback.

        Uses provider-native JSON schema output. Forced tool calling (method="function_calling")
        is rejected by OpenRouter's Claude models: tool_choice "any"/"tool" is unsupported.
        """
        def bind(model: BaseChatModel) -> Runnable:
            return model.with_structured_output(schema, method="json_schema")

        return bind(self.chat_model(role)).with_fallbacks([bind(self.fallback_model(role))])
