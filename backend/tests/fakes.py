"""Deterministic stand-ins for LLMs, so agent wiring can be tested without API keys."""

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from insightforge.config import Settings
from insightforge.llm.gateway import ModelGateway, Role


class RuleModel(BaseChatModel):
    """A chat model whose reply is a pure function of the messages it receives.

    Stateless, so one instance can safely serve parallel subagent runs. Every input
    is recorded in `seen` so tests can assert what an agent was (and wasn't) shown.
    """

    rule: Any
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "rule"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.rule(messages))])

    def bind_tools(self, tools, **kwargs) -> "RuleModel":
        return self


def tool_call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def ai_calls(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


class FakeGateway(ModelGateway):
    def __init__(self, models: dict[Role, BaseChatModel], structured: dict[Role, Callable[[Any], Any]] | None = None):
        super().__init__(Settings(_env_file=None, openrouter_api_key="sk-test"))
        self.models = models
        self.structured_fns = structured or {}

    def chat_model(self, role: Role, *, tier=None) -> BaseChatModel:
        return self.models[role]

    def fallback_model(self, role: Role) -> BaseChatModel:
        return self.models[role]

    def structured(self, role: Role, schema: type) -> Runnable:
        return RunnableLambda(self.structured_fns[role])
