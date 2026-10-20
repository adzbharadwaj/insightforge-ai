"""Shared builder for isolated subagents spawned through the Planner's `task` tool.

Every subagent gets the same guard rails: only the tools it is given plus read-only
`read_file`, a hard tool-call limit, a fallback model, and a typed final answer via
provider-native JSON schema (OpenRouter's Claude models reject the forced tool_choice
that ToolStrategy needs).
"""

from deepagents import FilesystemMiddleware, SubAgent
from langchain.agents.middleware import ModelFallbackMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from insightforge.llm.gateway import ModelGateway, Role


def isolated_subagent(
    *,
    name: str,
    description: str,
    system_prompt: str,
    gateway: ModelGateway,
    role: Role,
    tools: list[BaseTool],
    max_tool_calls: int,
    response_schema: type[BaseModel],
) -> SubAgent:
    return {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "model": gateway.chat_model(role),
        "tools": tools,
        "middleware": [
            # Replaces the default filesystem stack: no write/edit/ls/glob/grep. deepagents requires read_file,
            # which only reads this agent's own in-memory offloaded tool results.
            FilesystemMiddleware(tools=["read_file"]),
            ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="end"),
            ModelFallbackMiddleware(gateway.fallback_model(role)),
        ],
        "response_format": ProviderStrategy(response_schema),
    }
