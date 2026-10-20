"""Web Search subagent spec (component 07), spawned by the Planner via the `task` tool."""

from deepagents import SubAgent
from langchain_core.tools import BaseTool

from insightforge.agents.prompts import WEB_RESEARCHER_PROMPT
from insightforge.agents.subagent import isolated_subagent
from insightforge.graph.state import ResearchResult
from insightforge.llm.gateway import ModelGateway, Role

NAME = "web-researcher"


def web_researcher_spec(gateway: ModelGateway, web_search_tool: BaseTool, max_tool_calls: int) -> SubAgent:
    return isolated_subagent(
        name=NAME,
        description="Researches one sub-task on the live web and returns source references.",
        system_prompt=WEB_RESEARCHER_PROMPT,
        gateway=gateway,
        role=Role.RESEARCHER,
        tools=[web_search_tool],
        max_tool_calls=max_tool_calls,
        response_schema=ResearchResult,
    )
