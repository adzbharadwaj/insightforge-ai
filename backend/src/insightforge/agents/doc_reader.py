"""Doc Reader subagent spec (component 08), spawned by the Planner via the `task` tool."""

from deepagents import SubAgent
from langchain_core.tools import BaseTool

from insightforge.agents.prompts import DOC_READER_PROMPT
from insightforge.agents.subagent import isolated_subagent
from insightforge.graph.state import ResearchResult
from insightforge.llm.gateway import ModelGateway, Role

NAME = "doc-reader"


def doc_reader_spec(gateway: ModelGateway, doc_search_tool: BaseTool, max_tool_calls: int) -> SubAgent:
    return isolated_subagent(
        name=NAME,
        description="Answers one sub-task from the user's uploaded PDFs and returns chunk references.",
        system_prompt=DOC_READER_PROMPT,
        gateway=gateway,
        role=Role.RESEARCHER,
        tools=[doc_search_tool],
        max_tool_calls=max_tool_calls,
        response_schema=ResearchResult,
    )
