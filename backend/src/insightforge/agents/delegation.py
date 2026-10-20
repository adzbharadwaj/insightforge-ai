"""Delegation middleware: isolation enforced in code (component 06 → 07/08/09).

The Planner's `task` calls carry only an assignment id ("st-2", "entities"). This
middleware, driven by a Ledger that owns those ids:

- rejects unknown ids, duplicates and the wrong subagent type (including the
  auto-added general-purpose subagent, which is never allowed);
- replaces the description with the fixed prompt the ledger renders from typed
  data, so a subagent can never receive the brief or sibling output;
- applies a per-assignment time budget;
- hands the subagent's output to the ledger, which validates and records it and
  decides what the Planner sees.

Two ledgers exist: ResearchRecorder (wave 1, one assignment per sub-task) and
EntityWave in agents/entity_extractor.py (wave 2, source ids only).
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.agents.prompts import render_subtask_prompt
from insightforge.graph.state import ResearchResult, SubTaskSpec
from insightforge.llm.gateway import TokenBudgetExceeded
from insightforge.store.models import SourceType, SubTask, SubTaskStatus
from insightforge.store.run_log import RunLog
from insightforge.store.sources import SourceResolver

logger = logging.getLogger(__name__)

SUBAGENT_FOR_SOURCE: dict[SourceType, str] = {
    SourceType.WEB: "web-researcher",
    SourceType.DOCS: "doc-reader",
    # Prior sessions already reach the Planner as compressed summaries (and related entities).
    # No memory-reader subagent is registered, so `memory` sub-tasks are never planned.
    SourceType.MEMORY: "memory-reader",
}


class Ledger(Protocol):
    started: set[str]

    def assignment(self, task_id: str) -> tuple[str, str] | None:
        """(subagent_type, isolated prompt) for a valid id, else None."""

    def valid_ids(self) -> list[str]: ...

    async def mark_running(self, task_id: str) -> None: ...

    async def complete(self, task_id: str, output: str | None) -> str:
        """Validate and record the subagent's final output; return what the Planner sees."""

    async def fail(self, task_id: str, error: str) -> str: ...


class ResearchRecorder:
    """Wave-1 ledger: one ResearchResult per sub-task, status mirrored into Postgres for the UI."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        resolver: SourceResolver,
        log: RunLog,
        specs: list[SubTaskSpec],
    ):
        self._sessionmaker = sessionmaker
        self.resolver = resolver
        self.log = log
        self.specs = {s.id: s for s in specs}
        self.results: dict[str, ResearchResult] = {}
        self.started: set[str] = set()

    # --- Ledger -------------------------------------------------------------------

    def assignment(self, task_id: str) -> tuple[str, str] | None:
        spec = self.specs.get(task_id)
        return (SUBAGENT_FOR_SOURCE[spec.source_type], render_subtask_prompt(spec)) if spec else None

    def valid_ids(self) -> list[str]:
        return sorted(self.specs)

    async def mark_running(self, task_id: str) -> None:
        self.started.add(task_id)
        await self._set_status(task_id, status=SubTaskStatus.RUNNING, started_at=datetime.now(UTC))
        await self.log.progress("planner", f"delegated {task_id}", sub_task_id=task_id)

    async def complete(self, task_id: str, output: str | None) -> str:
        return (await self.record(_parse_result(output, task_id))).model_dump_json()

    async def fail(self, task_id: str, error: str) -> str:
        return (await self.record(ResearchResult(sub_task_id=task_id, status=SubTaskStatus.FAILED,
                                                 error=error))).model_dump_json()

    # --- recording ----------------------------------------------------------------

    async def _set_status(self, sub_task_id: str, **values: Any) -> None:
        async with self._sessionmaker() as session:
            await session.execute(
                update(SubTask)
                .where(SubTask.run_id == self.resolver.run_id, SubTask.task_key == sub_task_id)
                .values(**values)
            )
            await session.commit()

    async def record(self, result: ResearchResult) -> ResearchResult:
        resolved = {s.source_id: s for s in await self.resolver.resolve_many([ref.source_id for ref in result.refs])}
        invented = [ref.source_id for ref in result.refs if ref.source_id not in resolved]
        if invented:
            await self.log.emit("planner", "error", f"dropped {len(invented)} unknown source id(s)",
                                sub_task_id=result.sub_task_id, source_ids=invented)
        # Kind and credibility come from storage, not from what the model copied.
        refs = [
            ref.model_copy(update={"kind": resolved[ref.source_id].kind,
                                   "credibility": resolved[ref.source_id].credibility})
            for ref in result.refs if ref.source_id in resolved
        ]
        status = result.status
        if status is SubTaskStatus.DONE and not refs:
            status = SubTaskStatus.FAILED
        result = result.model_copy(update={
            "refs": refs,
            "status": status,
            "error": result.error or ("no usable sources" if status is SubTaskStatus.FAILED else None),
        })
        self.results[result.sub_task_id] = result
        await self._set_status(
            result.sub_task_id,
            status=result.status,
            summary=f"{len(refs)} source(s)",
            error=result.error,
            finished_at=datetime.now(UTC),
        )
        await self.log.progress(result.sub_task_id, f"{result.sub_task_id}: {result.status.value}",
                                sub_task_id=result.sub_task_id, sources=len(refs))
        return result

    async def finalize(self) -> list[ResearchResult]:
        """Every sub-task gets a result: ones the Planner never delegated are marked Failed."""
        for sub_task_id in self.specs:
            if sub_task_id not in self.results:
                await self.fail(sub_task_id, "not delegated")
        return [self.results[i] for i in self.specs]


class DelegationMiddleware(AgentMiddleware):
    name = "delegation"

    def __init__(self, ledger: Ledger, timeout_s: float):
        super().__init__()
        self.ledger = ledger
        self.timeout_s = timeout_s

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        call = request.tool_call
        if call["name"] != "task":
            return await handler(request)

        task_id = str(call["args"].get("description", "")).strip()
        assignment = self.ledger.assignment(task_id)
        if assignment is None:
            return _error(call, f"Unknown id {task_id!r}. Valid ids: {self.ledger.valid_ids()}. "
                                "Set description to exactly one id.")
        subagent_type, prompt = assignment
        if call["args"].get("subagent_type") != subagent_type:
            return _error(call, f"{task_id} must use subagent_type {subagent_type!r}.")
        if task_id in self.ledger.started:
            return _error(call, f"{task_id} was already delegated.")

        isolated = {**call, "args": {"description": prompt, "subagent_type": subagent_type}}
        await self.ledger.mark_running(task_id)
        try:
            outcome = await asyncio.wait_for(handler(request.override(tool_call=isolated)), self.timeout_s)
        except TimeoutError:
            return _message(call, await self.ledger.fail(task_id, f"timed out after {self.timeout_s:.0f}s"))
        except TokenBudgetExceeded:
            raise  # a run-level budget breach must stop the run, not just this task
        except Exception as exc:
            logger.exception("delegated task %s failed", task_id)
            return _message(call, await self.ledger.fail(task_id, f"{type(exc).__name__}: {exc}"[:300]))

        msg = _tool_message(outcome)
        content = await self.ledger.complete(task_id, msg.content if msg else None)
        return _replace_content(outcome, call, content)


def _tool_message(outcome: ToolMessage | Command) -> ToolMessage | None:
    if isinstance(outcome, ToolMessage):
        return outcome
    if isinstance(outcome, Command) and isinstance(outcome.update, dict):
        for msg in outcome.update.get("messages", []):
            if isinstance(msg, ToolMessage):
                return msg
    return None


def _parse_result(output: str | None, sub_task_id: str) -> ResearchResult:
    try:
        data = json.loads(output) if output else {}
        data["sub_task_id"] = sub_task_id  # the id we delegated wins over whatever the model wrote
        return ResearchResult.model_validate(data)
    except (ValueError, TypeError, ValidationError) as exc:
        return ResearchResult(sub_task_id=sub_task_id, status="Failed",
                              error=f"unparseable researcher output: {type(exc).__name__}")


def _replace_content(outcome: ToolMessage | Command, call: dict, content: str) -> ToolMessage | Command:
    """Return the ledger's validated content to the Planner, keeping any state update."""
    message = _message(call, content)
    if isinstance(outcome, Command) and isinstance(outcome.update, dict):
        others = [m for m in outcome.update.get("messages", []) if not isinstance(m, ToolMessage)]
        return Command(update={**outcome.update, "messages": [*others, message]})
    return message


def _message(call: dict, content: str) -> ToolMessage:
    return ToolMessage(content, tool_call_id=call["id"], name="task")


def _error(call: dict, text: str) -> ToolMessage:
    return ToolMessage(text, tool_call_id=call["id"], name="task", status="error")
