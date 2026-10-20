"""Research endpoints (component 15): submit a brief, follow it live, answer the clarifying question.

    POST /research/run                 multipart: brief, web, docs, memory, files[] → 202 + run id
    GET  /research/estimate            runtime estimate for the brief panel
    GET  /research/{run_id}            snapshot (status, sub-tasks, question, report id)
    GET  /research/status/{run_id}     SSE progress stream (token in ?token= for EventSource)
    POST /research/{run_id}/clarify    answer the Planner's question; resumes the run
    POST /research/{run_id}/resolve    choose a side for each conflicting fact (opt-in review)
    POST /research/{run_id}/retry      resume a failed run from its last checkpoint
    POST /research/{run_id}/cancel     cancel a queued or waiting run

Runs execute in the worker (insightforge.worker); this process only validates and queues.
"""

import datetime as dt
import logging
import uuid
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sse_starlette.sse import EventSourceResponse

from insightforge.agents.conflict_review import KEEP_BOTH
from insightforge.api.auth import CurrentUser
from insightforge.api.sse import run_event_stream
from insightforge.api.views import TERMINAL, RunSnapshot, estimate_runtime_s, owned_run, run_snapshot
from insightforge.config import get_settings
from insightforge.guardrails.input import InputRejected, check_brief, check_scope, check_source_prefs
from insightforge.guardrails.untrusted import clean_text, scan_injection
from insightforge.llm.gateway import ModelGateway
from insightforge.store.models import Run, RunStatus
from insightforge.store.run_log import RunLog
from insightforge.tools.credibility import normalize_domains
from insightforge.tools.docling_ingest import PdfRejected, safe_filename, save_upload, validate_pdfs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


class Queue(Protocol):
    async def enqueue_job(self, function: str, *args: Any, _job_id: str | None = None) -> Any: ...


class RunCreated(BaseModel):
    run_id: uuid.UUID
    status: RunStatus
    status_url: str
    eta_seconds: int


class ClarifyRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=500)


class ResolveRequest(BaseModel):
    choices: dict[str, str] = Field(default_factory=dict, description='conflict_id → option_id or "keep_both"')


def _reject(exc: InputRejected) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"code": exc.code, "message": exc.message})


async def _rate_limit(request: Request, user_id: uuid.UUID) -> None:
    s = get_settings()
    maker = request.app.state.sessionmaker
    async with maker() as session:
        active = await session.scalar(select(func.count()).select_from(Run).where(
            Run.user_id == user_id, Run.status.not_in(list(TERMINAL))))
    if active >= s.max_active_runs_per_user:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"You already have {active} run(s) in progress (limit {s.max_active_runs_per_user}).")
    key = f"rl:runs:{user_id}:{dt.datetime.now(dt.UTC):%Y%m%d%H}"
    count = await request.app.state.redis.incr(key)
    if count == 1:
        await request.app.state.redis.expire(key, 3600)
    if count > s.runs_per_hour_per_user:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Limit of {s.runs_per_hour_per_user} runs per hour reached.")


@router.get("/estimate")
async def estimate(sub_tasks: int = 5, pdfs: int = 0) -> dict:
    return {"eta_seconds": estimate_runtime_s(max(1, min(sub_tasks, 6)), max(0, pdfs))}


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    request: Request,
    user_id: CurrentUser,
    brief: Annotated[str, Form()],
    web: Annotated[bool, Form()] = True,
    docs: Annotated[bool, Form()] = False,
    memory: Annotated[bool, Form()] = True,
    blocked_domains: Annotated[str, Form(description="Never use these domains (comma or newline separated)")] = "",
    deprioritized_domains: Annotated[str, Form(description="Use, but rank lower (credibility halved)")] = "",
    review_conflicts: Annotated[bool, Form(description="Pause before writing to resolve conflicting facts")] = False,
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> RunCreated:
    s = get_settings()
    uploads = [(safe_filename(f.filename or "document.pdf"), await f.read()) for f in files or []]
    try:
        text = check_brief(brief, s)
        prefs = check_source_prefs({"web": web, "docs": docs, "memory": memory}, len(uploads))
        prefs["blocked_domains"] = normalize_domains(blocked_domains)
        prefs["deprioritized_domains"] = [d for d in normalize_domains(deprioritized_domains)
                                          if d not in prefs["blocked_domains"]]
        prefs["review_conflicts"] = review_conflicts
        validate_pdfs(uploads, s)
    except InputRejected as exc:
        raise _reject(exc) from exc
    except PdfRejected as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"code": "pdf_rejected", "message": str(exc)}) from exc

    await _rate_limit(request, user_id)
    gateway: ModelGateway = request.app.state.guardrail_gateway
    try:
        await check_scope(gateway, text)
    except InputRejected as exc:
        raise _reject(exc) from exc
    except Exception as exc:  # the scope check is a gate: if it can't run, don't queue the run
        logger.exception("scope check failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The brief check is unavailable; try again.") from exc

    maker = request.app.state.sessionmaker
    async with maker() as session:
        run = Run(user_id=user_id, brief=text, source_prefs=prefs)
        session.add(run)
        await session.commit()
    saved = [{"filename": name, "path": str(save_upload(user_id, run.id, name, data, s))} for name, data in uploads]
    await RunLog(maker, run.id).progress("api", "run queued", pdfs=len(saved))
    queue: Queue = request.app.state.queue
    await queue.enqueue_job("start_run", str(run.id), saved, _job_id=f"run:{run.id}")
    return RunCreated(run_id=run.id, status=RunStatus.QUEUED, status_url=f"/research/status/{run.id}",
                      eta_seconds=estimate_runtime_s(5, len(saved)))


async def _run_or_404(request: Request, run_id: uuid.UUID, user_id: uuid.UUID) -> Run:
    run = await owned_run(request.app.state.sessionmaker, run_id, user_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return run


@router.get("/status/{run_id}")
async def stream_status(
    request: Request,
    run_id: uuid.UUID,
    user_id: CurrentUser,
    last_event_id: Annotated[str | None, Header()] = None,
) -> EventSourceResponse:
    run = await _run_or_404(request, run_id, user_id)
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    stream = run_event_stream(request.app.state.sessionmaker, run, start, request.is_disconnected)
    return EventSourceResponse(stream, ping=15)


@router.get("/{run_id}")
async def get_run(request: Request, run_id: uuid.UUID, user_id: CurrentUser) -> RunSnapshot:
    return await run_snapshot(request.app.state.sessionmaker, await _run_or_404(request, run_id, user_id))


@router.post("/{run_id}/clarify", status_code=status.HTTP_202_ACCEPTED)
async def clarify(request: Request, run_id: uuid.UUID, body: ClarifyRequest, user_id: CurrentUser) -> RunSnapshot:
    run = await _run_or_404(request, run_id, user_id)
    answer = " ".join(clean_text(body.answer).split())
    if not answer:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"code": "empty_answer", "message": "Answer is empty."})
    if scan_injection(answer).suspicious:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            {"code": "answer_injection", "message": "Answer the question; don't instruct the system."})
    maker = request.app.state.sessionmaker
    async with maker() as session:  # conditional update: only one answer can win
        result = await session.execute(update(Run).where(
            Run.id == run.id, Run.status == RunStatus.AWAITING_CLARIFICATION).values(status=RunStatus.PLANNING))
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "This run is not waiting for an answer.")
    await request.app.state.queue.enqueue_job("resume_run", str(run.id), answer, _job_id=f"resume:{run.id}")
    return await run_snapshot(maker, run)


@router.post("/{run_id}/resolve", status_code=status.HTTP_202_ACCEPTED)
async def resolve(request: Request, run_id: uuid.UUID, body: ResolveRequest, user_id: CurrentUser) -> RunSnapshot:
    """Choose which version of each conflicting fact the report uses; unanswered conflicts keep both."""
    run = await _run_or_404(request, run_id, user_id)
    pending = {c["conflict_id"]: {o["option_id"] for o in c["options"]} for c in (run.review or {}).get("conflicts", [])}
    for cid, choice in body.choices.items():
        if cid not in pending or (choice != KEEP_BOTH and choice not in pending[cid]):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                {"code": "bad_choice", "message": f"Unknown conflict or option: {cid} → {choice}"})
    maker = request.app.state.sessionmaker
    async with maker() as session:  # conditional update: one decision wins
        result = await session.execute(update(Run).where(Run.id == run.id, Run.status == RunStatus.AWAITING_REVIEW)
                                       .values(status=RunStatus.SYNTHESIZING))
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "This run is not waiting for a review.")
    await request.app.state.queue.enqueue_job("resolve_run", str(run.id), body.choices, _job_id=f"resolve:{run.id}")
    return await run_snapshot(maker, run)


@router.post("/{run_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry(request: Request, run_id: uuid.UUID, user_id: CurrentUser) -> RunSnapshot:
    """Resume a failed run from its last checkpoint (finished stages and agents are not redone)."""
    run = await _run_or_404(request, run_id, user_id)
    s = get_settings()
    if run.attempts >= s.max_run_attempts:
        raise HTTPException(status.HTTP_409_CONFLICT, f"This run already had {run.attempts} attempts.")
    maker = request.app.state.sessionmaker
    async with maker() as session:  # conditional update: one retry at a time
        result = await session.execute(update(Run).where(Run.id == run.id, Run.status == RunStatus.FAILED)
                                       .values(status=RunStatus.QUEUED))
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only failed runs can be retried.")
    await request.app.state.queue.enqueue_job("retry_run", str(run.id), _job_id=f"retry:{run.id}:{run.attempts}")
    return await run_snapshot(maker, run)


@router.post("/{run_id}/cancel")
async def cancel(request: Request, run_id: uuid.UUID, user_id: CurrentUser) -> RunSnapshot:
    run = await _run_or_404(request, run_id, user_id)
    maker = request.app.state.sessionmaker
    cancellable = [RunStatus.QUEUED, RunStatus.AWAITING_CLARIFICATION]
    async with maker() as session:
        result = await session.execute(update(Run).where(Run.id == run.id, Run.status.in_(cancellable)).values(
            status=RunStatus.CANCELLED, finished_at=dt.datetime.now(dt.UTC), error="cancelled by the user"))
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only queued or waiting runs can be cancelled.")
    await RunLog(maker, run.id).progress("api", "run cancelled by the user")
    return await run_snapshot(maker, run)
