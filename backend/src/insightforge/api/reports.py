"""Report endpoints: list, read, export, and read-only share links.

    GET    /reports                         the user's reports, newest first
    GET    /reports/{id}                    report + citation data (sources, evidence passages, fact-check)
    GET    /reports/{id}/export?format=     md | pdf
    POST   /reports/{id}/share              signed read-only link, expires in 7 days
    DELETE /reports/{id}/share/{jti}        revoke a link early
    GET    /share/{token}                   the shared report (no login; token-scoped, revocable)
    GET    /share/{token}/export?format=
"""

import re
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from insightforge.api.auth import CurrentUser, issue_share_token, verify_share_token
from insightforge.api.views import ReportSummary, ReportView, report_view
from insightforge.config import get_settings
from insightforge.export.render import ExportUnavailable, report_pdf
from insightforge.store.models import Report, RevokedShareLink, Run

router = APIRouter(tags=["reports"])
Format = Literal["md", "pdf"]


class ShareLink(BaseModel):
    url: str
    token: str
    jti: str
    expires_at: str


async def _report(request: Request, report_id: uuid.UUID, user_id: uuid.UUID | None) -> tuple[Report, str]:
    async with request.app.state.sessionmaker() as session:
        report = await session.get(Report, report_id)
        if report is None or (user_id is not None and report.user_id != user_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
        brief = await session.scalar(select(Run.brief).where(Run.id == report.run_id))
    return report, brief or ""


def _filename(title: str, ext: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"
    return f"{slug}.{ext}"


def _export(report: Report, brief: str, fmt: Format) -> Response:
    if fmt == "md":
        return Response(report.markdown, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{_filename(report.title, "md")}"'})
    try:
        pdf = report_pdf(report.markdown, report.title, f"Brief: {brief} · {report.created_at:%Y-%m-%d}")
    except ExportUnavailable as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{_filename(report.title, "pdf")}"'})


@router.get("/reports")
async def list_reports(request: Request, user_id: CurrentUser, limit: int = 20, offset: int = 0) -> list[ReportSummary]:
    async with request.app.state.sessionmaker() as session:
        rows = (await session.execute(
            select(Report, Run.brief).join(Run, Run.id == Report.run_id).where(Report.user_id == user_id)
            .order_by(Report.created_at.desc()).limit(min(limit, 100)).offset(max(offset, 0)))).all()
    return [ReportSummary(id=r.id, run_id=r.run_id, title=r.title, brief=brief, created_at=r.created_at,
                          pass_rate=(r.factcheck_summary or {}).get("pass_rate")) for r, brief in rows]


@router.get("/reports/{report_id}")
async def get_report(request: Request, report_id: uuid.UUID, user_id: CurrentUser) -> ReportView:
    return report_view(*await _report(request, report_id, user_id))


@router.get("/reports/{report_id}/export")
async def export_report(request: Request, report_id: uuid.UUID, user_id: CurrentUser, format: Format = "md") -> Response:
    return _export(*await _report(request, report_id, user_id), format)


@router.post("/reports/{report_id}/share")
async def share_report(request: Request, report_id: uuid.UUID, user_id: CurrentUser) -> ShareLink:
    await _report(request, report_id, user_id)
    token, jti, expires = issue_share_token(report_id)
    return ShareLink(url=f"{get_settings().app_base_url}/share/{token}", token=token, jti=jti,
                     expires_at=expires.isoformat())


@router.delete("/reports/{report_id}/share/{jti}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_share(request: Request, report_id: uuid.UUID, jti: str, user_id: CurrentUser) -> Response:
    await _report(request, report_id, user_id)
    async with request.app.state.sessionmaker() as session:
        await session.execute(insert(RevokedShareLink).values(jti=jti[:64]).on_conflict_do_nothing())
        await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/share/{token}")
async def shared_report(request: Request, token: str) -> ReportView:
    report_id = await verify_share_token(token, request.app.state.sessionmaker)
    return report_view(*await _report(request, report_id, None))


@router.get("/share/{token}/export")
async def shared_export(request: Request, token: str, format: Format = "md") -> Response:
    report_id = await verify_share_token(token, request.app.state.sessionmaker)
    return _export(*await _report(request, report_id, None), format)
