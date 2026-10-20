"""Watchlist + alert endpoints. The scheduler process picks up changes on its next reconcile.

    GET    /watchlist                     items with last/next run and latest alert
    POST   /watchlist                     {topic, cadence, webhook_url?}: same guardrails as a brief
    PATCH  /watchlist/{id}                {active?, cadence?, webhook_url?}
    DELETE /watchlist/{id}
    POST   /watchlist/{id}/run-now        queue a run immediately (also how a new item gets its baseline fast)
    GET    /watchlist/{id}/diff           the latest change summary for the item
    GET    /alerts                        alert feed across all items (newest first)
    POST   /alerts/{id}/read
"""

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from insightforge.api.auth import CurrentUser
from insightforge.config import get_settings
from insightforge.guardrails.input import InputRejected, check_brief, check_scope
from insightforge.guardrails.webhooks import WebhookRejected, validate_webhook_url
from insightforge.store.models import Alert, Cadence, Report, WatchlistItem
from insightforge.watchlist.service import start_watch_run

router = APIRouter(tags=["watchlist"])
MAX_ITEMS_PER_USER = 25


class WatchCreate(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    cadence: Cadence = Cadence.WEEKLY
    webhook_url: str | None = None


class WatchUpdate(BaseModel):
    active: bool | None = None
    cadence: Cadence | None = None
    webhook_url: Annotated[str | None, Field(description='"" removes the webhook')] = None


class AlertView(BaseModel):
    id: uuid.UUID
    watchlist_item_id: uuid.UUID
    topic: str
    run_id: uuid.UUID
    report_id: uuid.UUID | None
    change_summary: dict
    read: bool
    webhook_status: str | None
    created_at: dt.datetime


class WatchView(BaseModel):
    id: uuid.UUID
    topic: str
    cadence: Cadence
    active: bool
    webhook_url: str | None
    last_run: dt.datetime | None
    next_run: dt.datetime | None
    created_at: dt.datetime
    latest_alert: AlertView | None = None


def _rejected(code: str, message: str) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"code": code, "message": message})


async def _owned(request: Request, item_id: uuid.UUID, user_id: uuid.UUID) -> WatchlistItem:
    async with request.app.state.sessionmaker() as session:
        item = await session.get(WatchlistItem, item_id)
    if item is None or item.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "watchlist item not found")
    return item


async def _alert_views(request: Request, user_id: uuid.UUID, *, item_id: uuid.UUID | None = None,
                       limit: int = 50) -> list[AlertView]:
    q = (select(Alert, WatchlistItem.topic, Report.id).join(WatchlistItem, WatchlistItem.id == Alert.watchlist_item_id)
         .outerjoin(Report, Report.run_id == Alert.run_id).where(Alert.user_id == user_id))
    if item_id:
        q = q.where(Alert.watchlist_item_id == item_id)
    async with request.app.state.sessionmaker() as session:
        rows = (await session.execute(q.order_by(Alert.created_at.desc()).limit(limit))).all()
    return [AlertView(id=a.id, watchlist_item_id=a.watchlist_item_id, topic=topic, run_id=a.run_id, report_id=rid,
                      change_summary=a.change_summary, read=a.read, webhook_status=a.webhook_status,
                      created_at=a.created_at) for a, topic, rid in rows]


def _view(item: WatchlistItem, latest: AlertView | None = None) -> WatchView:
    return WatchView(id=item.id, topic=item.topic, cadence=item.cadence, active=item.active,
                     webhook_url=item.webhook_url, last_run=item.last_run, next_run=item.next_run,
                     created_at=item.created_at, latest_alert=latest)


@router.get("/watchlist")
async def list_items(request: Request, user_id: CurrentUser) -> list[WatchView]:
    async with request.app.state.sessionmaker() as session:
        items = (await session.scalars(select(WatchlistItem).where(WatchlistItem.user_id == user_id)
                                       .order_by(WatchlistItem.created_at.desc()))).all()
    latest: dict[uuid.UUID, AlertView] = {}
    for a in await _alert_views(request, user_id, limit=200):
        latest.setdefault(a.watchlist_item_id, a)
    return [_view(i, latest.get(i.id)) for i in items]


@router.post("/watchlist", status_code=status.HTTP_201_CREATED)
async def create_item(request: Request, body: WatchCreate, user_id: CurrentUser) -> WatchView:
    try:
        topic = check_brief(body.topic, get_settings())
        await check_scope(request.app.state.guardrail_gateway, topic)
        webhook = await validate_webhook_url(body.webhook_url) if body.webhook_url else None
    except InputRejected as exc:
        raise _rejected(exc.code, exc.message) from exc
    except WebhookRejected as exc:
        raise _rejected("webhook_rejected", str(exc)) from exc
    async with request.app.state.sessionmaker() as session:
        count = len((await session.scalars(select(WatchlistItem.id).where(WatchlistItem.user_id == user_id))).all())
        if count >= MAX_ITEMS_PER_USER:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, f"Limit of {MAX_ITEMS_PER_USER} watchlist items.")
        item = WatchlistItem(user_id=user_id, topic=topic, cadence=body.cadence, webhook_url=webhook)
        session.add(item)
        await session.commit()
    return _view(item)


@router.patch("/watchlist/{item_id}")
async def update_item(request: Request, item_id: uuid.UUID, body: WatchUpdate, user_id: CurrentUser) -> WatchView:
    await _owned(request, item_id, user_id)
    values: dict = {}
    if body.active is not None:
        values["active"] = body.active
    if body.cadence is not None:
        values["cadence"] = body.cadence
        values["next_run"] = None  # the scheduler recomputes it on its next reconcile
    if body.webhook_url is not None:
        try:
            values["webhook_url"] = await validate_webhook_url(body.webhook_url) if body.webhook_url else None
        except WebhookRejected as exc:
            raise _rejected("webhook_rejected", str(exc)) from exc
    async with request.app.state.sessionmaker() as session:
        item = await session.get(WatchlistItem, item_id)
        for k, v in values.items():
            setattr(item, k, v)
        await session.commit()
    return _view(item)


@router.delete("/watchlist/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(request: Request, item_id: uuid.UUID, user_id: CurrentUser) -> None:
    await _owned(request, item_id, user_id)
    async with request.app.state.sessionmaker() as session:
        await session.delete(await session.get(WatchlistItem, item_id))
        await session.commit()


@router.post("/watchlist/{item_id}/run-now", status_code=status.HTTP_202_ACCEPTED)
async def run_now(request: Request, item_id: uuid.UUID, user_id: CurrentUser) -> dict:
    await _owned(request, item_id, user_id)
    run_id = await start_watch_run(request.app.state.sessionmaker, request.app.state.queue, item_id)
    if run_id is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This item is paused or already has a run in progress.")
    return {"run_id": str(run_id), "status_url": f"/research/status/{run_id}"}


@router.get("/watchlist/{item_id}/diff")
async def latest_diff(request: Request, item_id: uuid.UUID, user_id: CurrentUser) -> AlertView | None:
    await _owned(request, item_id, user_id)
    alerts = await _alert_views(request, user_id, item_id=item_id, limit=1)
    return alerts[0] if alerts else None


@router.get("/alerts")
async def alert_feed(request: Request, user_id: CurrentUser, limit: int = 50) -> list[AlertView]:
    return await _alert_views(request, user_id, limit=min(limit, 200))


@router.post("/alerts/{alert_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_read(request: Request, alert_id: uuid.UUID, user_id: CurrentUser) -> None:
    async with request.app.state.sessionmaker() as session:
        alert = await session.get(Alert, alert_id)
        if alert is None or alert.user_id != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "alert not found")
        alert.read = True
        await session.commit()
