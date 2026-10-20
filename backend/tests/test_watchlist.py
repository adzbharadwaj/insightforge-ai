import datetime as dt
import hashlib
import hmac
import json
import uuid

import httpx
import pytest
from fakes import FakeGateway
from sqlalchemy import select

from insightforge.agents.diff import Materiality, MaterialityVerdicts, diff_claims
from insightforge.agents.writer import render_report
from insightforge.config import Settings, get_settings
from insightforge.graph.state import FactCheckSummary, WriterOutput, WrittenSection, WrittenSentence
from insightforge.guardrails.webhooks import (
    WebhookRejected,
    deliver,
    is_slack,
    sign,
    slack_payload,
    validate_webhook_url,
)
from insightforge.llm.gateway import Role
from insightforge.store.models import Alert, Cadence, Run, RunStatus, WatchlistItem
from insightforge.store.reports import save_report
from insightforge.watchlist.service import process_watch_result, start_watch_run

PUBLIC = lambda host: ["93.184.216.34"]
PRIVATE = lambda host: ["10.0.0.5"]
SECRET = "w" * 48


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(get_settings(), "webhook_signing_secret", SecretStr(SECRET))


# --- webhook guardrail (03d) ----------------------------------------------------------


async def test_webhook_urls_must_be_https_and_public():
    assert await validate_webhook_url(" https://hooks.example.com/x ", resolve=PUBLIC) == "https://hooks.example.com/x"
    for bad, resolve in [("http://hooks.example.com/x", PUBLIC), ("https://internal.example/x", PRIVATE),
                         ("https://127.0.0.1/x", PUBLIC), ("https://example.com/" + "a" * 2001, PUBLIC)]:
        with pytest.raises(WebhookRejected):
            await validate_webhook_url(bad, resolve=resolve)


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_delivery_is_signed_and_retries_server_errors():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(503 if len(seen) < 3 else 200)

    result = await deliver(client_for(handler), "https://hooks.example.com/x", {"a": 1}, resolve=PUBLIC, backoff_s=0)
    assert (result.status, result.attempts) == ("sent", 3)
    req = seen[-1]
    expected = sign(req.content, req.headers["X-InsightForge-Timestamp"], SECRET)
    assert req.headers["X-InsightForge-Signature"] == expected
    mac = hmac.new(SECRET.encode(), f"{req.headers['X-InsightForge-Timestamp']}.".encode() + req.content, hashlib.sha256)
    assert expected == f"sha256={mac.hexdigest()}"  # receivers verify with the documented scheme


async def test_client_errors_and_redirects_are_not_retried_or_followed():
    calls = []

    def rejects(request):
        calls.append(1)
        return httpx.Response(400)

    assert (await deliver(client_for(rejects), "https://h.example.com/x", {}, resolve=PUBLIC, backoff_s=0)).status == "failed"
    assert len(calls) == 1

    def redirects(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})

    result = await deliver(client_for(redirects), "https://h.example.com/x", {}, resolve=PUBLIC, backoff_s=0)
    assert result.status == "failed" and result.detail == "HTTP 302"


async def test_dns_is_rechecked_at_send_time():
    def never(request):
        raise AssertionError("must not send to a private address")

    result = await deliver(client_for(never), "https://was-public.example.com/x", {}, resolve=PRIVATE)
    assert result.status == "blocked"


def test_slack_format():
    assert is_slack("https://hooks.slack.com/services/T/B/x") and not is_slack("https://example.com/hook")
    summary = {"added_findings": [{"text": f"New deal {i}"} for i in range(7)],
               "changed_findings": [{"before": "raised $7.6M", "after": "raised $9M"}], "removed_findings": []}
    text = slack_payload("SEA payroll", summary, "https://app/report/1")["text"]
    assert "*New (7)*" in text and "…and 2 more" in text and "raised $9M (was: raised $7.6M)" in text
    assert "<https://app/report/1|Open report>" in text


# --- Diff Agent ------------------------------------------------------------------------


KIND_TO_CLASS = {"added": "new", "changed": "changed", "removed": "removed"}


def materiality(material_ids: set[str] | None = None, seen: list | None = None):
    """Fake judge: material candidates keep their kind; others are not material."""
    def fn(messages):
        lines = [line for line in messages[-1].content.splitlines() if line and not line.startswith(" ")]
        if seen is not None:
            seen.append(messages[-1].content)
        verdicts = []
        for line in lines:
            cid, kind = line.split()[0], line.split()[1].strip("[]")
            material = material_ids is None or cid in material_ids
            verdicts.append(Materiality(id=cid, classification=KIND_TO_CLASS[kind] if material else "not_material"))
        return MaterialityVerdicts(verdicts=verdicts)
    return fn


async def test_first_report_is_the_baseline():
    assert (await diff_claims(FakeGateway({}, {}), [], ["Salary Hero raised US$7.6 million."])).empty


async def test_changed_figures_are_caught_in_code_and_rephrasing_is_ignored():
    seen: list[str] = []
    gw = FakeGateway({}, {Role.DIFF: materiality(set(), seen)})
    old = ["Salary Hero raised US$7.6 million in a Series A led by Cento Ventures.",
           "Talenox serves small businesses in Singapore and Malaysia."]
    new = ["Salary Hero raised US$9 million in a Series A led by Cento Ventures.",
           "Talenox serves small businesses across Singapore and Malaysia."]
    summary = await diff_claims(gw, old, new)
    assert [(c.before, c.after) for c in summary.changed_findings] == [(old[0], new[0])]
    assert not summary.added_findings and not summary.removed_findings
    assert all("US$9 million" not in s for s in seen)  # the figure change never needed the model


async def test_only_material_additions_and_removals_are_kept():
    old = ["Talenox serves small businesses in Singapore and Malaysia.",
           "Swingvy offers a free plan for small teams in Malaysia."]
    new = ["Talenox serves small businesses in Singapore and Malaysia.",
           "Payboy launched an AI payroll assistant for Malaysian employers in September 2026.",
           "Payroll software helps businesses pay employees."]

    def judge(messages):
        lines = [line for line in messages[-1].content.splitlines() if line and not line.startswith(" ")]
        return MaterialityVerdicts(verdicts=[
            Materiality(id=line.split()[0], classification=(KIND_TO_CLASS[line.split()[1].strip("[]")]
                                                           if "Payboy" in line or "Swingvy" in line else "not_material"))
            for line in lines])

    summary = await diff_claims(FakeGateway({}, {Role.DIFF: judge}), old, new)
    assert [f.text for f in summary.added_findings] == [new[1]]  # trivia ("helps businesses pay") dropped
    assert [f.text for f in summary.removed_findings] == [old[1]]


# --- watchlist service ------------------------------------------------------------------


class FakeQueue:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, function, *args, _job_id=None):
        self.jobs.append((function, args, _job_id))


async def add_item(maker, user_id, **values) -> uuid.UUID:
    async with maker() as s:
        item = WatchlistItem(user_id=user_id, topic="SEA payroll software market", cadence=Cadence.DAILY, **values)
        s.add(item)
        await s.commit()
        return item.id


async def test_start_watch_run_queues_once_and_skips_paused_items(committed_run):
    maker, user_id, _ = committed_run
    queue = FakeQueue()
    item_id = await add_item(maker, user_id)
    run_id = await start_watch_run(maker, queue, item_id)
    assert queue.jobs == [("start_run", (str(run_id), []), f"run:{run_id}")]
    async with maker() as s:
        run, item = await s.get(Run, run_id), await s.get(WatchlistItem, item_id)
    assert run.watchlist_item_id == item_id and run.brief == "SEA payroll software market"
    assert item.last_run and item.next_run - item.last_run == dt.timedelta(days=1)

    assert await start_watch_run(maker, queue, item_id) is None  # a run is still in flight
    paused = await add_item(maker, user_id, active=False)
    assert await start_watch_run(maker, queue, paused) is None
    assert len(queue.jobs) == 1


async def complete_watch_run(maker, user_id, item_id, sentences: list[str]) -> uuid.UUID:
    from test_report import FUNDING, bundle_of

    async with maker() as s:
        run = Run(user_id=user_id, brief="SEA payroll software market", status=RunStatus.COMPLETE,
                  watchlist_item_id=item_id)
        s.add(run)
        await s.commit()
    out = WriterOutput(title="T", executive_summary=[],
                       sections=[WrittenSection(sub_task_id="st-1", heading="H",
                                                sentences=[WrittenSentence(text=t, evidence_ids=["ev-1"]) for t in sentences])])
    b = bundle_of(FUNDING)
    draft = render_report(out, b)
    await save_report(maker, run_id=run.id, user_id=user_id, draft=draft, markdown=draft.markdown, bundle=b,
                      fact_check=FactCheckSummary(total_checked=0, passed=0, failed=0, pass_rate=1.0, unverified_claims=[]))
    return run.id


async def test_change_becomes_an_alert_and_a_signed_webhook(committed_run, monkeypatch):
    from insightforge.watchlist import service

    maker, user_id, _ = committed_run
    item_id = await add_item(maker, user_id, webhook_url="https://hooks.example.com/x")

    def judge(messages):  # an update when the figure moved; anything else is not material
        verdicts = []
        for line in (ln for ln in messages[-1].content.splitlines() if ln and not ln.startswith(" ")):
            cid, kind = line.split()[0], line.split()[1].strip("[]")
            moved = "US$9 million" in line and kind == "changed"
            verdicts.append(Materiality(id=cid, classification="changed" if moved else "not_material"))
        return MaterialityVerdicts(verdicts=verdicts)

    gw = FakeGateway({}, {Role.DIFF: judge})
    posted: list[httpx.Request] = []
    http = client_for(lambda request: posted.append(request) or httpx.Response(200))
    real_deliver = service.deliver
    monkeypatch.setattr(service, "deliver", lambda client, url, payload: real_deliver(client, url, payload, resolve=PUBLIC))

    first = await complete_watch_run(maker, user_id, item_id, ["Salary Hero raised US$7.6 million in a Series A."])
    assert await process_watch_result(maker, gw, http, first) is None  # baseline: no alert

    same = await complete_watch_run(maker, user_id, item_id, ["Salary Hero raised US$7.6 million in its Series A."])
    assert await process_watch_result(maker, gw, http, same) is None  # rephrasing only: no alert

    changed = await complete_watch_run(maker, user_id, item_id, ["Salary Hero raised US$9 million in its Series A."])
    alert = await process_watch_result(maker, gw, http, changed)
    assert alert is not None and alert.webhook_status == "sent"
    assert alert.change_summary["changed_findings"][0]["after"] == "Salary Hero raised US$9 million in its Series A."
    body = json.loads(posted[0].content)
    assert body["event"] == "watchlist.changed" and body["report_url"].startswith(get_settings().app_base_url)
    async with maker() as s:
        assert len((await s.scalars(select(Alert).where(Alert.watchlist_item_id == item_id))).all()) == 1


# --- scheduler -------------------------------------------------------------------------


async def test_reconcile_adds_updates_and_removes_jobs(committed_run, monkeypatch):
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from insightforge.watchlist import scheduler as sched

    maker, user_id, _ = committed_run
    monkeypatch.setattr(sched, "get_sessionmaker", lambda: maker)
    daily = await add_item(maker, user_id)
    paused = await add_item(maker, user_id, active=False)
    scheduler = AsyncIOScheduler(jobstores={"default": MemoryJobStore()}, timezone=dt.UTC)
    scheduler.start(paused=True)
    try:
        await sched.reconcile(scheduler)
        ids = {j.id for j in scheduler.get_jobs()}
        assert f"watch:{daily}" in ids and f"watch:{paused}" not in ids
        assert scheduler.get_job(f"watch:{daily}").trigger.interval == dt.timedelta(days=1)
        async with maker() as s:
            assert (await s.get(WatchlistItem, daily)).next_run is not None  # mirrored for the dashboard

        async with maker() as s:
            item = await s.get(WatchlistItem, daily)
            item.cadence = Cadence.WEEKLY
            await s.commit()
        await sched.reconcile(scheduler)
        assert scheduler.get_job(f"watch:{daily}").trigger.interval == dt.timedelta(days=7)

        async with maker() as s:
            (await s.get(WatchlistItem, daily)).active = False
            await s.commit()
        await sched.reconcile(scheduler)
        assert scheduler.get_job(f"watch:{daily}") is None
    finally:
        scheduler.shutdown(wait=False)


def test_jobs_persist_in_postgres_across_scheduler_restarts():
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    from insightforge.watchlist.scheduler import fire, jobstore_url

    url, job_id = jobstore_url(get_settings()), f"watch:test-{uuid.uuid4().hex[:8]}"
    first = BackgroundScheduler(jobstores={"default": SQLAlchemyJobStore(url=url, tablename="apscheduler_jobs")})
    first.start(paused=True)
    first.add_job(fire, IntervalTrigger(days=1), args=["x"], id=job_id)
    first.shutdown(wait=False)

    second = BackgroundScheduler(jobstores={"default": SQLAlchemyJobStore(url=url, tablename="apscheduler_jobs")})
    second.start(paused=True)
    try:
        job = second.get_job(job_id)
        assert job is not None and job.func_ref == "insightforge.watchlist.scheduler:fire"
    finally:
        second.remove_job(job_id)
        second.shutdown(wait=False)


def test_interval_override_for_demos():
    from insightforge.watchlist.service import interval

    assert interval("daily", Settings(_env_file=None)) == dt.timedelta(days=1)
    assert interval("weekly", Settings(_env_file=None)) == dt.timedelta(days=7)
    assert interval("weekly", Settings(_env_file=None, watchlist_interval_override_s=60)) == dt.timedelta(seconds=60)



def test_production_scheduler_never_doubles_up_runs():
    from insightforge.watchlist.scheduler import build_scheduler

    defaults = build_scheduler(Settings(_env_file=None, scheduler_misfire_grace_s=900))._job_defaults
    assert defaults == {"coalesce": True, "max_instances": 1, "misfire_grace_time": 900}


async def test_cadence_change_keeps_the_schedule_anchored_to_the_last_run(committed_run, monkeypatch):
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from insightforge.watchlist import scheduler as sched

    maker, user_id, _ = committed_run
    monkeypatch.setattr(sched, "get_sessionmaker", lambda: maker)
    ran = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    item_id = await add_item(maker, user_id, last_run=ran)
    scheduler = AsyncIOScheduler(jobstores={"default": MemoryJobStore()}, timezone=dt.UTC)
    scheduler.start(paused=True)
    try:
        await sched.reconcile(scheduler)
        nxt = scheduler.get_job(f"watch:{item_id}").next_run_time
        assert abs((nxt - (ran + dt.timedelta(days=1))).total_seconds()) < 5  # not "now"
    finally:
        scheduler.shutdown(wait=False)


async def test_different_companies_are_never_a_change():
    """Live regression: two startups' seed rounds were reported as one 'changed' finding."""
    old = ["GIMO, a Vietnamese payroll startup, raised $1.9M in seed funding led by Singapore's Integra Partners."]
    new = ["Gajiku, an Indonesia-based earned wages startup, raised US$1.1 million in a seed round led by AC Ventures."]
    summary = await diff_claims(FakeGateway({}, {Role.DIFF: materiality()}), old, new)
    assert not summary.changed_findings
    assert [f.text for f in summary.added_findings] == new


def test_subjects_ignore_places_and_filler():
    from insightforge.agents.diff import same_subject, subjects

    assert subjects("Salary Hero raised US$7.6 million in Thailand.") == {"Salary", "Hero"}
    assert same_subject("Salary Hero raised US$7.6M.", "Salary Hero's Series A was US$9M.")
    assert not same_subject("GIMO raised in Vietnam.", "Gajiku raised in Indonesia.")


async def test_removals_alone_do_not_alert(committed_run, monkeypatch):
    from insightforge.agents.diff import ChangeSummary, Finding
    from insightforge.watchlist import service

    maker, user_id, _ = committed_run
    item_id = await add_item(maker, user_id)
    first = await complete_watch_run(maker, user_id, item_id, ["Salary Hero raised US$7.6 million in a Series A."])
    await process_watch_result(maker, FakeGateway({}, {}), client_for(lambda r: httpx.Response(200)), first)
    second = await complete_watch_run(maker, user_id, item_id, ["Talenox serves SMBs in Singapore."])

    async def only_removed(gateway, old, new):
        return ChangeSummary(removed_findings=[Finding(text=old[0])])

    monkeypatch.setattr(service, "diff_claims", only_removed)
    assert await process_watch_result(maker, FakeGateway({}, {}), client_for(lambda r: httpx.Response(200)), second) is None



async def test_same_company_different_fact_is_new_not_changed():
    """Live regression: 'PayrollPanda secured US$700,000 seed' was reported as a change of its customer count."""
    old = ["PayrollPanda is part of LaunchPad and claims over 1,000 customers across Malaysia."]
    new = ["PayrollPanda, a payroll software startup in Malaysia, secured US$700,000 in seed funding."]

    def judge(messages):  # a judge that recognises the candidate is a new fact about the same company
        cid = messages[-1].content.split()[0]
        return MaterialityVerdicts(verdicts=[Materiality(id=cid, reason="different fact", classification="new")])

    summary = await diff_claims(FakeGateway({}, {Role.DIFF: judge}), old, new)
    assert not summary.changed_findings and [f.text for f in summary.added_findings] == new


async def test_near_identical_sentence_with_new_figure_is_changed_without_the_model():
    old = ["Salary Hero raised US$7.6 million in a Series A round led by Cento Ventures."]
    new = ["Salary Hero raised US$9 million in a Series A round led by Cento Ventures."]

    def no_model(messages):
        raise AssertionError("must be decided in code")

    summary = await diff_claims(FakeGateway({}, {Role.DIFF: no_model}), old, new)
    assert [(c.before, c.after) for c in summary.changed_findings] == [(old[0], new[0])]
