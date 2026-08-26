from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from ..utils import (
    CmdHandler,
    ColdDown,
    Config,
    HandlerContext,
    async_task,
    get_client_session,
    get_exc_desc,
    get_file_db,
    get_group_white_list,
    get_logger,
    send_group_msg_by_bot,
)


config = Config("codexreset")
logger = get_logger("CodexReset")
file_db = get_file_db("data/codexreset/db.json", logger)
gwl = get_group_white_list(file_db, logger, "codexreset")
cd = ColdDown(file_db, logger)

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_EVENT_KINDS = {"reset_completed", "reset_scheduled", "banked_reset"}
MAX_KNOWN_EVENT_KEYS = 256
CHINA_TIMEZONE = timezone(timedelta(hours=8))

TITLE_BY_EVENT = {
    ("reset_completed", "global"): "【Codex额度已重置】",
    ("reset_completed", "banked"): "【Codex备用重置已赠送】",
    ("reset_scheduled", "global"): "【Codex额度预计重置】",
    ("reset_scheduled", "banked"): "【Codex备用重置预计赠送】",
    ("banked_reset", "banked"): "【Codex备用重置将赠送】",
}
PLAN_LABELS = {
    "all": "全部套餐",
    "plus": "Plus",
    "pro": "Pro",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
}
WINDOW_LABELS = {
    "five_hour": "5小时额度",
    "5h": "5小时额度",
    "weekly": "周额度",
}


def _state() -> dict[str, Any]:
    return file_db.get_copy(
        "state",
        {
            "initialized": False,
            "known_event_keys": [],
            "pending_events": {},
        },
    )


def _save_state(state: dict[str, Any]) -> None:
    file_db.set("state", state)
    file_db.save()


def _event_key(event: dict[str, Any]) -> str:
    source = event.get("source")
    if not isinstance(source, dict) or not source.get("postId"):
        raise ValueError("事件缺少 source.postId")
    return f'{event["kind"]}:{event.get("resetType", "unknown")}:{source["postId"]}'


def _validate_event(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("kind") not in SUPPORTED_EVENT_KINDS:
        return None
    if not event.get("announcedAt"):
        logger.warning("忽略缺少 announcedAt 的 Codex Reset 事件")
        return None
    try:
        _event_key(event)
    except ValueError as e:
        logger.warning(f"忽略无效 Codex Reset 事件: {e}")
        return None
    return event


def _collect_events(feed: dict[str, Any]) -> list[dict[str, Any]]:
    events = feed.get("events")
    if not isinstance(events, list):
        raise ValueError("状态 JSON 缺少 events 数组")

    collected: dict[str, dict[str, Any]] = {}
    for raw_event in events:
        event = _validate_event(raw_event)
        if event is not None:
            collected[_event_key(event)] = event

    timeline = feed.get("resetTimeline")
    if isinstance(timeline, dict):
        next_schedule = timeline.get("nextSchedule")
        if isinstance(next_schedule, dict):
            raw_schedule = next_schedule.get("schedule", next_schedule)
            event = _validate_event(raw_schedule)
            if event is not None:
                event = dict(event)
                window = next_schedule.get("window")
                if isinstance(window, dict):
                    event["_scheduleWindow"] = window
                collected[_event_key(event)] = event

    return sorted(collected.values(), key=lambda item: item["announcedAt"])


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CHINA_TIMEZONE)


def _format_time(value: str) -> str:
    return _parse_time(value).strftime("%Y-%m-%d %H:%M")


def _format_event_time(event: dict[str, Any]) -> str:
    if event["kind"] == "reset_scheduled":
        window = event.get("_scheduleWindow")
        if isinstance(window, dict) and window.get("startAt"):
            start = _format_time(window["startAt"])
            end_at = window.get("endAt")
            if end_at and end_at != window["startAt"]:
                return f"预计：{start}～{_format_time(end_at)}（北京时间）"
            return f"预计：{start}（北京时间）"
        if event.get("effectiveAt"):
            return f'预计：{_format_time(event["effectiveAt"])}（北京时间）'
    value = event.get("effectiveAt") or event["announcedAt"]
    return f"时间：{_format_time(value)}（北京时间）"


def _scope_parts(event: dict[str, Any]) -> list[str]:
    reset_type = event.get("resetType")
    parts = ["全局"] if reset_type == "global" else []
    scope = event.get("scope")
    if isinstance(scope, dict):
        plans = [
            PLAN_LABELS.get(str(value), str(value))
            for value in scope.get("plans", [])
            if value != "unknown"
        ]
        windows = [
            WINDOW_LABELS.get(str(value), str(value))
            for value in scope.get("windows", [])
            if value != "unknown"
        ]
        if plans:
            parts.append("/".join(plans))
        if windows:
            parts.append("/".join(windows))
    confidence = event.get("confidence")
    if isinstance(confidence, (int, float)):
        parts.append(f"置信度{round(confidence * 100)}%")
    return parts


def format_event_message(event: dict[str, Any]) -> str:
    title = TITLE_BY_EVENT.get(
        (event["kind"], event.get("resetType")),
        "【Codex额度更新】",
    )
    lines = [title]
    if parts := _scope_parts(event):
        lines.append("｜".join(parts))
    lines.append(_format_event_time(event))

    source = event["source"]
    if source_url := source.get("url"):
        handle = f'@{source["handle"]} ' if source.get("handle") else ""
        lines.append(f"{handle}{source_url}")
    elif source.get("origin") == "operator":
        lines.append("来源：人工确认")
    return "\n".join(lines)


async def _deliver_pending_events() -> None:
    state = _state()
    pending_events = state["pending_events"]
    changed = False
    for event_key in list(pending_events):
        pending = pending_events[event_key]
        event = pending["event"]
        for group_id in list(pending["remaining_group_ids"]):
            if not gwl.check_id(group_id):
                pending["remaining_group_ids"].remove(group_id)
                changed = True
                continue
            try:
                result = await send_group_msg_by_bot(group_id, format_event_message(event))
                if result is None:
                    logger.warning(f"Codex Reset 事件 {event_key} 向群 {group_id} 投递未完成")
                    continue
                pending["remaining_group_ids"].remove(group_id)
                changed = True
            except Exception as e:
                logger.warning(
                    f"Codex Reset 事件 {event_key} 向群 {group_id} 投递失败: "
                    f"{get_exc_desc(e)}"
                )
        if not pending["remaining_group_ids"]:
            del pending_events[event_key]
            changed = True
    if changed:
        _save_state(state)


async def _fetch_feed() -> dict[str, Any]:
    feed_url = str(config.get("feed_url")).strip()
    if not feed_url:
        raise ValueError("config/codexreset.yaml 中的 feed_url 不能为空")
    timeout_seconds = float(config.get("request_timeout_seconds"))
    if timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds 必须大于 0")
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with get_client_session().get(feed_url, timeout=timeout) as response:
        response.raise_for_status()
        feed = await response.json(content_type=None)
    if not isinstance(feed, dict):
        raise ValueError("状态 JSON 顶层必须是对象")
    if feed.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f'不支持的 schemaVersion: {feed.get("schemaVersion")}')
    monitor = feed.get("monitor")
    if not isinstance(monitor, dict) or monitor.get("status") != "ok":
        error_code = monitor.get("errorCode") if isinstance(monitor, dict) else None
        raise ValueError(f"上游监控状态异常: {error_code or 'unknown'}")
    return feed


async def _poll_once() -> None:
    await _deliver_pending_events()
    feed = await _fetch_feed()
    events = _collect_events(feed)
    state = _state()
    event_keys = [_event_key(event) for event in events]

    if not state["initialized"]:
        state["initialized"] = True
        state["known_event_keys"] = event_keys[-MAX_KNOWN_EVENT_KEYS:]
        _save_state(state)
        logger.info(f"Codex Reset 初始化完成，已记录 {len(event_keys)} 个现有事件")
        return

    known_keys = set(state["known_event_keys"])
    new_events = [event for event in events if _event_key(event) not in known_keys]
    if not new_events:
        return

    target_group_ids = list(gwl.get())
    for event in new_events:
        event_key = _event_key(event)
        state["known_event_keys"].append(event_key)
        if target_group_ids:
            state["pending_events"][event_key] = {
                "event": event,
                "remaining_group_ids": target_group_ids.copy(),
            }
    state["known_event_keys"] = state["known_event_keys"][-MAX_KNOWN_EVENT_KEYS:]
    _save_state(state)
    logger.info(
        f"发现 {len(new_events)} 个 Codex Reset 新事件，"
        f"目标群数 {len(target_group_ids)}"
    )
    await _deliver_pending_events()


query_latest = CmdHandler(["/codexreset"], logger, priority=1, block=False)
query_latest.check_cdrate(cd).check_wblist(gwl).check_group()


@query_latest.handle()
async def _(ctx: HandlerContext) -> None:
    if ctx.get_args().strip():
        return
    try:
        events = _collect_events(await _fetch_feed())
    except Exception as e:
        logger.warning(f"手动查询 Codex Reset 失败: {get_exc_desc(e)}")
        return await ctx.asend_reply_msg("Codex额度查询失败，请稍后重试")
    if not events:
        return await ctx.asend_reply_msg("暂无Codex额度事件")
    return await ctx.asend_reply_msg(format_event_message(events[-1]))


def _next_poll_delay_seconds() -> float:
    minimum = float(config.get("poll_interval_minutes.min"))
    maximum = float(config.get("poll_interval_minutes.max"))
    if minimum <= 0 or maximum < minimum:
        raise ValueError("轮询间隔必须满足 0 < min <= max")
    return random.uniform(minimum, maximum) * 60


@async_task("Codex Reset轮询", logger)
async def _poll_worker() -> None:
    while True:
        try:
            await _poll_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Codex Reset 轮询失败: {get_exc_desc(e)}")
        await asyncio.sleep(_next_poll_delay_seconds())
