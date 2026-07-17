from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from ..utils import *
from .models import (
    BattleOptions,
    BattleRecorder,
    BattleRecord,
    battle_summary_text,
)
from .render import RenderConfig, render_battle_report
from .storage import GbfrStorage


config = Config("gbfr_act")
logger = get_logger("GBFR-ACT")
file_db = get_file_db("data/gbfr/db.json", logger)
cd = ColdDown(file_db, logger)
gwl = get_group_white_list(file_db, logger, "gbfr")
report_sub = GroupSubHelper("gbfr_report", file_db, logger)

_storage: GbfrStorage | None = None
_storage_base_dir: str | None = None
_recorder: BattleRecorder | None = None
_ws_connected = False
_last_ws_event_at: datetime | None = None


def _battle_options() -> BattleOptions:
    return BattleOptions.from_config(config.get("battle", {}))


def _render_config() -> RenderConfig:
    render_cfg = dict(config.get("render", {}))
    if not render_cfg.get("font_path"):
        render_cfg["font_path"] = global_config.get("font.path", "")
    return RenderConfig.from_config(render_cfg)


def _storage_keep_battles() -> int:
    return int(config.get("storage.keep_battles", 100))


def get_storage() -> GbfrStorage:
    global _storage, _storage_base_dir
    base_dir = str(config.get("storage.base_dir", "data/gbfr"))
    if _storage is None or _storage_base_dir != base_dir:
        _storage = GbfrStorage(base_dir)
        _storage_base_dir = base_dir
    return _storage


def get_recorder() -> BattleRecorder:
    global _recorder
    options = _battle_options()
    if _recorder is None:
        _recorder = BattleRecorder(options)
    else:
        _recorder.options = options
    return _recorder


def get_latest_record_for_query(use_active: bool = True) -> BattleRecord | None:
    if use_active:
        active_record = get_recorder().snapshot()
        if active_record:
            return active_record
    return get_storage().latest_battle(options=_battle_options())


async def _handle_ws_event(event: dict[str, Any]) -> None:
    global _last_ws_event_at
    _last_ws_event_at = datetime.now()
    get_storage().append_raw_event(event)
    record = get_recorder().ingest(event)
    if record:
        await _archive_and_notify(record)


async def _archive_and_notify(record: BattleRecord) -> None:
    path = get_storage().save_battle(record, keep_battles=_storage_keep_battles())
    logger.info(
        f"保存 GBFR 战斗记录 {record.battle_id}: "
        f"{record.total_damage} damage, {record.duration_seconds:.1f}s -> {path}"
    )
    if not config.get("subscribe.auto_send", True):
        return
    await _send_record_to_subscribers(record)


async def _send_record_to_subscribers(record: BattleRecord) -> None:
    group_ids = [gid for gid in report_sub.get_all() if gwl.check_id(gid)]
    if not group_ids:
        return

    delay = float(config.get("subscribe.auto_send_delay_seconds", 1))
    if delay > 0:
        await asyncio.sleep(delay)

    render_cfg = _render_config()
    img = await run_in_pool(render_battle_report, record, render_cfg)
    img_cq = await get_image_cq(
        img,
        logger=logger,
        low_quality=bool(config.get("render.low_quality", True)),
    )
    text = "【GBFR 战斗结束】\n" + battle_summary_text(
        record,
        show_username=render_cfg.show_username,
    )
    msg = f"{text}\n{img_cq}"
    for group_id in group_ids:
        try:
            await send_group_msg_by_bot(group_id, msg)
        except Exception as e:
            logger.print_exc(f"向群 {group_id} 推送 GBFR 输出报告失败: {get_exc_desc(e)}")


@async_task("GBFR ACT WebSocket采集", logger, delay=3)
async def _gbfr_ws_worker():
    if not config.get("websocket.enable", True):
        logger.info("GBFR ACT WebSocket采集未启用")
        return

    import aiohttp

    global _ws_connected
    error_count = 0
    while True:
        url = str(config.get("websocket.url", "ws://127.0.0.1:24399"))
        heartbeat = float(config.get("websocket.heartbeat_seconds", 30))
        reconnect_seconds = float(config.get("websocket.reconnect_seconds", 5))
        try:
            async with get_client_session().ws_connect(url, heartbeat=heartbeat) as ws:
                _ws_connected = True
                error_count = 0
                logger.info(f"已连接 GBFR ACT WebSocket: {url}")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(msg.data)
                            if isinstance(event, dict):
                                await _handle_ws_event(event)
                        except Exception as e:
                            logger.print_exc(f"处理 GBFR ACT WebSocket 事件失败: {get_exc_desc(e)}")
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_count += 1
            if error_count <= int(config.get("websocket.error_log_limit", 5)):
                logger.warning(f"连接 GBFR ACT WebSocket 失败: {get_exc_desc(e)}")
        finally:
            if _ws_connected:
                logger.warning("GBFR ACT WebSocket 已断开")
            _ws_connected = False
        await asyncio.sleep(reconnect_seconds)


@repeat_with_interval(config.item("battle.idle_check_interval_seconds"), "GBFR ACT空闲战斗归档", logger, delay=10)
async def _finish_idle_battle():
    if not config.get("websocket.enable", True):
        return
    idle_seconds = int(config.get("battle.idle_finish_seconds", 120))
    record = get_recorder().finish_if_idle(
        int(datetime.now().timestamp() * 1000),
        idle_seconds * 1000,
    )
    if record:
        await _archive_and_notify(record)


@on_shutdown()
async def _save_active_battle_on_shutdown():
    record = get_recorder().finish("shutdown")
    if record:
        get_storage().save_battle(record, keep_battles=_storage_keep_battles())


query_output = CmdHandler(["/gbfr 查输出", "/gbfr dps", "/gbfr output"], logger)
query_output.check_cdrate(cd).check_wblist(gwl)


@query_output.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().lower()
    use_active = args not in {"last", "latest", "最近归档", "归档"}
    record = get_latest_record_for_query(use_active=use_active)
    assert_and_reply(record, "暂无 GBFR 战斗记录，请先运行 Windows 上的 GBFR ACT 并完成一次战斗")
    await ctx.block("gbfr_act_render", timeout=120, err_msg="正在生成 GBFR 输出图，请稍后再试")
    render_cfg = _render_config()
    img = await run_in_pool(render_battle_report, record, render_cfg)
    return await ctx.asend_reply_msg(
        await get_image_cq(
            img,
            logger=logger,
            low_quality=bool(config.get("render.low_quality", True)),
        )
    )


sub_cmd = CmdHandler(["/gbfr sub"], logger)
sub_cmd.check_cdrate(cd).check_wblist(gwl).check_group().check_superuser()


@sub_cmd.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().lower()
    if args in {"on", "enable", "开启", "订阅"}:
        if report_sub.sub(ctx.group_id):
            return await ctx.asend_reply_msg("已开启本群 GBFR 战斗结束自动推送")
        return await ctx.asend_reply_msg("本群已经开启 GBFR 战斗结束自动推送")
    if args in {"off", "disable", "关闭", "取消订阅"}:
        if report_sub.unsub(ctx.group_id):
            return await ctx.asend_reply_msg("已关闭本群 GBFR 战斗结束自动推送")
        return await ctx.asend_reply_msg("本群没有开启 GBFR 战斗结束自动推送")

    status = "开启" if report_sub.is_subbed(ctx.group_id) else "关闭"
    ws_status = "已连接" if _ws_connected else "未连接"
    last_event = _last_ws_event_at.strftime("%Y-%m-%d %H:%M:%S") if _last_ws_event_at else "无"
    return await ctx.asend_reply_msg(
        f"本群自动推送: {status}\n"
        f"WebSocket: {ws_status}\n"
        f"最近事件: {last_event}\n"
        "用法: /gbfr sub on 或 /gbfr sub off"
    )
