import aiohttp

from ..utils import (
    CmdHandler,
    ColdDown,
    Config,
    HandlerContext,
    ReplyException,
    get_file_db,
    get_group_white_list,
    get_image_cq,
    get_logger,
)
from .data import DIMENSIONS, RadarClient, RadarDataError
from .render import render_radar


config = Config("codexradar")
logger = get_logger("CodexRadar")
file_db = get_file_db("data/codexradar/db.json", logger)
gwl = get_group_white_list(file_db, logger, "codexradar")
cd = ColdDown(file_db, logger)
client = RadarClient(config, logger)

query = CmdHandler("/codexradar", logger).check_wblist(gwl).check_cdrate(cd)


@query.handle()
async def query_radar(ctx: HandlerContext):
    dimension = ctx.get_args().strip().lower() or "total"
    if dimension not in DIMENSIONS:
        raise ReplyException(
            "使用 /codexradar 查询总分；可添加 se（软件工程）、vis（视觉空间）、"
            "term（终端与系统操作）、sci（科学研究）查询分项。"
        )

    # 同一会话生成图片时阻止重复请求。
    await ctx.block(f"codexradar:{ctx.group_id or ctx.user_id}")
    try:
        snapshot = await client.get_snapshot()
    except RadarDataError as exc:
        raise ReplyException(str(exc)) from exc
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning(f"雷达数据请求失败：{type(exc).__name__}")
        raise ReplyException("雷达站暂时没有响应，请稍后再试") from exc

    image = await render_radar(
        snapshot, dimension, config=config, logger=logger, group_id=ctx.group_id
    )
    await ctx.asend_reply_msg(
        await get_image_cq(image, logger=logger, low_quality=False)
    )
