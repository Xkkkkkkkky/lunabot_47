"""HK/MO/TW 服务器查询入口；业务解析由 service 负责。"""

from ...utils import HandlerContext, get_image_cq
from ..handler import BdonCmdHandler
from .. import service


event = BdonCmdHandler(['/bdon event'], help_command='/bdon event')


@event.handle()
async def handle_event(ctx: HandlerContext):
    img = await service.query_event(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


events = BdonCmdHandler(['/bdon events'], help_command='/bdon events')


@events.handle()
async def handle_events(ctx: HandlerContext):
    img = await service.query_event(ctx.get_args().strip(), list_mode=True)
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


story = BdonCmdHandler(['/bdon story'], help_command='/bdon story')


@story.handle()
async def handle_story(ctx: HandlerContext):
    async def progress():
        await ctx.asend_reply_msg("正在生成剧情总结…")

    img = await service.query_story(ctx.get_args().strip(), progress=progress)
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


song = BdonCmdHandler(['/bdon song'], help_command='/bdon song')


@song.handle()
async def handle_song(ctx: HandlerContext):
    img = await service.query_music(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


cover = BdonCmdHandler(['/bdon cover'], help_command='/bdon cover')


@cover.handle()
async def handle_cover(ctx: HandlerContext):
    img = await service.query_cover(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


songs = BdonCmdHandler(['/bdon songs'], help_command='/bdon songs')


@songs.handle()
async def handle_songs(ctx: HandlerContext):
    img = await service.query_music(ctx.get_args().strip(), list_mode=True)
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


meta = BdonCmdHandler(['/bdon meta'], help_command='/bdon meta')


@meta.handle()
async def handle_meta(ctx: HandlerContext):
    img = await service.query_meta(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


card = BdonCmdHandler(['/bdon card'], help_command='/bdon card')


@card.handle()
async def handle_card(ctx: HandlerContext):
    img = await service.query_card(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


card_art = BdonCmdHandler(['/bdon card art'], help_command='/bdon card art')


@card_art.handle()
async def handle_card_art(ctx: HandlerContext):
    img = await service.query_card_art(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))


comic = BdonCmdHandler(['/bdon comic'], help_command='/bdon comic')


@comic.handle()
async def handle_comic(ctx: HandlerContext):
    img = await service.query_comic(ctx.get_args().strip())
    await ctx.asend_reply_msg(await get_image_cq(img, send_url_as_is=True))
