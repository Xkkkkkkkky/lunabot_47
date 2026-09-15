from ...utils import *
from ..common import *
from ..handler import *
from ..asset import *
from ..draw import *
from .resbox import get_res_box_info

# ======================= 处理逻辑 ======================= #

async def get_vlive_widget(ctx: SekaiHandlerContext, vlive: dict) -> Frame:
    """
    从单个vlive数据生成一个vlive控件(Frame)
    """
    vlive["current"] = None
    vlive["living"] = False
    for start, end in vlive["schedule"]:
        if datetime.now() < end:
            vlive["current"] = (start, end)
            vlive["living"] = datetime.now() >= start
            break
    vlive["rest_num"] = 0
    for start, end in vlive["schedule"]:
        if datetime.now() < start:
            vlive["rest_num"] += 1

    with Frame().set_bg(roundrect_bg()).set_padding(16) as f:
        with VSplit().set_content_align('l').set_item_align('l').set_sep(8):
            # 标题
            TextBox(
                f"【{vlive['id']}】{vlive['name']}", 
                TextStyle(font=DEFAULT_BOLD_FONT, size=20, color=(20, 20, 20)), 
                line_count=2, use_real_line_count=True
            ).set_w(750)
            Spacer(w=1, h=4)

            with HSplit().set_content_align('c').set_item_align('c').set_sep(8):
                # 图片
                asset_name = vlive['asset_name']
                img = await ctx.rip.img(f"virtual_live/select/banner/{asset_name}_rip/{asset_name}.png", allow_error=True)
                if img:
                    ImageBox(img, size=(None, 100), use_alphablend=True)

                # 各种时间
                with VSplit().set_content_align('l').set_item_align('l').set_sep(8):
                    start_text  = f"开始于 {get_readable_datetime(vlive['start'])}"
                    end_text    = f"结束于 {get_readable_datetime(vlive['end'])}"
                    if vlive['living']:
                        current_text = "当前Live进行中!"
                    elif vlive["current"]:
                        current_text = f"下一场: {get_readable_datetime(vlive['current'][0], show_original_time=False)}"
                    else:
                        current_text = "已结束"
                    rest_text = f" | 剩余场次: {vlive['rest_num']}"

                    TextBox(start_text, TextStyle(font=DEFAULT_FONT, size=18, color=(50, 50, 50)))
                    TextBox(end_text, TextStyle(font=DEFAULT_FONT, size=18, color=(50, 50, 50)))
                    TextBox(current_text + rest_text, TextStyle(font=DEFAULT_FONT, size=18, color=(50, 50, 50)))

            with HSplit().set_content_align('t').set_item_align('t').set_sep(16):
                # 参与奖励
                res_size = 64
                res_info_list = []
                try:
                    for reward in vlive['rewards']:
                        if reward['virtualLiveType'] == 'normal':
                            res_info_list = await get_res_box_info(ctx, "virtual_live_reward", reward['resourceBoxId'], res_size)
                            break
                except:
                    logger.print_exc(f"获取虚拟Live奖励失败")
                if res_info_list:
                    with VSplit().set_content_align('l').set_item_align('l').set_sep(8):
                        TextBox("参与奖励", TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=(50, 50, 50)))
                        with HSplit().set_content_align('l').set_item_align('l').set_sep(6):
                            for res_info in res_info_list:
                                image, quantity = res_info['image'], res_info['quantity']
                                w, h = max(image.width, res_size), max(image.height, res_size)
                                with Frame().set_size((w, h)):
                                    ImageBox(res_info['image'], use_alphablend=True).set_offset((w//2, h//2)).set_offset_anchor('c')
                                    if quantity > 1:
                                        t = TextBox(f"x{quantity}", TextStyle(font=DEFAULT_BOLD_FONT, size=12, color=(50, 50, 50)))
                                        t.set_offset((w//2, h)).set_offset_anchor('b')

                # 出演角色
                chara_icons = []
                for item in vlive['characters']:
                    if 'virtualLivePerformanceType' in item and \
                        item['virtualLivePerformanceType'] not in ['main_only', 'both']:
                        continue
                    cuid = item.get('gameCharacterUnitId')
                    scid = item.get('subGameCharacter2dId')
                    if cuid:
                        cu = await ctx.md.game_character_units.find_by_id(cuid)
                        chara_icons.append(get_chara_icon_by_chara_id(cid=cu['gameCharacterId'], unit=cu['unit']))
                if chara_icons:
                    with VSplit().set_content_align('l').set_item_align('l').set_sep(8):
                        TextBox("出演角色", TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=(50, 50, 50)))
                        with Grid(col_count=10).set_content_align('c').set_sep(4, 4).set_padding(0):
                            for icon in chara_icons:
                                ImageBox(icon, size=(30, 30), use_alphablend=True)  

    return f

async def compose_vlive_list_image(ctx: SekaiCmdHandler, vlives, title=None, title_style=None, bg=SEKAI_BLUE_BG) -> Image.Image: 
    """
    从给定的多个vlive生成一个vlive列表图片
    """
    with Canvas(bg=bg).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(16):
            if title and title_style:
                TextBox(title, title_style).set_bg(roundrect_bg()).set_padding(16)
            for vlive in vlives:
                await get_vlive_widget(ctx, vlive)
    add_watermark(canvas)
    return await canvas.get_img()


# ======================= 指令处理 ======================= #

# 获取最近的vlive信息
pjsk_live = SekaiCmdHandler(['/pjsk live', '/虚拟live', '/pjsk vlive', "/vlive",])
pjsk_live.check_cdrate(cd).check_wblist(gbl)
@pjsk_live.handle()
async def _(ctx: SekaiHandlerContext):
    now = datetime.now()
    vlives = [
        vlive for vlive in await ctx.md.vlives.get() 
        if now < vlive['end'] and vlive['start'] - now < timedelta(days=7) \
        and vlive['end'] - vlive['start'] < timedelta(days=30)
    ]
    if len(vlives) == 0:
        return await ctx.asend_reply_msg("当前没有虚拟Live")
    return await ctx.asend_reply_msg(await get_image_cq(await compose_vlive_list_image(ctx, vlives)))
