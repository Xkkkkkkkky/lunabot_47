from ..utils import *
from .common import *
from .handler import SekaiHandlerContext
from ..draw.bg import generate_organic_gradient_background
from copy import copy as _copy


# 通过角色ID获取角色头像
def get_chara_icon_by_chara_id(cid: int, size: int = None, raise_exc=True, default=None, unit=None):
    """
    通过角色ID获取角色头像
    """
    nickname = get_character_first_nickname(cid)
    if not nickname:
        if raise_exc: raise Exception(f"要获取的icon的角色ID={cid}错误")
        else: return default
    path = f"chara_icon/{nickname}"
    if unit is not None and unit != "piapro" and nickname == "miku":
        path += f"_{unit}"
    path += ".png"
    img = SekaiHandlerContext.from_region(DEFAULT_SERVER_REGION).static_imgs.get(path)
    if size is not None:
        img = img.resize((size, size))
    return img
    
# 通过角色昵称获取角色头像
def get_chara_icon_by_nickname(nickname: str, size: int = None, raise_exc=True, default=None, unit=None):
    """
    通过角色昵称获取角色头像
    """
    cid = get_cid_by_nickname(nickname)
    if not cid:
        if raise_exc: raise Exception(f"要获取的icon的角色昵称错误")
        else: return default
    return get_chara_icon_by_chara_id(cid, size, raise_exc, default, unit)

# 获取团logo
def get_unit_logo(unit: str, size: int = None):
    img = SekaiHandlerContext.from_region(DEFAULT_SERVER_REGION).static_imgs.get(f"logo_{unit}.png")
    if size is not None:
        img = img.resize((size, size))
    return img

# 获取属性图标
def get_attr_icon(attr: str, size: int = None):
    img = SekaiHandlerContext.from_region(DEFAULT_SERVER_REGION).static_imgs.get(f"card/attr_icon_{attr}.png")
    if size is not None:
        img = img.resize((size, size))
    return img


# 无专用图片背景的绘图使用通用报告背景；有图片的功能仍传入原 ImageBg。
SEKAI_BLUE_BG = default_page_background(resolve_draw_theme(config))

BG_PADDING = 20
WIDGET_BG_COLOR_CFG = config.item("draw.widget_bg_color")
WIDGET_BG_RADIUS_CFG = config.item("draw.widget_bg_radius")
BLURGLASS_CFG = config.item("draw.blurglass")

# 统一的半透明白色圆角矩形背景
def roundrect_bg(
    fill: Color | ConfigItem=WIDGET_BG_COLOR_CFG, 
    radius: int | ConfigItem=WIDGET_BG_RADIUS_CFG, 
    alpha: int=None,
    blurglass: bool=None, 
    blurglass_kwargs: dict=None
):
    """
    统一的半透明白色圆角矩形背景
    """
    if blurglass is None:
        blurglass = BLURGLASS_CFG.get()
    fill = get_cfg_or_value(fill)
    radius = get_cfg_or_value(radius)
    if alpha is not None:
        fill = (*fill[:3], alpha)
    return report_card_background(
        fill=fill,
        radius=radius,
        blurglass=blurglass,
        blurglass_kwargs=blurglass_kwargs,
        theme=resolve_draw_theme(config),
    )


def create_sekai_canvas(
    *,
    background: WidgetBg = None,
    width: int = None,
    height: int = None,
    padding: int = BG_PADDING,
) -> Canvas:
    """沿用原画布尺寸、内边距和左上对齐，使用通用绘图库的画布实现。"""

    canvas = create_report_canvas(
        background=background,
        width=width,
        padding=padding,
        theme=resolve_draw_theme(config),
    ).set_content_align('lt')
    if height is not None:
        canvas.set_h(height)
    return canvas


def _matte_background(background):
    """仅调整 Sekai 方框的装饰，保留颜色、透明度、圆角与自定义模糊。"""
    if not isinstance(background, RoundRectBg):
        return background
    background = _copy(background)
    background.stroke = None
    background.blurglass_kwargs = {
        "blur": 10,
        **(background.blurglass_kwargs or {}),
        "shadow_width": 2,
        "shadow_alpha": 0.0,
        "edge_strength": 0.0,
    }
    return background


def _matte_item_background(factory):
    # 表格允许通过回调生成单元格背景，延迟到绘制时处理，避免提前执行回调。
    def background(*args):
        return _matte_background(factory(*args))
    return background


async def render_sekai_canvas(canvas: Canvas, scale=None, cache_key=None) -> Image.Image:
    """统一磨砂方框；没有明确图片背景时，用页面已有素材生成有机渐变。"""
    # 先完成父容器布局，再读取素材显示尺寸，避免提前缓存子控件尺寸影响排版。
    canvas_size = canvas._get_self_size()
    images = {}
    needs_background = not isinstance(canvas.bg, ImageBg)
    stack = [canvas]
    while stack:
        widget = stack.pop()
        widget.bg = _matte_background(widget.bg)
        if hasattr(widget, 'item_bg'):
            widget.item_bg = (
                _matte_item_background(widget.item_bg)
                if callable(widget.item_bg) else _matte_background(widget.item_bg)
            )
        if needs_background and isinstance(widget, ImageBox):
            # 按页面实际显示面积挑选主素材，避免大量小图标主导配色。
            width, height = widget._get_self_size()
            key = id(widget.image)
            previous = images.get(key)
            if previous is None or width * height > previous[0]:
                images[key] = (width * height, widget.image)
        stack.extend(reversed(getattr(widget, 'items', ())))

    # 剧情页已生成的背景、活动原图及用户配置的背景图片均直接保留。
    if needs_background:
        size_limit = global_config.get('plot.canvas_size_limit')
        assert canvas_size[0] * canvas_size[1] <= size_limit[0] * size_limit[1], \
            f'Canvas size is too large ({canvas_size[0]}x{canvas_size[1]})'
        sources = sorted(images.values(), key=lambda item: item[0], reverse=True)[:16]
        if sources:
            # 小尺寸拼图只用于取色，不改变原素材或页面上的图片。
            source = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
            for index, (_, img) in enumerate(sources):
                thumb = img.convert('RGBA')
                thumb.thumbnail((32, 32), Image.Resampling.LANCZOS)
                source.paste(thumb, ((index % 4) * 32, (index // 4) * 32))
        else:
            # 纯文字/表格页面没有可取色图片，使用当前 Sekai 主题色生成同款渐变。
            palette = resolve_draw_palette(config)
            source = Image.new('RGBA', (len(palette), 1))
            source.putdata(palette)
        background = await asyncio.to_thread(
            generate_organic_gradient_background,
            source,
            canvas_size,
            saturation=0.75,
            distortion=1.20,
            blur=0.035,
        )
        canvas.set_bg(ImageBg(background, mode='fill', fade=0))
    return await canvas.get_img(scale=scale, cache_key=cache_key)


# 使用已部署的标题及团体背景，避免随机选中未提供的旧区域背景。
COMMON_BG_NAMES = ["bg/title_background.png"]
GROUP_BG_NAMES = {
    UNIT_LN:   ["bg/bg_light_sound.png"],
    UNIT_MMJ:  ["bg/bg_idol.png"],
    UNIT_VBS:  ["bg/bg_street.png"],
    UNIT_WS:   ["bg/bg_theme_park.png"],
    UNIT_25:   ["bg/bg_school_refusal.png"],
    UNIT_VS:   ["bg/bg_virtual_singer.png"]
}

# 随机选择团队背景
def random_unit_bg(unit: str = None):
    """
    随机选择团队背景
    unit为None时随机选择一个通用背景
    """
    ctx = SekaiHandlerContext.from_region(DEFAULT_SERVER_REGION)
    if unit is None:
        bg_name = random.choice(COMMON_BG_NAMES)
        img = ctx.static_imgs.get(bg_name)
    else:
        bg_name = random.choice(GROUP_BG_NAMES.get(unit, COMMON_BG_NAMES))
        img = ctx.static_imgs.get(bg_name)
    return ImageBg(img)


DEFAULT_WATERMARK_CFG = config.item("draw.default_watermark")

# 在画布上添加水印
def add_watermark(canvas: Canvas, text: str | ConfigItem=DEFAULT_WATERMARK_CFG, size=12):
    """
    在画布上添加水印
    """
    text = get_cfg_or_value(text)
    frame_watermark = Frame().set_content_align('rb').set_padding(0)
    frame_canvas = Frame().set_content_align(canvas.get_content_align()).set_padding(0).set_size((canvas.w, canvas.h))
    s1 = TextStyle(font=DEFAULT_FONT, size=size, color=(255, 255, 255, 256))
    s2 = TextStyle(font=DEFAULT_FONT, size=size, color=(75, 75, 75, 256))
    offset1 = (int(16 - BG_PADDING * 0.5), 16)
    offset2 = (offset1[0] + 1, offset1[1] + 1)
    text1 = TextBox(text, style=s1).set_omit_parent_bg(True).set_offset(offset1)
    text2 = TextBox(text, style=s2).set_omit_parent_bg(True).set_offset(offset2)
    items = canvas.items
    canvas.set_items([])
    canvas.set_padding(BG_PADDING)
    for item in items:
        frame_canvas.add_item(item)
    frame_watermark.add_item(frame_canvas)
    frame_watermark.add_item(text2)
    frame_watermark.add_item(text1)
    canvas.add_item(frame_watermark).set_size(None)


DIFF_COLORS = {
    "easy": (102, 221, 17, 255),
    "normal": (51,187, 238, 255),
    "hard": (255, 170, 0, 255),
    "expert": (238, 68, 102, 255),
    "master": (187, 51, 238, 255),
    "append": LinearGradient((182, 144, 247, 255), (243, 132, 220, 255), (0, 0), (1, 1)),
}
PLAY_RESULT_COLORS = {
    'not_clear': (69, 67, 104, 255),
    'clear': (255, 226, 118, 255),
    'fc': (253, 167, 249, 255),
    'ap': (63, 230, 228, 255),
}
