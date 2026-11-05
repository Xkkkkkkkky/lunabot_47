"""Our Notes 资料绘图。"""

from PIL import Image, ImageFilter, ImageOps
from zhconv import convert

from ..draw.img_utils import mix_image_by_color
from ..draw.painter import DEFAULT_FONT, DEFAULT_BOLD_FONT, Painter
from ..draw.plot import (
    Canvas, Frame, HSplit, VSplit, TextBox, TextStyle, ImageBox,
    RoundRectBg, WidgetBg,
)


INK = (30, 43, 68, 255)
MUTED = (88, 99, 122, 255)
ACCENT = (102, 100, 153, 255)
WIDTH = 1000


class _Backdrop(WidgetBg):
    def __init__(self, source: Image.Image | None):
        self.source = source

    def draw(self, p: Painter):
        if self.source is not None:
            bg = ImageOps.fit(self.source.convert('RGB'), p.size)
            bg = bg.filter(ImageFilter.GaussianBlur(4))
            bg = mix_image_by_color(bg, (239, 242, 249, 64))
        else:
            bg = Image.new('RGB', (1, 256))
            bg.putdata([
                (round(239 - i * 9 / 255), round(242 - i * 10 / 255), 250)
                for i in range(256)
            ])
            bg = bg.resize(p.size)
        p.paste(bg, (0, 0))


def _glass():
    return RoundRectBg(
        (255, 255, 255, 160), 36, blurglass=True,
        blurglass_kwargs=dict(blur=12, shadow_width=10, shadow_alpha=0.12, edge_strength=0.65),
    )


def _text(text, width=944, size=27, color=INK, bold=False):
    # 仅转换绘制文字，保留原始数据和素材。
    return TextBox(
        convert(str(text), 'zh-cn'), TextStyle(font=DEFAULT_BOLD_FONT if bold else DEFAULT_FONT,
                             size=size, color=color),
        use_real_line_count=True, line_sep=8,
    ).set_w(width).set_padding(0)


def _card():
    return VSplit(sep=18, item_align='lt').set_w(WIDTH).set_padding(28).set_bg(_glass())


def _header(title, subtitle, eyebrow=''):
    with _card():
        with HSplit(sep=24, item_align='lt'):
            with VSplit(sep=8).set_w(776):
                _text('BanG Dream! Our Notes', 776, 23, ACCENT, True)
                if eyebrow:
                    _text(eyebrow, 776, 22, MUTED)
            with Frame().set_w(144).set_padding(12).set_bg(
                RoundRectBg((230, 230, 244, 235), 18)
            ):
                _text('HK/MO/TW', 120, 21, ACCENT, True)
        _text(title, size=43, bold=True)
        if subtitle:
            _text(subtitle, size=25, color=MUTED)


def _footer(note):
    if not note:
        return
    with VSplit(sep=12).set_w(WIDTH).set_padding((12, 8)):
        _text(note, 976, 21, MUTED)


def _art(image, max_height):
    scale = min(944 / image.width, max_height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    artwork = image.resize(size, Image.Resampling.LANCZOS)
    with Frame().set_w(944).set_content_align('c'):
        ImageBox(artwork, use_alphablend=True)


def _canvas(background):
    return Canvas(w=1080, bg=_Backdrop(background)).set_padding(40)


async def render_detail(title: str, subtitle: str, hero: Image.Image | None,
                        sections: list[tuple[str, list[tuple[str, str]]]], *,
                        background: Image.Image | None = None,
                        eyebrow: str = '', note: str = '') -> Image.Image:
    canvas = _canvas(background if background is not None else hero)
    with canvas:
        with VSplit(sep=24, item_align='lt').set_w(WIDTH):
            _header(title, subtitle, eyebrow)
            if hero is not None:
                with _card():
                    _art(hero, 570)
            for heading, entries in sections:
                with _card():
                    _text(heading, size=30, color=ACCENT, bold=True)
                    for label, value in entries:
                        with HSplit(sep=24, item_align='lt'):
                            _text(label, 244, 24, MUTED)
                            _text(value, 676, 26)
            _footer(note)
    return await canvas.get_img()


async def render_list(title: str, subtitle: str, rows: list[dict], *,
                      background: Image.Image | None = None, note: str = '') -> Image.Image:
    if len(rows) > 8:
        raise ValueError('Our Notes 列表每页最多 8 项，请先分页')
    source = background
    if source is None:
        source = next((row.get('image') for row in rows if row.get('image') is not None), None)
    canvas = _canvas(source)
    with canvas:
        with VSplit(sep=24, item_align='lt').set_w(WIDTH):
            _header(title, subtitle)
            for row in rows:
                with _card():
                    with HSplit(sep=24, item_align='lt'):
                        image = row.get('image')
                        if image is not None:
                            thumb = image.copy()
                            thumb.thumbnail((120, 120), Image.Resampling.LANCZOS)
                            with Frame().set_size((120, 120)).set_content_align('c'):
                                ImageBox(thumb, use_alphablend=True)
                        content_w = 800 if image is not None else 944
                        with VSplit(sep=10).set_w(content_w):
                            with HSplit(sep=16, item_align='lt'):
                                _text(row.get('title', ''), content_w - 176, 30, bold=True)
                                _text(row.get('tag', ''), 160, 21, ACCENT, True)
                            if row.get('subtitle'):
                                _text(row['subtitle'], content_w, 23, MUTED)
                            if row.get('detail'):
                                _text(row['detail'], content_w, 24)
            _footer(note)
    return await canvas.get_img()


async def render_story(title: str, subtitle: str, chapters: list[tuple[str, str]], *,
                       hero: Image.Image | None = None,
                       background: Image.Image | None = None, note: str = '') -> Image.Image:
    canvas = _canvas(background if background is not None else hero)
    with canvas:
        with VSplit(sep=24, item_align='lt').set_w(WIDTH):
            with _card():
                _text('AI生成 · 请勿转载', size=31, color=ACCENT, bold=True)
            _header(title, subtitle)
            if hero is not None:
                with _card():
                    _art(hero, 400)
            for heading, body in chapters:
                with _card():
                    _text(heading, size=31, color=ACCENT, bold=True)
                    _text(body, size=27)
            _footer(note)
    return await canvas.get_img()
