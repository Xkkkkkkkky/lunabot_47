"""图片识鸟的后端调用、结果解析与报告渲染。"""

from __future__ import annotations

import asyncio
import html
import io
import math
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import quote, urljoin

import aiohttp
from PIL import Image, ImageOps

from ..utils import (
    CmdHandler,
    ColdDown,
    Config,
    Grid,
    HSplit,
    HandlerContext,
    ReplyException,
    TempBotOrInternetFilePath,
    TempFilePath,
    TextBox,
    VSplit,
    create_report_canvas,
    create_report_column,
    get_client_session,
    get_file_db,
    get_group_white_list,
    get_image_cq,
    get_logger,
    loads_json,
    ranked_info_card,
    report_badge,
    report_card,
    report_description_panel,
    report_header,
    report_image_placeholder,
    report_info_row,
    report_section_title,
    resolve_configured_draw_background,
    resolve_draw_theme,
    rounded_cover_image,
    run_in_pool,
    themed_text_style,
)


config = Config("bird")
logger = get_logger("BirdIdentify")
file_db = get_file_db("data/bird/db.json", logger)
group_cd = ColdDown(
    file_db,
    logger,
    default_interval=config.item("recognition.cooldown.group_seconds"),
    cold_down_name="recognition_group",
    key_mode="group",
)
user_cd = ColdDown(
    file_db,
    logger,
    default_interval=config.item("recognition.cooldown.user_seconds"),
    cold_down_name="recognition_user",
)
gwl = get_group_white_list(file_db, logger, "bird")


class BirdRecognitionError(Exception):
    """识别后端返回不可用结果或请求失败。"""


@dataclass(frozen=True)
class BirdCandidate:
    """不同识别后端共用的单个候选鸟种信息。"""

    name_cn: str
    probability: float
    scientific_name: str = ""
    name_en: str = ""
    category_cn: str = ""
    common_names: str = ""
    description: str = ""
    representative_image_url: Optional[str] = None


@dataclass(frozen=True)
class BirdRecognitionResult:
    """不同识别后端共用的识别结果。"""

    candidates: List[BirdCandidate]
    source_name: str
    source_url: str


class BirdRecognitionBackend(ABC):
    """图片识鸟后端接口；正式 API 后端应实现此接口。"""

    @abstractmethod
    async def recognize(self, image_path: str) -> BirdRecognitionResult:
        """识别本地图片并返回按概率降序排列的候选鸟种。"""


BackendFactory = Callable[[Config], BirdRecognitionBackend]
_BACKEND_FACTORIES: Dict[str, BackendFactory] = {}


def register_recognition_backend(name: str, factory: BackendFactory) -> None:
    """注册识鸟后端，供未来正式 API 或其他识别服务接入。"""

    _BACKEND_FACTORIES[name] = factory


def create_recognition_backend() -> BirdRecognitionBackend:
    """根据热加载配置创建当前识鸟后端。"""

    provider = str(config.get("recognition.provider", "web")).strip()
    factory = _BACKEND_FACTORIES.get(provider)
    if factory is None:
        available = ", ".join(sorted(_BACKEND_FACTORIES)) or "无"
        raise BirdRecognitionError(
            f"未注册识鸟后端 {provider!r}，当前可用后端：{available}"
        )
    return factory(config)


_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


def _clean_description(value: object) -> str:
    """将网页结果中的简单 HTML 简介转换为安全的纯文本。"""

    text = _HTML_BREAK_RE.sub("\n", str(value or ""))
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class WebRecognitionBackend(BirdRecognitionBackend):
    """调用由配置定义的网页识别后端。"""

    def __init__(
        self,
        base_url: str,
        upload_path: str,
        classify_path: str,
        image_path_prefix: str,
        source_name: str,
        timeout_seconds: float,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.upload_url = urljoin(self.base_url, upload_path.lstrip("/"))
        self.classify_url = urljoin(self.base_url, classify_path.lstrip("/"))
        self.image_path_prefix = image_path_prefix.strip("/")
        self.source_name = source_name
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": self.base_url,
            "User-Agent": "LunaBot_47 bird recognition",
        }

    @classmethod
    def from_config(cls, cfg: Config) -> "WebRecognitionBackend":
        """从配置构造网页后端，地址和接口路径缺失时拒绝启动请求。"""

        web_cfg = cfg.get("recognition.web")
        if not isinstance(web_cfg, dict):
            raise BirdRecognitionError("识鸟配置 recognition.web 格式不正确")

        def required_text(key: str) -> str:
            value = str(web_cfg.get(key, "")).strip()
            if not value:
                raise BirdRecognitionError(
                    f"识鸟配置缺少 recognition.web.{key}"
                )
            return value

        return cls(
            base_url=required_text("base_url"),
            upload_path=required_text("upload_path"),
            classify_path=required_text("classify_path"),
            image_path_prefix=required_text("image_path_prefix"),
            source_name=required_text("source_name"),
            timeout_seconds=float(web_cfg.get("timeout_seconds", 30)),
        )

    async def _read_json_response(
        self,
        response: aiohttp.ClientResponse,
        action: str,
    ) -> dict:
        """解析站点以 text/plain 返回的 JSON，并生成稳定的错误信息。"""

        body = await response.read()
        if response.status != 200:
            raise BirdRecognitionError(
                f"识别服务{action}请求失败（HTTP {response.status}）"
            )
        try:
            payload = loads_json(body)
        except Exception as exc:
            raise BirdRecognitionError(f"识别服务{action}返回了无法解析的数据") from exc
        if not isinstance(payload, dict):
            raise BirdRecognitionError(f"识别服务{action}返回格式不正确")
        return payload

    async def _upload_image(self, image_path: str) -> str:
        """上传图片并返回网页接口用于后续分类的 imageId。"""

        # 站点上传解析器只兼容浏览器风格的 WebKit boundary；aiohttp 默认的
        # UUID boundary 会得到“上传图片失败”。图片上限仅 4 MB，直接构造请求
        # 体可以稳定复现网页 FormData，同时保持准确的 Content-Length。
        boundary = "----WebKitFormBoundary" + uuid.uuid4().hex[:16]
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
        prefix = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="0"; filename="bird.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")
        suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
        request_headers = {
            **self.headers,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        async with get_client_session().post(
            self.upload_url,
            data=prefix + image_content + suffix,
            headers=request_headers,
            timeout=self.timeout,
        ) as response:
            payload = await self._read_json_response(response, "图片上传")

        image_id = str(payload.get("imageId", "")).strip()
        if not image_id:
            message = str(payload.get("message", "")).strip()
            detail = f"：{message}" if message else ""
            raise BirdRecognitionError(f"识别服务未返回图片编号{detail}")
        return image_id

    async def _classify_image(self, image_id: str) -> dict:
        """调用网页分类端点并返回原始识别结果。"""

        async with get_client_session().post(
            self.classify_url,
            data={"image_id": image_id, "data_mode": "id"},
            headers=self.headers,
            timeout=self.timeout,
        ) as response:
            payload = await self._read_json_response(response, "识别")

        return_code = payload.get("returnCode")
        if return_code not in (None, 200, "200"):
            message = str(payload.get("message", "未知错误"))
            raise BirdRecognitionError(f"识别服务返回失败：{message}")
        return payload

    def _parse_candidate(self, raw: dict) -> BirdCandidate:
        """将网页接口字段转换为稳定的通用候选结果。"""

        name_cn = str(raw.get("nameCn", "")).strip()
        if not name_cn:
            raise BirdRecognitionError("识别服务返回的候选鸟种缺少中文名")
        try:
            probability = float(raw.get("prob", 0))
        except (TypeError, ValueError) as exc:
            raise BirdRecognitionError("识别服务返回了无效的识别概率") from exc
        probability = min(max(probability, 0.0), 1.0)

        image_filename = Path(str(raw.get("imageFilename", ""))).name
        image_url = None
        if image_filename:
            image_url = urljoin(
                self.base_url,
                f"{self.image_path_prefix}/{quote(image_filename)}",
            )

        return BirdCandidate(
            name_cn=name_cn,
            probability=probability,
            scientific_name=str(raw.get("nameScience", "")).strip(),
            name_en=str(raw.get("nameEn", "")).strip(),
            category_cn=str(raw.get("catCn", "")).strip(),
            common_names=str(raw.get("commonName", "")).strip(),
            description=_clean_description(raw.get("description", "")),
            representative_image_url=image_url,
        )

    async def recognize(self, image_path: str) -> BirdRecognitionResult:
        image_id = await self._upload_image(image_path)
        payload = await self._classify_image(image_id)
        raw_candidates = payload.get("classifyResult")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise BirdRecognitionError("识别服务没有返回候选鸟种")

        candidates = [
            self._parse_candidate(item)
            for item in raw_candidates
            if isinstance(item, dict)
        ]
        if not candidates:
            raise BirdRecognitionError("识别服务没有返回有效候选鸟种")
        candidates.sort(key=lambda item: item.probability, reverse=True)
        return BirdRecognitionResult(
            candidates=candidates,
            source_name=self.source_name,
            source_url=self.base_url,
        )


register_recognition_backend("web", WebRecognitionBackend.from_config)


def _prepare_image_for_upload(
    input_path: str,
    output_path: str,
    max_bytes: int,
    max_edge: int,
    max_pixels: int,
) -> None:
    """规范化识别图片，并逐步压缩到网页上传限制以内。

    处理过程固定取动图首帧、应用 EXIF 方向、用白底合成透明像素，再转换为
    JPEG。压缩时先降低质量；若仍然超限，则按当前文件体积估算缩放比例并
    继续缩小尺寸，避免直接使用极低画质损失鸟类羽色和纹理特征。
    """

    with Image.open(input_path) as source:
        if source.width * source.height > max_pixels:
            raise BirdRecognitionError(
                f"图片像素过大，最多允许 {max_pixels:,} 像素"
            )
        try:
            source.seek(0)
        except EOFError:
            pass
        source = ImageOps.exif_transpose(source)

        if source.mode in ("RGBA", "LA") or "transparency" in source.info:
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), resampling)

    for attempt in range(10):
        quality = max(48, 92 - attempt * 7)
        image.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=False,
        )
        current_bytes = os.path.getsize(output_path)
        if current_bytes <= max_bytes:
            return

        # 体积大致随像素数变化，因此用平方根估计下一轮的边长比例。
        scale = math.sqrt(max_bytes / current_bytes) * 0.94
        scale = min(0.88, max(0.65, scale))
        next_size = (
            max(320, int(image.width * scale)),
            max(320, int(image.height * scale)),
        )
        if next_size == image.size:
            break
        image = image.resize(next_size, resampling)

    raise BirdRecognitionError(
        f"图片压缩后仍超过 {max_bytes / 1024 / 1024:.1f} MB，无法上传识别"
    )


async def _report_background(group_id=None, *, theme=None):
    """通过统一绘图库加载识鸟报告背景。"""

    theme = theme or resolve_draw_theme(config, group_id=group_id)
    return await resolve_configured_draw_background(
        config,
        theme=theme,
        logger=logger,
        label="识鸟报告",
    )


async def _download_representative_image(
    image_url: Optional[str],
    timeout_seconds: float,
    max_bytes: int,
) -> Optional[Image.Image]:
    """下载并完整解码代表图片，供统一绘图控件直接使用。"""

    if not image_url or not image_url.startswith(("http://", "https://")):
        return None
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with get_client_session().get(image_url, timeout=timeout) as response:
            if response.status != 200:
                raise BirdRecognitionError(f"HTTP {response.status}")
            content = await response.read()
            if not content or len(content) > max_bytes:
                raise BirdRecognitionError("图片为空或过大")
            image = Image.open(io.BytesIO(content))
            image.load()
            return image.convert("RGBA")
    except Exception as exc:
        logger.warning(f"下载识鸟代表图片失败: {exc}")
        return None


def _format_probability(probability: float) -> str:
    percentage = probability * 100
    if 0 < percentage < 0.01:
        return "低于 0.01%"
    return f"{percentage:.2f}%"


def _truncate_description(description: str, max_chars: int) -> str:
    """将简介限制到配置长度，保持原报告的内容密度。"""

    description = description.strip() or "暂无简介"
    if len(description) > max_chars:
        description = description[: max_chars - 1].rstrip() + "…"
    return description


async def render_recognition_report(
    result: BirdRecognitionResult,
    *,
    group_id: int | str | None = None,
) -> Image.Image:
    """
    使用统一绘图模板生成识鸟报告。

    布局继续保持原报告的 940px 内容列、290px 主图、玻璃主卡、
    两列候选卡与页脚；仅将 HTML/CSS 实现替换为可被其他模块
    复用的 Pillow 控件。
    """

    render_cfg = config.get("recognition.render", {})
    width = max(720, int(render_cfg.get("width", 1000)))
    top_k = max(1, int(render_cfg.get("top_k", 5)))
    description_max_chars = max(
        120,
        int(render_cfg.get("description_max_chars", 700)),
    )
    image_timeout = float(render_cfg.get("image_timeout_seconds", 15))
    image_max_bytes = int(render_cfg.get("representative_image_max_bytes", 8_388_608))

    candidates = result.candidates[:top_k]
    primary = candidates[0]
    other_candidates = candidates[1:]
    representative_image = await _download_representative_image(
        result.candidates[0].representative_image_url,
        timeout_seconds=image_timeout,
        max_bytes=image_max_bytes,
    )
    theme = resolve_draw_theme(config, group_id=group_id)
    background = await _report_background(group_id, theme=theme)
    content_width = min(theme.content_width, width - theme.page_padding * 2)
    card_padding = 22
    primary_inner_width = content_width - card_padding * 2
    cover_size = 290 if content_width >= 900 else 240
    primary_gap = 25
    info_width = primary_inner_width - cover_size - primary_gap

    with create_report_canvas(background=background, width=width, theme=theme) as canvas:
        with create_report_column(content_width, theme=theme):
            report_header(
                "识鸟结果",
                width=content_width,
                eyebrow="B I R D   I D E N T I F I C A T I O N",
                meta=f"识别来源 · {result.source_name}",
                theme=theme,
            )

            with report_card(
                content_width,
                padding=card_padding,
                theme=theme,
            ):
                with (
                    HSplit()
                    .set_w(primary_inner_width)
                    .set_sep(primary_gap)
                    .set_item_align("t")
                ):
                    if representative_image is not None:
                        rounded_cover_image(
                            representative_image,
                            size=(cover_size, cover_size),
                        )
                    else:
                        report_image_placeholder(
                            size=(cover_size, cover_size),
                            text="暂无代表图片",
                            theme=theme,
                        )

                    with (
                        VSplit()
                        .set_w(info_width)
                        .set_sep(8)
                        .set_content_and_item_align("l")
                    ):
                        with HSplit().set_w(info_width).set_sep(12).set_item_align("c"):
                            TextBox(
                                primary.name_cn,
                                themed_text_style("title", size=31, theme=theme),
                                use_real_line_count=True,
                            ).set_w(max(1, info_width - 135)).set_padding(0)
                            report_badge(
                                _format_probability(primary.probability),
                                theme=theme,
                            )

                        TextBox(
                            primary.scientific_name or primary.name_en or "暂无学名",
                            themed_text_style("secondary", size=16, theme=theme),
                            line_count=2,
                        ).set_w(info_width).set_padding(0)
                        TextBox(
                            primary.category_cn or "分类信息暂缺",
                            themed_text_style(
                                "accent",
                                size=14,
                                color=theme.accent_dark,
                                theme=theme,
                            ),
                            line_count=2,
                        ).set_w(info_width).set_padding(0)
                        report_info_row(
                            "英文",
                            primary.name_en or "暂无",
                            width=info_width,
                            theme=theme,
                        )
                        if primary.common_names:
                            report_info_row(
                                "俗名",
                                primary.common_names,
                                width=info_width,
                                theme=theme,
                            )
                        report_description_panel(
                            _truncate_description(
                                primary.description,
                                description_max_chars,
                            ),
                            width=info_width,
                            theme=theme,
                        )

            report_section_title("其他可能", width=content_width, theme=theme)
            if other_candidates:
                candidate_gap = 11
                candidate_width = (content_width - candidate_gap) // 2
                with Grid(
                    col_count=2,
                    item_size_mode="fixed",
                    item_align="lt",
                    hsep=candidate_gap,
                    vsep=candidate_gap,
                ):
                    for index, candidate in enumerate(other_candidates, 2):
                        ranked_info_card(
                            index,
                            candidate.name_cn,
                            trailing=_format_probability(candidate.probability),
                            subtitle=(
                                candidate.scientific_name
                                or candidate.name_en
                                or "暂无学名"
                            ),
                            detail=candidate.category_cn or "分类信息暂缺",
                            width=candidate_width,
                            theme=theme,
                        )
            else:
                with report_card(
                    content_width,
                    padding=18,
                    compact=True,
                    theme=theme,
                ):
                    TextBox(
                        "没有返回其他候选鸟种",
                        themed_text_style("muted", theme=theme),
                    ).set_w(content_width - 36).set_padding(0).set_content_align("c")

            TextBox(
                f"结果仅供辅助辨识 · {result.source_url}",
                themed_text_style("muted", size=12, theme=theme),
                line_count=2,
            ).set_w(content_width).set_padding(0).set_content_align("c")

    return await canvas.get_img()


async def _recognize_attached_image(ctx: HandlerContext) -> BirdRecognitionResult:
    """获取消息中的唯一图片、规范化后交由当前配置的后端识别。"""

    image_datas = await ctx.aget_image_datas(max_count=1)
    image_data = image_datas[0]
    image_ref = image_data.get("url") or image_data.get("file")
    if not image_ref:
        raise ReplyException("图片消息缺少可下载地址")

    max_bytes = int(config.get("recognition.image.max_bytes", 4_096_000))
    max_edge = int(config.get("recognition.image.max_edge", 2400))
    max_pixels = int(config.get("recognition.image.max_pixels", 40_000_000))

    async with TempBotOrInternetFilePath("image", image_ref, ctx.bot) as input_path:
        with TempFilePath("jpg") as output_path:
            await run_in_pool(
                _prepare_image_for_upload,
                input_path,
                output_path,
                max_bytes,
                max_edge,
                max_pixels,
            )
            backend = create_recognition_backend()
            return await backend.recognize(output_path)


identify_bird = CmdHandler("/识鸟", logger)
# 先检查群冷却，避免同群其他用户因群冷却失败而消耗个人冷却。
identify_bird.check_cdrate(group_cd).check_cdrate(user_cd).check_wblist(gwl)


@identify_bird.handle()
async def handle_identify_bird(ctx: HandlerContext):
    """识别当前消息或回复消息中附带的一张鸟类图片。"""

    await ctx.block(f"bird-recognition-{ctx.user_id}")
    try:
        result = await _recognize_attached_image(ctx)
        report = await render_recognition_report(
            result,
            group_id=ctx.group_id,
        )
        return await ctx.asend_reply_msg(
            await get_image_cq(report, low_quality=True, logger=logger)
        )
    except ReplyException:
        raise
    except BirdRecognitionError as exc:
        raise ReplyException(f"识鸟失败：{exc}") from exc
    except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
        raise ReplyException("识鸟服务连接超时，请稍后重试") from exc
    except Exception as exc:
        logger.print_exc("识鸟指令处理失败")
        raise ReplyException("识鸟失败，请稍后重试") from exc
