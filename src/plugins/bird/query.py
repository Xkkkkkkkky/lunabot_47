"""使用大模型规范化鸟名，并从多语言 Wikipedia 生成鸟类档案。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse
from urllib.request import getproxies, proxy_bypass

import httpx
from bs4 import BeautifulSoup
from PIL import Image

from ..llm import ChatSession, ChatSessionResponse, get_model_preset
from ..utils import (
    CmdHandler,
    ColdDown,
    Config,
    HandlerContext,
    HSplit,
    ReplyException,
    TextBox,
    VSplit,
    create_report_canvas,
    create_report_column,
    dumps_json,
    get_file_db,
    get_group_white_list,
    get_image_cq,
    get_logger,
    loads_json,
    report_card,
    report_description_panel,
    report_header,
    report_image_placeholder,
    report_info_row,
    report_section_title,
    report_text_section,
    resolve_configured_draw_background,
    resolve_draw_theme,
    rounded_cover_image,
    run_in_pool,
    themed_text_style,
)
from .alias import BirdAliasManager, BirdAliasRecord


config = Config("bird")
logger = get_logger("BirdQuery")
file_db = get_file_db("data/bird/db.json", logger)
cd = ColdDown(file_db, logger)
gwl = get_group_white_list(file_db, logger, "bird")


class BirdQueryError(Exception):
    """鸟名解析或 Wiki 查询未得到可用结果。"""


def _resolution_from_alias(record: BirdAliasRecord) -> "BirdNameResolution":
    """把持久化鸟种记录转换为与 LLM 输出相同的后续查询输入。"""

    return BirdNameResolution(
        status="resolved",
        confidence=1.0,
        bird=BirdNameCandidate(
            name_zh=record.name_zh,
            name_en=record.name_en,
            name_ja=record.name_ja,
            scientific_name=record.canonical_name,
            reason="本地鸟类别名索引",
        ),
        candidates=(),
    )


@dataclass(frozen=True)
class BirdNameCandidate:
    """大模型返回的规范化候选鸟种。"""

    name_zh: str = ""
    name_en: str = ""
    name_ja: str = ""
    scientific_name: str = ""
    reason: str = ""

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name_en or self.name_ja or self.scientific_name


@dataclass(frozen=True)
class BirdNameResolution:
    """经过字段清洗和类型校验的大模型鸟名解析结果。"""

    status: str
    confidence: float
    bird: BirdNameCandidate
    candidates: tuple[BirdNameCandidate, ...]
    message: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved" and bool(self.bird.scientific_name)


@dataclass(frozen=True)
class WikiLanguage:
    code: str
    label: str
    api_url: str
    variant: str = ""


@dataclass(frozen=True)
class WikiSection:
    title: str
    text: str


@dataclass(frozen=True)
class WikiBirdArticle:
    """与 Wikipedia 语言和原始 HTML 解耦的绘图数据。"""

    title: str
    url: str
    language: WikiLanguage
    image_url: Optional[str]
    summary: str
    sections: tuple[WikiSection, ...]
    facts: tuple[tuple[str, str], ...] = ()


_SPACE_RE = re.compile(r"\s+")
_REFERENCE_RE = re.compile(r"\[(?:\d+|需要引文|來源請求|citation needed)\]", re.I)
_REFRESH_RE = re.compile(r"(?:^|\s)refresh\s*$", re.I)
_UNSAFE_CACHE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def _clean_text(value: object, max_chars: int = 160) -> str:
    """将模型和 Wiki 的任意标量收敛为长度受限的单行文本。"""

    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    return text[:max_chars]


def _candidate_from_payload(payload: object) -> BirdNameCandidate:
    if not isinstance(payload, dict):
        return BirdNameCandidate()
    return BirdNameCandidate(
        name_zh=_clean_text(payload.get("name_zh")),
        name_en=_clean_text(payload.get("name_en")),
        name_ja=_clean_text(payload.get("name_ja")),
        scientific_name=_clean_text(payload.get("scientific_name")),
        reason=_clean_text(payload.get("reason"), 240),
    )


def _extract_json_object(text: str) -> dict:
    """只接受回复中最外层 JSON 对象，兼容模型偶发添加的代码围栏。"""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("回复中没有 JSON 对象")
    payload = loads_json(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return payload


def _normalize_resolution(response: ChatSessionResponse) -> BirdNameResolution:
    """验证大模型固定 schema，并将所有输入规范化为稳定的内部类型。"""

    payload = _extract_json_object(response.result)
    status = _clean_text(payload.get("status"), 24).lower()
    if status not in {"resolved", "ambiguous", "not_bird"}:
        raise ValueError(f"未知解析状态: {status or '空'}")

    try:
        confidence = float(payload.get("confidence", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence 不是数字") from exc
    confidence = min(1.0, max(0.0, confidence))

    bird = _candidate_from_payload(payload.get("bird"))
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates 必须是数组")
    max_candidates = max(1, int(config.get("query.parser.max_candidates", 5)))
    candidates = tuple(
        candidate
        for candidate in (
            _candidate_from_payload(item) for item in raw_candidates[:max_candidates]
        )
        if candidate.display_name
    )

    min_confidence = float(config.get("query.parser.min_confidence", 0.72))
    if status == "resolved" and (
        not bird.scientific_name or confidence < min_confidence
    ):
        # 低置信度结果不能驱动后续 Wiki 查询，转为候选提示更安全。
        status = "ambiguous"
        if bird.scientific_name and all(item != bird for item in candidates):
            candidates = (bird, *candidates)[:max_candidates]

    return BirdNameResolution(
        status=status,
        confidence=confidence,
        bird=bird,
        candidates=candidates,
        message=_clean_text(payload.get("message"), 300),
    )


async def parse_bird_name(user_input: str) -> BirdNameResolution:
    """按 cron 插件的策略，通过文件提示词、模型预设和重试解析输入。"""

    prompt_path = Path(str(config.get("query.parser.system_prompt_path")))
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BirdQueryError(f"无法读取鸟名解析提示词: {prompt_path}") from exc

    model_name = get_model_preset(str(config.get("query.parser.model_preset", "bird.query")))
    max_retries = max(1, int(config.get("query.parser.max_retries", 2)))
    last_error: Optional[Exception] = None
    # 每次重试都创建新会话，避免上一次不合规输出污染下一次请求。
    for retry_index in range(max_retries):
        try:
            session = ChatSession(system_prompt)
            session.append_user_content(
                dumps_json({"query": _clean_text(user_input, 200)}, indent=False)
            )
            return await session.get_response(
                model_name,
                process_func=_normalize_resolution,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"鸟名解析第 {retry_index + 1}/{max_retries} 次失败: {exc}"
            )
    raise BirdQueryError("大模型未能返回规范的鸟名解析结果") from last_error


async def resolve_bird_name(
    user_input: str,
) -> tuple[BirdNameResolution, bool, Optional[str]]:
    """优先解析本地索引，并把 LLM 的唯一候选提升为直接查询结果。

    返回值依次为解析结果、是否命中本地索引、建议保存的原始别名。只有
    LLM 未确定但给出唯一候选时，第三项才非空。
    """

    alias_record = BirdAliasManager.get().resolve(user_input)
    if alias_record is not None:
        return _resolution_from_alias(alias_record), True, None

    resolution = await parse_bird_name(user_input)
    if not resolution.is_resolved and len(resolution.candidates) == 1:
        candidate = resolution.candidates[0]
        if candidate.scientific_name:
            resolution = BirdNameResolution(
                status="resolved",
                confidence=resolution.confidence,
                bird=candidate,
                candidates=(),
                message=resolution.message,
            )
            return resolution, False, user_input
    return resolution, False, None


def _wiki_languages() -> tuple[WikiLanguage, ...]:
    raw_languages = config.get("query.wiki.languages")
    if not isinstance(raw_languages, list) or not raw_languages:
        raise BirdQueryError("query.wiki.languages 配置为空或格式不正确")
    languages = []
    for item in raw_languages:
        if not isinstance(item, dict):
            continue
        code = _clean_text(item.get("code"), 12)
        label = _clean_text(item.get("label"), 40)
        api_url = _clean_text(item.get("api_url"), 500)
        variant = _clean_text(item.get("variant"), 20)
        if code and label and api_url.startswith(("http://", "https://")):
            languages.append(WikiLanguage(code, label, api_url, variant))
    if not languages:
        raise BirdQueryError("没有可用的 Wikipedia 语言配置")
    return tuple(languages)


def _localized_terms(
    bird: BirdNameCandidate,
    language_code: str,
    original_input: str,
) -> tuple[str, ...]:
    localized = {
        "zh": bird.name_zh,
        "en": bird.name_en,
        "ja": bird.name_ja,
    }.get(language_code, "")
    terms = [localized, bird.scientific_name]
    if language_code == "zh":
        terms.append(original_input)
    return tuple(dict.fromkeys(term for term in map(_clean_text, terms) if term))


def _wiki_proxy(target_url: str) -> Optional[str]:
    """根据 Wiki 配置选择代理，并可显式继承进程环境代理。

    项目共享的 ``aiohttp.ClientSession`` 默认不启用 ``trust_env``，因此即使
    运行环境提供了 HTTP_PROXY/HTTPS_PROXY，也必须在单次请求中传入代理。
    """

    configured = str(config.get("query.wiki.proxy", "", raise_exc=False) or "").strip()
    if configured:
        return configured
    if not bool(config.get("query.wiki.use_env_proxy", False)):
        return None

    parsed = urlparse(target_url)
    if parsed.hostname and proxy_bypass(parsed.hostname):
        return None
    proxies = getproxies()
    return proxies.get(parsed.scheme) or proxies.get("all")


def _wiki_headers(*, accept: str, referer: str = "") -> dict[str, str]:
    """构造 Wikimedia 可识别的统一请求头。"""

    headers = {
        "User-Agent": str(config.get("query.wiki.user_agent")),
        "Accept": accept,
    }
    if referer:
        headers["Referer"] = referer
    return headers


async def _request_wiki(language: WikiLanguage, params: dict) -> dict:
    timeout = float(config.get("query.wiki.timeout_seconds", 20))
    headers = _wiki_headers(accept="application/json")
    max_attempts = max(1, int(config.get("query.wiki.request_retries", 3)))
    retry_backoff = max(
        0.1,
        float(config.get("query.wiki.retry_backoff_seconds", 1.0)),
    )
    retry_statuses = {429, 500, 502, 503, 504}
    # aiohttp 通过部分 HTTP CONNECT 代理读取较大的 parse 响应时会被重置；
    # Wiki 查询使用独立 httpx 客户端，避免影响项目其他共享 HTTP 会话。
    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        proxy=_wiki_proxy(language.api_url),
        trust_env=False,
        follow_redirects=True,
    ) as client:
        for attempt in range(max_attempts):
            response = await client.get(
                language.api_url,
                params={
                    "format": "json",
                    "formatversion": "2",
                    **({"variant": language.variant, "uselang": language.variant}
                       if language.variant else {}),
                    **params,
                },
            )
            body = response.content
            if response.status_code == 200:
                payload = loads_json(body)
                if not isinstance(payload, dict) or payload.get("error"):
                    raise BirdQueryError(f"{language.label} Wikipedia 返回异常数据")
                return payload
            if response.status_code not in retry_statuses or attempt == max_attempts - 1:
                raise BirdQueryError(
                    f"{language.label} Wikipedia 请求失败（HTTP {response.status_code}）"
                )

            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = max(retry_backoff, float(retry_after))
            except ValueError:
                delay = retry_backoff * (attempt + 1)
            # 避免异常 Retry-After 令指令长时间占用处理器。
            delay = min(delay, 8.0)
            logger.warning(
                f"{language.label} Wikipedia 返回 HTTP {response.status_code}，"
                f"{delay:.1f} 秒后重试（{attempt + 2}/{max_attempts}）"
            )
            await asyncio.sleep(delay)
    raise BirdQueryError(f"{language.label} Wikipedia 请求失败")


def _usable_page(page: object) -> bool:
    return (
        isinstance(page, dict)
        and not page.get("missing")
        and "disambiguation" not in (page.get("pageprops") or {})
    )


async def _find_wiki_page(
    language: WikiLanguage,
    terms: tuple[str, ...],
) -> Optional[dict]:
    """优先查精确标题；失败后才使用 Wikipedia 自身全文搜索。"""

    common_params = {
        "action": "query",
        "prop": "pageprops|pageimages|info",
        "inprop": "url",
        "piprop": "original|thumbnail",
        "pithumbsize": int(config.get("query.wiki.image_width", 1000)),
        "redirects": "1",
    }
    # 逐项精确查询以保留名称优先级：本地语言规范名应先于拉丁学名，
    # 不能依赖 API 按 pageid 返回多个标题时的非输入顺序。
    for term in terms:
        payload = await _request_wiki(
            language,
            {**common_params, "titles": term},
        )
        pages = payload.get("query", {}).get("pages", [])
        for page in pages:
            if _usable_page(page):
                return page

    search_limit = max(1, int(config.get("query.wiki.search_limit", 3)))
    for term in terms:
        payload = await _request_wiki(
            language,
            {
                **common_params,
                "generator": "search",
                "gsrsearch": term,
                "gsrnamespace": "0",
                "gsrlimit": search_limit,
            },
        )
        for page in payload.get("query", {}).get("pages", []):
            if _usable_page(page):
                return page
    return None


def _append_section_text(sections: list[dict], text: str) -> None:
    text = _REFERENCE_RE.sub("", _SPACE_RE.sub(" ", text)).strip()
    if text and text not in sections[-1]["paragraphs"]:
        sections[-1]["paragraphs"].append(text)


def _extract_infobox_facts(soup: BeautifulSoup) -> tuple[tuple[str, str], ...]:
    """从 Wiki infobox 中提取分类、保护状态等学术化键值信息。"""

    table = soup.select_one("table.infobox")
    if table is None:
        return ()
    max_facts = max(0, int(config.get("query.render.max_facts", 8)))
    if max_facts == 0:
        return ()
    facts = []
    for row in table.select("tr"):
        label_node = row.find("th", recursive=False)
        value_node = row.find("td", recursive=False)
        if label_node is None or value_node is None:
            continue
        label = _clean_text(label_node.get_text(" ", strip=True), 40)
        value = _clean_text(
            _REFERENCE_RE.sub("", value_node.get_text(" ", strip=True)),
            220,
        )
        if label and value and (label, value) not in facts:
            facts.append((label, value))
        if len(facts) >= max_facts:
            break
    return tuple(facts)


def _extract_wiki_sections(
    html_text: str,
) -> tuple[str, tuple[WikiSection, ...], tuple[tuple[str, str], ...]]:
    """从 Wiki 正文提取概述和分节文字，删除引用、表格与导航模板。"""

    soup = BeautifulSoup(html_text, "html.parser")
    facts = _extract_infobox_facts(soup)
    for selector in (
        "table",
        "style",
        "script",
        "sup.reference",
        ".mw-editsection",
        ".navbox",
        ".vertical-navbox",
        ".metadata",
        ".ambox",
        ".thumb",
        ".gallery",
    ):
        for element in soup.select(selector):
            element.decompose()

    skip_titles = {
        _clean_text(title).casefold()
        for title in config.get("query.wiki.skip_sections", [])
    }
    sections: list[dict] = [{"title": "概述", "paragraphs": []}]
    skipping = False
    for node in soup.select("h2, h3, p, li"):
        if node.name in {"h2", "h3"}:
            title = _clean_text(node.get_text(" ", strip=True), 80)
            skipping = title.casefold() in skip_titles
            if title and not skipping:
                sections.append({"title": title, "paragraphs": []})
            continue
        if skipping or node.find_parent(("li", "p")) is not None:
            continue
        text = node.get_text(" ", strip=True)
        if node.name == "li" and text:
            text = f"• {text}"
        _append_section_text(sections, text)

    max_sections = max(1, int(config.get("query.render.max_sections", 6)))
    section_max_chars = max(
        120,
        int(config.get("query.render.section_max_chars", 900)),
    )
    total_max_chars = max(
        section_max_chars,
        int(config.get("query.render.total_max_chars", 4800)),
    )
    remaining = total_max_chars
    normalized: list[WikiSection] = []
    for section in sections:
        text = "\n".join(section["paragraphs"]).strip()
        if not text or remaining <= 0:
            continue
        limit = min(section_max_chars, remaining)
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        normalized.append(WikiSection(section["title"], text))
        remaining -= len(text)
        if len(normalized) >= max_sections:
            break

    if not normalized:
        return "", (), facts
    # 首段固定作为主卡摘要；其余分节使用自适应正文卡，避免重复绘制。
    summary = normalized.pop(0).text
    return summary, tuple(normalized), facts


async def _fetch_wiki_article(
    language: WikiLanguage,
    page: dict,
) -> Optional[WikiBirdArticle]:
    title = _clean_text(page.get("title"), 200)
    if not title:
        return None
    payload = await _request_wiki(
        language,
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "disabletoc": "1",
            "redirects": "1",
        },
    )
    html_text = payload.get("parse", {}).get("text", "")
    if not isinstance(html_text, str):
        return None
    summary, sections, facts = _extract_wiki_sections(html_text)
    min_chars = max(1, int(config.get("query.wiki.min_content_chars", 80)))
    if len(summary) + sum(len(item.text) for item in sections) < min_chars:
        return None
    # 优先缩略图以避开 SVG 等 Pillow 无法直接解码的原始媒体。
    image_data = page.get("thumbnail") or page.get("original") or {}
    return WikiBirdArticle(
        title=title,
        url=_clean_text(page.get("fullurl"), 1000),
        language=language,
        image_url=_clean_text(image_data.get("source"), 1000) or None,
        summary=summary,
        sections=sections,
        facts=facts,
    )


async def find_wiki_article(
    resolution: BirdNameResolution,
    original_input: str,
) -> WikiBirdArticle:
    """依配置顺序查询 Wiki；默认顺序即中文、英文、日文回退。"""

    errors = []
    attempted_languages = 0
    for language in _wiki_languages():
        terms = _localized_terms(resolution.bird, language.code, original_input)
        if not terms:
            continue
        attempted_languages += 1
        try:
            page = await _find_wiki_page(language, terms)
            if page and (article := await _fetch_wiki_article(language, page)):
                return article
        except Exception as exc:
            error_desc = f"{type(exc).__name__}: {exc}"
            errors.append(f"{language.code}: {error_desc}")
            logger.warning(f"查询 {language.label} Wikipedia 失败: {error_desc}")
    detail = f"（{'；'.join(errors)}）" if errors else ""
    if errors and len(errors) == attempted_languages:
        raise BirdQueryError(f"Wikipedia 请求全部失败{detail}")
    if errors:
        raise BirdQueryError(f"未找到条目，且部分 Wikipedia 请求失败{detail}")
    raise BirdQueryError(f"中、英、日 Wikipedia 均未找到可用条目{detail}")


async def _download_article_image(
    image_url: Optional[str],
    referer_url: str = "",
) -> Optional[Image.Image]:
    """受超时和体积限制地下载 Wiki 首图，失败不影响正文报告。"""

    if not image_url:
        return None
    timeout = float(config.get("query.wiki.image_timeout_seconds", 15))
    max_bytes = int(config.get("query.wiki.image_max_bytes", 10_485_760))
    try:
        async with httpx.AsyncClient(
            headers=_wiki_headers(
                accept="image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                referer=referer_url,
            ),
            timeout=timeout,
            proxy=_wiki_proxy(image_url),
            trust_env=False,
            follow_redirects=True,
        ) as client:
            response = await client.get(image_url)
            if response.status_code != 200:
                raise BirdQueryError(f"HTTP {response.status_code}")
            content = response.content
            if not content or len(content) > max_bytes:
                raise BirdQueryError("图片为空或超过体积限制")
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGBA")
    except Exception as exc:
        logger.warning(f"下载 Wiki 鸟类首图失败: {exc}")
        return None


async def _report_background(group_id=None, *, theme=None):
    theme = theme or resolve_draw_theme(config, group_id=group_id)
    return await resolve_configured_draw_background(
        config,
        theme=theme,
        logger=logger,
        label="查鸟报告",
    )


def _report_cache_path(
    resolution: BirdNameResolution,
    group_id: int | str | None = None,
) -> Optional[Path]:
    """返回规范鸟种对应的缓存文件，目录名保持可读且不可路径穿越。"""

    if not bool(config.get("query.cache.enabled", True)):
        return None
    cache_root = str(config.get("query.cache.directory", "")).strip()
    if not cache_root:
        return None

    identity = resolution.bird.scientific_name
    if not identity:
        return None
    safe_name = _UNSAFE_CACHE_NAME_RE.sub("_", identity).strip(". ")[:80]
    if not safe_name:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        safe_name = f"bird-{digest}"
    cache_version = max(1, int(config.get("query.cache.version", 1)))
    # 群级主题可不同，缓存必须隔离，否则其他群会命中错误配色。
    group_cache_key = str(group_id if group_id is not None else 0)
    return (
        Path(cache_root).expanduser()
        / safe_name
        / f"report-v{cache_version}-g{group_cache_key}.jpg"
    )


def _save_report_cache(image: Image.Image, cache_path: Path, quality: int) -> None:
    """将报告以 JPEG 原子写入缓存，避免并发读取到半成品文件。"""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        image.convert("RGB").save(
            temp_path,
            format="JPEG",
            quality=min(95, max(50, quality)),
            optimize=True,
            progressive=False,
        )
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _remember_resolution(resolution: BirdNameResolution) -> None:
    """保存已成功取得报告的规范鸟种，失败不影响本次结果发送。"""

    bird = resolution.bird
    if not bird.scientific_name:
        logger.warning("解析结果缺少学名，已跳过鸟类别名索引保存")
        return
    try:
        BirdAliasManager.get().remember(
            bird.scientific_name,
            name_zh=bird.name_zh,
            name_en=bird.name_en,
            name_ja=bird.name_ja,
        )
    except Exception as exc:
        logger.warning(f"保存鸟类别名索引失败: {exc}")


def _alias_suggestion_text(
    alias: Optional[str],
    resolution: BirdNameResolution,
) -> str:
    """生成唯一候选成功查询后的 alias 保存提示。"""

    if not alias:
        return ""
    canonical_name = resolution.bird.scientific_name
    if not canonical_name or alias == canonical_name:
        return ""

    def quote(value: str) -> str:
        # 始终使用引号，兼容含空格名称；全角括号避免用户输入形成 CQ 码。
        value = value.replace("[", "［").replace("]", "］")
        value = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{value}"'

    command = f"/bird add alias {quote(canonical_name)} {quote(alias)}"
    return f"本次按“{canonical_name}”查询；可发送 {command} 保存别名。\n"


def _metadata_rows(bird: BirdNameCandidate, article: WikiBirdArticle) -> Iterable[tuple[str, str]]:
    rows = [
        ("学名", bird.scientific_name),
        ("英文", bird.name_en),
        ("日文", bird.name_ja),
        ("中文", bird.name_zh if bird.name_zh != article.title else ""),
        ("条目", article.title),
        ("来源", f"{article.language.label} Wikipedia"),
        *article.facts,
    ]
    seen_values = set()
    for label, value in rows:
        normalized_value = _clean_text(value, 220)
        value_key = normalized_value.casefold()
        if not label or not normalized_value or value_key in seen_values:
            continue
        seen_values.add(value_key)
        yield _clean_text(label, 40), normalized_value


async def render_bird_article(
    resolution: BirdNameResolution,
    article: WikiBirdArticle,
    *,
    group_id: int | str | None = None,
) -> Image.Image:
    """使用统一 draw 库生成随 Wiki 内容长度自动增高的学术风格档案。"""

    render_cfg = config.get("query.render", {})
    width = max(720, int(render_cfg.get("width", 1000)))
    theme = resolve_draw_theme(config, group_id=group_id)
    content_width = min(theme.content_width, width - theme.page_padding * 2)
    cover_size = min(300, max(220, int(render_cfg.get("cover_size", 290))))
    card_padding = 22
    gap = 24
    inner_width = content_width - card_padding * 2
    info_width = inner_width - cover_size - gap
    image, background = await asyncio.gather(
        _download_article_image(article.image_url, article.url),
        _report_background(group_id, theme=theme),
    )

    display_name = resolution.bird.scientific_name or article.title
    with create_report_canvas(background=background, width=width, theme=theme) as canvas:
        with create_report_column(content_width, theme=theme):
            report_header(
                display_name,
                width=content_width,
                eyebrow="O R N I T H O L O G Y   N O T E",
                meta=f"",
                theme=theme,
            )

            with report_card(content_width, padding=card_padding, theme=theme):
                with HSplit().set_w(inner_width).set_sep(gap).set_item_align("t"):
                    if image is None:
                        report_image_placeholder(
                            size=(cover_size, cover_size),
                            text="暂无条目图片",
                            theme=theme,
                        )
                    else:
                        rounded_cover_image(image, size=(cover_size, cover_size))

                    with (
                        VSplit()
                        .set_w(info_width)
                        .set_sep(7)
                        .set_content_and_item_align("l")
                    ):
                        TextBox(
                            display_name,
                            themed_text_style("title", size=30, theme=theme),
                            use_real_line_count=True,
                        ).set_w(info_width).set_padding(0)
                        common_name = resolution.bird.name_zh or article.title
                        if common_name and common_name != display_name:
                            TextBox(
                                common_name,
                                themed_text_style("accent", size=17, theme=theme),
                                line_count=2,
                            ).set_w(info_width).set_padding(0)
                        for label, value in _metadata_rows(resolution.bird, article):
                            report_info_row(
                                label,
                                value,
                                width=info_width,
                                label_width=44,
                                theme=theme,
                            )
                        if article.summary:
                            report_description_panel(
                                article.summary,
                                width=info_width,
                                label="条目摘要",
                                theme=theme,
                            )

            if article.sections:
                report_section_title("物种资料", width=content_width, theme=theme)
                for section in article.sections:
                    report_text_section(
                        section.title,
                        section.text,
                        width=content_width,
                        theme=theme,
                    )

            TextBox(
                f"资料来源：{article.url}",
                themed_text_style("muted", size=12, theme=theme),
                use_real_line_count=True,
            ).set_w(content_width).set_padding(0).set_content_align("c")
            TextBox(
                "Wikipedia 内容可能随时间更新；分类与保护信息请以权威名录为准",
                themed_text_style("muted", size=12, theme=theme),
                line_count=2,
            ).set_w(content_width).set_padding(0).set_content_align("c")

    return await canvas.get_img()


def _format_candidates(resolution: BirdNameResolution) -> str:
    lines = []
    for candidate in resolution.candidates:
        names = [candidate.display_name]
        if candidate.scientific_name and candidate.scientific_name != candidate.display_name:
            names.append(candidate.scientific_name)
        suffix = f"：{candidate.reason}" if candidate.reason else ""
        lines.append(f"- {' / '.join(names)}{suffix}")
    return "\n".join(lines)


def _unresolved_reply(resolution: BirdNameResolution) -> str:
    if resolution.status == "not_bird":
        heading = resolution.message or "输入内容不像一个可识别的鸟类名称。"
    else:
        heading = resolution.message or "暂时无法确定你指的是哪一种鸟。"
    candidates = _format_candidates(resolution)
    if candidates:
        return f"{heading}\n其他可能的鸟类：\n{candidates}"
    return f"{heading}\n请尝试补充标准中文名、英文名或拉丁学名。"


bird = CmdHandler(["/bird", "/查鸟", "/鸟"], logger)
bird.check_cdrate(cd).check_wblist(gwl)


@bird.handle()
async def handle_bird(ctx: HandlerContext):
    """解析用户鸟名、查询多语言 Wiki，并回复 draw 生成的档案图。"""

    raw_name = ctx.get_args().strip()
    # refresh 保留旧指令语义：跳过现有报告缓存并重新查询、绘制。
    refresh = bool(_REFRESH_RE.search(raw_name))
    bird_name = _REFRESH_RE.sub("", raw_name).strip()
    if not bird_name:
        raise ReplyException("请输入要查询的鸟类名称")

    await ctx.block(f"bird-query-{ctx.user_id}")
    try:
        resolution, resolved_locally, suggested_alias = await resolve_bird_name(bird_name)
        if not resolution.is_resolved:
            return await ctx.asend_reply_msg(_unresolved_reply(resolution))

        cache_path = _report_cache_path(resolution, ctx.group_id)
        if cache_path is not None and cache_path.is_file() and not refresh:
            try:
                cached_message = await get_image_cq(
                    str(cache_path),
                    low_quality=True,
                    logger=logger,
                )
                if not resolved_locally:
                    _remember_resolution(resolution)
                return await ctx.asend_reply_msg(
                    _alias_suggestion_text(suggested_alias, resolution)
                    + cached_message
                )
            except Exception as exc:
                # 损坏或不可读缓存不阻断查询，成功后会被新报告覆盖。
                logger.warning(f"读取鸟类报告缓存失败，将重新生成: {exc}")

        try:
            article = await find_wiki_article(resolution, bird_name)
        except BirdQueryError as exc:
            # 鸟种已确定时不再重复展示 LLM 候选；候选仅用于解析不确定路径。
            return await ctx.asend_reply_msg(f"查阅 Wikipedia 失败：{exc}")

        report = await render_bird_article(
            resolution,
            article,
            group_id=ctx.group_id,
        )
        if not resolved_locally:
            _remember_resolution(resolution)
        if cache_path is not None:
            try:
                await run_in_pool(
                    _save_report_cache,
                    report,
                    cache_path,
                    int(config.get("query.cache.jpeg_quality", 88)),
                )
            except Exception as exc:
                # 缓存属于优化路径，写入失败时仍发送本次已生成的报告。
                logger.warning(f"写入鸟类报告缓存失败: {exc}")
        return await ctx.asend_reply_msg(
            _alias_suggestion_text(suggested_alias, resolution)
            + await get_image_cq(report, low_quality=True, logger=logger)
        )
    except ReplyException:
        raise
    except BirdQueryError as exc:
        raise ReplyException(f"查鸟失败：{exc}") from exc
    except (asyncio.TimeoutError, httpx.HTTPError) as exc:
        raise ReplyException("查鸟服务连接超时，请稍后重试") from exc
    except Exception as exc:
        logger.print_exc("查鸟指令处理失败")
        raise ReplyException("查鸟失败，请稍后重试") from exc
