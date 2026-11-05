"""查询与图片输出。"""

from __future__ import annotations

import asyncio
import ast
import html
import operator
import random
import re
from dataclasses import dataclass

from ..utils import ReplyException
from .asset import assets
from .data import ATTRIBUTES, DIFF_NAMES, data, parse_date, released, search
from .draw import render_detail, render_list, render_story
from .metrics import build_metrics, format_duration, format_number


PAGE_SIZE = 8
META_NOTE = "理论值 · 同综合力 · 5×100%／5秒技能 · 音频时长+30秒 · 覆盖率取 Free"


@dataclass
class QueryOptions:
    query: str = ""
    page: int = 1
    difficulty: str | None = None
    level: tuple[float, float] | None = None
    mode: str = "gekisou"
    sort: str = "efficiency"
    refresh: bool = False


def parse_options(query: str, *, chart: bool = False, story: bool = False) -> QueryOptions:
    options = QueryOptions()
    words = query.strip().split()
    remaining = []
    index = 0
    aliases = {"ez": "easy", "nm": "normal", "hd": "hard", "ex": "expert", "exp": "expert"}
    while index < len(words):
        word = words[index]
        lowered = word.lower()
        page = re.fullmatch(r"(?:p|page=)(\d+)", lowered)
        level = re.fullmatch(
            r"(easy|normal|hard|expert|ez|nm|hd|ex|exp)?(?:lv)?(\d{1,2}(?:\.\d+)?)(?:-(\d{1,2}(?:\.\d+)?))?",
            lowered,
        ) if chart else None
        if lowered == "--page":
            index += 1
            if index >= len(words) or not words[index].isdigit():
                raise ReplyException("请在 --page 后输入页码，例如 --page 2。")
            options.page = int(words[index])
        elif page:
            options.page = int(page[1])
        elif chart and lowered in (*DIFF_NAMES, *aliases):
            options.difficulty = aliases.get(lowered, lowered)
        elif chart and (level or lowered.startswith("lv")):
            if not level:
                raise ReplyException("等级格式：26、25-28 或 lv26。")
            if level[1]:
                options.difficulty = aliases.get(level[1], level[1])
            low, high = float(level[2]), float(level[3] or level[2])
            if low > high:
                raise ReplyException("难度范围的下限不能高于上限。")
            options.level = (low, high)
        elif chart and lowered in ("free", "gekisou"):
            options.mode = lowered
        elif chart and lowered.startswith("sort="):
            options.sort = lowered[5:]
            if options.sort not in ("efficiency", "score", "duration", "nps", "coverage"):
                raise ReplyException("排序支持 efficiency、score、duration、nps、coverage。")
        elif story and lowered == "--refresh":
            options.refresh = True
        else:
            remaining.append(word)
        index += 1
    if options.page < 1:
        raise ReplyException("页码必须从 1 开始。")
    options.query = " ".join(remaining)
    return options


def _page(rows: list, page: int):
    if not rows:
        raise ReplyException("找不到符合条件的 HK/MO/TW 资料，请调整名称、ID 或筛选条件。")
    pages = (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE
    if page > pages:
        raise ReplyException(f"只有 {pages} 页，请使用 p1 至 p{pages}。")
    return rows[(page - 1) * PAGE_SIZE:page * PAGE_SIZE], f"共 {len(rows)} 项 · 第 {page}/{pages} 页"


def _date(value: str) -> str:
    parsed = parse_date(value)
    return parsed.strftime("%Y/%m/%d %H:%M") if parsed else "未提供"


def _levels(song: dict) -> str:
    return "  ·  ".join(f"{DIFF_NAMES[chart['difficulty']]} {chart['display_level']:g}" for chart in song["charts"])


def _charts(song: dict, options: QueryOptions):
    return [chart for chart in song["charts"]
            if (options.difficulty is None or chart["difficulty"] == options.difficulty)
            and (options.level is None or options.level[0] <= chart["display_level"] <= options.level[1])]


async def _music_matches(options: QueryOptions) -> list[dict]:
    rows = await data.music()
    if not options.query:
        rows = [row for row in rows if released(row)]
    rows = search(rows, options.query, limit=len(rows))
    return [row for row in rows if _charts(row, options)]


async def _choices(title: str, rows: list[dict], page: int, *, kind: str):
    selected, subtitle = _page(rows, page)
    images = await asyncio.gather(*(assets.image(row.get("thumbnail_key", row["image_key"])) for row in selected))
    rendered = []
    for row, image in zip(selected, images):
        if kind == "music":
            secondary = row["band"] or row["vocalists"]
            detail = _levels(row)
        elif kind == "card":
            secondary = f"{row['character']} · {row['band']}"
            detail = f"{row['rarity']} · {row['attribute']}属性 · 综合力 {row['total_power']:,}"
        else:
            secondary = f"{row['status']} · {_date(row['start_at'])}"
            detail = f"结束 {_date(row['end_at'])}"
        rendered.append({"title": row["title"], "subtitle": secondary, "detail": detail, "tag": f"#{row['id']}", "image": image})
    return await render_list(title, subtitle, rendered)


async def query_music(query: str, *, list_mode: bool = False):
    options = parse_options(query, chart=True)
    rows = await _music_matches(options)
    if list_mode or len(rows) != 1:
        return await _choices("歌曲列表" if list_mode else "歌曲查询结果", rows, options.page, kind="music")
    song = rows[0]
    hero = await assets.image(song["image_key"])
    sections = [
        ("歌曲资料", [("演出乐队", song["band"] or "未提供"), ("演唱", song["vocalists"] or "未提供"),
                      ("作曲", song["composer"] or "未提供"), ("作词", song["lyricist"] or "未提供"),
                      ("编曲", song["arranger"] or "未提供"), ("歌曲属性", song["attribute"]),
                      ("公开时间", _date(song["start_at"]))]),
        ("谱面难度", [(DIFF_NAMES[chart["difficulty"]], f"Lv.{chart['display_level']:g}  ·  {chart['notes']:,} COMBO")
                      for chart in song["charts"]]),
    ]
    return await render_detail(song["title"], f"歌曲 #{song['id']}", hero, sections, eyebrow="MUSIC")


async def query_cover(query: str):
    options = parse_options(query)
    if not options.query:
        raise ReplyException("请输入要查询曲绘的歌曲名称或 ID。")
    rows = search(await data.music(), options.query)
    if len(rows) != 1:
        return await _choices("请选择歌曲 ID", rows, options.page, kind="music")
    song = rows[0]
    return await assets.image_path(song["image_key"])


def _bpm(metrics: dict) -> str:
    low, high = metrics.get("bpmMin"), metrics.get("bpmMax")
    if low is not None and high is not None:
        return f"{low:g}" if low == high else f"{low:g}–{high:g}"
    return format_number(metrics.get("bpm"), 1)


def _percent(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


async def query_meta(query: str):
    options = parse_options(query, chart=True)
    songs, raw = await asyncio.gather(_music_matches(options), assets.music_meta())
    metrics = build_metrics(raw, mode=options.mode)
    mode_name = "Free Live" if options.mode == "free" else "激奏 Live／第1名"
    if len(songs) == 1 and options.query:
        song = songs[0]
        sections = []
        for chart in _charts(song, options):
            item = metrics.get(chart["score_id"], {})
            sections.append((f"{DIFF_NAMES[chart['difficulty']]} · Lv.{chart['display_level']:g}", [
                ("音频／谱面时长", f"{format_duration(item.get('durationMs'))} ／ {format_duration(item.get('chartDurationMs'))}"),
                ("BPM／音符数", f"{_bpm(item)} ／ {format_number(item.get('notes'), 0)}"),
                ("音符密度", f"{format_number(item.get('nps'))} 音符／秒"),
                ("分数倍率／每分钟", f"{format_number(item.get('rate'), 4)} ／ {format_number(item.get('perMinute'), 4)}"),
                ("Free 技能覆盖率", _percent(item.get("skillCoverage"))),
                ("Just／Luck 期望", f"{format_number(item.get('justNotes'), 0)} ／ {format_number(item.get('luckPoints'), 1)}"),
            ]))
        return await render_detail(song["title"], f"歌曲 META · #{song['id']} · {mode_name}", await assets.image(song["image_key"]),
                                   sections, eyebrow="CHART METRICS", note=META_NOTE)
    difficulty = options.difficulty or "expert"
    ranking = []
    for song in songs:
        for chart in _charts(song, options):
            if chart["difficulty"] == difficulty:
                ranking.append((song, chart, metrics.get(chart["score_id"], {})))
    field = {"efficiency": "perMinute", "score": "rate", "duration": "durationMs", "nps": "nps", "coverage": "skillCoverage"}[options.sort]
    ascending = options.sort == "duration"
    ranking.sort(key=lambda row: (row[2].get(field) is None,
                                 (row[2].get(field) or 0) * (1 if ascending else -1), row[0]["id"]))
    selected, subtitle = _page(ranking, options.page)
    images = await asyncio.gather(*(assets.image(song["image_key"]) for song, _, _ in selected))
    rendered = []
    for index, ((song, chart, item), image) in enumerate(zip(selected, images), start=(options.page - 1) * PAGE_SIZE + 1):
        rendered.append({
            "title": f"{index:02d}  {song['title']}", "subtitle": f"{song['band']} · {DIFF_NAMES[chart['difficulty']]} Lv.{chart['display_level']:g}",
            "tag": f"#{song['id']}", "image": image,
            "detail": f"效率 {format_number(item.get('perMinute'), 4)}／分  ·  倍率 {format_number(item.get('rate'), 4)}\n"
                      f"{format_duration(item.get('durationMs'))}  ·  BPM {_bpm(item)}  ·  NPS {format_number(item.get('nps'))}  ·  覆盖 {_percent(item.get('skillCoverage'))}",
        })
    labels = {"efficiency": "每分钟效率", "score": "分数倍率", "duration": "音频时长（短→长）", "nps": "音符密度", "coverage": "技能覆盖率"}
    return await render_list("歌曲 META", f"{mode_name} · {labels[options.sort]}\n{subtitle}", rendered, note=META_NOTE)


def _clean_text(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", "", re.sub(r"<br\s*/?>", "\n", text, flags=re.I))).strip()


def _skill_description(template: str, effects: list[dict]) -> str:
    """只解释主表的数字占位与四则运算，绝不执行模板中的 Python 代码。"""
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](number(node.left), number(node.right))
        raise ValueError("不支持的技能表达式")

    def replace(match):
        expression, digits = match[1], int(match[2] or 1)

        def field(item):
            effect = effects[int(item[1])]
            return str(effect["effectValue"] if item[2] == "value" else effect["activationTimeSecond"])

        expression = re.sub(r"effects\[(\d+)\]\.(value|time)", field, expression)
        return f"{number(ast.parse(expression, mode='eval').body):.{digits}f}"

    return _clean_text(re.sub(r"\{([^{}:]+)(?::F(\d+))?\}", replace, template))


async def _card_skills(card: dict) -> list[tuple[str, str]]:
    (texts, _, _), live, live_effects, leaders, leader_effects, gekisou = await asyncio.gather(
        data.core(), assets.table("MasterLiveSkill"), assets.table("MasterLiveSkillEffect"),
        assets.table("MasterLeaderSkill"), assets.table("MasterLeaderSkillEffect"), assets.table("MasterGekisouSkill")
    )
    result = []
    for label, skill_id, definitions, effects, foreign_key in (
        ("演奏技能", card["live_skill_id"], live, live_effects, "liveSkillID"),
        ("队长技能", card["leader_skill_id"], leaders, leader_effects, "leaderSkillID"),
    ):
        definition = next((row for row in definitions if row["id"] == skill_id), None)
        rows = [row for row in effects if row[foreign_key] == skill_id]
        if definition is None or not rows:
            result.append((label, "主资料尚未提供技能说明"))
            continue
        level = max(row["level"] for row in rows)
        rows = [row for row in rows if row["level"] == level]
        name = data.text(texts, definition["nameTextID"])
        template = data.text(texts, definition["descriptionTextFormatID"])
        result.append((f"{label} · Lv.{level}", f"{name}\n{_skill_description(template, rows)}"))
    definition = next((row for row in gekisou if row["id"] == card["gekisou_skill_id"]), None)
    if definition is not None:
        mission = {1: "COMBO", 2: "LUCK", 3: "JUST"}.get(definition["gekisouMissionType"], "未提供")
        result.append(("激奏技能", f"{data.text(texts, definition['nameTextID'])} · {mission}激奏"))
    return result


async def query_card(query: str):
    options = parse_options(query)
    rows = await data.cards()
    rows = search(rows, options.query, limit=len(rows))
    if len(rows) != 1:
        return await _choices("卡牌查询", rows, options.page, kind="card")
    card = rows[0]
    image, skills = await asyncio.gather(assets.image(card["image_key"]), _card_skills(card))
    sections = [
        ("卡牌资料", [("角色／乐队", f"{card['character']} ／ {card['band']}"),
                      ("稀有度／属性", f"{card['rarity']} ／ {card['attribute']}"), ("公开时间", _date(card["start_at"]))]),
        ("满级基础能力", [("Performance", f"{card['powers'][0]:,}"), ("Technic", f"{card['powers'][1]:,}"),
                            ("Visual", f"{card['powers'][2]:,}"), ("综合力", f"{card['total_power']:,}")]),
        ("技能", skills),
    ]
    if card["voice"]:
        sections.append(("招募台词", [(card["character"], _clean_text(card["voice"]))]))
    return await render_detail(card["title"], f"卡牌 #{card['id']} · {card['character']}", image, sections,
                               background=image, eyebrow="MEMBER CARD")


async def query_card_art(query: str):
    options = parse_options(query)
    if not options.query:
        raise ReplyException("请输入要查询卡面的卡牌名称或 ID。")
    rows = search(await data.cards(), options.query)
    if len(rows) != 1:
        return await _choices("请选择卡牌 ID", rows, options.page, kind="card")
    card = rows[0]
    return await assets.image_path(card["image_key"])


async def query_event(query: str, *, list_mode: bool = False):
    options = parse_options(query)
    if list_mode:
        rows = await data.events()
        rows = search(rows, options.query, limit=len(rows))
    else:
        rows = await data.event(options.query)
    if list_mode or len(rows) != 1:
        return await _choices("活动列表", rows, options.page, kind="event")
    event = rows[0]
    hero = await assets.image(event["image_key"]) if event["image_key"] else None
    types = {1: "挑战演出活动"}
    sections = [("活动资讯", [("进度", event["status"]), ("类型", types.get(event["event_type"], f"类型 {event['event_type']}")),
                              ("开始", _date(event["start_at"])), ("结束", _date(event["end_at"])),
                              ("展示结束", _date(event["display_end_at"]))])]
    if event["music_id"]:
        sections.append(("活动歌曲", [(f"#{event['music_id']}", event["music_title"] or "未提供")]))
    if event["description"]:
        sections.append(("剧情简介", [(event["band"] or "故事", _clean_text(event["description"]))]))
    episodes = await data.episodes(event["chapter_id"]) if event["chapter_id"] else []
    sections.append(("活动剧情", [("章节", f"{len(episodes)} 话" if episodes else "暂无剧情")]))
    effects, (texts, characters, bands), cards, support_cards = await asyncio.gather(
        assets.table("MasterEventEffect"), data.core(), data.cards(), assets.table("MasterSupportCard")
    )
    card_map = {card["id"]: card for card in cards}
    support_map = {card["id"]: card for card in support_cards}
    bonuses = {}
    for effect in effects:
        if effect["eventId"] != event["id"]:
            continue
        target = ""
        specific = ""
        if effect.get("memberCardId"):
            card = card_map.get(effect["memberCardId"])
            target = f"卡牌 #{effect['memberCardId']}"
            specific = f"{card['character']} · {card['title']}" if card else ""
        elif effect.get("supportCardId"):
            card = support_map.get(effect["supportCardId"], {})
            target = f"卡牌 #{effect['supportCardId']}"
            specific = " · ".join(filter(None, (data.text(texts, card.get("nameTextID", "")),
                                                  data.text(texts, card.get("descriptionTextID", "")))))
        elif effect.get("characterId"):
            target = data.text(texts, characters.get(effect["characterId"], {}).get("nameTextID", ""))
        elif effect.get("bandId"):
            target = data.text(texts, bands.get(effect["bandId"], {}).get("nameTextID", ""))
        elif effect.get("cardType"):
            target = ATTRIBUTES.get(effect["cardType"], "") + "属性"
        if target:
            # 0/1 为能力值加成，2 为道具加成。
            kind = {2: "团员", 3: "支援"}.get(effect["resourceTypeConstraint"], "卡牌")
            label = {0: "能力值", 1: "能力值", 2: "活动道具"}.get(effect["eventBonusType"], f"加成{effect['eventBonusType']}")
            key = (f"{kind} · {target}", specific)
            value = f"{label} +{effect['rank1EffectValue']/100:g}%～{effect['rank5EffectValue']/100:g}%"
            bonuses.setdefault(key, []).append(value)
    if bonuses:
        for specific_only, title in ((True, "指定卡牌加成（Rank 1～5）"), (False, "活动加成（Rank 1～5）")):
            entries = [(target, (specific + "\n" if specific else "") + "／".join(dict.fromkeys(values)))
                       for (target, specific), values in bonuses.items() if bool(specific) == specific_only]
            if entries:
                sections.append((title, entries))
    return await render_detail(event["title"], f"活动 #{event['id']} · {event['status']} · UTC+8", hero, sections,
                               background=hero, eyebrow="EVENT")


async def query_comic(query: str = ""):
    options = parse_options(query)
    rows = await data.comics()
    rows = search(rows, options.query, limit=len(rows)) if options.query else rows
    if not rows:
        raise ReplyException("找不到符合条件的 HK/MO/TW 漫画。")
    comic = random.choice(rows)
    return await assets.image_path(comic["image_key"])


async def query_story(query: str, *, progress=None):
    from .story import summarize_event

    options = parse_options(query, story=True)
    rows = await data.event(options.query)
    if len(rows) != 1:
        return await _choices("请选择活动 ID", rows, options.page, kind="event")
    event = rows[0]
    summary = await summarize_event(event, refresh=options.refresh, progress=progress)
    hero = await assets.image(event["image_key"]) if event["image_key"] else None
    chapters = [("整体故事", summary["summary"]), *summary["chapters"]]
    subtitle = f"活动剧情 · #{event['id']}" + (" · 已公开章节" if summary["incomplete"] else "")
    return await render_story(event["title"], subtitle, chapters, hero=hero, background=hero)
