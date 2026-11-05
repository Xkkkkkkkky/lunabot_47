"""只使用 hk-tw-mo 主表，保留游戏原有 ID 与繁中字段。"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

from rapidfuzz import fuzz
from zhconv import convert

from ..common.config import Config
from .asset import assets


TW_TZ = timezone(timedelta(hours=8))
DIFFICULTIES = ("easy", "normal", "hard", "expert")
DIFF_NAMES = {"easy": "EASY", "normal": "NORMAL", "hard": "HARD", "expert": "EXPERT"}
ATTRIBUTES = {1: "红", 2: "蓝", 3: "绿", 4: "黄", 5: "紫"}
RARITIES = {2: "R", 3: "SR", 4: "SSR", 20: "BD"}
ALIAS_CONFIGS = {kind: Config(f"bdon.{kind}_aliases") for kind in ("character", "band", "music")}


def _get_aliases(kind: str) -> dict[int, list[str]]:
    """复用配置热重载；别名绑定主表 ID，不依赖显示语言。"""
    return {item["id"]: item["aliases"] for item in ALIAS_CONFIGS[kind].get("aliases", [])}


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    # 主表为 UTC+8，小时可能不补零。
    return datetime.strptime(value.strip().replace("/", "-"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TW_TZ)


def released(row: dict, now: datetime | None = None) -> bool:
    start = parse_date(row.get("start_at", ""))
    return start is None or start <= (now or datetime.now(TW_TZ))


def normalize_query(value: str) -> str:
    return re.sub(r"[\s・·!！?？～~\-—_\[\]【】“”‘’]+", "", convert(value, "zh-hant").casefold())


def search(rows: list[dict], query: str, *, limit: int | None = None) -> list[dict]:
    """精确 ID/名称优先，其次关键词与模糊匹配；返回多项由调用者展示选择。"""
    query = query.strip()
    match = re.fullmatch(r"[mceMCE]?(\d+)", query)
    if match:
        return [row for row in rows if row["id"] == int(match[1])]
    needle = normalize_query(query)
    if not needle:
        minimum = datetime.min.replace(tzinfo=TW_TZ)
        return sorted(rows, key=lambda row: (parse_date(row.get("start_at", "")) or minimum, row["id"]), reverse=True)[:limit]
    # 歌曲别名与完整曲名同等精确匹配，避免短别名被其他条目的关键词淹没。
    exact = [row for row in rows if any(normalize_query(name) == needle
                                      for name in (row.get("title", ""), *row.get("aliases", [])))]
    if exact:
        return exact[:limit]
    parts = [normalize_query(part) for part in query.split()]
    contains = [row for row in rows if all(part in normalize_query(row["search_text"]) for part in parts)]
    if contains:
        return sorted(contains, key=lambda row: row["id"], reverse=True)[:limit]
    scored = [(fuzz.WRatio(needle, normalize_query(row["search_text"])), row) for row in rows]
    scored = sorted((item for item in scored if item[0] >= 65), key=lambda item: (-item[0], -item[1]["id"]))
    return [row for score, row in scored[:limit]]


class BdonData:
    async def core(self):
        texts, characters, bands = await asyncio.gather(
            assets.table("MasterText"), assets.table("MasterCharacter"), assets.table("MasterBand")
        )
        text_map = {row["id"]: row for row in texts}
        return text_map, {row["id"]: row for row in characters}, {row["id"]: row for row in bands}

    @staticmethod
    def text(text_map: dict, key: str) -> str:
        value = text_map.get(key, {}).get("traditionalChinese", "")
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def search_names(text_map: dict, *keys: str) -> str:
        return " ".join(str(value) for key in keys for value in text_map.get(key, {}).values())

    async def music(self) -> list[dict]:
        (texts, characters, bands), music, scores = await asyncio.gather(
            self.core(), assets.table("MasterLiveMusic"), assets.table("MasterLiveMusicScore")
        )
        text = lambda key: self.text(texts, key)
        score_map = {row["id"]: row for row in scores}
        character_aliases = _get_aliases("character")
        band_aliases = _get_aliases("band")
        music_aliases = _get_aliases("music")
        result = []
        for song in music:
            band_id = next(iter(song["bandIDs"]), 0)
            band = text(song.get("bandNameTextID") or bands.get(band_id, {}).get("nameTextID", ""))
            title = text(song["titleTextID"])
            vocalists = [text(characters.get(cid, {}).get("nameTextID", "")) for cid in song["vocalCharacterIDs"]]
            charts = []
            for difficulty in DIFFICULTIES:
                score = score_map.get(song[difficulty + "ID"])
                if score:
                    charts.append({
                        "difficulty": difficulty, "score_id": score["id"],
                        "level": score["musicScoreLevel"], "display_level": score["musicScoreDisplayLevel"],
                        "notes": score["fullComboCount"],
                    })
            composer, lyricist, arranger = (text(song[field]) for field in ("composerTextID", "lyricistTextID", "arrangerTextID"))
            aliases = music_aliases.get(song["id"], [])
            result.append({
                "id": song["id"], "title": title or f"歌曲 #{song['id']}", "band": band,
                "aliases": aliases,
                "band_id": band_id, "attribute": ATTRIBUTES.get(song["musicType"], "未提供"),
                "composer": composer, "lyricist": lyricist, "arranger": arranger,
                "vocalists": "、".join(vocalists), "start_at": song["startAt"], "charts": charts,
                "image_key": f"Image/Jacket/{song['jacketAssetName']}",
                "search_text": " ".join([str(song["id"]), title, band, composer, lyricist, arranger, *vocalists,
                                           self.search_names(texts, song["titleTextID"]), *aliases,
                                           *(text(bands.get(bid, {}).get("nameTextID", "")) for bid in song["bandIDs"]),
                                           *(alias for bid in song["bandIDs"] for alias in band_aliases.get(bid, [])),
                                           *(alias for cid in song["vocalCharacterIDs"] for alias in character_aliases.get(cid, []))]),
            })
        return result

    async def cards(self) -> list[dict]:
        (texts, characters, bands), cards = await asyncio.gather(self.core(), assets.table("MasterMemberCard"))
        text = lambda key: self.text(texts, key)
        character_aliases = _get_aliases("character")
        band_aliases = _get_aliases("band")
        result = []
        for card in cards:
            character_row = characters.get(card["characterID"], {})
            band_id = character_row.get("bandID", 0)
            character = text(card["nameTextID"])
            band = text(bands.get(band_id, {}).get("nameTextID", ""))
            title = text(card["subtitleTextID"])
            powers = [card[field] for field in ("performancePowerMax", "technicPowerMax", "visualPowerMax")]
            result.append({
                "id": card["id"], "asset_id": card["assetID"], "title": title or f"卡牌 #{card['id']}",
                "character": character, "character_id": card["characterID"], "band": band, "band_id": band_id,
                "rarity": RARITIES[card["rarity"]], "attribute": ATTRIBUTES[card["cardType"]],
                "powers": powers, "total_power": sum(powers), "voice": text(card.get("gachaVoiceTextId", "")),
                "live_skill_id": card["liveSkillID"], "leader_skill_id": card["leaderSkillID"],
                "gekisou_skill_id": card["gekisouSkillID"], "start_at": card["startAt"],
                "image_key": f"MemberCard/{card['assetID']}/member_full",
                "thumbnail_key": f"MemberCard/{card['assetID']}/member_thumbnail",
                "search_text": " ".join([str(card["id"]), title, character, band, RARITIES[card["rarity"]],
                                          ATTRIBUTES[card["cardType"]], self.search_names(texts, card["subtitleTextID"], card["nameTextID"]),
                                          *character_aliases.get(card["characterID"], []), *band_aliases.get(band_id, [])]),
            })
        return result

    async def events(self) -> list[dict]:
        (texts, characters, bands), events, chapters, music = await asyncio.gather(
            self.core(), assets.table("MasterEvent"), assets.table("MasterStoryChapter"), assets.table("MasterLiveMusic")
        )
        text = lambda key: self.text(texts, key)
        chapter_map = {row["id"]: row for row in chapters}
        song_map = {row["id"]: row for row in music}
        band_aliases = _get_aliases("band")
        music_aliases = _get_aliases("music")
        result = []
        now = datetime.now(TW_TZ)
        for event in events:
            chapter = chapter_map.get(event["storyChapterId"], {})
            band_id = chapter.get("bandId", 0)
            band = text(bands.get(band_id, {}).get("nameTextID", ""))
            music_title = text(song_map.get(event["musicId"], {}).get("titleTextID", ""))
            title = text(event["nameTextId"])
            start, end = parse_date(event["startAt"]), parse_date(event["endAt"])
            status = "即将开始" if start and start > now else "进行中" if end and now <= end else "已结束"
            background = f"Image/Event/{event['backgroundAsset']}" if event.get("backgroundAsset") else ""
            banner = (f"Story/Banner/Chapter/{chapter['banner']}" if chapter.get("banner")
                      else f"Image/Event/{event['bannerAsset']}" if event.get("bannerAsset") else background)
            result.append({
                "id": event["id"], "title": title or f"活动 #{event['id']}", "chapter_id": event["storyChapterId"],
                "description": text(chapter.get("descriptionTextId", "")),
                "band": band, "band_id": band_id,
                "start_at": event["startAt"], "end_at": event["endAt"], "display_end_at": event["displayEndAt"],
                "status": status, "event_type": event["eventType"],
                "music_id": event["musicId"], "music_title": music_title,
                "image_key": banner,
                "search_text": " ".join([str(event["id"]), title, text(chapter.get("nameTextId", "")), band, music_title,
                                          self.search_names(texts, event["nameTextId"]),
                                          *band_aliases.get(band_id, []), *music_aliases.get(event["musicId"], [])]),
            })
        return result

    async def event(self, query: str) -> list[dict]:
        rows = await self.events()
        if query.strip() in ("", "当前", "當前", "本期", "最新"):
            current = [row for row in rows if row["status"] == "进行中"]
            return current or sorted([row for row in rows if released(row)], key=lambda row: parse_date(row["start_at"]), reverse=True)[:1]
        if query.strip() in ("上期", "上次"):
            past = [row for row in rows if row["status"] == "已结束"]
            return sorted(past, key=lambda row: parse_date(row["start_at"]), reverse=True)[:1]
        return search(rows, query)

    async def episodes(self, chapter_id: int) -> list[dict]:
        (texts, _, _), episodes, advs = await asyncio.gather(self.core(), assets.table("MasterStoryEpisode"), assets.table("MasterAdv"))
        adv_map = {row["id"]: row for row in advs}
        result = []
        for episode in episodes:
            if episode["chapterId"] != chapter_id:
                continue
            adv = adv_map.get(episode["advId"], {})
            result.append({
                "id": episode["id"], "adv_id": episode["advId"], "number": episode["episodeNumber"],
                "title": self.text(texts, adv.get("titleTextId", "")) or f"第 {episode['episodeNumber']} 话",
                "description": self.text(texts, episode.get("descriptionTextId", "")),
                "another": episode["isAnotherEpisode"], "extra": episode["isExtraEpisode"],
                "start_at": episode.get("startAt", ""),
            })
        return sorted(result, key=lambda row: (row["another"], row["extra"], row["number"], row["id"]))

    async def comics(self) -> list[dict]:
        (texts, characters, _), comics = await asyncio.gather(self.core(), assets.table("MasterLoadingComics"))
        character_aliases = _get_aliases("character")
        now = datetime.now(TW_TZ)
        result = []
        for comic in comics:
            if not released({"start_at": comic["startAt"]}, now):
                continue
            end = parse_date(comic.get("endAt", ""))
            if end and end < now:
                continue
            names = "、".join(self.text(texts, characters.get(cid, {}).get("nameTextID", "")) for cid in comic["characterIds"])
            result.append({"id": comic["id"], "title": f"漫画 #{comic['id']}", "character": names,
                           "image_key": f"Image/Comic/{comic['imageAsset']}",
                           "search_text": " ".join([names, *(alias for cid in comic["characterIds"]
                                                             for alias in character_aliases.get(cid, []))])})
        return result


data = BdonData()
