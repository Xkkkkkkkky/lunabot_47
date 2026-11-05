"""Moenotes 繁中主数据与素材的按需缓存。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import time
from pathlib import Path
from urllib.parse import quote

import aiohttp
import orjson
from PIL import Image

from ..utils import ReplyException, get_client_session, run_in_pool
from .common import DATA_DIR, config


class BdonSourceError(ReplyException):
    """资源读取错误。"""


class BdonAssets:
    def __init__(self):
        self.root = Path(DATA_DIR)
        self._locks: dict[str, asyncio.Lock] = {}
        self._json: dict[str, tuple[int, object]] = {}
        self._tables: dict[str, tuple[object, list[dict]]] = {}
        self._requests = asyncio.Semaphore(4)

    def configured_url(self, key: str) -> str:
        value = config.get(key, "", raise_exc=False).strip().rstrip("/")
        if not value.startswith(("https://", "http://")):
            raise BdonSourceError(f"BDON 尚未配置 {key}，请联系管理员。")
        return value

    async def _read(self, url: str, label: str) -> bytes:
        try:
            async with self._requests:
                async with get_client_session().get(
                    url, timeout=aiohttp.ClientTimeout(total=config.get("request_timeout", 45))
                ) as response:
                    if response.status != 200:
                        raise BdonSourceError(f"BDON {label}读取失败（HTTP {response.status}）。")
                    return await response.read()
        except BdonSourceError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise BdonSourceError(f"BDON {label}暂时无法连接，请稍后重试。") from exc

    async def cached_bytes(self, relative: str, url: str, max_age: float, label: str) -> bytes:
        """按有效期缓存，原子写入。"""
        async with self._locks.setdefault(relative, asyncio.Lock()):
            path = self.root / relative
            if path.is_file() and time.time() - path.stat().st_mtime < max_age:
                return await run_in_pool(path.read_bytes)
            content = await self._read(url, label)
            try:
                if path.suffix == ".json":
                    orjson.loads(content)
                else:
                    with Image.open(io.BytesIO(content)) as image:
                        image.verify()
            except (ValueError, OSError) as exc:
                raise BdonSourceError(f"BDON {label}格式无效，请稍后重试。") from exc

            def save():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)

            await run_in_pool(save)
            return content

    async def json(self, relative: str, url: str, max_age: float | None = None) -> object:
        max_age = config.get("cache_seconds", 1800) if max_age is None else max_age
        path = self.root / relative
        cached = self._json.get(relative)
        if cached is not None and path.is_file():
            stat = path.stat()
            if cached[0] == stat.st_mtime_ns and time.time() - stat.st_mtime < max_age:
                return cached[1]
        content = await self.cached_bytes(
            relative, url, max_age, "资料"
        )
        mtime = (self.root / relative).stat().st_mtime_ns
        cached = self._json.get(relative)
        if cached is None or cached[0] != mtime:
            self._json[relative] = (mtime, await run_in_pool(orjson.loads, content))
        return self._json[relative][1]

    async def table(self, name: str) -> list[dict]:
        if not re.fullmatch(r"Master[A-Za-z0-9]+", name):
            raise ValueError("无效主数据表名")
        raw = await self.json(f"masterdata/{name}.json", f"{self.configured_url('masterdata_url')}/{name}.json")
        if not isinstance(raw, dict) or not isinstance(raw.get("_allData"), list):
            raise BdonSourceError(f"BDON {name} 数据结构无效。")
        cached = self._tables.get(name)
        if cached is None or cached[0] is not raw:
            # 去除主表字段的前导下划线。
            rows = [{key.removeprefix("_"): value for key, value in row.items()} for row in raw["_allData"]]
            self._tables[name] = (raw, rows)
        return self._tables[name][1]

    def image_url(self, key: str) -> str:
        """Addressables 逻辑路径对应发布的完整 WebP；卡牌 atlas 名不代表裁剪。"""
        key = key.removesuffix(".png").strip("/")
        key = re.sub(r"^(MemberCard/\d+/[^/]+)_atlas$", r"\1", key)
        language = config.get("asset_language", "zh-Hant")
        if language != "zh-Hant":
            raise BdonSourceError("BDON 仅支持繁体中文素材，请设置 asset_language: zh-Hant。")
        parts = [language, *key.split("/"), key.rsplit("/", 1)[-1] + ".webp"]
        return self.configured_url("asset_url") + "/" + "/".join(quote(part, safe="") for part in parts)

    async def image_path(self, key: str) -> str:
        url = self.image_url(key)
        digest = hashlib.sha256(url.encode()).hexdigest()
        relative = f"assets/{digest}.webp"
        await self.cached_bytes(relative, url, 7 * 86400, "素材")
        return str(self.root / relative)

    async def image(self, key: str) -> Image.Image:
        path = await self.image_path(key)

        def decode():
            with Image.open(path) as image:
                return image.convert("RGBA")

        return await run_in_pool(decode)

    async def music_meta(self) -> dict:
        raw = await self.json("music_meta.json", self.configured_url("music_meta_url"))
        if not isinstance(raw, dict) or raw.get("format") != "nnnotes.music-data/2":
            raise BdonSourceError("BDON 歌曲统计格式无效。")
        # 上游将 HK/MO/TW 标记为 tw。
        if raw.get("provenance", {}).get("region") != "tw":
            raise BdonSourceError("BDON 歌曲统计来源不是 HK/MO/TW 服务器。")
        return raw


assets = BdonAssets()
