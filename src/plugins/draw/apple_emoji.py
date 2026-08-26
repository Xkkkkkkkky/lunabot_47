import base64
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image
from pilmoji.source import BaseSource


_ASSET_DIR = Path(__file__).with_name("assets") / "apple_emoji"
_INDEX_PATH = _ASSET_DIR / "emoji.json"
_SHEET_PATH = _ASSET_DIR / "sheet.png"
_EMOJI_SIZE = 32
_CELL_SIZE = _EMOJI_SIZE + 2


def _unified_to_text(unified: str) -> str:
    return "".join(chr(int(codepoint, 16)) for codepoint in unified.split("-"))


@lru_cache(maxsize=1)
def _emoji_index() -> dict[str, tuple[int, int]]:
    rows = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    index: dict[str, tuple[int, int]] = {}

    for row in rows:
        variants = (row, *row.get("skin_variations", {}).values())
        for variant in variants:
            if not variant.get("has_img_apple"):
                continue
            position = (variant["sheet_x"], variant["sheet_y"])
            index[_unified_to_text(variant["unified"])] = position
            if non_qualified := variant.get("non_qualified"):
                index[_unified_to_text(non_qualified)] = position
    return index


@lru_cache(maxsize=1)
def _emoji_sheet() -> Image.Image:
    sheet = Image.open(_SHEET_PATH).convert("RGBA")
    sheet.load()
    return sheet


@lru_cache(maxsize=512)
def get_apple_emoji_png(characters: str) -> bytes | None:
    position = _emoji_index().get(characters)
    if position is None:
        return None

    left = position[0] * _CELL_SIZE + 1
    top = position[1] * _CELL_SIZE + 1
    image = _emoji_sheet().crop(
        (left, top, left + _EMOJI_SIZE, top + _EMOJI_SIZE)
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def get_apple_emoji_data_uri(characters: str) -> str | None:
    data = get_apple_emoji_png(characters)
    if data is None:
        return None
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def get_apple_emoji_asset_signature() -> str:
    return ":".join(
        str(path.stat().st_mtime_ns) for path in (_INDEX_PATH, _SHEET_PATH)
    )


class LocalAppleEmojiSource(BaseSource):
    def get_emoji(self, emoji: str, /) -> BytesIO | None:
        data = get_apple_emoji_png(emoji)
        return BytesIO(data) if data is not None else None

    def get_discord_emoji(self, id: int, /) -> None:
        return None


LOCAL_APPLE_EMOJI_SOURCE = LocalAppleEmojiSource()
