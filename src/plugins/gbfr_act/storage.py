from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import BattleOptions, BattleRecord


class GbfrStorage:
    """
    GBFR ACT 本地持久化。

    raw 目录保存完整 WebSocket 事件流，logs 目录保存按战斗切分后的可重建记录。
    查询和渲染优先读取 logs，避免依赖 bot 当前内存状态。
    """

    def __init__(self, base_dir: str | Path = "data/gbfr"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.log_dir = self.base_dir / "logs"
        self.index_path = self.base_dir / "index.json"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def append_raw_event(self, event: dict[str, Any]) -> Path:
        self.ensure_dirs()
        time_ms = int(event.get("time_ms") or datetime.now().timestamp() * 1000)
        day = datetime.fromtimestamp(time_ms / 1000).strftime("%Y-%m-%d")
        path = self.raw_dir / f"{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        return path

    def save_battle(self, record: BattleRecord, keep_battles: int = 100) -> Path:
        self.ensure_dirs()
        file_name = f"{record.start_time.strftime('%Y%m%d_%H%M%S')}_{record.battle_id}.json"
        path = self.log_dir / file_name
        data = record.to_dict()
        data["source_path"] = str(path)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        self._update_index(path, record, keep_battles)
        return path

    def latest_battle(self, options: BattleOptions | None = None) -> BattleRecord | None:
        for item in reversed(self._read_index()):
            path = Path(item.get("path", ""))
            if path.is_file():
                return self.load_battle(path, options=options)
        paths = sorted(self.log_dir.glob("*.json"))
        if not paths:
            return None
        return self.load_battle(paths[-1], options=options)

    def load_battle(self, path: str | Path, options: BattleOptions | None = None) -> BattleRecord:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source_path"] = str(path)
        return BattleRecord.from_dict(data, options=options)

    def list_battles(self, limit: int = 20) -> list[dict[str, Any]]:
        items = self._read_index()
        return list(reversed(items[-limit:]))

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _update_index(self, path: Path, record: BattleRecord, keep_battles: int) -> None:
        items = [
            item
            for item in self._read_index()
            if item.get("battle_id") != record.battle_id and item.get("path") != str(path)
        ]
        items.append({
            "battle_id": record.battle_id,
            "path": str(path),
            "start_time_ms": record.start_time_ms,
            "end_time_ms": record.end_time_ms,
            "duration_seconds": record.duration_seconds,
            "total_damage": record.total_damage,
            "dps": record.dps,
            "finish_reason": record.finish_reason,
        })
        if keep_battles > 0:
            overflow = items[:-keep_battles]
            items = items[-keep_battles:]
            for item in overflow:
                old_path = Path(item.get("path", ""))
                if old_path.is_file():
                    try:
                        old_path.unlink()
                    except OSError:
                        pass
        self.index_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
