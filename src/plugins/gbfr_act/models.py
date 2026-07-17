from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


EVENT_DAMAGE = "damage"
EVENT_LOAD_PARTY = "load_party"
EVENT_ENTER_AREA = "enter_area"
EVENT_INC_DEATH_CNT = "inc_death_cnt"

KNOWN_EVENT_TYPES = {
    EVENT_DAMAGE,
    EVENT_LOAD_PARTY,
    EVENT_ENTER_AREA,
    EVENT_INC_DEATH_CNT,
}

COMMON_ACTION_NAMES = {
    -1: "Link Attack",
    -2: "Skybound Art",
    -3: "Supplementary Damage",
    -0x100: "DoT",
    5000: "Ether Round",
    5010: "Charged Shot",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, str) and value.lower().startswith("0x"):
            return int(value, 16)
        return int(value)
    except Exception:
        return default


def _normalize_hex(value: Any, width: int = 8) -> str:
    return f"{_safe_int(value) & ((1 << (width * 4)) - 1):0{width}X}"


def _event_time_ms(event: Dict[str, Any]) -> int:
    return _safe_int(event.get("time_ms"), int(datetime.now().timestamp() * 1000))


def format_number(value: float | int) -> str:
    return f"{value:,.0f}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minute, sec = divmod(seconds, 60)
    hour, minute = divmod(minute, 60)
    if hour:
        return f"{hour:d}:{minute:02d}:{sec:02d}"
    return f"{minute:d}:{sec:02d}"


def action_display_name(action_id: int) -> str:
    return COMMON_ACTION_NAMES.get(action_id, f"Action {action_id}")


def event_action_id(data: Dict[str, Any]) -> int:
    flags = _safe_int(data.get("flags"))
    action_id = _safe_int(data.get("action_id"))
    if flags & (1 << 15):
        return -3
    return action_id


@dataclass
class BattleOptions:
    min_damage: int = 0
    ignore_non_party_sources: bool = True
    ignored_target_type_ids: set[int] = field(default_factory=set)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None) -> "BattleOptions":
        cfg = cfg or {}
        ignored = set()
        for value in cfg.get("ignored_target_type_ids", []) or []:
            ignored.add(_safe_int(value))
        return cls(
            min_damage=max(0, _safe_int(cfg.get("min_damage"), 0)),
            ignore_non_party_sources=bool(cfg.get("ignore_non_party_sources", True)),
            ignored_target_type_ids=ignored,
        )


@dataclass
class ActorRef:
    type_name: str = "UNKNOWN"
    idx: int = 0
    type_id: int = 0
    party_idx: int = -1

    @classmethod
    def from_raw(cls, raw: Any) -> "ActorRef":
        if isinstance(raw, ActorRef):
            return raw
        if not isinstance(raw, (list, tuple)):
            return cls()
        data = list(raw) + [None] * 4
        return cls(
            type_name=str(data[0] or "UNKNOWN").upper(),
            idx=_safe_int(data[1]),
            type_id=_safe_int(data[2]),
            party_idx=_safe_int(data[3], -1),
        )

    @property
    def key(self) -> str:
        if self.party_idx >= 0:
            return f"party:{self.party_idx}"
        return f"actor:{self.type_name}:{self.idx}:{self.type_id}"

    @property
    def short_name(self) -> str:
        suffix = f"#{self.idx}" if self.idx else ""
        return f"{self.type_name}{suffix}"

    def to_list(self) -> list[Any]:
        return [self.type_name, self.idx, self.type_id, self.party_idx]


@dataclass
class ActionStats:
    action_id: int
    hit: int = 0
    damage: int = 0
    min_damage: int = -1
    max_damage: int = -1

    @property
    def avg_damage(self) -> int:
        if self.hit <= 0:
            return 0
        return self.damage // self.hit

    @property
    def name(self) -> str:
        return action_display_name(self.action_id)

    def add_damage(self, damage: int) -> None:
        self.hit += 1
        self.damage += damage
        if self.min_damage < 0 or damage < self.min_damage:
            self.min_damage = damage
        if damage > self.max_damage:
            self.max_damage = damage

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "hit": self.hit,
            "damage": self.damage,
            "min_damage": self.min_damage,
            "max_damage": self.max_damage,
            "avg_damage": self.avg_damage,
        }


@dataclass
class TargetStats:
    ref: ActorRef
    damage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_list(),
            "name": self.ref.short_name,
            "damage": self.damage,
        }


@dataclass
class DamagePoint:
    time_ms: int
    actor_key: str
    damage: int
    action_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_ms": self.time_ms,
            "actor_key": self.actor_key,
            "damage": self.damage,
            "action_id": self.action_id,
        }


@dataclass
class ActorStats:
    ref: ActorRef
    damage: int = 0
    hit: int = 0
    death_cnt: int = 0
    member_info: dict[str, Any] | None = None
    actions: Dict[int, ActionStats] = field(default_factory=dict)
    targets: Dict[str, TargetStats] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.ref.key

    def display_name(self, show_username: bool = True) -> str:
        prefix = f"[{self.ref.party_idx}]" if self.ref.party_idx >= 0 else ""
        if self.member_info:
            c_name = str(self.member_info.get("c_name") or "").strip()
            d_name = str(self.member_info.get("d_name") or "").strip()
            if show_username and d_name:
                return f"{prefix}{d_name}({self.ref.type_name})"
            if c_name:
                return f"{prefix}{c_name}({self.ref.type_name})"
        return f"{prefix}{self.ref.short_name}"

    def add_damage(self, target: ActorRef, damage: int, action_id: int) -> None:
        self.hit += 1
        self.damage += damage
        if action_id not in self.actions:
            self.actions[action_id] = ActionStats(action_id=action_id)
        self.actions[action_id].add_damage(damage)
        if target.key not in self.targets:
            self.targets[target.key] = TargetStats(ref=target)
        self.targets[target.key].damage += damage

    def top_actions(self, limit: int = 5) -> list[ActionStats]:
        return sorted(self.actions.values(), key=lambda item: item.damage, reverse=True)[:limit]

    def to_dict(self, total_damage: int = 0, duration_seconds: float = 0) -> dict[str, Any]:
        share = self.damage / total_damage if total_damage > 0 else 0
        dps = self.damage / duration_seconds if duration_seconds > 0 else 0
        return {
            "ref": self.ref.to_list(),
            "key": self.key,
            "name": self.display_name(),
            "damage": self.damage,
            "share": share,
            "dps": dps,
            "hit": self.hit,
            "death_cnt": self.death_cnt,
            "member_info": self.member_info,
            "actions": [item.to_dict() for item in self.top_actions(50)],
            "targets": [item.to_dict() for item in sorted(self.targets.values(), key=lambda v: v.damage, reverse=True)],
        }


@dataclass
class BattleRecord:
    battle_id: str
    start_time_ms: int
    end_time_ms: int
    events: list[dict[str, Any]]
    actors: Dict[str, ActorStats]
    damage_points: list[DamagePoint]
    total_damage: int
    hit: int
    archived: bool = False
    finish_reason: str = ""
    source_path: str | None = None

    @classmethod
    def from_events(
        cls,
        events: Iterable[dict[str, Any]],
        battle_id: str | None = None,
        options: BattleOptions | None = None,
        archived: bool = False,
        finish_reason: str = "",
    ) -> "BattleRecord":
        options = options or BattleOptions()
        event_list = [dict(event) for event in events]
        if not event_list:
            now_ms = int(datetime.now().timestamp() * 1000)
            event_list = [{"time_ms": now_ms, "type": "empty"}]
        start_time_ms = min(_event_time_ms(event) for event in event_list)
        end_time_ms = start_time_ms
        actors: dict[str, ActorStats] = {}
        damage_points: list[DamagePoint] = []
        total_damage = 0
        hit = 0

        def get_actor(ref: ActorRef) -> ActorStats:
            if ref.key not in actors:
                actors[ref.key] = ActorStats(ref=ref)
            return actors[ref.key]

        for event in event_list:
            event_type = event.get("type")
            time_ms = _event_time_ms(event)
            data = event.get("data") or {}
            end_time_ms = max(end_time_ms, time_ms)

            if event_type == EVENT_LOAD_PARTY and isinstance(data, list):
                for member in data:
                    if not member:
                        continue
                    ref = ActorRef.from_raw(member.get("common_info"))
                    actor = get_actor(ref)
                    actor.member_info = member

            elif event_type == EVENT_INC_DEATH_CNT and isinstance(data, dict):
                ref = ActorRef.from_raw(data.get("actor"))
                actor = get_actor(ref)
                actor.death_cnt = max(actor.death_cnt, _safe_int(data.get("death_cnt")))

            elif event_type == EVENT_DAMAGE and isinstance(data, dict):
                damage = _safe_int(data.get("damage"))
                if damage <= options.min_damage:
                    continue
                source = ActorRef.from_raw(data.get("source"))
                target = ActorRef.from_raw(data.get("target"))
                if options.ignore_non_party_sources and source.party_idx < 0:
                    continue
                if target.type_id in options.ignored_target_type_ids:
                    continue
                action_id = event_action_id(data)
                actor = get_actor(source)
                actor.add_damage(target, damage, action_id)
                damage_points.append(DamagePoint(time_ms, actor.key, damage, action_id))
                total_damage += damage
                hit += 1

        if damage_points:
            end_time_ms = max(point.time_ms for point in damage_points)
        if battle_id is None:
            battle_id = make_battle_id(start_time_ms)
        return cls(
            battle_id=battle_id,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            events=event_list,
            actors=actors,
            damage_points=damage_points,
            total_damage=total_damage,
            hit=hit,
            archived=archived,
            finish_reason=finish_reason,
        )

    @property
    def duration_seconds(self) -> float:
        if self.end_time_ms <= self.start_time_ms:
            return 1.0
        return max(1.0, (self.end_time_ms - self.start_time_ms) / 1000)

    @property
    def start_time(self) -> datetime:
        return datetime.fromtimestamp(self.start_time_ms / 1000)

    @property
    def end_time(self) -> datetime:
        return datetime.fromtimestamp(self.end_time_ms / 1000)

    @property
    def dps(self) -> float:
        return self.total_damage / self.duration_seconds if self.duration_seconds > 0 else 0

    @property
    def is_meaningful(self) -> bool:
        return self.total_damage > 0 and bool(self.damage_points)

    def actor_rank(self, limit: Optional[int] = None) -> list[ActorStats]:
        actors = sorted(self.actors.values(), key=lambda item: item.damage, reverse=True)
        actors = [actor for actor in actors if actor.damage > 0]
        return actors if limit is None else actors[:limit]

    def to_dict(self) -> dict[str, Any]:
        duration = self.duration_seconds
        return {
            "battle_id": self.battle_id,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_seconds": duration,
            "archived": self.archived,
            "finish_reason": self.finish_reason,
            "total_damage": self.total_damage,
            "dps": self.dps,
            "hit": self.hit,
            "events": self.events,
            "damage_points": [point.to_dict() for point in self.damage_points],
            "actors": [
                actor.to_dict(self.total_damage, duration)
                for actor in self.actor_rank()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], options: BattleOptions | None = None) -> "BattleRecord":
        record = cls.from_events(
            data.get("events", []),
            battle_id=data.get("battle_id"),
            options=options,
            archived=bool(data.get("archived", True)),
            finish_reason=str(data.get("finish_reason") or ""),
        )
        record.source_path = data.get("source_path")
        return record


class BattleRecorder:
    def __init__(self, options: BattleOptions | None = None):
        self.options = options or BattleOptions()
        self.current_events: list[dict[str, Any]] = []
        self.current_id: str | None = None
        self.last_event_time_ms = 0
        self.last_damage_time_ms = 0

    @property
    def has_current(self) -> bool:
        return bool(self.current_events)

    def ingest(self, event: dict[str, Any]) -> BattleRecord | None:
        event_type = event.get("type")
        time_ms = _event_time_ms(event)
        self.last_event_time_ms = max(self.last_event_time_ms, time_ms)

        if event_type == EVENT_ENTER_AREA:
            return self.finish("enter_area")

        if event_type not in KNOWN_EVENT_TYPES:
            return None

        if not self.current_events:
            self.current_id = make_battle_id(time_ms)
        self.current_events.append(dict(event))
        if event_type == EVENT_DAMAGE:
            self.last_damage_time_ms = time_ms
        return None

    def finish_if_idle(self, now_ms: int, idle_ms: int) -> BattleRecord | None:
        if not self.current_events or self.last_damage_time_ms <= 0:
            return None
        if now_ms - self.last_damage_time_ms < idle_ms:
            return None
        return self.finish("idle")

    def finish(self, reason: str) -> BattleRecord | None:
        if not self.current_events:
            return None
        record = BattleRecord.from_events(
            self.current_events,
            battle_id=self.current_id,
            options=self.options,
            archived=True,
            finish_reason=reason,
        )
        self.current_events = []
        self.current_id = None
        self.last_damage_time_ms = 0
        if not record.is_meaningful:
            return None
        return record

    def snapshot(self) -> BattleRecord | None:
        if not self.current_events:
            return None
        record = BattleRecord.from_events(
            self.current_events,
            battle_id=self.current_id,
            options=self.options,
            archived=False,
            finish_reason="active",
        )
        return record if record.is_meaningful else None


def make_battle_id(time_ms: int | None = None) -> str:
    if time_ms is None:
        time_ms = int(datetime.now().timestamp() * 1000)
    dt = datetime.fromtimestamp(time_ms / 1000)
    return f"{dt.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def battle_summary_text(record: BattleRecord, show_username: bool = True, limit: int = 4) -> str:
    lines = [
        f"战斗时间: {record.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"持续: {format_duration(record.duration_seconds)}  总伤害: {format_number(record.total_damage)}  DPS: {format_number(record.dps)}",
    ]
    for idx, actor in enumerate(record.actor_rank(limit), start=1):
        share = actor.damage / record.total_damage * 100 if record.total_damage else 0
        lines.append(
            f"{idx}. {actor.display_name(show_username)} "
            f"{format_number(actor.damage)} ({share:.1f}%)"
        )
    return "\n".join(lines)
