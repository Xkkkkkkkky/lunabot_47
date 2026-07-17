"""鸟类规范名称与用户别名的持久化索引。"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..utils import (
    CmdHandler,
    ColdDown,
    Config,
    HandlerContext,
    ReplyException,
    dump_json,
    get_file_db,
    get_group_white_list,
    get_logger,
    load_json,
)


config = Config("bird")
logger = get_logger("BirdAlias")
file_db = get_file_db("data/bird/db.json", logger)
cd = ColdDown(file_db, logger)
gwl = get_group_white_list(file_db, logger, "bird")


@dataclass
class BirdAliasRecord:
    """一个以拉丁学名为唯一主键、可绕过 LLM 的鸟种记录。"""

    canonical_name: str
    name_zh: str = ""
    name_en: str = ""
    name_ja: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def scientific_name(self) -> str:
        return self.canonical_name

    def dump(self) -> dict:
        return {
            "aliases": sorted(set(self.aliases)),
            "name_zh": self.name_zh,
            "name_en": self.name_en,
            "name_ja": self.name_ja,
            "scientific_name": self.canonical_name,
        }


def _normalize_token(value: object, desc: str) -> str:
    token = " ".join(str(value or "").split()).strip()
    max_length = max(1, int(config.get("query.alias.max_name_length", 100)))
    if not token:
        raise ReplyException(f"{desc}不能为空")
    if len(token) > max_length:
        raise ReplyException(f"{desc}不能超过{max_length}个字符")
    if any(ord(char) < 32 for char in token):
        raise ReplyException(f"{desc}不能包含控制字符")
    return token


class BirdAliasManager:
    """仿照 NSY aliases 总表维护鸟种名称与别名的双向索引。"""

    _manager: Optional["BirdAliasManager"] = None

    def __init__(self, alias_file: Optional[Path] = None):
        if alias_file is None:
            configured = str(config.get("query.alias.file", "")).strip()
            if not configured:
                raise RuntimeError("query.alias.file 配置为空")
            alias_file = Path(configured)
        self.alias_file = alias_file
        self.records: dict[str, BirdAliasRecord] = {}
        self.bird_by_alias: dict[str, str] = {}
        self._ensure_file()
        self._load()

    @classmethod
    def get(cls) -> "BirdAliasManager":
        if cls._manager is None:
            cls._manager = cls()
        return cls._manager

    def _ensure_file(self) -> None:
        if not self.alias_file.exists():
            dump_json({}, str(self.alias_file))

    def _load(self) -> None:
        """加载学名主表并迁移旧的中文主键结构。"""

        payload = load_json(str(self.alias_file))
        if not isinstance(payload, dict):
            raise RuntimeError("鸟类 aliases.json 顶层必须是对象")

        pending_records: dict[str, BirdAliasRecord] = {}
        migrated = False
        for raw_name, raw_record in payload.items():
            try:
                if not isinstance(raw_record, dict):
                    raise ValueError("鸟类记录不是对象")
                old_key = _normalize_token(raw_name, "鸟类名称")
                scientific_name = _normalize_token(
                    raw_record.get("scientific_name") or old_key,
                    "鸟类学名",
                )
                if scientific_name != old_key:
                    migrated = True

                raw_aliases = raw_record.get("aliases", [])
                if not isinstance(raw_aliases, list):
                    raise ValueError("aliases 不是数组")
                record = pending_records.get(scientific_name)
                if record is None:
                    record = BirdAliasRecord(canonical_name=scientific_name)
                    pending_records[scientific_name] = record

                name_zh = " ".join(str(raw_record.get("name_zh", "")).split())
                if not name_zh and old_key != scientific_name:
                    # 旧格式以解析出的中文名为键，迁移时保留为中文名与 alias。
                    name_zh = old_key
                for field_name, value in (
                    ("name_zh", name_zh),
                    ("name_en", raw_record.get("name_en", "")),
                    ("name_ja", raw_record.get("name_ja", "")),
                ):
                    normalized = " ".join(str(value or "").split()).strip()
                    if normalized and not getattr(record, field_name):
                        setattr(record, field_name, normalized)

                alias_values = [*raw_aliases, old_key, name_zh, record.name_en, record.name_ja]
                for raw_alias in alias_values:
                    if not raw_alias:
                        continue
                    alias = _normalize_token(raw_alias, "鸟类别名")
                    if alias != scientific_name and alias not in record.aliases:
                        record.aliases.append(alias)
            except Exception as exc:
                logger.warning(f'加载鸟类别名记录"{raw_name}"失败: {exc}')

        self.records = pending_records
        self.bird_by_alias = {}
        canonical_names = set(self.records)
        for scientific_name, record in self.records.items():
            valid_aliases = []
            for alias in record.aliases:
                if alias in canonical_names:
                    logger.warning(f'忽略与学名主键冲突的鸟类别名"{alias}"')
                    continue
                owner = self.bird_by_alias.get(alias)
                if owner is not None and owner != scientific_name:
                    logger.warning(
                        f'忽略重复鸟类别名"{alias}"，已属于"{owner}"'
                    )
                    continue
                self.bird_by_alias[alias] = scientific_name
                valid_aliases.append(alias)
            record.aliases = valid_aliases

        if migrated:
            self._save()
            logger.info("已将旧鸟类别名索引迁移为学名主键结构")

        logger.info(
            f"成功加载鸟类别名索引: {len(self.records)}个鸟种, "
            f"{len(self.bird_by_alias)}个别名"
        )

    def _save(self) -> None:
        payload = {
            name: record.dump()
            for name, record in sorted(self.records.items())
        }
        dump_json(payload, str(self.alias_file))

    def resolve(self, name_or_alias: str) -> Optional[BirdAliasRecord]:
        """将已保存的规范名称或别名解析为鸟种记录，不执行模糊匹配。"""

        token = " ".join(str(name_or_alias or "").split()).strip()
        canonical_name = token if token in self.records else self.bird_by_alias.get(token)
        return self.records.get(canonical_name) if canonical_name else None

    def resolve_required(self, name_or_alias: str) -> BirdAliasRecord:
        record = self.resolve(name_or_alias)
        if record is None:
            token = " ".join(str(name_or_alias or "").split()).strip()
            raise ReplyException(
                f'鸟类"{token}"尚未保存，请先使用 /查鸟 {token} 成功查询一次'
            )
        return record

    def remember(
        self,
        scientific_name: str,
        *,
        name_zh: str = "",
        name_en: str = "",
        name_ja: str = "",
    ) -> BirdAliasRecord:
        """登记学名，并自动把解析出的中英日名称加入该学名的 alias。"""

        scientific_name = _normalize_token(scientific_name, "鸟类学名")
        owner = self.bird_by_alias.get(scientific_name)
        if owner and owner != scientific_name:
            raise ReplyException(
                f'鸟类学名"{scientific_name}"已被用作"{owner}"的别名'
            )
        record = self.records.get(scientific_name)
        created = record is None
        if record is None:
            record = BirdAliasRecord(canonical_name=scientific_name)
            self.records[scientific_name] = record

        changed = False
        for field_name, value in (
            ("name_zh", name_zh),
            ("name_en", name_en),
            ("name_ja", name_ja),
        ):
            normalized = " ".join(str(value or "").split()).strip()
            if normalized and getattr(record, field_name) != normalized:
                setattr(record, field_name, normalized)
                changed = True

            if not normalized or normalized == scientific_name:
                continue
            alias_owner = self.bird_by_alias.get(normalized)
            if normalized in self.records and normalized != scientific_name:
                logger.warning(f'自动添加鸟类别名"{normalized}"与其他学名冲突，已跳过')
            elif alias_owner is not None and alias_owner != scientific_name:
                logger.warning(
                    f'自动添加鸟类别名"{normalized}"已属于"{alias_owner}"，已跳过'
                )
            elif normalized not in record.aliases:
                record.aliases.append(normalized)
                self.bird_by_alias[normalized] = scientific_name
                changed = True
        if created or changed:
            self._save()
        return record

    def get_aliases(self, name_or_alias: str) -> tuple[str, list[str]]:
        record = self.resolve_required(name_or_alias)
        return record.canonical_name, sorted(record.aliases)

    def add_alias(self, name_or_alias: str, alias: str) -> tuple[str, str]:
        record = self.resolve_required(name_or_alias)
        alias = _normalize_token(alias, "鸟类别名")
        if alias in self.records:
            raise ReplyException(f'鸟类别名"{alias}"已是规范鸟类名称')
        if alias in self.bird_by_alias:
            raise ReplyException(f'鸟类别名"{alias}"已存在')
        record.aliases.append(alias)
        self.bird_by_alias[alias] = record.canonical_name
        self._save()
        return record.canonical_name, alias

    def delete_alias(self, name_or_alias: str, alias: str) -> tuple[str, str]:
        record = self.resolve_required(name_or_alias)
        alias = _normalize_token(alias, "鸟类别名")
        if self.bird_by_alias.get(alias) != record.canonical_name:
            raise ReplyException(f'鸟类"{record.canonical_name}"没有别名"{alias}"')
        record.aliases.remove(alias)
        del self.bird_by_alias[alias]
        self._save()
        return record.canonical_name, alias


def _parse_one_arg(ctx: HandlerContext, usage: str) -> str:
    value = ctx.get_args().strip()
    if not value:
        raise ReplyException(f"使用方式: {usage}")
    return value


def _parse_two_args(ctx: HandlerContext, usage: str) -> tuple[str, str]:
    try:
        args = shlex.split(ctx.get_args().strip())
    except ValueError as exc:
        raise ReplyException(f"参数引号不完整，使用方式: {usage}") from exc
    if len(args) != 2:
        raise ReplyException(f"使用方式: {usage}")
    return args[0], args[1]


def _alias_command(commands: list[str], *, superuser: bool = False) -> CmdHandler:
    handler = CmdHandler(commands, logger, priority=2)
    handler.check_cdrate(cd).check_wblist(gwl)
    if superuser:
        handler.check_superuser()
    return handler


bird_alias = _alias_command(["/bird alias", "/鸟别名"])


@bird_alias.handle()
async def handle_bird_alias(ctx: HandlerContext):
    name = _parse_one_arg(ctx, "/bird alias 鸟类名称/别名")
    canonical_name, aliases = BirdAliasManager.get().get_aliases(name)
    if not aliases:
        return await ctx.asend_reply_msg(f'鸟类"{canonical_name}"还没有别名')
    return await ctx.asend_reply_msg(
        f'鸟类"{canonical_name}"的别名: ' + "，".join(aliases)
    )


bird_add_alias = _alias_command(
    ["/bird add alias", "/添加鸟别名"]
)


@bird_add_alias.handle()
async def handle_bird_add_alias(ctx: HandlerContext):
    name, alias = _parse_two_args(ctx, '/bird add alias 鸟类名称/别名 "新别名"')
    canonical_name, alias = BirdAliasManager.get().add_alias(name, alias)
    return await ctx.asend_reply_msg(
        f'鸟类"{canonical_name}"添加别名"{alias}"成功'
    )


bird_delete_alias = _alias_command(
    ["/bird del alias", "/删除鸟别名"],
    superuser=True,
)


@bird_delete_alias.handle()
async def handle_bird_delete_alias(ctx: HandlerContext):
    name, alias = _parse_two_args(ctx, '/bird del alias 鸟类名称/别名 "别名"')
    canonical_name, alias = BirdAliasManager.get().delete_alias(name, alias)
    return await ctx.asend_reply_msg(
        f'鸟类"{canonical_name}"删除别名"{alias}"成功'
    )
