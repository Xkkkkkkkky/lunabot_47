from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..utils import *
from .render import render_attendance_heatmap
from .storage import TravelStorage


config = Config("travel")
logger = get_logger("Travel")
file_db = get_file_db("data/travel/db.json", logger)
cd = ColdDown(file_db, logger, cold_down_name="travel")
gwl = get_group_white_list(file_db, logger, "travel")
storage = TravelStorage(
    str(config.get("storage.path", "data/travel/travel.sqlite"))
)


ISO_ALPHA2_CODES = frozenset(
    """
    AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ
    BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ
    CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ
    DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR
    GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY
    HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT
    JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ
    LA LB LC LI LK LR LS LT LU LV LY
    MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ
    NA NC NE NF NG NI NL NO NP NR NU NZ OM
    PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA
    RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ
    TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ
    UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW
    """.split()
)
COUNTRY_QUERY_RE = re.compile(
    r"^\s*([a-zA-Z]{2})\s*(?:(\d+)\s*(?:个?\s*月)?)?\s*$"
)
MIN_FLAT_QUERY_MONTHS = 6


def _configured_timezone() -> ZoneInfo:
    timezone_name = str(config.get("timezone", "Asia/Shanghai")).strip()
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        raise ReplyException(
            f'travel.timezone 配置无效: "{timezone_name}"'
        )


def _normalize_country_code(raw_code: str) -> str:
    code = raw_code.strip().strip(",，").upper()
    aliases = config.get("country_aliases", {}, raise_exc=False) or {}
    if isinstance(aliases, dict):
        alias_target = aliases.get(code.lower(), aliases.get(code))
        if alias_target:
            code = str(alias_target).strip().upper()

    extra_codes = {
        str(item).strip().upper()
        for item in (
            config.get("extra_country_codes", [], raise_exc=False) or []
        )
    }
    if not re.fullmatch(r"[A-Z]{2}", code):
        raise ReplyException("国家缩写必须是两个英文字母，例如 JP、CN")
    if code not in ISO_ALPHA2_CODES and code not in extra_codes:
        raise ReplyException(
            f'不支持的国家缩写“{code}”，需使用 ISO 3166-1 两位代码'
        )
    return code


def _parse_checkin_args(args_text: str) -> str:
    tokens = args_text.strip().split()
    if len(tokens) != 1:
        raise ReplyException("使用方式: /打卡 国家缩写，例如 /打卡 JP")
    return _normalize_country_code(tokens[0])


def _parse_attendance_args(args_text: str) -> tuple[str, int]:
    match = COUNTRY_QUERY_RE.fullmatch(args_text)
    if match is None:
        raise ReplyException(
            "使用方式: /查出勤 国家缩写 [月数]，例如 /查出勤 JP 3"
        )
    country = _normalize_country_code(match.group(1))
    default_months = int(config.get("default_query_months", 1))
    max_months = int(config.get("max_query_months", 12))
    months = int(match.group(2) or default_months)
    if not 1 <= months <= max_months:
        raise ReplyException(f"查询月数必须在 1～{max_months} 之间")
    return country, months


def _subtract_months(day: date, months: int) -> date:
    month_index = day.year * 12 + day.month - 1 - months
    target_year, target_month_index = divmod(month_index, 12)
    target_month = target_month_index + 1
    target_day = min(
        day.day,
        calendar.monthrange(target_year, target_month)[1],
    )
    return date(target_year, target_month, target_day)


def _query_date_range(today: date, months: int) -> tuple[date, date]:
    # 左端点开区间、右端点闭区间：过去一个月在月末也保持直观。
    return _subtract_months(today, months) + timedelta(days=1), today


def _travel_cmd(commands: str | list[str]) -> CmdHandler:
    handler = CmdHandler(commands, logger)
    handler.check_group().check_wblist(gwl).check_cdrate(cd)
    return handler


checkin = _travel_cmd("/打卡")


@checkin.handle()
async def _(ctx: HandlerContext):
    country = _parse_checkin_args(ctx.get_args())
    checked_at = datetime.now(_configured_timezone())
    result = await storage.record_checkin(
        group_id=ctx.group_id,
        user_id=ctx.user_id,
        country=country,
        checkin_date=checked_at.date(),
        checked_at=checked_at,
    )
    if result.status == "created":
        message = f"{country} 打卡成功 · {checked_at:%Y-%m-%d}"
    elif result.status == "updated":
        message = (
            f"今日位置已从 {result.previous_country} 更新为 {country}"
            f" · {checked_at:%Y-%m-%d}"
        )
    else:
        message = f"今天已经在 {country} 打过卡了 · {checked_at:%Y-%m-%d}"
    await ctx.asend_reply_msg(message)


attendance = _travel_cmd("/查出勤")


@attendance.handle()
async def _(ctx: HandlerContext):
    country, months = _parse_attendance_args(ctx.get_args())
    today = datetime.now(_configured_timezone()).date()
    start_date, end_date = _query_date_range(today, months)
    await ctx.block(f"travel:attendance:{ctx.group_id}:{country}")
    stats = await storage.query_attendance(
        group_id=ctx.group_id,
        country=country,
        start_date=start_date,
        end_date=end_date,
    )
    flat_months = max(months, MIN_FLAT_QUERY_MONTHS)
    if flat_months == months:
        flat_stats = stats
    else:
        flat_start_date, flat_end_date = _query_date_range(
            today,
            flat_months,
        )
        flat_stats = await storage.query_attendance(
            group_id=ctx.group_id,
            country=country,
            start_date=flat_start_date,
            end_date=flat_end_date,
        )
    image = await render_attendance_heatmap(
        stats,
        flat_stats=flat_stats,
        months=months,
        group_id=ctx.group_id,
        config=config,
        logger=logger,
    )
    image_cq = await get_image_cq(
        image,
        logger=logger,
        low_quality=bool(config.get("draw.low_quality", False)),
    )
    await ctx.asend_reply_msg(image_cq)


@on_shutdown()
async def _close_travel_storage():
    await storage.close()
