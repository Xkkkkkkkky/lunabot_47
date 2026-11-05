"""只读导出已保存的群消息，不访问媒体资源或机器人接口。"""

from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import orjson


BEIJING_TZ = ZoneInfo('Asia/Shanghai')
EXPORT_USAGE = '用法：/record [@成员或QQ号] [YYYY-MM-DD] [YYYY-MM-DD]；一次只能指定一名成员，结束日期不能早于开始日期'


def parse_export_args(args: str, at_user_ids: list[int]) -> tuple[datetime | None, datetime | None, int | None]:
    """解析指定成员和日期，把包含结束日的范围转为北京时间半开区间。"""
    try:
        members = {int(user_id) for user_id in at_user_ids}
        dates = []
        for part in args.split():
            if re.fullmatch(r'[0-9]+', part):
                members.add(int(part))
            elif re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', part):
                dates.append(datetime.strptime(part, '%Y-%m-%d').replace(tzinfo=BEIJING_TZ))
            else:
                raise ValueError
        if len(members) > 1 or any(not 0 < user_id <= 2**63 - 1 for user_id in members) or len(dates) > 2:
            raise ValueError
        user_id = next(iter(members), None)
        if not dates:
            return None, None, user_id
        start, last = dates[0], dates[-1]
        if last < start:
            raise ValueError
        return start, last + timedelta(days=1), user_id
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError(EXPORT_USAGE) from exc


def _readable_content(content: str) -> str:
    segments = orjson.loads(content)
    if not isinstance(segments, list):
        raise ValueError('聊天记录消息内容必须是消息段数组')
    result = []
    placeholders = {
        'image': '图片', 'record': '语音', 'audio': '语音', 'video': '视频',
        'forward': '合并转发', 'json': 'JSON卡片',
    }
    for segment in segments:
        kind, data = segment['type'], segment['data']
        if kind == 'text':
            result.append(data['text'])
        elif kind == 'at':
            result.append('@全体成员' if str(data['qq']) == 'all' else f"@{data['qq']}")
        elif kind == 'reply':
            result.append(f"[引用消息ID：{data['id']}]")
        elif kind == 'face':
            result.append(f"[表情ID：{data['id']}]")
        elif kind == 'file':
            result.append(f"[文件：{data['name']}]" if data.get('name') else '[文件]')
        else:
            result.append(f"[{placeholders.get(kind, '其他消息类型：' + kind)}]")
    return ''.join(result)


def export_chat_history(
    db_path: str, group_id: int, file_path: str,
    start_time: datetime | None = None, end_time: datetime | None = None,
    user_id: int | None = None,
) -> int:
    """逐条读取指定群的已保存消息并写入 TXT，返回导出条数。"""
    group_id = int(group_id)
    if group_id <= 0:
        raise ValueError('群号必须为正整数')
    if user_id is not None:
        user_id = int(user_id)
        if not 0 < user_id <= 2**63 - 1:
            raise ValueError('成员QQ号必须为有效的正整数')
    database = Path(db_path).resolve()
    if not database.exists():
        return 0
    table = f'msg_{group_id}'
    # 只读连接不会创建数据库；表名仅由已校验的整数构造。
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as conn:
        if conn.execute('SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?', ('table', table)).fetchone() is None:
            return 0
        conditions, params = [], []
        if start_time is not None:
            conditions.append('time >= ?')
            params.append(start_time.timestamp())
        if end_time is not None:
            conditions.append('time < ?')
            params.append(end_time.timestamp())
        if user_id is not None:
            conditions.append('user_id = ?')
            params.append(user_id)
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        query = f'SELECT time, nickname, user_id, msg_id, content FROM {table}{where} ORDER BY time ASC, id ASC'
        count = 0
        with open(file_path, 'w', encoding='utf-8-sig') as output:
            time_range = '全部记录'
            if start_time is not None or end_time is not None:
                start = start_time.astimezone(BEIJING_TZ).isoformat() if start_time else '不限'
                end = end_time.astimezone(BEIJING_TZ).isoformat() if end_time else '不限'
                time_range = f'{start} 至 {end}（不含结束时刻）'
            member_label = '全体成员' if user_id is None else str(user_id)
            output.write(f'群号：{group_id}\n成员：{member_label}\n范围：{time_range}\n时间为北京时间（Asia/Shanghai）\n仅导出已保存记录；媒体仅占位，不含媒体文件；合并转发不展开。\n\n')
            with closing(conn.execute(query, params)) as rows:
                for timestamp, nickname, user_id, msg_id, content in rows:
                    time = datetime.fromtimestamp(timestamp, BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
                    output.write(f'时间：{time}\n昵称：{nickname}\n用户ID：{user_id}\n消息ID：{msg_id}\n{_readable_content(content)}\n\n')
                    count += 1
        return count
