"""缓存谱面指标的纯计算；仅支持 nnnotes.music-data/2 的当前统计结构。"""

from math import floor, isfinite


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) else None


def _positive(value):
    value = _number(value)
    return value if value is not None and value > 0 else None


def _center(value):
    return _number(value[0]) if value else None


def _kind_weights(weights, kind):
    if weights is None or kind is None or kind >= len(weights):
        return None
    return weights[kind]


def _figures(deck, kind, power, mode):
    if not deck or kind is None or power is None:
        return None
    if mode == 'gekisou':
        if deck.get('unplayable'):
            return None
        expectation = deck.get('expectation') or {}
        score = _center(expectation.get('score'))
        raw = _kind_weights(expectation.get('weights'), kind)
        weights = [_center(value) for value in raw] if raw is not None else None
        base = _number(score / power) if score is not None else None
    else:
        seeds = deck.get('offSeeds')
        if not seeds:
            return None
        scores = [_number(seed.get('score')) for seed in seeds]
        rows = [_kind_weights(seed.get('weights'), kind) for seed in seeds]
        if any(score is None for score in scores) or any(row is None or len(row) != 5 for row in rows):
            return None
        if any(_number(value) is None for row in rows for value in row):
            return None
        base = _number(sum(score / power / len(seeds) for score in scores))
        weights = [_number(sum(row[i] / len(rows) for row in rows)) for i in range(5)]
    if base is None or weights is None or len(weights) != 5 or any(value is None for value in weights):
        return None
    total = _number(sum(weights))
    rate = _number(base + total) if total is not None else None
    return (base, total, rate) if rate is not None else None


def chart_metrics(data: dict, song: dict, chart: dict, *, mode: str = 'gekisou') -> dict:
    """固定五个全队 +100%／5秒技能，激奏第1名；区间取中心值。

    时长为毫秒，密度为音符/秒，分数为每单位综合力估计值。
    """
    if mode not in ('gekisou', 'free'):
        raise ValueError('mode 必须为 gekisou 或 free')
    if data['format'] != 'nnnotes.music-data/2':
        raise ValueError('不支持的 music-data 格式')
    model = data['deck']
    power = _positive(model['model']['power'])
    kind = next((item['id'] for item in model['kinds']
                 if item.get('effectType') == 2000 and item.get('durationMs') == 5000
                 and not any(item.get(key) for key in (
                     'skillTargetIds', 'skillConditionGroup', 'skillReleaseConditionGroup',
                     'effectLimitCount', 'effectExecuteLimitCount'))), None)
    deck = chart.get('deck')
    free = _figures(deck, kind, power, 'free')
    figures = free if mode == 'free' else _figures(deck, kind, power, mode)
    base, rate = (figures[0], figures[2]) if figures else (None, None)
    duration = _positive(((song.get('bgm') or {}).get('length') or {}).get('durationMs'))
    notes = _number((chart.get('notes') or {}).get('judged'))
    first, last = _number(chart.get('firstNoteMs')), _number(chart.get('lastJudgedNoteMs'))
    span = _positive(last - first) if first is not None and last is not None else None
    bpm = chart.get('bpm') or {}
    luck = None
    if deck is not None and deck.get('ranges') is not None:
        indices = [i for i, item in enumerate(deck['ranges']) if item['mission'] == 2]
        ranges = (deck.get('expectation') or {}).get('ranges')
        if not indices:
            luck = 0
        elif ranges is not None:
            values = [_center(ranges[i].get('luckPoints')) if i < len(ranges) else None for i in indices]
            if all(value is not None for value in values):
                luck = _number(sum(values))
    return {
        'scoreId': chart['scoreId'], 'musicId': song['id'], 'difficulty': chart['difficulty'],
        'durationMs': duration, 'chartDurationMs': _positive(chart.get('musicLengthMs')),
        'notes': notes, 'nps': _number(notes * 1000 / span) if notes is not None and span else None,
        'bpm': _number(bpm.get('main')), 'bpmMin': _number(bpm.get('min')), 'bpmMax': _number(bpm.get('max')),
        'skillCoverage': _number(free[1] / free[0]) if free and free[0] > 0 else None,
        'base': base, 'rate': rate,
        'perMinute': _number(rate * 60000 / (duration + 30000)) if rate is not None and duration else None,
        'justNotes': _number(deck.get('justNotes')) if deck else None,
        'luckPoints': luck, 'missions': song.get('gekisouMissions', []),
    }


def build_metrics(data: dict, *, mode: str = 'gekisou') -> dict[int, dict]:
    """按 scoreId 建立指标索引；结构错误直接暴露，不静默跳过整张谱面。"""
    return {chart['scoreId']: chart_metrics(data, song, chart, mode=mode)
            for song in data['songs'] for chart in song['charts']}


def format_duration(ms: float | None) -> str:
    """毫秒转换为 m:ss，按秒四舍五入；缺失或非正时长显示破折号。"""
    if _positive(ms) is None:
        return '—'
    seconds = floor(ms / 1000 + 0.5)
    return f'{seconds // 60}:{seconds % 60:02d}'


def format_number(value: float | None, digits: int = 2) -> str:
    """有限数值保留指定小数位；缺失显示破折号。"""
    return f'{value:,.{digits}f}' if _number(value) is not None else '—'
