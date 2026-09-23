"""按轻量模式开关筛选 Sekai 帮助，不加载 Sekai 插件及其后台任务。

文档用 ``<!-- sekai-light: 分类.指令 ... -->`` 和 ``<!-- /sekai-light -->``
包裹目录、章节或行内命令；缺失的开关视为关闭，普通模式仅移除标记。
"""

import re

from .config import Config


_config = Config('sekai.sekai')
# 整行标记连同行尾移除；行内标记不吞掉正文的换行。
_markers = re.compile(
    r'^[ \t]*<!--\s*/?sekai-light\b[^<>\n]*?-->[ \t]*(?:\n|$)'
    r'|<!--\s*/?sekai-light\b.*?-->',
    re.MULTILINE,
)
_condition = re.compile(r'<!--\s*(/?)sekai-light(?:\s*:\s*(.*?))?\s*-->')


def render_sekai_help(text: str) -> str:
    """每次显示时读取当前配置；嵌套条件取交集，同一条件内的开关取并集。"""
    cfg = _config.get_all()
    if cfg.get('sekai-light') is not True:
        return _markers.sub('', text)

    commands = cfg.get('light_commands', {})

    def enabled(key: str) -> bool:
        group, name = key.split('.', 1)
        values = commands.get(group, {}) if isinstance(commands, dict) else {}
        return isinstance(values, dict) and values.get(name) is True

    output = []
    active = [True]
    start = 0
    for marker in _markers.finditer(text):
        if active[-1]:
            output.append(text[start:marker.start()])
        condition = _condition.search(marker.group())
        if condition is None:
            raise ValueError('Sekai 帮助中存在无效的轻量模式标记')
        closing, keys = condition.groups()
        if closing:
            if len(active) == 1:
                raise ValueError('Sekai 帮助中存在多余的轻量模式结束标记')
            active.pop()
        else:
            if not keys:
                raise ValueError('Sekai 帮助的轻量模式条件缺少指令开关')
            active.append(active[-1] and any(enabled(key) for key in keys.split()))
        start = marker.end()
    if len(active) != 1:
        raise ValueError('Sekai 帮助的轻量模式条件未闭合')
    output.append(text[start:])
    rendered = ''.join(output)
    # 抽卡记录在现有文档中使用一级标题，仍按一个指令项目判断。
    if not re.search(r'^###\s|^# 查询抽卡记录\s*$', rendered, re.MULTILINE):
        rendered = rendered.replace('## 指令目录\n', '## 指令目录\n\n当前没有已开启的帮助项目。\n', 1)
    return rendered
