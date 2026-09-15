from ..common import config

from .card_extractor import *
from .card import *
from .chart import *
from .entertainment import *
from .event import *
from .gacha import *
from .honor import *
from .misc import *
from .music import *
from .profile import *
from .resbox import *
from .stamp import *
from .stamp_maker import *
from .vlive import *

# 按服务开关导入模块，关闭时不注册指令和后台任务。
if config.get('ranking_enabled', False, raise_exc=False):
    from .sk import *
if config.get('deck_enabled', False, raise_exc=False):
    from .deck import *
    from .score import *
