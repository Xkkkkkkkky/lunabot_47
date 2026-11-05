"""BanG Dream! Our Notes 繁中资料查询。"""

from ..common.config import Config


if Config("bdon").get("enabled", False, raise_exc=False):
    from .modules import *
