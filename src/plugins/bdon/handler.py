"""繁中 BanG Dream! 群聊指令的共用检查。"""

from ..utils import CmdHandler
from .common import cd, gwl, logger


class BdonCmdHandler(CmdHandler):
    def __init__(self, commands, logger=logger, *, help_command, **kwargs):
        kwargs["use_seg_cmd"] = False
        kwargs.setdefault("force_whitespace", True)
        super().__init__(commands, logger, help_command=help_command, **kwargs)
        # 先检查群白名单，再检查帮助与冷却。
        self.check_group()
        self.check_wblist(gwl, allow_private=False, allow_super=False)
        self.check_cdrate(cd)
