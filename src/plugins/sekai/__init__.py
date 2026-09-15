from ..common.config import Config


config = Config("sekai.sekai")

# 只有显式启用才加载命令及其后台任务。
if config.get("enabled", False, raise_exc=False):
    from .modules import *
