from ..common.config import Config


config = Config("sekai.sekai")

if config.get("enabled", True, raise_exc=False):
    from .modules import *
