import os
import os.path as osp
from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Callable, List
import yaml
import inspect
import time


CONFIG_DIR = "config/"
CONFIG_UPDATE_CHECK_INTERVAL = 3.0


@dataclass
class ConfigData:
    mtime: int = 0
    path: str = None
    data: dict = field(default_factory=dict)


class _GlobalConfigState:
    """
    全局配置状态管理单例，用于存储内存中的配置数据和回调函数
    """
    _cache: Dict[str, ConfigData] = {}
    _callbacks: Dict[str, List[Callable]] = {}

    @classmethod
    def get_data(cls, name: str) -> dict:
        return cls._cache.get(name, ConfigData()).data
    
    @classmethod
    def update_cache(cls, name: str, path: str, force_load=False):
        """加载或重新加载配置文件"""
        if not osp.exists(path):
            print(f"配置文件 {path} 不存在，跳过加载")
            return
        try:
            # 使用纳秒时间戳，避免编辑器在同一秒内连续保存时漏掉热重载。
            mtime = os.stat(path).st_mtime_ns
            if force_load or name not in cls._cache or cls._cache[name].mtime != mtime:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                cls._cache[name] = ConfigData(mtime=mtime, path=path, data=data)
                cls.trigger_callbacks(name)
                return True
        except Exception as e:
            print(f"读取配置文件 {path} 失败: {e}")
        return False

    @classmethod
    def register_callback(cls, name: str, func: Callable):
        if inspect.iscoroutinefunction(func):
            raise RuntimeError("不支持注册异步回调函数")
        if name not in cls._callbacks:
            cls._callbacks[name] = []
        cls._callbacks[name].append(func)

    @classmethod
    def trigger_callbacks(cls, name: str):
        """触发回调，支持同步函数"""
        if name in cls._callbacks:
            current_data = cls.get_data(name)
            for func in cls._callbacks[name]:
                try:
                    func(current_data)
                except Exception as e:
                    print(f"执行配置 {name} 的更新回调 {func.__name__} 失败: {e}")


class ConfigItem:
    """
    配置项类，用于动态延迟获取配置文件中的单个配置项
    """
    def __init__(self, config: 'Config', key: str | tuple[str] | Any):
        self.config = config
        if isinstance(key, str):
            self.keys = key.split('.')
        elif isinstance(key, (list, tuple)):
            self.keys = key
        else:
            self.keys = [key]
            
    def get(self, default=None, raise_exc: Optional[bool]=None) -> Any:
        return self.config.get(self.keys, default, raise_exc)


class Config:
    def __init__(self, name: str):
        """
        初始化配置类
        name: 配置名称，格式为 "module" 或 "module.submodule"
        """
        self.name = name
        self.path = pjoin(CONFIG_DIR, name.replace('.', '/') + '.yaml')
        self._last_check_time = 0.0
        self._check_interval = CONFIG_UPDATE_CHECK_INTERVAL
        if self.name not in _GlobalConfigState._cache:
            _GlobalConfigState.update_cache(self.name, self.path, force_load=True)

    def _ensure_updated(self):
        current_time = time.time()
        if current_time - self._last_check_time > self._check_interval:
            self._last_check_time = current_time
            _GlobalConfigState.update_cache(self.name, self.path)

    def get_all(self) -> dict:
        """
        获取配置项的所有数据
        """
        self._ensure_updated()
        return _GlobalConfigState.get_data(self.name)

    def get(self, key: str | tuple[str] | Any, default=None, raise_exc: Optional[bool]=None) -> Any:
        """
        获取配置项的值
        """
        self._ensure_updated()
        if raise_exc is None:
            raise_exc = default is None
        
        if isinstance(key, str):
            keys = key.split('.')
        elif isinstance(key, (list, tuple)):
            keys = key
        else:
            keys = [key]
            
        ret = _GlobalConfigState.get_data(self.name)
        
        for k in keys:
            if isinstance(ret, dict) and k in ret:
                ret = ret[k]
            else:
                if raise_exc:
                    raise KeyError(f"配置 {self.name} 中不存在 {key}")
                return default
        return ret
    
    def mtime(self) -> int:
        self._ensure_updated()
        return _GlobalConfigState._cache.get(self.name, ConfigData()).mtime
    
    def item(self, key: str | tuple[str] | Any) -> ConfigItem:
        return ConfigItem(self, key)
    

def get_cfg_or_value(obj: Union[ConfigItem, Any], default=None, raise_exc: Optional[bool]=None) -> Any:
    """
    如果是 ConfigItem 对象则返回值，否则返回原对象
    """
    if isinstance(obj, ConfigItem):
        return obj.get(default, raise_exc)
    return obj


def parse_cfg_num(x: str) -> Union[int, float]:
    """
    解析配置中的数字字符串，支持数字和数字四则运算
    """
    if isinstance(x, (int, float)):
        return x
    try:
        return eval(x, {'__builtins__': None}, {})
    except Exception as e:
        raise ValueError(f"无法解析配置数字 '{x}': {e}")


global_config = Config('global')


def get_nonebot_env_config(name: str, default=None) -> Any:
    """读取 NoneBot 合并后的系统环境变量与 ``.env`` 配置。

    NoneBot/Pydantic 会把自定义环境变量规范化为小写字段，但不会把
    ``.env`` 内容写回 ``os.environ``，因此不能用 ``os.getenv`` 读取。
    """
    try:
        from nonebot import get_driver
        return getattr(get_driver().config, name.strip().lower(), default)
    except ValueError:
        # 允许配置模块在 NoneBot 初始化前被独立工具导入。
        return default


def get_shared_draw_background_image() -> str:
    """返回通用绘图背景路径，优先使用 ``DRAW_BACKGROUND_IMAGE``。

    ``global.draw.background_image`` 仅作为旧部署迁移兼容项；一旦
    ``DRAW_BACKGROUND_IMAGE`` 存在（包括显式留空），就以环境配置为准。
    """
    missing = object()
    value = get_nonebot_env_config('DRAW_BACKGROUND_IMAGE', missing)
    if value is not missing:
        return str(value or '').strip()
    return str(global_config.get(
        'draw.background_image',
        '',
        raise_exc=False,
    ) or '').strip()
