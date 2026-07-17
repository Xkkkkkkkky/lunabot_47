"""鸟类名称查询与图片识别指令。"""

# 两个实现分文件注册指令，避免名称查询与图片识别的后端、绘图逻辑耦合。
from . import query as _query
from . import recognition as _recognition

