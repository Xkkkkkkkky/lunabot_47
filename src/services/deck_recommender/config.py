from utils import *


# 统一从仓库根目录定位配置和数据，避免从不同工作目录启动时写错位置。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
with open(pjoin(PROJECT_ROOT, 'config', 'sekai', 'sekai.yaml'), 'r', encoding='utf-8') as f:
    sekai_config = yaml.safe_load(f) or {}
    SEKAI_ENABLED = sekai_config.get('enabled', False)
    DECK_ENABLED = sekai_config.get('deck_enabled', False)

CONFIG_PATH = pjoin(PROJECT_ROOT, 'config', 'sekai', 'deck_recommender.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f) or {}

HOST = CONFIG['host']
PORT = int(CONFIG['port'])
WORKER_NUM = int(CONFIG['worker_num'])
USERDATA_CACHE_NUM = int(CONFIG['userdata_cache_num'])
DATA_DIR = pjoin(PROJECT_ROOT, CONFIG['data_dir'])
DB_PATH = pjoin(DATA_DIR, 'deckrec.json')
