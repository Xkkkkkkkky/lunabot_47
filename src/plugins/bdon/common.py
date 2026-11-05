"""插件配置。"""

from ..utils import ColdDown, Config, get_file_db, get_group_white_list, get_logger


DATA_DIR = "data/bdon"
config = Config("bdon")
logger = get_logger("Bdon")
file_db = get_file_db(f"{DATA_DIR}/db.json", logger)
cd = ColdDown(file_db, logger, default_interval=config.item("cd"))
gwl = get_group_white_list(file_db, logger, "bdon")
