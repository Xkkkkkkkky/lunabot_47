from ..utils import *
from nonebot import on_message
import hashlib
from dataclasses import replace

from .image_dedup import (
    ImageFeature,
    SimilarityGroup,
    compute_phash,
    find_similarity_groups,
    hamming_distance,
    normalize_phash,
    render_similarity_report,
    select_best_image,
)


config = Config('nsy')
logger = get_logger('NSY')

DATA_DIR = 'data/nsy'
INDEX_FILE = pjoin(DATA_DIR, 'index.json')
ALIASES_FILE = pjoin(DATA_DIR, 'aliases.json')
OPERATION_LOG_FILE = pjoin(DATA_DIR, 'operations.log')
STATE_DB_FILE = pjoin(DATA_DIR, 'state.json')
RANDOM_HISTORY_DB_KEY = 'random_image_history'

file_db = get_file_db(STATE_DB_FILE, logger)
cd = ColdDown(file_db, logger, cold_down_name='nsy')
gbl = get_group_black_list(file_db, logger, 'nsy')

IMAGE_EXT_BY_FORMAT = {
    'JPEG': '.jpg',
    'PNG': '.png',
    'GIF': '.gif',
    'WEBP': '.webp',
    'BMP': '.bmp',
}
RESERVED_DATA_NAMES = {
    osp.basename(INDEX_FILE),
    osp.basename(ALIASES_FILE),
    osp.basename(OPERATION_LOG_FILE),
    osp.basename(STATE_DB_FILE),
}
GALLERY_SUGGESTION_LIMIT = 5
GALLERY_SUGGESTION_MAX_DISTANCE = 2
DEFAULT_RECENT_IMAGE_WEIGHT_MULTIPLIERS = (0.05, 0.2, 0.4, 0.6, 0.8)
CREATED_AT_FORMAT = '%Y-%m-%d %H:%M:%S'


@dataclass
class ImageInfo:
    """本地图片文件经过真实内容校验后的基础信息。"""
    fmt: str
    ext: str
    width: int
    height: int
    size: int


@dataclass
class NsyImage:
    """索引 JSON 中单张图片的记录。"""
    pid: int
    gallery: str
    filename: str
    hash: str
    format: str
    phash: str = ''
    size: int = 0
    width: int = 0
    height: int = 0
    created_at: str = ''
    uploader_id: int | None = None
    group_id: int | None = None
    linked_to_pid: int | None = None
    linked_pids: list[int] = field(default_factory=list)

    @classmethod
    def load(cls, gallery: str, pid: int, data: dict) -> 'NsyImage':
        """从图库分组下的 pid 子项加载图片，父级键不在记录中重复保存。"""
        return cls(
            pid=pid,
            gallery=gallery,
            filename=str(data['filename']),
            hash=str(data['hash']),
            format=str(data.get('format', '')),
            phash=str(data.get('phash', '')),
            size=int(data.get('size', 0)),
            width=int(data.get('width', 0)),
            height=int(data.get('height', 0)),
            created_at=str(data.get('created_at', '')),
            uploader_id=data.get('uploader_id', None),
            group_id=data.get('group_id', None),
            linked_to_pid=(
                int(data['linked_to_pid'])
                if data.get('linked_to_pid') is not None else None
            ),
            linked_pids=[int(linked_pid) for linked_pid in data.get('linked_pids', [])],
        )

    def dump(self) -> dict:
        """输出 pid 子项内容；pid 和 gallery 分别由两级 JSON 键表达。"""
        data = asdict(self)
        del data['pid']
        del data['gallery']
        return data


@dataclass
class ReloadImageCandidate:
    """重载扫描得到的有效图片及其原索引信息。"""
    gallery: str
    path: Path
    info: ImageInfo
    old_image: NsyImage | None
    inode: tuple[int, int]
    created_at: str
    pid: int = 0


@dataclass
class SimilarUploadRejection:
    """因 pHash 命中而被拒绝的图片上传项，用于生成相似对比报告。"""

    upload_index: int
    gallery: str
    staged_path: str
    info: ImageInfo
    phash: str
    similar_images: list[tuple[NsyImage, int]] = field(default_factory=list)


@dataclass
class SimilarUploadLink:
    """跨图库 pHash 命中后已自动创建的硬链接加图项。"""

    upload_index: int
    gallery: str
    staged_path: str
    info: ImageInfo
    image_hash: str
    phash: str
    link_root_pid: int
    linked_pid: int
    similar_images: list[tuple[NsyImage, int]] = field(default_factory=list)


@dataclass
class GalleryUploadResult:
    """一次批量加图中，单个目标图库的有序处理结果。"""
    gallery: str
    added: list[NsyImage] = field(default_factory=list)
    repeats: list[tuple[int, NsyImage]] = field(default_factory=list)
    similar_rejections: list[SimilarUploadRejection] = field(default_factory=list)
    phash_links: list[SimilarUploadLink] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GalleryUploadTarget:
    """用户输入的单个加图目标及其图库索引结果。"""
    token: str
    gallery: str | None = None
    error: str = ''


@dataclass(frozen=True)
class GlobalDedupPlan:
    """一组全局相似图片的索引保留、硬链接合并与删除计划。"""

    group: SimilarityGroup
    root_pid: int
    retained_pids: tuple[int, ...]
    relink_pids: tuple[int, ...]
    delete_pids: tuple[int, ...]


def _now_str() -> str:
    return datetime.now().strftime(CREATED_AT_FORMAT)


def _event_group_id(event: MessageEvent) -> int:
    return int(event.group_id) if is_group_msg(event) else 0


def _sha256_file(path: str) -> str:
    """以文件真实字节计算 SHA-256，用于上传阶段阻止完全相同文件。"""
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def _validate_gallery_token(name: str, desc: str = '图库名称') -> str:
    """
    校验图库名称/别名，避免路径穿越、空白分隔歧义和与 pid 纯数字冲突。
    """
    name = name.strip()
    max_len = int(config.get('max_name_length', 64))
    if not name:
        raise ReplyException(f'{desc}不能为空')
    if len(name) > max_len:
        raise ReplyException(f'{desc}不能超过{max_len}个字符')
    if name in ('.', '..') or name.startswith('.'):
        raise ReplyException(f'{desc}"{name}"无效')
    if name.isdigit():
        raise ReplyException(f'{desc}"{name}"不能是纯数字')
    if any(c.isspace() for c in name) or any(c in r'\/:*?"<>|' for c in name):
        raise ReplyException(f'{desc}"{name}"不能包含空白或路径特殊字符')
    if name in RESERVED_DATA_NAMES:
        raise ReplyException(f'{desc}"{name}"与数据文件名冲突')
    return name


def _inspect_image_file(path: str, check_size: bool = True) -> ImageInfo:
    """
    通过 Pillow 打开并 verify 图片内容，而不是信任扩展名或上游消息类型。

    返回的扩展名来自图片真实格式，后续落盘时统一使用 pid.ext 命名。
    """
    size = os.path.getsize(path)
    if check_size:
        max_mb = float(config.get('max_image_size_mb', 20))
        if size > max_mb * 1024 * 1024:
            raise ReplyException(f'图片超过大小限制 {max_mb:g}MB')

    try:
        with Image.open(path) as img:
            fmt = img.format
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
    except ReplyException:
        raise
    except Exception as e:
        raise ReplyException(f'文件不是有效图片: {get_exc_desc(e)}')

    allowed = [str(x).upper() for x in config.get('allowed_formats', list(IMAGE_EXT_BY_FORMAT.keys()))]
    if fmt not in allowed or fmt not in IMAGE_EXT_BY_FORMAT:
        raise ReplyException(f'不支持的图片格式: {fmt or "未知"}')
    return ImageInfo(fmt=fmt, ext=IMAGE_EXT_BY_FORMAT[fmt], width=width, height=height, size=size)


def _append_operation_log(event: MessageEvent, operation: str):
    """
    追加 JSON Lines 操作日志，记录操作者、群号、时间和操作内容。
    """
    create_parent_folder(OPERATION_LOG_FILE)
    item = {
        'time': _now_str(),
        'user_id': int(event.user_id),
        'group_id': _event_group_id(event),
        'operation': operation,
    }
    with open(OPERATION_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(dumps_json(item, indent=False) + '\n')


def _format_log_line(line: str) -> str:
    try:
        item = loads_json(line)
        return (
            f"{item.get('time', '')} | "
            f"qq={item.get('user_id', 0)} | "
            f"group={item.get('group_id', 0)} | "
            f"{item.get('operation', '')}"
        )
    except Exception:
        return line.strip()


def _format_upload_result(
    gallery: str,
    total: int,
    added: list['NsyImage'],
    repeats: list[tuple[int, 'NsyImage']],
    similar_rejections: list[SimilarUploadRejection],
    phash_links: list[SimilarUploadLink],
    errors: list[str],
    images_by_pid: dict[int, NsyImage],
) -> str:
    lines = [
        f'图库"{gallery}": 添加{len(added)}张，共{total}张'
    ]
    if added:
        lines.append('pid: ' + ' '.join(str(image.pid) for image in added))
        linked = [image for image in added if image.linked_to_pid is not None]
        if linked:
            for image in linked:
                existing_galleries = sorted({
                    indexed.gallery
                    for indexed in images_by_pid.values()
                    if indexed.pid != image.pid and indexed.hash == image.hash
                })
                gallery_names = '、'.join(f'"{name}"' for name in existing_galleries)
                lines.append(
                    f'已存在于 {gallery_names}，链接"{image.gallery}"（根 pid={image.linked_to_pid}）'
                )
    if repeats:
        lines.extend(
            f'第{idx}张与 pid={duplicated.pid}（图库"{duplicated.gallery}"）重复'
            for idx, duplicated in repeats
        )
    if similar_rejections:
        lines.append(f'相似图: {len(similar_rejections)}张，已拒绝添加')
    if phash_links:
        lines.append(f'跨图库相似图: {len(phash_links)}张，已链接')
    if errors:
        lines.append('失败:')
        lines.extend(errors)
    return '\n'.join(lines)


def _format_reload_result(result: dict) -> str:
    lines = [
        '图库重载完成',
        f'图库数: {result["gallery_count"]}',
        f'图片数: {result["image_count"]}',
        f'新增索引: {len(result["added_pids"])}',
        f'删除失效索引: {len(result["removed_pids"])}',
        f'重命名文件: {len(result["renamed_files"])}',
        f'无效图片: {len(result["invalid_files"])}',
        f'删除失效别名: {len(result["removed_aliases"])}',
    ]
    if result['added_pids']:
        lines.append('新增 pid: ' + ' '.join(str(pid) for pid in result['added_pids']))
    if result['removed_pids']:
        lines.append('移除 pid: ' + ' '.join(str(pid) for pid in result['removed_pids']))
    if result['removed_aliases']:
        lines.append('移除别名: ' + '，'.join(result['removed_aliases']))
    if result['invalid_files']:
        lines.append('无效图片:')
        lines.extend(result['invalid_files'][:20])
        if len(result['invalid_files']) > 20:
            lines.append(f'... 还有{len(result["invalid_files"]) - 20}项')
    return '\n'.join(lines)


def _read_random_weight(value, default: float = 1.0, desc: str = '随机权重') -> float:
    """读取非负随机权重；配置错误时回退默认值，避免随机看图中断。"""
    try:
        weight = float(value)
    except Exception:
        logger.warning(f'{desc}配置无效: {value!r}，使用默认值 {default:g}')
        return default
    if weight != weight or weight == float('inf') or weight < 0:
        logger.warning(f'{desc}配置无效: {value!r}，使用默认值 {default:g}')
        return default
    return weight


def _get_random_weight_config() -> tuple[dict[str, float], float]:
    """
    读取随机看图权重配置。

    格式权重按图片真实 format 匹配；链接权重用于已链接或被链接的图片，
    最终图片权重 = 格式权重 * 链接关系权重。
    """
    raw_format_weights = config.get('random_weights.formats', {}, raise_exc=False)
    format_weights: dict[str, float] = {}
    if raw_format_weights is None:
        raw_format_weights = {}
    if not isinstance(raw_format_weights, dict):
        logger.warning('random_weights.formats 不是对象，使用默认格式权重')
        raw_format_weights = {}
    for fmt, weight in raw_format_weights.items():
        format_weights[str(fmt).upper()] = _read_random_weight(
            weight,
            desc=f'{fmt}格式随机权重',
        )

    linked_weight = _read_random_weight(
        config.get('random_weights.linked_image', 1, raise_exc=False),
        desc='链接图片随机权重',
    )
    return format_weights, linked_weight


def _get_recent_image_weight_multipliers() -> list[float]:
    """
    读取最近图片的降权系数，配置顺序为最近一次到第 5 次。

    列表长度同时决定历史窗口大小。配置结构错误时整体回退默认值，单个
    系数错误时仅回退该位置，避免错误配置导致随机看图不可用。
    """
    defaults = list(DEFAULT_RECENT_IMAGE_WEIGHT_MULTIPLIERS)
    raw_weights = config.get('random_weights.recent_images', defaults, raise_exc=False)
    if not isinstance(raw_weights, list) or not raw_weights:
        logger.warning('NSY random_weights.recent_images 不是非空数组，使用默认近期图片权重')
        return defaults
    return [
        _read_random_weight(
            value,
            default=defaults[idx] if idx < len(defaults) else 1.0,
            desc=f'最近第{idx + 1}张图片随机权重系数',
        )
        for idx, value in enumerate(raw_weights)
    ]


def _get_phash_distance_threshold() -> int:
    """读取 pHash 汉明距离严格上界，配置错误时回退为 4。"""
    value = config.get('phash_distance_threshold', 4, raise_exc=False)
    try:
        threshold = int(value)
    except Exception:
        threshold = 4
    if not 1 <= threshold <= 64:
        logger.warning(f'pHash Hamming distance 阈值配置无效: {value!r}，使用默认值 4')
        return 4
    return threshold


def _get_image_random_weight(
    image: NsyImage,
    format_weights: dict[str, float],
    linked_weight: float,
) -> float:
    """计算单张图片的随机权重。"""
    weight = format_weights.get(str(image.format).upper(), 1.0)
    if image.linked_to_pid is not None or image.linked_pids:
        weight *= linked_weight
    return weight


def _extract_exact_gallery_text(event: MessageEvent) -> str | None:
    """
    提取“整条消息只有文本”的图库触发词；含图片、表情、at 等消息均不触发看图。
    """
    msg = get_msg(event)
    if not msg or any(seg['type'] != 'text' for seg in msg):
        return None
    text = ''.join(seg['data'].get('text', '') for seg in msg).strip()
    if not text or text.startswith('/'):
        return None
    return text


class NsyManager:
    """
    NSY 图库索引管理器。

    aliases.json 是图库入口总表，负责将图库名称或别名解析为规范图库名；
    index.json 只保存规范图库名下的图片。磁盘目录仅在重载时用于重建两张表。
    """
    _mgr: 'NsyManager' = None

    def __init__(self):
        self.pid_top = 0
        self.images_by_gallery: dict[str, dict[int, NsyImage]] = {}
        self.images_by_pid: dict[int, NsyImage] = {}
        self.aliases_by_gallery: dict[str, list[str]] = {}
        self.gallery_by_alias: dict[str, str] = {}
        create_folder(DATA_DIR)
        self._ensure_json_files()
        self._load()

    @classmethod
    def get(cls) -> 'NsyManager':
        if cls._mgr is None:
            cls._mgr = NsyManager()
        return cls._mgr

    def _ensure_json_files(self):
        if not osp.exists(INDEX_FILE):
            dump_json({}, INDEX_FILE)
        if not osp.exists(ALIASES_FILE):
            dump_json({}, ALIASES_FILE)

    def _load(self):
        """
        加载图库优先的持久化结构，并构建 pid、别名的 O(1) 运行时反向索引。

        文件只接受当前结构，不解析旧版 pid_top/images 或 alias->gallery 表。
        """
        self._ensure_json_files()
        index = load_json(INDEX_FILE)
        alias_index = load_json(ALIASES_FILE)
        self.images_by_gallery = {}
        self.images_by_pid = {}

        for gallery, pid_items in index.items():
            gallery = str(gallery)
            self.images_by_gallery[gallery] = {}
            if not isinstance(pid_items, dict):
                logger.warning(f'加载 NSY 图库索引 gallery={gallery} 失败: pid 子项不是对象')
                continue
            for pid_text, data in pid_items.items():
                try:
                    pid = int(pid_text)
                    if pid in self.images_by_pid:
                        raise ValueError(f'pid={pid} 在多个图库中重复')
                    image = NsyImage.load(gallery, pid, data)
                    self.images_by_gallery[gallery][pid] = image
                    self.images_by_pid[pid] = image
                except Exception as e:
                    logger.warning(f'加载 NSY 图片索引 gallery={gallery} pid={pid_text} 失败: {get_exc_desc(e)}')

        self.aliases_by_gallery = {}
        self.gallery_by_alias = {}
        for gallery, data in alias_index.items():
            gallery = str(gallery)
            self.aliases_by_gallery[gallery] = []
            try:
                aliases = data['aliases']
                if not isinstance(aliases, list):
                    raise ValueError('aliases 子项不是数组')
                for alias in aliases:
                    alias = str(alias)
                    if alias in self.gallery_by_alias:
                        raise ValueError(f'别名"{alias}"在多个图库中重复')
                    self.gallery_by_alias[alias] = gallery
                    self.aliases_by_gallery.setdefault(gallery, []).append(alias)
            except Exception as e:
                logger.warning(f'加载别名索引 gallery={gallery} 失败: {get_exc_desc(e)}')

        self.pid_top = max(self.images_by_pid, default=0)
        logger.info(
            f'成功加载图库索引: {len(self.aliases_by_gallery)}个图库, '
            f'{len(self.images_by_pid)}张图片, {len(self.gallery_by_alias)}个别名, '
            f'pid_top={self.pid_top}'
        )

    def _save_index(self):
        data = {}
        for gallery, images in sorted(self.images_by_gallery.items()):
            data[gallery] = {
                str(pid): image.dump()
                for pid, image in sorted(images.items())
            }
        dump_json(data, INDEX_FILE)

    def _save_aliases(self):
        data = {
            gallery: {'aliases': sorted(aliases)}
            for gallery, aliases in sorted(self.aliases_by_gallery.items())
        }
        dump_json(data, ALIASES_FILE)

    def _gallery_dir(self, gallery: str) -> str:
        return pjoin(DATA_DIR, gallery)

    def _image_path(self, image: NsyImage) -> str:
        return pjoin(self._gallery_dir(image.gallery), image.filename)

    def _list_gallery_names(self) -> list[str]:
        create_folder(DATA_DIR)
        names = []
        for path in Path(DATA_DIR).iterdir():
            if path.is_dir():
                names.append(path.name)
        return sorted(names)

    def _check_token_free(self, token: str, desc: str):
        if token in self.aliases_by_gallery:
            raise ReplyException(f'{desc}"{token}"已是图库名称')
        if token in self.gallery_by_alias:
            raise ReplyException(f'{desc}"{token}"已是图库别名')
        if osp.exists(pjoin(DATA_DIR, token)):
            raise ReplyException(f'{desc}"{token}"与 data/nsy 下已有文件冲突')

    def _pid_file_exists(self, pid: int) -> bool:
        for gallery in self._list_gallery_names():
            if glob.glob(pjoin(self._gallery_dir(gallery), f'{pid}.*')):
                return True
        return False

    def _allocate_pid(self) -> int:
        """
        分配不会撞上索引和现有磁盘文件的 pid。
        """
        self.pid_top = max(self.pid_top, max(self.images_by_pid, default=0))
        while True:
            self.pid_top += 1
            if self.pid_top not in self.images_by_pid and not self._pid_file_exists(self.pid_top):
                return self.pid_top

    @staticmethod
    def _reload_file_sort_key(path: Path) -> tuple:
        """数字文件名按数值排序，其余文件名按文本排序，保证重载编号稳定。"""
        if path.stem.isdigit():
            return 0, int(path.stem), path.suffix.lower(), path.name
        return 1, path.name

    @staticmethod
    def _reload_candidate_sort_key(candidate: ReloadImageCandidate) -> tuple:
        """
        按添加时间生成重载编号顺序，并为时间相同或缺失的记录提供稳定顺序。

        旧索引可能没有 created_at。此类记录视为早期遗留图片，排在有明确
        时间的记录之前；同一时间优先保持旧 pid 顺序，新发现的图片则依靠
        Python 稳定排序保留磁盘扫描顺序。
        """
        created_at = candidate.created_at.strip()
        try:
            created_time = datetime.strptime(created_at, CREATED_AT_FORMAT)
            has_created_at = True
        except ValueError:
            created_time = datetime.min
            has_created_at = False

        old_pid = candidate.old_image.pid if candidate.old_image is not None else 0
        return (
            has_created_at,
            created_time,
            candidate.old_image is None,
            old_pid,
        )

    def _renumber_reload_files(self, candidates: list[ReloadImageCandidate]) -> list[str]:
        """
        将所有有效图片全局连续编号，并用两阶段重命名避免文件名交换时覆盖。

        第一阶段把需要改名的图片移到同目录临时文件；第二阶段再统一移到
        pid.ext。若任一阶段失败，会尽量恢复所有原文件名后抛出异常。
        """
        source_paths = {osp.abspath(str(candidate.path)) for candidate in candidates}
        moves: list[tuple[ReloadImageCandidate, Path, Path, Path]] = []
        renamed_files: list[str] = []

        for pid, candidate in enumerate(candidates, 1):
            candidate.pid = pid
            target = candidate.path.parent / f'{pid}{candidate.info.ext}'
            if target.exists() and osp.abspath(str(target)) not in source_paths:
                raise ReplyException(
                    f'重载失败: 目标文件 {candidate.gallery}/{target.name} '
                    '被无效图片或其他文件占用'
                )
            if osp.abspath(str(target)) == osp.abspath(str(candidate.path)):
                continue

            temp = candidate.path.parent / f'.nsy_reload_{pid}.tmp'
            suffix = 0
            while temp.exists():
                suffix += 1
                temp = candidate.path.parent / f'.nsy_reload_{pid}_{suffix}.tmp'
            moves.append((candidate, candidate.path, target, temp))
            renamed_files.append(
                f'{candidate.gallery}/{candidate.path.name} -> {target.name}'
            )

        staged: list[tuple[ReloadImageCandidate, Path, Path, Path]] = []
        try:
            for move in moves:
                _, source, _, temp = move
                os.rename(source, temp)
                staged.append(move)
        except Exception:
            for _, source, _, temp in reversed(staged):
                os.rename(temp, source)
            raise

        finalized: list[tuple[ReloadImageCandidate, Path, Path, Path]] = []
        try:
            for move in moves:
                candidate, _, target, temp = move
                os.rename(temp, target)
                candidate.path = target
                finalized.append(move)
        except Exception:
            for candidate, _, target, temp in finalized:
                os.rename(target, temp)
                candidate.path = temp
            for candidate, source, _, temp in moves:
                os.rename(temp, source)
                candidate.path = source
            raise

        return renamed_files

    def resolve_gallery(self, name_or_alias: str, raise_if_missing: bool = False) -> str | None:
        """仅通过 aliases 总表把图库名称或别名解析为规范图库名。"""
        token = name_or_alias.strip()
        gallery = token if token in self.aliases_by_gallery else self.gallery_by_alias.get(token)
        if gallery is not None:
            return gallery
        if raise_if_missing:
            if not token:
                raise ReplyException('图库名称不能为空')
            msg = f'图库"{token}"不存在'
            suggestions = self.suggest_gallery_tokens(token)
            if suggestions:
                msg += f'\n模糊匹配：{"、".join(suggestions)}'
            raise ReplyException(msg)
        return None

    def suggest_gallery_tokens(
        self,
        name_or_alias: str,
        limit: int = GALLERY_SUGGESTION_LIMIT,
        max_distance: int = GALLERY_SUGGESTION_MAX_DISTANCE,
    ) -> list[str]:
        """
        按编辑距离推荐相近的图库名称或别名。

        图库名称和别名都是有效的命令参数，因此统一参与匹配；距离相同时按
        文本排序，确保每次返回顺序稳定。
        """
        token = name_or_alias.strip()
        if not token or limit <= 0:
            return []

        candidates = set(self.aliases_by_gallery) | set(self.gallery_by_alias)
        ranked = sorted(
            (
                (candidate, levenshtein_distance(token, candidate))
                for candidate in candidates
            ),
            key=lambda item: (item[1], item[0]),
        )
        return [
            candidate
            for candidate, distance in ranked
            if distance <= max_distance
        ][:limit]

    def get_aliases(self, name_or_alias: str) -> tuple[str, list[str]]:
        gallery = self.resolve_gallery(name_or_alias, raise_if_missing=True)
        return gallery, sorted(self.aliases_by_gallery.get(gallery, []))

    def create_gallery(self, name: str) -> str:
        name = _validate_gallery_token(name)
        self._check_token_free(name, '图库名称')
        create_folder(self._gallery_dir(name))
        self.images_by_gallery.setdefault(name, {})
        self.aliases_by_gallery.setdefault(name, [])
        self._save_index()
        self._save_aliases()
        return name

    def add_alias(self, name_or_alias: str, alias: str) -> tuple[str, str]:
        gallery = self.resolve_gallery(name_or_alias, raise_if_missing=True)
        alias = _validate_gallery_token(alias, '图库别名')
        self._check_token_free(alias, '图库别名')
        self.aliases_by_gallery.setdefault(gallery, []).append(alias)
        self.gallery_by_alias[alias] = gallery
        self._save_aliases()
        return gallery, alias

    def delete_alias(self, name_or_alias: str, alias: str) -> tuple[str, str]:
        gallery = self.resolve_gallery(name_or_alias, raise_if_missing=True)
        if self.gallery_by_alias.get(alias) != gallery:
            raise ReplyException(f'图库"{gallery}"没有别名"{alias}"')
        self.aliases_by_gallery[gallery].remove(alias)
        del self.gallery_by_alias[alias]
        self._save_aliases()
        return gallery, alias

    def find_image(self, pid: int, raise_if_missing: bool = False) -> NsyImage | None:
        image = self.images_by_pid.get(pid)
        if image is None and raise_if_missing:
            raise ReplyException(f'图片 pid={pid} 不存在')
        return image

    def find_images_by_hash(self, image_hash: str) -> list[NsyImage]:
        """返回全部同 hash 索引，查询图片信息时可能需要展示多个图库条目。"""
        return sorted(
            (image for image in self.images_by_pid.values() if image.hash == image_hash),
            key=lambda image: image.pid,
        )

    def _to_image_feature(self, image: NsyImage) -> ImageFeature:
        """把索引图片转换为 pHash 比较和报告使用的不可变特征。"""
        return ImageFeature(
            identity=f'pid:{image.pid}',
            gallery=image.gallery,
            path=self._image_path(image),
            phash=image.phash,
            width=image.width,
            height=image.height,
            pid=image.pid,
        )

    def _refresh_all_image_features(self) -> bool:
        """
        从真实文件刷新全部 SHA-256、pHash、格式和分辨率。

        `/图库查重` 会据此生成删除计划，因此不能只信任索引中
        格式合法但可能过期的 pHash。同一 inode 只读取和计算一次，
        避免对已有硬链接重复执行昂贵计算。
        """
        feature_by_inode: dict[tuple[int, int], tuple[ImageInfo, str, str]] = {}
        changed = False
        errors = []
        for image in sorted(self.images_by_pid.values(), key=lambda item: item.pid):
            path = self._image_path(image)
            if not osp.exists(path):
                continue
            try:
                stat = os.stat(path)
                inode = (stat.st_dev, stat.st_ino)
                if inode not in feature_by_inode:
                    info = _inspect_image_file(path, check_size=False)
                    feature_by_inode[inode] = (
                        info,
                        _sha256_file(path),
                        compute_phash(path),
                    )
                info, image_hash, image_phash = feature_by_inode[inode]
                values = {
                    'hash': image_hash,
                    'format': info.fmt,
                    'phash': image_phash,
                    'size': info.size,
                    'width': info.width,
                    'height': info.height,
                }
                for field_name, value in values.items():
                    if getattr(image, field_name) != value:
                        setattr(image, field_name, value)
                        changed = True
            except Exception as e:
                errors.append(f'pid={image.pid}: {get_exc_desc(e)}')
        if errors:
            raise ReplyException(
                '图库查重中止：以下图片无法刷新真实特征，请先重载图库\n'
                + '\n'.join(errors[:20])
            )
        if changed:
            self._save_index()
        return changed

    def find_global_similar_images(
        self,
        image_phash: str,
    ) -> list[tuple[NsyImage, int]]:
        """
        在全部图库的现有索引中查找与给定 pHash 相似的有效图片。

        汉明距离上界由 ``phash_distance_threshold`` 配置读取，使用
        严格小于语义；配置为 4 时，距离 0~3 判定为相似。
        该查询不读取或刷新现有图片特征，图库特征的权威刷新由
        ``/图库查重`` 或 ``/重载图库`` 负责。
        """
        image_phash = normalize_phash(image_phash)
        if image_phash is None:
            raise ReplyException('待比较图片的 pHash 无效')
        threshold = _get_phash_distance_threshold()
        matches = []
        for image in self.images_by_pid.values():
            path = self._image_path(image)
            if not osp.exists(path) or normalize_phash(image.phash) is None:
                continue
            distance = hamming_distance(image_phash, image.phash)
            if distance < threshold:
                matches.append((image, distance))
        return sorted(matches, key=lambda item: (item[1], item[0].pid))

    def select_best_phash_link_root(
        self,
        matches: list[tuple[NsyImage, int]],
    ) -> NsyImage:
        """
        从跨图库 pHash 命中中选出分辨率最高的可用硬链接根。

        候选索引可能本身是硬链接子节点，因此先解析到现有根节点；
        断链候选不参与选择，避免为新索引继续扩散无效关系。
        """
        roots: dict[int, NsyImage] = {}
        for image, _ in matches:
            root = self._get_link_root(image)
            if root is not None and osp.exists(self._image_path(root)):
                roots[root.pid] = root
        if not roots:
            raise ReplyException('跨图库相似图片没有可用的硬链接根文件')
        return max(
            roots.values(),
            key=lambda image: (
                image.width * image.height,
                image.width,
                image.height,
                -image.pid,
            ),
        )

    def find_global_similarity_groups(
        self,
        name_or_alias: str | None = None,
    ) -> tuple[str | None, list[SimilarityGroup]]:
        """
        对所有图库图片执行全局 pHash 比较。

        传入图库时仍会全局扫描，但只返回包含该图库图片的相似组；
        组内的其他图库图片不会被截断，以便正确合并跨图库索引。
        """
        filter_gallery = (
            self.resolve_gallery(name_or_alias, raise_if_missing=True)
            if name_or_alias is not None else None
        )
        self._refresh_all_image_features()
        features = [
            self._to_image_feature(image)
            for image in self.images_by_pid.values()
            if osp.exists(self._image_path(image)) and normalize_phash(image.phash) is not None
        ]
        groups = find_similarity_groups(features, _get_phash_distance_threshold())
        if filter_gallery is not None:
            groups = [
                group for group in groups
                if any(image.gallery == filter_gallery for image in group.images)
            ]
        return filter_gallery, groups

    def build_global_dedup_plans(
        self,
        name_or_alias: str | None = None,
    ) -> tuple[str | None, list[GlobalDedupPlan]]:
        """
        为全局相似组生成可执行去重计划。

        同图库相似图片只留下分辨率最高的图片，其余图片的所有
        指向与被指向关系先归并到最高分辨率根图片再删除。跨图库
        索引保留原 pid 和上传信息，但物理文件统一硬链接到全局根图片。
        已经是标准硬链接结构的组不再重复报告。
        """
        filter_gallery, groups = self.find_global_similarity_groups(name_or_alias)
        plans = []
        for group in groups:
            images_by_gallery: dict[str, list[ImageFeature]] = {}
            for image in group.images:
                images_by_gallery.setdefault(image.gallery, []).append(image)

            retained = {
                select_best_image(images).identity
                for images in images_by_gallery.values()
            }
            root = group.keep
            if root.pid is None or root.identity not in retained:
                logger.warning(f'NSY 全局去重组根节点无效: {root.identity}')
                continue
            retained_pids = tuple(sorted(
                image.pid for image in group.images
                if image.identity in retained and image.pid is not None
            ))
            delete_pids = tuple(sorted(
                image.pid for image in group.images
                if image.identity not in retained and image.pid is not None
            ))
            root_image = self.images_by_pid[root.pid]
            root_path = self._image_path(root_image)
            root_ext = self._dedup_root_extension(root_image)
            relink_pids = []
            metadata_changed = False
            expected_linked_pids = sorted(pid for pid in retained_pids if pid != root.pid)

            if root_image.linked_to_pid is not None or root_image.linked_pids != expected_linked_pids:
                metadata_changed = True
            for pid in expected_linked_pids:
                image = self.images_by_pid[pid]
                path = self._image_path(image)
                target_filename = f'{pid}{root_ext}'
                try:
                    same_file = osp.samefile(path, root_path)
                except OSError:
                    same_file = False
                if image.filename != target_filename or not same_file:
                    relink_pids.append(pid)
                if (
                    image.hash != root_image.hash
                    or image.format != root_image.format
                    or image.phash != root_image.phash
                    or image.size != root_image.size
                    or image.width != root_image.width
                    or image.height != root_image.height
                    or image.linked_to_pid != root.pid
                    or image.linked_pids
                ):
                    metadata_changed = True

            if not delete_pids and not relink_pids and not metadata_changed:
                continue

            relink_pid_set = set(relink_pids)
            annotated_images = []
            for image in group.images:
                if image.identity == root.identity:
                    action = '保留根文件'
                elif image.identity in retained:
                    action = (
                        '保留索引 · 链接'
                        if image.pid in relink_pid_set else
                        '保留索引 · 已是链接'
                    )
                else:
                    action = '归并链接后删除图片'
                annotated_images.append(replace(image, action=action))
            action_order = {
                '保留根文件': 0,
                '保留索引 · 链接': 1,
                '保留索引 · 已是链接': 1,
                '归并链接后删除图片': 2,
            }
            annotated_images.sort(key=lambda image: (
                action_order[image.action], image.gallery, image.pid or 0,
            ))
            annotated_root = next(
                image for image in annotated_images
                if image.identity == root.identity
            )
            plans.append(GlobalDedupPlan(
                group=SimilarityGroup(tuple(annotated_images), annotated_root),
                root_pid=root.pid,
                retained_pids=retained_pids,
                relink_pids=tuple(relink_pids),
                delete_pids=delete_pids,
            ))
        return filter_gallery, plans

    def _get_link_root(self, image: NsyImage) -> NsyImage | None:
        """沿 linked_to_pid 找到根文件；断链或循环索引返回 None。"""
        seen: set[int] = set()
        while image.linked_to_pid is not None:
            if image.pid in seen:
                return None
            seen.add(image.pid)
            image = self.images_by_pid.get(image.linked_to_pid)
            if image is None:
                return None
        return image

    @staticmethod
    def _dedup_root_extension(root: NsyImage) -> str:
        """返回去重根文件的规范扩展名。"""
        ext = IMAGE_EXT_BY_FORMAT.get(str(root.format).upper())
        if ext is None:
            ext = Path(root.filename).suffix.lower()
        if not ext:
            raise ReplyException(f'图片 pid={root.pid} 缺少可用扩展名')
        return ext

    def _validate_hardlink_index(
        self,
        images_by_pid: dict[int, NsyImage] | None = None,
    ):
        """
        校验索引硬链接关系与真实 inode 分组完全一致。

        每个多路径 inode 组必须恰好有一个根，根的 ``linked_pids``
        与所有其他成员完全一致，子节点必须直接指向根。单路径
        inode 不得携带链接元数据。任一路径缺失都视为校验失败。
        """
        images_by_pid = images_by_pid if images_by_pid is not None else self.images_by_pid
        components: dict[tuple[int, int], list[NsyImage]] = {}
        for image in images_by_pid.values():
            path = self._image_path(image)
            if not osp.exists(path):
                raise ReplyException(f'链接校验失败: pid={image.pid} 文件不存在')
            stat = os.stat(path)
            components.setdefault((stat.st_dev, stat.st_ino), []).append(image)

        for members in components.values():
            members.sort(key=lambda image: image.pid)
            roots = [image for image in members if image.linked_to_pid is None]
            if len(members) == 1:
                image = members[0]
                if len(roots) != 1 or image.linked_pids:
                    raise ReplyException(
                        f'链接校验失败: 单文件 pid={image.pid} 携带链接关系'
                    )
                continue
            if len(roots) != 1:
                raise ReplyException(
                    '链接校验失败: inode 组根数量不是 1，'
                    f'pids={[image.pid for image in members]}'
                )
            root = roots[0]
            children = [image for image in members if image.pid != root.pid]
            expected_child_pids = [image.pid for image in children]
            if root.linked_pids != expected_child_pids:
                raise ReplyException(
                    f'链接校验失败: 根 pid={root.pid} 的反向索引不完整'
                )
            for child in children:
                if child.linked_to_pid != root.pid or child.linked_pids:
                    raise ReplyException(
                        f'链接校验失败: 子 pid={child.pid} 未直接指向根 pid={root.pid}'
                    )

    def add_image_from_file(
        self,
        name_or_alias: str,
        src_path: str,
        uploader_id: int,
        group_id: int,
        info: ImageInfo | None = None,
        image_hash: str | None = None,
        image_phash: str | None = None,
        prechecked_hash_matches: list[NsyImage] | None = None,
        preferred_link_root: NsyImage | None = None,
    ) -> tuple[NsyImage | None, NsyImage | None]:
        """
        校验并阻止同一图库加入完全相同文件。

        其他图库已有相同 SHA-256 时创建硬链接；上层也可通过
        ``preferred_link_root`` 指定跨图库 pHash 相似图片的硬链接根。
        指定根时表示上层已完成 pHash 比较并选定根，本方法只校验根有效性
        并执行硬链接，不再扫描全图库。
        ``prechecked_hash_matches`` 可传入上层已完成的全库 SHA-256
        比对快照，避免落盘时再扫描一次全图库。
        创建链接时索引的文件特征以根文件为准，上传者等索引信息仍独立保留。

        返回 (新增图片, 重复图片)，二者只会有一个非 None。
        """
        gallery = self.resolve_gallery(name_or_alias, raise_if_missing=True)
        info = info or _inspect_image_file(src_path)
        image_hash = image_hash or _sha256_file(src_path)
        image_phash = normalize_phash(image_phash)
        link_root = None
        if preferred_link_root is not None:
            preferred_link_root = self.images_by_pid.get(preferred_link_root.pid)
            if preferred_link_root is None:
                raise ReplyException('pHash 链接根索引已失效')
            link_root = self._get_link_root(preferred_link_root)
            if link_root is None or not osp.exists(self._image_path(link_root)):
                raise ReplyException('pHash 链接根文件已失效')
        else:
            match_candidates = (
                self.find_images_by_hash(image_hash)
                if prechecked_hash_matches is None
                else prechecked_hash_matches
            )
            # 快照中的索引可能在落盘前被删除，因此只重新解析
            # 已命中的 pid，不再枚举整个图库。
            matches = [
                current
                for image in match_candidates
                if (current := self.images_by_pid.get(image.pid)) is not None
                and current.hash == image_hash
                and osp.exists(self._image_path(current))
            ]
            if duplicated := next(
                (image for image in matches if image.gallery == gallery),
                None,
            ):
                return None, duplicated
            for matched in matches:
                root = self._get_link_root(matched)
                if root is not None and osp.exists(self._image_path(root)):
                    link_root = root
                    break

        # 精确 hash 或上层 pHash 已选定链接根时，索引特征直接继承根文件；
        # 只有确实需要保存上传原图时才计算其 pHash。
        if link_root is None and image_phash is None:
            image_phash = compute_phash(src_path)

        pid = self._allocate_pid()
        ext = self._dedup_root_extension(link_root) if link_root else info.ext
        filename = f'{pid}{ext}'
        dst_path = create_parent_folder(pjoin(self._gallery_dir(gallery), filename))
        if link_root is None:
            shutil.copy2(src_path, dst_path)
        else:
            try:
                os.link(self._image_path(link_root), dst_path)
            except OSError as e:
                raise ReplyException(f'创建链接失败: {get_exc_desc(e)}')

        image = NsyImage(
            pid=pid,
            gallery=gallery,
            filename=filename,
            hash=link_root.hash if link_root else image_hash,
            format=link_root.format if link_root else info.fmt,
            phash=link_root.phash if link_root else image_phash,
            size=link_root.size if link_root else info.size,
            width=link_root.width if link_root else info.width,
            height=link_root.height if link_root else info.height,
            created_at=_now_str(),
            uploader_id=int(uploader_id),
            group_id=int(group_id),
            linked_to_pid=link_root.pid if link_root else None,
        )
        self.images_by_gallery.setdefault(gallery, {})[pid] = image
        self.images_by_pid[pid] = image
        if link_root is not None:
            link_root.linked_pids = sorted(set(link_root.linked_pids) | {pid})
        self._save_index()
        return image, None

    def random_image(self, gallery: str) -> NsyImage:
        """
        从规范图库中按格式、链接关系和近期发送历史共同计算权重后随机取图。

        近期历史按图片哈希记录，因此图库重载导致 pid 变化时仍然有效，且相同
        内容的硬链接图片共享近期降权效果。
        """
        images = [
            image
            for image in self.images_by_gallery.get(gallery, {}).values()
            if osp.exists(self._image_path(image))
        ]
        if not images:
            raise ReplyException(f'图库"{gallery}"没有图片')
        format_weights, linked_weight = _get_random_weight_config()
        recent_multipliers = _get_recent_image_weight_multipliers()
        recent_hashes = self._get_random_image_history(gallery, len(recent_multipliers))
        recent_weight_by_hash = dict(zip(recent_hashes, recent_multipliers))
        weights = [
            _get_image_random_weight(image, format_weights, linked_weight)
            * recent_weight_by_hash.get(image.hash, 1.0)
            for image in images
        ]
        total_weight = sum(weights)
        if total_weight != total_weight or total_weight == float('inf') or total_weight <= 0:
            logger.warning(f'图库"{gallery}"随机权重总和无效，回退等概率随机')
            return random.choice(images)
        return random.choices(images, weights=weights, k=1)[0]

    def _get_random_image_history(self, gallery: str, limit: int) -> list[str]:
        """读取某图库最近发送过的不同图片哈希，按新到旧返回。"""
        history_by_gallery = file_db.get(RANDOM_HISTORY_DB_KEY, {})
        if not isinstance(history_by_gallery, dict):
            logger.warning('随机图片历史格式无效，本次忽略近期降权')
            return []
        raw_history = history_by_gallery.get(gallery, [])
        if not isinstance(raw_history, list):
            return []

        history = []
        for image_hash in raw_history:
            image_hash = str(image_hash)
            if image_hash and image_hash not in history:
                history.append(image_hash)
            if len(history) >= limit:
                break
        return history

    def record_random_image_sent(self, image: NsyImage):
        """在图片成功发送后，将其移到所属图库近期历史的首位并持久化。"""
        history_limit = len(_get_recent_image_weight_multipliers())
        history_by_gallery = file_db.get(RANDOM_HISTORY_DB_KEY, {})
        if not isinstance(history_by_gallery, dict):
            history_by_gallery = {}
        old_history = self._get_random_image_history(image.gallery, history_limit)
        history_by_gallery[image.gallery] = [image.hash] + [
            image_hash for image_hash in old_history
            if image_hash != image.hash
        ][:max(0, history_limit - 1)]
        file_db.set(RANDOM_HISTORY_DB_KEY, history_by_gallery)

    def _delete_image_records(self, pid: int) -> list[NsyImage]:
        """删除 pid；若它是硬链接根文件，则同时删除全部指向它的记录。"""
        image = self.find_image(pid, raise_if_missing=True)
        dependent_pids = set(image.linked_pids)
        dependent_pids.update(
            item.pid
            for item in self.images_by_pid.values()
            if item.linked_to_pid == pid
        )
        remove_pids = dependent_pids | {pid}
        removed = [
            self.images_by_pid[item_pid]
            for item_pid in sorted(remove_pids)
            if item_pid in self.images_by_pid
        ]

        # 删除普通链接时，从仍保留的根记录中移除反向引用。
        for item in removed:
            if item.linked_to_pid is None or item.linked_to_pid in remove_pids:
                continue
            root = self.images_by_pid.get(item.linked_to_pid)
            if root is not None:
                root.linked_pids = [linked_pid for linked_pid in root.linked_pids if linked_pid != item.pid]

        for item in removed:
            path = self._image_path(item)
            if osp.exists(path):
                os.remove(path)
            self.images_by_gallery.get(item.gallery, {}).pop(item.pid, None)
            self.images_by_pid.pop(item.pid, None)
        return removed

    def delete_image(self, pid: int) -> tuple[NsyImage, list[NsyImage]]:
        requested = self.find_image(pid, raise_if_missing=True)
        removed = self._delete_image_records(pid)
        self._save_index()
        return requested, removed

    def consolidate_global_dedup_plan(
        self,
        plan: GlobalDedupPlan,
    ) -> tuple[list[NsyImage], list[NsyImage]]:
        """
        执行一组全局去重计划，返回（删除索引，重建硬链接索引）。

        同图库低分辨率图片的指向和被指向关系会先归并到根图片，
        然后删除其文件与索引。跨图库保留索引的 pid、图库、上传者和
        创建时间不变，文件内容元数据统一为最高分辨率根文件。
        文件先移到同目录临时路径，硬链接
        和索引全部成功后才删除备份；中途失败时尽量原样回滚。
        """
        retained_pids = tuple(dict.fromkeys(plan.retained_pids))
        delete_pids = tuple(dict.fromkeys(plan.delete_pids))
        retained_pid_set = set(retained_pids)
        delete_pid_set = set(delete_pids)
        if plan.root_pid not in retained_pid_set:
            raise ReplyException('去重计划中的根 pid 未被保留')
        if retained_pid_set & delete_pid_set:
            raise ReplyException('去重计划中存在同时保留和删除的 pid')

        affected_pids = retained_pid_set | delete_pid_set
        records: dict[int, NsyImage] = {}
        feature_by_pid = {
            image.pid: image for image in plan.group.images
            if image.pid is not None
        }
        actual_by_inode: dict[tuple[int, int], tuple[ImageInfo, str, str]] = {}
        for pid in sorted(affected_pids):
            image = self.images_by_pid.get(pid)
            if image is None:
                raise ReplyException(f'去重计划已过期: pid={pid} 不存在')
            path = self._image_path(image)
            if not osp.exists(path):
                raise ReplyException(f'去重计划已过期: pid={pid} 文件不存在')
            feature = feature_by_pid.get(pid)
            if feature is None or (
                image.phash != feature.phash
                or image.width != feature.width
                or image.height != feature.height
            ):
                raise ReplyException(f'去重计划已过期: pid={pid} 特征已变化')
            try:
                stat = os.stat(path)
                inode = (stat.st_dev, stat.st_ino)
                if inode not in actual_by_inode:
                    actual_by_inode[inode] = (
                        _inspect_image_file(path, check_size=False),
                        _sha256_file(path),
                        compute_phash(path),
                    )
                info, actual_hash, actual_phash = actual_by_inode[inode]
            except Exception as e:
                raise ReplyException(
                    f'去重计划已过期: pid={pid} 无法复核真实文件: {get_exc_desc(e)}'
                )
            if (
                image.hash != actual_hash
                or image.phash != actual_phash
                or image.format != info.fmt
                or image.size != info.size
                or image.width != info.width
                or image.height != info.height
            ):
                raise ReplyException(f'去重计划已过期: pid={pid} 文件内容已变化')
            records[pid] = image

        retained_galleries = [records[pid].gallery for pid in retained_pids]
        if len(retained_galleries) != len(set(retained_galleries)):
            raise ReplyException('去重计划无效: 同一图库保留了多个索引')

        root = records[plan.root_pid]
        root_path = self._image_path(root)
        root_ext = self._dedup_root_extension(root)
        nonroot_pids = [pid for pid in retained_pids if pid != root.pid]
        expected_paths = {
            pid: pjoin(self._gallery_dir(records[pid].gallery), f'{pid}{root_ext}')
            for pid in nonroot_pids
        }

        # 确认时重新检查 inode，避免在五分钟窗口内重复改写已处理文件。
        relink_pids = []
        for pid in nonroot_pids:
            old_path = self._image_path(records[pid])
            try:
                same_file = osp.samefile(old_path, root_path)
            except OSError:
                same_file = False
            if osp.abspath(old_path) != osp.abspath(expected_paths[pid]) or not same_file:
                relink_pids.append(pid)

        staged_pids = sorted(delete_pid_set | set(relink_pids))
        original_paths = {pid: self._image_path(records[pid]) for pid in staged_pids}
        if len({osp.abspath(path) for path in original_paths.values()}) != len(original_paths):
            raise ReplyException('去重失败: 多个索引指向同一文件路径')
        for pid in relink_pids:
            target_path = expected_paths[pid]
            old_path = original_paths[pid]
            if osp.exists(target_path) and osp.abspath(target_path) != osp.abspath(old_path):
                raise ReplyException(
                    f'去重失败: 目标文件 {records[pid].gallery}/{Path(target_path).name} 已存在'
                )

        snapshots = {
            pid: replace(image, linked_pids=list(image.linked_pids))
            for pid, image in records.items()
        }
        staged: list[tuple[int, str, str]] = []
        created_links: list[str] = []

        def rollback():
            for path in reversed(created_links):
                try:
                    if osp.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f'去重回滚删除硬链接失败: {get_exc_desc(e)}')
            for _, original_path, temp_path in reversed(staged):
                try:
                    if osp.exists(temp_path):
                        os.rename(temp_path, original_path)
                except Exception as e:
                    logger.warning(f'去重回滚恢复文件失败: {get_exc_desc(e)}')
            for pid, snapshot in snapshots.items():
                self.images_by_pid[pid] = snapshot
                self.images_by_gallery.setdefault(snapshot.gallery, {})[pid] = snapshot
            try:
                self._save_index()
            except Exception as e:
                logger.warning(f'去重回滚恢复索引失败: {get_exc_desc(e)}')

        try:
            for pid in staged_pids:
                original_path = original_paths[pid]
                temp_path = pjoin(
                    osp.dirname(original_path),
                    f'.nsy_dedup_{pid}.tmp',
                )
                suffix = 0
                while osp.exists(temp_path):
                    suffix += 1
                    temp_path = pjoin(
                        osp.dirname(original_path),
                        f'.nsy_dedup_{pid}_{suffix}.tmp',
                    )
                os.rename(original_path, temp_path)
                staged.append((pid, original_path, temp_path))

            for pid in relink_pids:
                target_path = create_parent_folder(expected_paths[pid])
                os.link(root_path, target_path)
                created_links.append(target_path)

            root.linked_to_pid = None
            root.linked_pids = sorted(nonroot_pids)
            for pid in nonroot_pids:
                image = records[pid]
                image.filename = f'{pid}{root_ext}'
                image.hash = root.hash
                image.format = root.format
                image.phash = root.phash
                image.size = root.size
                image.width = root.width
                image.height = root.height
                image.linked_to_pid = root.pid
                image.linked_pids = []

            for pid in delete_pids:
                image = records[pid]
                self.images_by_gallery.get(image.gallery, {}).pop(pid, None)
                self.images_by_pid.pop(pid, None)
            self._validate_hardlink_index()
            self._save_index()
        except Exception as e:
            rollback()
            if isinstance(e, ReplyException):
                raise
            raise ReplyException(f'全局图像去重失败: {get_exc_desc(e)}')

        for _, _, temp_path in staged:
            try:
                if osp.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f'去重删除临时备份失败: {get_exc_desc(e)}')

        removed = [snapshots[pid] for pid in delete_pids]
        relinked = [self.images_by_pid[pid] for pid in relink_pids]
        return removed, relinked

    def delete_gallery(self, name_or_alias: str) -> tuple[str, list[int], list[str]]:
        gallery = self.resolve_gallery(name_or_alias, raise_if_missing=True)
        gallery_pids = sorted(self.images_by_gallery.get(gallery, {}))
        removed: dict[int, NsyImage] = {}
        for pid in gallery_pids:
            if pid not in self.images_by_pid:
                continue
            for image in self._delete_image_records(pid):
                removed[image.pid] = image
        removed_pids = sorted(removed)
        self.images_by_gallery.pop(gallery, None)

        removed_aliases = sorted(self.aliases_by_gallery.pop(gallery, []))
        for alias in removed_aliases:
            del self.gallery_by_alias[alias]

        gallery_dir = self._gallery_dir(gallery)
        if osp.isdir(gallery_dir):
            shutil.rmtree(gallery_dir)

        self._save_index()
        self._save_aliases()
        return gallery, removed_pids, removed_aliases

    def reload_from_disk(self) -> dict:
        """
        以 data/nsy 的实际目录同步 aliases 总表并重建图片索引。

        - 每个现有目录都会写入 aliases 总表，原有有效别名会保留；
        - 所有有效图片按 created_at 从早到晚全局重新编号为 1...N，并重命名为 pid.ext；
        - 索引中有、文件已不存在的图片会从索引移除；
        - 文件存在但索引缺失的图片会加入索引；
        - 已有索引直接复用 SHA-256 和 pHash，仅新物理文件计算特征；
        - 指向不存在图库的别名会被删除。
        """
        self._load()
        old_images = self.images_by_pid
        old_by_path = {
            osp.abspath(self._image_path(image)): image
            for image in old_images.values()
        }
        reload_created_at = _now_str()
        gallery_names = set(self._list_gallery_names())
        candidates: list[ReloadImageCandidate] = []
        invalid_files: list[str] = []

        for gallery in sorted(gallery_names):
            gallery_dir = Path(self._gallery_dir(gallery))
            paths = sorted(gallery_dir.iterdir(), key=self._reload_file_sort_key)
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    info = _inspect_image_file(str(path), check_size=False)
                except Exception as e:
                    invalid_files.append(f'{gallery}/{path.name}: {get_exc_desc(e)}')
                    continue
                stat = path.stat(follow_symlinks=False)
                old_image = old_by_path.get(osp.abspath(str(path)))
                candidates.append(ReloadImageCandidate(
                    gallery=gallery,
                    path=path,
                    info=info,
                    old_image=old_image,
                    inode=(stat.st_dev, stat.st_ino),
                    created_at=(
                        old_image.created_at
                        if old_image is not None else reload_created_at
                    ),
                ))

        candidates.sort(key=self._reload_candidate_sort_key)
        renamed_files = self._renumber_reload_files(candidates)
        new_images_by_gallery: dict[str, dict[int, NsyImage]] = {
            gallery: {} for gallery in gallery_names
        }
        new_images_by_pid: dict[int, NsyImage] = {}
        added_pids: list[int] = []

        # 同一 inode 的多个目录项即为硬链接：最小 pid 作为根，其余均直接指向根。
        root_pid_by_inode: dict[tuple[int, int], int] = {}
        linked_to_by_pid: dict[int, int] = {}
        linked_pids_by_root: dict[int, list[int]] = {}
        for candidate in candidates:
            root_pid = root_pid_by_inode.setdefault(candidate.inode, candidate.pid)
            if root_pid != candidate.pid:
                linked_to_by_pid[candidate.pid] = root_pid
                linked_pids_by_root.setdefault(root_pid, []).append(candidate.pid)

        # index.json 是已有图片特征的信任源。先按 inode 收集
        # 旧索引特征，使新发现的硬链接也能直接复用根文件特征。
        feature_by_inode: dict[tuple[int, int], tuple[str, str]] = {}
        for candidate in candidates:
            if candidate.old_image is not None:
                feature_by_inode.setdefault(
                    candidate.inode,
                    (candidate.old_image.hash, candidate.old_image.phash),
                )

        for candidate in candidates:
            if candidate.inode not in feature_by_inode:
                # 只有无任何旧索引的新物理文件需要计算特征；
                # 同 inode 的多个新硬链接仍只计算一次。
                feature_by_inode[candidate.inode] = (
                    _sha256_file(str(candidate.path)),
                    compute_phash(candidate.path),
                )
            image_hash, image_phash = feature_by_inode[candidate.inode]
            old_image = candidate.old_image
            image = NsyImage(
                pid=candidate.pid,
                gallery=candidate.gallery,
                filename=candidate.path.name,
                hash=image_hash,
                format=candidate.info.fmt,
                phash=image_phash,
                size=candidate.info.size,
                width=candidate.info.width,
                height=candidate.info.height,
                created_at=candidate.created_at,
                uploader_id=old_image.uploader_id if old_image else None,
                group_id=old_image.group_id if old_image else None,
                linked_to_pid=linked_to_by_pid.get(candidate.pid),
                linked_pids=linked_pids_by_root.get(candidate.pid, []),
            )
            new_images_by_gallery[candidate.gallery][candidate.pid] = image
            new_images_by_pid[candidate.pid] = image
            if old_image is None:
                added_pids.append(candidate.pid)

        retained_old_pids = {
            candidate.old_image.pid
            for candidate in candidates
            if candidate.old_image is not None
        }
        removed_pids = sorted(set(old_images) - retained_old_pids)
        removed_aliases = sorted(
            alias
            for gallery, aliases in self.aliases_by_gallery.items()
            if gallery not in gallery_names
            for alias in aliases
        )
        for alias in removed_aliases:
            del self.gallery_by_alias[alias]
        self.aliases_by_gallery = {
            gallery: self.aliases_by_gallery.get(gallery, [])
            for gallery in gallery_names
        }

        self._validate_hardlink_index(new_images_by_pid)
        self.images_by_gallery = new_images_by_gallery
        self.images_by_pid = new_images_by_pid
        self.pid_top = len(self.images_by_pid)
        self._save_index()
        self._save_aliases()
        return {
            'added_pids': sorted(added_pids),
            'removed_pids': removed_pids,
            'renamed_files': renamed_files,
            'invalid_files': invalid_files,
            'removed_aliases': removed_aliases,
            'gallery_count': len(gallery_names),
            'image_count': len(self.images_by_pid),
        }

    def get_image_path(self, image: NsyImage) -> str:
        return self._image_path(image)


def _get_image_temp_file(ctx: HandlerContext, image_data: dict):
    """根据消息图片数据创建统一的临时文件上下文。"""
    url = image_data.get('url')
    file = image_data.get('file')
    if url:
        return TempDownloadFilePath(url, 'img')
    if file:
        return TempBotOrInternetFilePath('image', file, ctx.bot)
    raise ReplyException('图片消息缺少可下载地址')


async def _batch_add_images(
    ctx: HandlerContext,
    manager: NsyManager,
    galleries: list[str],
    image_datas: list[dict],
    check_phash: bool = True,
) -> list[GalleryUploadResult]:
    """
    每张新图只下载并计算一次特征，再与全图库现有索引比较。

    同图库 pHash 命中时拒绝；仅跨图库命中时立即创建硬链接。
    加图不刷新现有图片特征；强制模式仅跳过 pHash 相似检查。
    """
    results = [GalleryUploadResult(gallery=gallery) for gallery in galleries]
    # 同一批次中，同一图库只保留一个指向相同根的 pHash 链接。
    phash_link_by_target: dict[tuple[str, int], SimilarUploadLink] = {}
    for idx, image_data in enumerate(image_datas, 1):
        processed_results: set[int] = set()
        report_items_for_image: list[
            SimilarUploadRejection | SimilarUploadLink
        ] = []
        try:
            temp = _get_image_temp_file(ctx, image_data)
            async with temp as path:
                info = _inspect_image_file(path)
                image_hash = _sha256_file(path)
                # 先完成全图库 SHA-256 查询；只在精确 hash 零命中时
                # 才计算并检查 pHash，后续目标均复用本次查询结果。
                hash_matches = [
                    image
                    for image in manager.find_images_by_hash(image_hash)
                    if osp.exists(manager.get_image_path(image))
                ]
                image_phash = None
                similar_images = []
                if not hash_matches:
                    image_phash = await run_in_pool(compute_phash, path)
                    if check_phash:
                        similar_images = await run_in_pool(
                            manager.find_global_similar_images,
                            image_phash,
                        )
                for result_idx, result in enumerate(results):
                    try:
                        duplicated = next((
                            image
                            for image in hash_matches
                            if image.gallery == result.gallery
                        ), None)
                        if duplicated is not None:
                            result.repeats.append((idx, duplicated))
                            continue

                        same_gallery_similar = [
                            (image, distance)
                            for image, distance in similar_images
                            if image.gallery == result.gallery
                        ]
                        if same_gallery_similar:
                            rejection = SimilarUploadRejection(
                                upload_index=idx,
                                gallery=result.gallery,
                                staged_path='',
                                info=info,
                                phash=image_phash,
                                similar_images=same_gallery_similar,
                            )
                            result.similar_rejections.append(rejection)
                            report_items_for_image.append(rejection)
                            continue

                        # SHA-256 全局零命中且仅在其他图库命中 pHash 时，
                        # 才指定 pHash 根文件。
                        phash_link_root = None
                        if check_phash and similar_images:
                            phash_link_root = manager.select_best_phash_link_root(
                                similar_images,
                            )

                        if phash_link_root is not None:
                            link_target = (result.gallery, phash_link_root.pid)
                            previous_link = phash_link_by_target.get(link_target)
                            if previous_link is not None:
                                result.errors.append(
                                    f'第{idx}张: 与本批第{previous_link.upload_index}张'
                                    '会创建相同硬链接，已跳过'
                                )
                                continue
                            image, duplicated = manager.add_image_from_file(
                                result.gallery,
                                path,
                                uploader_id=ctx.user_id,
                                group_id=ctx.group_id or 0,
                                info=info,
                                image_hash=image_hash,
                                image_phash=image_phash,
                                preferred_link_root=phash_link_root,
                            )
                            if duplicated is not None:
                                result.repeats.append((idx, duplicated))
                                continue
                            if image is None:
                                raise ReplyException('创建链接失败')
                            link = SimilarUploadLink(
                                upload_index=idx,
                                gallery=result.gallery,
                                staged_path='',
                                info=info,
                                image_hash=image_hash,
                                phash=image_phash,
                                link_root_pid=phash_link_root.pid,
                                linked_pid=image.pid,
                                similar_images=list(similar_images),
                            )
                            phash_link_by_target[link_target] = link
                            result.added.append(image)
                            result.phash_links.append(link)
                            report_items_for_image.append(link)
                            continue

                        image, duplicated = manager.add_image_from_file(
                            result.gallery,
                            path,
                            uploader_id=ctx.user_id,
                            group_id=ctx.group_id or 0,
                            info=info,
                            image_hash=image_hash,
                            image_phash=image_phash,
                            prechecked_hash_matches=hash_matches,
                        )
                        if image:
                            result.added.append(image)
                            # 同一条命令可向多个图库加图；把刚新增的
                            # 索引加入快照，后续图库可直接创建硬链接。
                            hash_matches.append(image)
                        elif duplicated:
                            result.repeats.append((idx, duplicated))
                    except ReplyException as e:
                        result.errors.append(f'第{idx}张: {get_exc_desc(e)}')
                    except Exception as e:
                        logger.print_exc(f'添加第{idx}张图片到图库"{result.gallery}"失败')
                        result.errors.append(f'第{idx}张: {get_exc_desc(e)}')
                    finally:
                        processed_results.add(result_idx)

                if report_items_for_image:
                    with TempFilePath(
                        info.ext,
                        remove_after=timedelta(minutes=5),
                    ) as staged_path:
                        shutil.copy2(path, staged_path)
                        for report_item in report_items_for_image:
                            report_item.staged_path = staged_path
        except ReplyException as e:
            for result_idx, result in enumerate(results):
                if result_idx not in processed_results:
                    result.errors.append(f'第{idx}张: {get_exc_desc(e)}')
        except Exception as e:
            logger.print_exc(f'下载或校验第{idx}张图片失败')
            for result_idx, result in enumerate(results):
                if result_idx not in processed_results:
                    result.errors.append(f'第{idx}张: {get_exc_desc(e)}')
    return results


def _upload_similarity_feature(
    item: SimilarUploadRejection | SimilarUploadLink,
) -> ImageFeature:
    """把 pHash 拒绝/自动链接项转换为对比报告的距离基准。"""
    is_phash_link = isinstance(item, SimilarUploadLink)
    return ImageFeature(
        identity=f'upload:{type(item).__name__}:{item.gallery}:{item.upload_index}',
        gallery=item.gallery,
        path=item.staged_path,
        phash=item.phash,
        width=item.info.width,
        height=item.info.height,
        pending=True,
        action=(
            f'已链接 · pid={item.linked_pid}'
            if is_phash_link else '已拒绝添加'
        ),
    )


def _build_upload_similarity_groups(
    manager: NsyManager,
    rejections: list[SimilarUploadRejection],
    phash_links: list[SimilarUploadLink],
) -> list[SimilarityGroup]:
    """为拒绝/自动链接项构建“待上传图 + 已有相似图”对比组。"""
    groups = []
    for item in [*rejections, *phash_links]:
        candidate = _upload_similarity_feature(item)
        features = [candidate]
        features.extend(
            manager._to_image_feature(image)
            for image, _ in item.similar_images
            if manager.find_image(image.pid) is not None
            and osp.exists(manager.get_image_path(image))
        )
        if len(features) >= 2:
            groups.append(SimilarityGroup(tuple(features), candidate))
    return groups


def _cancel_phash_link(
    manager: NsyManager,
    link: SimilarUploadLink,
) -> NsyImage:
    """删除一项仍保持原关系的跨图库 pHash 子链接。"""
    image = manager.find_image(link.linked_pid)
    if image is None:
        raise ReplyException(f'pid={link.linked_pid} 已不存在')
    if (
        image.gallery != link.gallery
        or image.linked_to_pid != link.link_root_pid
        or image.linked_pids
    ):
        raise ReplyException(f'pid={link.linked_pid} 的链接关系已变化')
    requested, _ = manager.delete_image(link.linked_pid)
    return requested


def _parse_one_arg(ctx: HandlerContext, usage: str) -> str:
    args = ctx.get_args().strip().split()
    if len(args) != 1:
        raise ReplyException(f'使用方式: {usage}')
    return args[0]


def _parse_many_args(ctx: HandlerContext, usage: str) -> list[str]:
    args = ctx.get_args().strip().split()
    if not args:
        raise ReplyException(f'使用方式: {usage}')
    return args


def _parse_two_args(ctx: HandlerContext, usage: str) -> tuple[str, str]:
    args = ctx.get_args().strip().split()
    if len(args) != 2:
        raise ReplyException(f'使用方式: {usage}')
    return args[0], args[1]


def _parse_dedup_group_selection(args_text: str, group_count: int) -> list[int]:
    """
    解析去重确认的相似组编号。

    空参数代表确认全部组；传入组号时去重并按报告顺序执行。
    不合法组号直接拒绝，由通用确认机制保留原待确认操作供重试。
    """
    tokens = args_text.strip().split()
    if not tokens:
        return list(range(1, group_count + 1))

    selected = set()
    for token in tokens:
        try:
            group_number = int(token)
        except ValueError:
            raise ReplyException(
                f'相似组编号“{token}”无效，请使用：/确认 1 2 4'
            )
        if not 1 <= group_number <= group_count:
            raise ReplyException(
                f'相似组编号 {group_number} 超出范围 1~{group_count}'
            )
        selected.add(group_number)
    return sorted(selected)


def _batch_create_galleries(
    manager: NsyManager,
    names: list[str],
) -> list[tuple[str, bool, str]]:
    """按输入顺序逐个建库，单项失败不阻断后续图库。"""
    results = []
    for name in names:
        try:
            gallery = manager.create_gallery(name)
            results.append((gallery, True, '创建成功'))
        except ReplyException as e:
            results.append((name, False, get_exc_desc(e)))
        except Exception as e:
            logger.print_exc(f'创建图库"{name}"失败')
            results.append((name, False, get_exc_desc(e)))
    return results


def _resolve_gallery_upload_targets(
    manager: NsyManager,
    gallery_tokens: list[str],
) -> list[GalleryUploadTarget]:
    """逐项索引加图目标；单项失败不阻断其他有效图库。"""
    targets = []
    seen_galleries = set()
    for token in gallery_tokens:
        try:
            gallery = manager.resolve_gallery(token, raise_if_missing=True)
            if gallery in seen_galleries:
                raise ReplyException(f'图库"{gallery}"重复输入')
            seen_galleries.add(gallery)
            targets.append(GalleryUploadTarget(token=token, gallery=gallery))
        except ReplyException as e:
            targets.append(
                GalleryUploadTarget(token=token, error=get_exc_desc(e))
            )
    return targets


def _nsy_cmd(
    commands: str | list[str],
    *,
    superuser: bool = False,
    priority: int = 0,
    force_whitespace: str | bool | None = None,
) -> CmdHandler:
    handler = CmdHandler(
        commands,
        logger,
        priority=priority,
        force_whitespace=force_whitespace,
    )
    handler.check_cdrate(cd).check_wblist(gbl)
    if superuser:
        handler.check_superuser()
    return handler


# ======================= 指令处理 ======================= #

async def _handle_nsy_add(ctx: HandlerContext, force: bool = False):
    """处理普通/强制加图；强制模式只跳过 pHash，不跳过 SHA-256。"""
    command = '/强制加图' if force else '/加图'
    gallery_tokens = _parse_many_args(ctx, f'{command} 图库名称/别名... [图片]')
    manager = NsyManager.get()
    targets = _resolve_gallery_upload_targets(manager, gallery_tokens)
    galleries = [target.gallery for target in targets if target.gallery is not None]
    for gallery in galleries:
        await ctx.block(f'nsy:add:{gallery}')

    image_datas = []
    results = []
    if galleries:
        image_datas = await ctx.aget_image_datas(
            max_count=int(config.get('max_upload_images', 20)),
        )
        results = await _batch_add_images(
            ctx,
            manager,
            galleries,
            image_datas,
            check_phash=not force,
        )
    result_by_gallery = {result.gallery: result for result in results}

    sections = []
    for target in targets:
        if target.gallery is None:
            _append_operation_log(
                ctx.event,
                (
                    f'{"强制加图" if force else "加图"} '
                    f'gallery_token="{target.token}" status=failed '
                    f'error="{target.error}"'
                ),
            )
            sections.append(
                f'图库"{target.token}": 索引失败\n{target.error}'
            )
            continue

        result = result_by_gallery[target.gallery]
        _append_operation_log(
            ctx.event,
            (
                f'{"强制加图" if force else "加图"} gallery="{result.gallery}" '
                f'added={[image.pid for image in result.added]} '
                f'repeated={[dup.pid for _, dup in result.repeats]} '
                f'phash_rejected={len(result.similar_rejections)} '
                f'phash_linked={[link.linked_pid for link in result.phash_links]} '
                f'failed={len(result.errors)}'
            ),
        )
        sections.append(_format_upload_result(
            result.gallery,
            len(image_datas),
            result.added,
            result.repeats,
            result.similar_rejections,
            result.phash_links,
            result.errors,
            manager.images_by_pid,
        ))
    msg = '\n\n'.join(sections)
    rejections = [
        rejection
        for result in results
        for rejection in result.similar_rejections
    ]
    phash_links = [
        link
        for result in results
        for link in result.phash_links
    ]
    if not rejections and not phash_links:
        return await ctx.asend_fold_msg_adaptive(msg)

    report_items = [*rejections, *phash_links]
    staged_paths = {item.staged_path for item in report_items}
    try:
        groups = _build_upload_similarity_groups(
            manager,
            rejections,
            phash_links,
        )
        report_pages = await render_similarity_report(
            groups,
            '加图相似检查',
            show_decisions=False,
        )
        for page in report_pages:
            msg += '\n' + await get_image_cq(page, low_quality=True)
    except Exception as e:
        logger.print_exc('生成加图检查报告失败')
        msg += f'\n相似图对比报告生成失败: {get_exc_desc(e)}'
    finally:
        for path in staged_paths:
            remove_file(path)
    force_usage = '/强制加图 ' + ' '.join(gallery_tokens) + ' [图片]'
    msg += f'\n可使用：{force_usage} 跳过 pHash 检查。'
    if not phash_links:
        return await ctx.asend_fold_msg_adaptive(msg)
    msg += '\n回复“/取消”将删除跨图库相似图链接。'

    async def cancel_phash_actions(cancel_ctx: HandlerContext):
        affected_galleries = {link.gallery for link in phash_links}
        for gallery in sorted(affected_galleries):
            await cancel_ctx.block(f'nsy:add:{gallery}')

        removed: list[NsyImage] = []
        skipped = []
        for link in phash_links:
            try:
                removed.append(_cancel_phash_link(manager, link))
            except ReplyException as e:
                skipped.append(
                    f'图库“{link.gallery}”第{link.upload_index}张: {get_exc_desc(e)}'
                )
            except Exception as e:
                logger.print_exc(f'取消图库“{link.gallery}”pHash 链接失败')
                skipped.append(
                    f'图库“{link.gallery}”第{link.upload_index}张: {get_exc_desc(e)}'
                )

        _append_operation_log(
            cancel_ctx.event,
            (
                '取消pHash链接 '
                f'removed_links={[image.pid for image in removed]} '
                f'skipped={skipped}'
            ),
        )
        result_lines = [
            '已取消本次跨图库 pHash 链接',
            f'撤销链接: {len(removed)}/{len(phash_links)}',
        ]
        if removed:
            result_lines.append(
                '删除链接 pid: ' + ' '.join(str(image.pid) for image in removed)
            )
        if skipped:
            result_lines.append('未完成:')
            result_lines.extend(skipped)
        await cancel_ctx.asend_fold_msg_adaptive('\n'.join(result_lines))

    await add_cancellable_action(
        ctx,
        cancel_phash_actions,
        additional_msg=msg,
        timeout=timedelta(minutes=5),
    )


nsy_add = _nsy_cmd(['/加图'])
@nsy_add.handle()
async def _(ctx: HandlerContext):
    await _handle_nsy_add(ctx)


nsy_force_add = _nsy_cmd(['/强制加图'])
@nsy_force_add.handle()
async def _(ctx: HandlerContext):
    await _handle_nsy_add(ctx, force=True)


nsy_alias = _nsy_cmd(['/alias'])
@nsy_alias.handle()
async def _(ctx: HandlerContext):
    name = _parse_one_arg(ctx, '/alias 图库名称/别名')
    gallery, aliases = NsyManager.get().get_aliases(name)
    if not aliases:
        return await ctx.asend_reply_msg(f'图库"{gallery}"还没有别名')
    msg = f'图库"{gallery}"的别名: ' + '，'.join(aliases)
    await ctx.asend_fold_msg_adaptive(msg.strip())


nsy_create_gallery = _nsy_cmd(['/加图库'])
@nsy_create_gallery.handle()
async def _(ctx: HandlerContext):
    names = _parse_many_args(ctx, '/加图库 图库名称...')
    manager = NsyManager.get()
    results = _batch_create_galleries(manager, names)
    for name, success, message in results:
        if success:
            operation = f'加图库 gallery="{name}" status=success'
        else:
            operation = f'加图库 gallery="{name}" status=failed error="{message}"'
        _append_operation_log(ctx.event, operation)

    success_count = sum(success for _, success, _ in results)
    lines = [f'图库创建完成: {success_count}/{len(results)}']
    lines.extend(
        f'{idx}. 图库"{name}": {message}'
        for idx, (name, _, message) in enumerate(results, 1)
    )
    await ctx.asend_fold_msg_adaptive('\n'.join(lines))


nsy_add_alias = _nsy_cmd(['/add alias'], priority=1)
@nsy_add_alias.handle()
async def _(ctx: HandlerContext):
    name, alias = _parse_two_args(ctx, '/add alias 图库名称 别名')
    gallery, alias = NsyManager.get().add_alias(name, alias)
    _append_operation_log(ctx.event, f'add alias gallery="{gallery}" alias="{alias}"')
    await ctx.asend_reply_msg(f'图库"{gallery}"添加别名"{alias}"成功')


nsy_image_dedup = _nsy_cmd(['/图像去重', '/图库查重'], superuser=True)
@nsy_image_dedup.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().split()
    if len(args) > 1:
        raise ReplyException('使用方式: /图像去重 或 /图库查重 [图库名称/别名]')

    manager = NsyManager.get()
    gallery_token = args[0] if args else None
    distance_threshold = _get_phash_distance_threshold()
    await ctx.block('nsy:image-dedup')

    filter_gallery, plans = await run_in_pool(
        manager.build_global_dedup_plans,
        gallery_token,
    )
    scope = f'图库“{filter_gallery}”相关的全局图片' if filter_gallery else '全部图库'
    if not plans:
        return await ctx.asend_reply_msg(
            f'{scope}检查完成，未发现需要处理的相似图片'
            f'（Hamming distance 小于 {distance_threshold}）'
        )

    report_pages = await render_similarity_report(
        [plan.group for plan in plans],
        '全局图像去重检查',
    )
    delete_count = sum(len(plan.delete_pids) for plan in plans)
    relink_count = sum(len(plan.relink_pids) for plan in plans)
    msg = (
        f'{scope}发现 {len(plans)} 组需要处理的相似图片。'
        f'确认后将删除 {delete_count} 张图片，'
        f'重建 {relink_count} 个链接，'
    )
    for page in report_pages:
        msg += '\n' + await get_image_cq(page, low_quality=True)
    msg += (
        '\n引用本消息发送“/确认”处理全部相似组；'
        '发送“/确认 1 2 4”则只处理指定组。'
    )

    def validate_confirmation(confirm_ctx: HandlerContext):
        _parse_dedup_group_selection(confirm_ctx.get_args(), len(plans))

    async def confirm(new_ctx: HandlerContext):
        await new_ctx.block('nsy:image-dedup')
        selected_group_numbers = _parse_dedup_group_selection(
            new_ctx.get_args(),
            len(plans),
        )
        removed: list[NsyImage] = []
        relinked: list[NsyImage] = []
        completed_groups = 0
        skipped_errors = []
        for group_number in selected_group_numbers:
            plan = plans[group_number - 1]
            try:
                group_removed, group_relinked = manager.consolidate_global_dedup_plan(plan)
                removed.extend(group_removed)
                relinked.extend(group_relinked)
                completed_groups += 1
            except ReplyException as e:
                skipped_errors.append(f'第{group_number}组: {get_exc_desc(e)}')
        removed_pids = [image.pid for image in removed]
        relinked_pids = [image.pid for image in relinked]
        _append_operation_log(
            new_ctx.event,
            (
                f'pHash全局图像去重 filter_gallery={filter_gallery!r} '
                f'selected_groups={selected_group_numbers} '
                f'groups={completed_groups}/{len(selected_group_numbers)} removed={removed_pids} '
                f'relinked={relinked_pids} skipped={skipped_errors}'
            ),
        )
        result_msg = (
            f'全局图像去重完成 {completed_groups}/{len(selected_group_numbers)} 个已选组，'
            f'删除 {len(removed)} 张图片，'
            f'重建 {len(relinked)} 个链接'
        )
        unselected_count = len(plans) - len(selected_group_numbers)
        if unselected_count:
            result_msg += f'\n未选择的 {unselected_count} 组本次未处理'
        if removed:
            result_msg += '\n删除 pid: ' + ' '.join(str(image.pid) for image in removed)
        if relinked:
            result_msg += '\n重建链接 pid: ' + ' '.join(str(image.pid) for image in relinked)
        if skipped_errors:
            result_msg += f'\n跳过 {len(skipped_errors)} 组:'
            result_msg += '\n' + '\n'.join(skipped_errors)
        await new_ctx.asend_fold_msg_adaptive(result_msg)

    await add_need_confirm_action(
        ctx,
        confirm,
        additional_msg=msg,
        timeout=timedelta(minutes=5),
        allow_cancel_command=False,
        confirmation_validator=validate_confirmation,
    )


nsy_reload = _nsy_cmd(['/重载图库'], superuser=True)
@nsy_reload.handle()
async def _(ctx: HandlerContext):
    await ctx.block('nsy:reload')
    result = NsyManager.get().reload_from_disk()
    _append_operation_log(
        ctx.event,
        (
            '重载图库 '
            f'galleries={result["gallery_count"]} images={result["image_count"]} '
            f'added={result["added_pids"]} removed={result["removed_pids"]} '
            f'renamed={len(result["renamed_files"])} invalid={len(result["invalid_files"])} '
            f'removed_aliases={result["removed_aliases"]}'
        ),
    )
    await ctx.asend_fold_msg_adaptive(_format_reload_result(result))


nsy_delete_image = _nsy_cmd(['/删图'], superuser=True)
@nsy_delete_image.handle()
async def _(ctx: HandlerContext):
    pid_text = _parse_one_arg(ctx, '/删图 pid')
    try:
        pid = int(pid_text)
    except Exception:
        raise ReplyException('pid必须是整数')

    image, removed = NsyManager.get().delete_image(pid)
    removed_pids = [item.pid for item in removed]
    _append_operation_log(
        ctx.event,
        f'删图 pid={image.pid} gallery="{image.gallery}" removed_pids={removed_pids}',
    )
    msg = f'图片 pid={image.pid} 已从图库"{image.gallery}"删除'
    linked_pids = [removed_pid for removed_pid in removed_pids if removed_pid != image.pid]
    if linked_pids:
        msg += '\n同时删除关联链接: ' + ' '.join(f'pid={linked_pid}' for linked_pid in linked_pids)
    await ctx.asend_reply_msg(msg)


nsy_delete_gallery = _nsy_cmd(['/删图库'], superuser=True)
@nsy_delete_gallery.handle()
async def _(ctx: HandlerContext):
    name = _parse_one_arg(ctx, '/删图库 图库名称/别名')
    gallery, removed_pids, removed_aliases = NsyManager.get().delete_gallery(name)
    _append_operation_log(
        ctx.event,
        f'删图库 gallery="{gallery}" removed_pids={removed_pids} removed_aliases={removed_aliases}',
    )
    msg = f'图库"{gallery}"删除成功，删除图片{len(removed_pids)}张'
    if removed_aliases:
        msg += '\n同时删除别名: ' + '，'.join(removed_aliases)
    await ctx.asend_reply_msg(msg)


nsy_delete_alias = _nsy_cmd(['/del alias', '/delete alias', '/删除图集别名'], superuser=True)
@nsy_delete_alias.handle()
async def _(ctx: HandlerContext):
    name, alias = _parse_two_args(ctx, '/del alias 图库名称/别名 别名')
    gallery, alias = NsyManager.get().delete_alias(name, alias)
    _append_operation_log(ctx.event, f'del alias gallery="{gallery}" alias="{alias}"')
    await ctx.asend_reply_msg(f'图库"{gallery}"删除别名"{alias}"成功')


nsy_query_log = _nsy_cmd(['/查询图库日志'], superuser=True)
@nsy_query_log.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    if args:
        try:
            count = int(args)
        except Exception:
            raise ReplyException('日志条数必须是整数')
    else:
        count = int(config.get('default_log_query_count', 10))

    max_count = int(config.get('max_log_query_count', 50))
    if count <= 0:
        raise ReplyException('日志条数必须大于0')
    if count > max_count:
        raise ReplyException(f'一次最多查询{max_count}条日志')

    if not osp.exists(OPERATION_LOG_FILE):
        return await ctx.asend_reply_msg('暂无图库日志')

    with open(OPERATION_LOG_FILE, 'r', encoding='utf-8') as f:
        lines = [line for line in f.readlines() if line.strip()]
    if not lines:
        return await ctx.asend_reply_msg('暂无图库日志')

    selected = lines[-count:]
    msg = f'最近{len(selected)}条图库日志:\n'
    msg += '\n'.join(_format_log_line(line) for line in selected)
    await ctx.asend_fold_msg_adaptive(msg.strip())


# ======================= 无斜杠看图 ======================= #

nsy_pick = on_message(priority=20, block=False)
@nsy_pick.handle()
async def _(bot: Bot, event: MessageEvent):
    if not isinstance(event, MessageEvent):
        return
    if check_self(event):
        return
    if on_safe_mode() and not check_superuser(event):
        return
    if is_group_msg(event) and check_group_disabled(event.group_id):
        return
    if check_in_blacklist(event.user_id):
        return
    if not gbl.check(event, allow_private=True):
        return

    text = _extract_exact_gallery_text(event)
    if text is None:
        return

    manager = NsyManager.get()
    gallery = manager.resolve_gallery(text)
    if gallery is None:
        return
    if not check_send_msg_daily_limit(int(bot.self_id)) and not check_superuser(event):
        return
    if not await cd.check(event):
        return

    try:
        image = manager.random_image(gallery)
        await send_msg(
            nsy_pick,
            event,
            await get_image_cq(manager.get_image_path(image), send_url_as_is=True),
        )
        try:
            manager.record_random_image_sent(image)
        except Exception:
            # 图片已经发送成功，历史写入失败不应再向用户报告“发送图片失败”。
            logger.print_exc(f'记录 NSY 图库"{gallery}"随机发送历史失败')
    except ReplyException as e:
        await send_reply_msg(nsy_pick, event, get_exc_desc(e))
    except Exception as e:
        logger.print_exc(f'随机发送 NSY 图库"{gallery}"图片失败')
        await send_reply_msg(nsy_pick, event, f'发送图片失败: {get_exc_desc(e)}')


# 查询模块依赖本文件中的索引模型和命令工厂，必须在它们完成定义后加载。
from . import query as _query
