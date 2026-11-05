import re

from ..utils import (
    CmdHandler,
    ColdDown,
    Config,
    HandlerContext,
    ReplyException,
    get_file_db,
    get_group_white_list,
    get_logger,
    run_in_pool,
)
from .algorithm import SearchLimits, generate_polynomial


logger = get_logger('Kyn')
config = Config('kyn')
file_db = get_file_db('data/kyn/db.json', logger)
gwl = get_group_white_list(file_db, logger, 'kyn')
cd = ColdDown(file_db, logger)

query = CmdHandler('/947', logger).check_wblist(gwl, allow_super=False).check_cdrate(cd)


@query.handle()
async def query_polynomial(ctx: HandlerContext):
    args = ctx.get_args().strip()
    if not re.fullmatch(r'[+-]?[0-9]+', args):
        raise ReplyException('用法：/947 <整数>，例如 /947 123 或 /947 -123')

    try:
        number = int(args)
    except ValueError as exc:
        raise ReplyException('整数过长，无法处理。') from exc

    # 按当前算法分组读取搜索参数，复用配置热加载。
    try:
        algorithm_name = config.get('algorithm')
        if algorithm_name != 'dp':
            raise ValueError('当前仅支持 dp 算法')
        parameters = config.get(('algorithms', algorithm_name))
        if not isinstance(parameters, dict):
            raise ValueError('algorithms.dp 必须是参数对象')
        limits = SearchLimits(**parameters)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplyException(f'947 计算配置无效：{exc}') from exc

    # 将 CPU 密集的表达式搜索移出事件循环，避免阻塞其他消息处理。
    result = await run_in_pool(generate_polynomial, number, limits)
    await ctx.asend_reply_msg(f"{number} = {result['expression']}  [[{result['elapsed_ms']:.3f} ms]]")
