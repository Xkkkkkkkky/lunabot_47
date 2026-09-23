from ..utils import *
from .common import *
from .handler import *

# ======================= 群聊订阅 ======================= #

class SekaiGroupSubHelper:
    all_subs: List['SekaiGroupSubHelper'] = []

    def __init__(self, id: str, name: str, regions: List[str], hide: bool = False):
        self.id = id
        self.name = name
        self.regions = filter_enabled_server_regions(regions)
        self.hide = hide
        self.subs = {
            region: GroupSubHelper(
                f"{name}({get_region_name(region)})_群聊",
                file_db,
                logger,
            ) for region in self.regions
        }
        SekaiGroupSubHelper.all_subs.append(self)

    @classmethod
    def find_by_ids(cls, ids: List[str]) -> list['SekaiGroupSubHelper']:
        return find_by_predicate(cls.all_subs, lambda s: s.id in ids, mode='all')

    def do_sub(self, ctx: SekaiHandlerContext) -> str:
        if ctx.region not in self.subs:
            return f"❌ 群聊订阅 {self.name} 不支持服务器 {ctx.region}"
        self.subs[ctx.region].sub(ctx.group_id)
        return f"✅ 成功开启本群 {self.name}({get_region_name(ctx.region)})"
    
    def do_unsub(self, ctx: SekaiHandlerContext) -> str:
        if ctx.region not in self.subs:
            return f"❌ 群聊订阅 {self.name} 不支持服务器 {ctx.region}"
        self.subs[ctx.region].unsub(ctx.group_id)
        return f"✅ 成功关闭本群 {self.name}({get_region_name(ctx.region)})"
    
    @classmethod
    def do_get_sub(cls, ctx: SekaiHandlerContext) -> str:
        msg = "当前群聊开启:\n"
        has_sub = False
        for sub in cls.all_subs:
            sub_regions = []
            for region in sub.regions:
                if sub.is_subbed(region, ctx.group_id):
                    sub_regions.append(region)
            if sub_regions:
                msg += f"{sub.name}({', '.join(sub_regions)})\n"
                has_sub = True
        if not has_sub:
            msg += "无\n"
        msg += "---\n"
        msg += "所有可开启项目:\n"
        for sub in cls.all_subs:
            if not sub.hide and sub.regions:
                msg += f"{sub.id}: {sub.name}({', '.join(sub.regions)})\n"
        msg += "---\n"
        msg += "使用\"/pjsk开启 英文项目名\"开启订阅\n"
        return msg.strip()

    def _check_region(self, region):
        if region not in self.regions:
            raise Exception(f"群聊订阅 {self.name} 不支持服务器 {region}")

    def is_subbed(self, region, group_id):
        self._check_region(region)
        return self.subs[region].is_subbed(group_id)

    def sub(self, region, group_id):
        self._check_region(region)
        return self.subs[region].sub(group_id)

    def unsub(self, region, group_id):
        self._check_region(region)
        return self.subs[region].unsub(group_id)

    def get_all(self, region):
        self._check_region(region)
        return self.subs[region].get_all()

    def clear(self, region):
        self._check_region(region)
        return self.subs[region].clear()


sekai_group_sub = SekaiCmdHandler([
    "/pjsk group sub", "/pjsk 群订阅", "/pjsk 群聊订阅", "/pjsk 开启",
], command_key='misc.sekai_group_sub')
sekai_group_sub.check_cdrate(cd).check_wblist(gwl).check_superuser()
@sekai_group_sub.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip()
    msg = ""
    subs = None
    # 提取订阅项目
    if args:
        if not (subs := SekaiGroupSubHelper.find_by_ids(args.split())):
            msg += f"无效的订阅项目: {args}\n"
    if not subs:
        # 无订阅项目则列出所有订阅状态
        msg += SekaiGroupSubHelper.do_get_sub(ctx)
    else:
        # 订阅指定项目
        for sub in subs:
            msg += sub.do_sub(ctx) + "\n"
    return await ctx.asend_reply_msg(msg.strip())


sekai_group_unsub = SekaiCmdHandler([
    "/pjsk group unsub", "/pjsk 取消群订阅", "/pjsk 取消群聊订阅", "/pjsk 关闭",
], command_key='misc.sekai_group_unsub')
sekai_group_unsub.check_cdrate(cd).check_wblist(gwl).check_superuser()
@sekai_group_unsub.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip()
    msg = ""
    subs = None
    # 提取订阅项目
    if args:
        if not (subs := SekaiGroupSubHelper.find_by_ids(args.split())):
            msg += f"无效的订阅项目: {args}\n"
    if not subs:
        # 无订阅项目则列出所有订阅状态
        msg += SekaiGroupSubHelper.do_get_sub(ctx)
    else:
        # 取消订阅指定项目
        for sub in subs:
            msg += sub.do_unsub(ctx) + "\n"
    return await ctx.asend_reply_msg(msg.strip())
