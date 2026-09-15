from ..utils import *
from .common import *
from .handler import *

gameapi_config = Config('sekai.gameapi')

@dataclass
class GameApiConfig:
    api_status_url: Optional[str] = None
    profile_api_url: Optional[str] = None 
    suite_api_url: Optional[str] = None
    ranking_top100_api_url: Optional[str] = None
    ranking_border_api_url: Optional[str] = None


# 获取游戏api相关配置
def get_gameapi_config(ctx: SekaiHandlerContext) -> GameApiConfig:
    return GameApiConfig(**(gameapi_config.get(ctx.region, {})))


# 请求游戏API data_type: json/bytes/None
async def request_gameapi(url: str, method: str = 'GET', data_type: str | None = 'json', authenticated: bool = False, **kwargs):
    # Toolbox 公共数据无需令牌；游戏代理凭据仅发送给显式标记的受保护接口。
    headers = {}
    if authenticated:
        token = config.get('gameapi_token', '')
        assert_and_reply(token, '尚未配置 Haruki 游戏 API 令牌，请联系部署者')
        headers['X-Haruki-Sekai-Token'] = token
    timeout = aiohttp.ClientTimeout(total=config.get('gameapi_timeout', 30))
    try:
        async with get_client_session().request(method, url, headers=headers, timeout=timeout, **kwargs) as resp:
            if resp.status != 200:
                detail = resp.reason or '请求失败'
                try:
                    detail = await resp.text()
                    detail = loads_json(detail)
                    if isinstance(detail, dict):
                        detail = detail.get('detail', detail.get('message', detail))
                except:
                    pass
                utils_logger.error(f"请求游戏API后端失败: HTTP {resp.status}")
                raise HttpError(resp.status, detail)
            
            if data_type is None:
                return resp
            elif data_type == 'json':
                if "text/plain" in resp.content_type:
                    return loads_json(await resp.text())
                elif "application/octet-stream" in resp.content_type:
                    import io
                    return loads_json(io.BytesIO(await resp.read()).read())
                else:
                    return await resp.json()
            elif data_type == 'bytes':
                return await resp.read()
            else:
                raise Exception(f"不支持的数据类型: {data_type}")
                
    except aiohttp.ClientConnectionError as e:
        raise Exception(f"连接游戏API后端失败，请稍后再试")
    except asyncio.TimeoutError:
        raise ReplyException('游戏 API 请求超时，请稍后再试')
