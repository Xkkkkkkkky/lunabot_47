from .utils import *
from aiohttp import ClientSession, ClientConnectionError, ClientTimeout

REQUEST_TIMEOUT = 30.0

_session: ClientSession | None = None

def get_session() -> ClientSession:
    global _session
    if _session is None or _session.closed:
        # 使用环境代理，但保留 aiohttp 默认的 TLS 证书校验。
        _session = ClientSession(
            trust_env=True,
            timeout=ClientTimeout(total=REQUEST_TIMEOUT),
        )
    return _session

async def close_session():
    global _session
    if _session is not None and not _session.closed:
        await _session.close()


gameapi_config = Config('sekai.gameapi')

@dataclass
class GameApiConfig:
    api_status_url: Optional[str] = None
    profile_api_url: Optional[str] = None 
    suite_api_url: Optional[str] = None
    ranking_top100_api_url: Optional[str] = None
    ranking_border_api_url: Optional[str] = None


# 获取游戏api相关配置
def get_gameapi_config(region: str) -> GameApiConfig:
    return GameApiConfig(**(gameapi_config.get(region, {})))


# 请求游戏API data_type: json/bytes/None
async def request_gameapi(
    url: str,
    method: str = 'GET',
    data_type: str | None = 'json',
    authenticated: bool = False,
    **kwargs,
):
    debug(f"请求游戏API后端: {method} {url}")
    headers = dict(kwargs.pop('headers', {}) or {})
    if authenticated:
        token = str(config.get('gameapi_token', '') or '').strip()
        if not token:
            raise RuntimeError('未配置 Haruki 游戏 API 令牌，事件榜线服务不会开始记录')
        # Haruki-Sekai-API 的受保护接口使用专用请求头。
        headers['X-Haruki-Sekai-Token'] = token
    try:
        async with get_session().request(method, url, headers=headers, **kwargs) as resp:
            if resp.status != 200:
                detail = resp.reason or '请求失败'
                try:
                    detail = await resp.text()
                    detail_json = loads_json(detail)
                    if isinstance(detail_json, dict):
                        detail = detail_json.get('detail', detail_json.get('message', detail_json))
                except:
                    pass
                error(f"请求游戏API后端 {url} 失败: {resp.status} {detail}")
                raise Exception(f"请求游戏API后端失败: {resp.status} {detail}")
            
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
                
    except ClientConnectionError as e:
        raise Exception(f"连接游戏API后端失败")
