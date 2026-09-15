from ..utils import *
from .api_provider import ApiProvider, LlmModel
from .api_provider_manager import api_provider_mgr

config = Config('llm.llm')
logger = get_logger("Llm")
file_db = get_file_db("data/llm/db.json", logger)

# -------------------------------- 获取模型预设 -------------------------------- #

def get_model_preset(key: str) -> Union[str, List[str], dict]:
    ret = Config('llm.model_preset').get(key)
    def parse_ref(s: str):
        return get_model_preset(s[1:]) if s.startswith("&") else s
    if isinstance(ret, str):
        ret = parse_ref(ret)
    if isinstance(ret, list):
        ret = [parse_ref(s) for s in ret]
    if isinstance(ret, dict):
        ret = {k: parse_ref(v) for k, v in ret.items()}
    return ret


# -------------------------------- 聊天相关 -------------------------------- #

CHAT_TIMEOUT_CFG = config.item('chat_timeout')
CHAT_MODEL_SWITCH_INTERVAL_CFG = config.item('chat_model_switch_interval')
CHAT_MAX_TOKENS_CFG = config.item('chat_max_tokens')
session_id_top = 0

# ChatSeesion回复结果类型
@dataclass
class ChatSessionResponse:
    result: str
    provider: ApiProvider
    model: LlmModel
    prompt_tokens: int
    completion_tokens: int
    cost: float
    quota: float
    reasoning: Optional[str] = None
    images: List[Image.Image] = field(default_factory=list)
    result_list: List[Union[str, Image.Image]] = field(default_factory=list)
        

# 会话类型
class ChatSession:
    # 检查模型名，不存在或不支持多模态抛出异常
    @staticmethod
    def check_model_name(model_name: Union[str, List[str]], mode="text"):
        if not isinstance(model_name, str):
            for name in model_name:
                ChatSession.check_model_name(name, mode=mode)
        else:
            model = api_provider_mgr.find_model(model_name)
            if mode == "mm" and not model.is_multimodal:
                raise Exception(f"模型 {model_name} 不支持多模态输入")
            if mode == "image" and not model.image_response:
                raise Exception(f"模型 {model_name} 不支持图片回复")

    def __init__(self, system_prompt=None):
        global session_id_top
        session_id_top += 1
        self.id = session_id_top
        logger.info(f"创建会话{self.id}")

        self.content = []
        self.has_image = False
        if system_prompt:
            self.append_system_content(system_prompt, verbose=False)

        self.update_time = datetime.now()

    # 添加一条消息
    def append_content(self, role, text, imgs=None, verbose=True):
        if not text and not imgs:
            logger.warning(f"会话{self.id}跳过添加空消息")
            return
        if imgs is None: 
            imgs = []

        for i in range(len(imgs)):
            if isinstance(imgs[i], Image.Image):
                imgs[i] = get_image_b64(imgs[i])
        
        if len(imgs) > 0:
            content = [{"type": "text", "text": text}]
            for img in imgs:
                content.append({
                    "type": "image_url",
                    "image_url": { "url": img }
                })
            self.has_image = True
        else:
            content = text
        self.content.append({
            "role": role, 
            "content": content
        })
        if verbose:
            log_text = f"会话{self.id}添加{role}_content: "
            log_text += "\"" + text.replace('\n', '\\n') + f"\""
            if imgs: 
                log_text += f" + {len(imgs)}img(s)"
            log_text += f", 目前会话长度:{len(self)}"
            logger.info(log_text)
        self.update_time = datetime.now()

    # 添加系统消息
    def append_system_content(self, text, verbose=True):
        self.append_content("system", text, verbose=verbose)

    # 添加用户消息
    def append_user_content(self, text, imgs=None, verbose=True):
        self.append_content("user", text, imgs, verbose=verbose)
    
    # 添加assistant消息
    def append_bot_content(self, text, imgs=None, verbose=True):
        self.append_content("assistant", text, imgs, verbose=verbose)

    # 限制会话长度
    def limit_length(self, limit: int, drop='oldest'):
        assert drop in ['oldest', 'newest']
        system_content = None
        if self.content[0]['role'] == 'system':
            system_content = self.content[0]
            self.content = self.content[1:]
        if len(self.content) >= limit:
            if drop == 'oldest':
                logger.info(f"会话{self.id}长度超过限制({len(self.content)}>{limit}), 删除最旧的消息")
                self.content = self.content[-limit:]
            else:
                logger.info(f"会话{self.id}长度超过限制({len(self.content)}>{limit}), 删除最新的消息")
                self.content = self.content[:limit]
        if system_content:
            self.content.insert(0, system_content)

    # 会话长度
    def __len__(self):
        return len(self.content)

    # 清空消息
    def clear_content(self):
        logger.info(f"会话{self.id}清空消息")
        self.content = []
        self.has_image = False
        self.update_time = datetime.now()

    # 是否存在多模态消息
    def has_multimodal_content(self):
        return self.has_image

    # 获取回复 并且自动添加回复到消息列表
    async def get_response(
        self, 
        model_name: Union[str, List[str]],
        process_func=None,
        image_response=False,
        timeout: Union[int, ConfigItem]=CHAT_TIMEOUT_CFG,
        model_switch_interval: Union[int, ConfigItem]=CHAT_MODEL_SWITCH_INTERVAL_CFG,
        max_tokens: Union[int, ConfigItem]=CHAT_MAX_TOKENS_CFG,
        response_format: Optional[dict]=None,
        reasoning_effort: Optional[str]=None,
    ):
        if isinstance(model_name, str):
            model_name = [model_name]
        errs: List[Tuple[str, str]] = []
        for idx, name in enumerate(model_name):
            try:
                model = api_provider_mgr.find_model(name)
                name = model.get_full_name()
                provider = model.provider
                if not model.is_multimodal and self.has_image:
                    raise Exception(f"模型 {name} 不支持多模态输入")
                
                logger.info(f"会话{self.id}请求回复, 模型名: {name} ({idx+1}/{len(model_name)})")

                provider.check_qps_limit()

                # 附加请求体参数
                extra_body = model.extra_body.copy()
                if model.image_response:
                    extra_body["image_response"] = image_response
                    extra_body["modalities"] = ["image", "text"]
                client_kwargs = model.client_kwargs.copy()
                if response_format is not None:
                    client_kwargs["response_format"] = response_format
                if reasoning_effort is not None:
                    client_kwargs["reasoning_effort"] = reasoning_effort

                client = provider.get_client()

                # 请求回复
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model.get_model_id(),
                            messages=self.content,
                            extra_body=extra_body,
                            max_tokens=get_cfg_or_value(max_tokens),
                            **client_kwargs,
                        ), 
                        timeout=get_cfg_or_value(timeout),
                    )
                except TimeoutError:
                    raise Exception(f"等待回复超时")

                if not isinstance(response, dict):
                    response = response.model_dump()

                if response.get('error'):
                    raise Exception(response['error'])

                # 解析回复
                message             = response['choices'][0]['message']
                prompt_tokens       = response['usage']['prompt_tokens']
                completion_tokens   = response['usage']['completion_tokens']

                # 回复内容
                resp_content = message['content']
                result: str = ""
                images: List[Image.Image] = []
                if isinstance(resp_content, str):
                    # 纯文本回复
                    result = resp_content
                    result_list = [result]
                else:
                    # 多段回复（文本+图片）
                    for part in resp_content:
                        if isinstance(part, str):
                            result += part
                        elif isinstance(part, Image.Image):
                            images.append(part)
                    result_list = resp_content
                # 额外的图像内容
                for item in message.get('images', []):
                    img = b64_to_image(item['image_url']['url'])
                    images.append(img)
                    result_list.append(img)

                # 推理内容
                reasoning: str = None
                if 'reasoning_content' in message:
                    reasoning = message['reasoning_content']
                elif 'reasoning' in message:
                    reasoning = message['reasoning']

                log_text = f"会话{self.id}获取回复，使用token: {prompt_tokens}+{completion_tokens}，内容:\n"
                if reasoning: log_text += f"【思考】" + truncate(reasoning.replace('\n', '\\n'), 128) + "\n"
                for part in result_list:
                    log_text += truncate(part.replace('\n', '\\n'), 128) if isinstance(part, str) else "[图片]"
                logger.info(log_text)

                # 添加到对话记录
                self.append_bot_content(result, imgs=[get_image_b64(img) for img in images], verbose=False)

                # 计算并更新额度
                cost = model.calc_price(prompt_tokens, completion_tokens)
                quota = await provider.aupdate_quota(-cost)

                self.update_time = datetime.now()

                ret = ChatSessionResponse(
                    result=result,
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    quota=quota,
                    reasoning=reasoning,
                    images=images,
                    result_list=result_list,
                )

                if process_func:
                    if asyncio.iscoroutinefunction(process_func):
                        ret = await process_func(ret)
                    else:
                        ret = process_func(ret)
                return ret
            
            except Exception as e:
                logger.print_exc(f"会话{self.id}获取回复失败, 使用模型 {name}: {get_exc_desc(e)}")
                errs.append((name, get_exc_desc(e)))
                await asyncio.sleep(get_cfg_or_value(model_switch_interval))

        if len(errs) == 1:
            # logger.error(f"会话{self.id}调用模型{errs[0][0]}失败:\n{errs[0][1]}")
            raise ReplyException(f"调用模型{errs[0][0]}失败:\n{truncate(errs[0][1], 64)}")
        else:
            err_str = "\n".join([f"[{err[0]}] {err[1]}" for err in errs])
            # logger.error(f"会话{self.id}调用模型{errs[0][0]}失败:\n{err_str}")
            raise ReplyException(f"调用多个模型失败:\n{truncate(err_str, 64)}")


# -------------------------------- 文本翻译相关 -------------------------------- #

async def translate_text(text, additional_info=None, dst_lang="中文", timeout=20, default=None, model=None, cache=True):
    if model is None:
        model = get_model_preset("translation")
    text_translation_db = get_file_db("data/llm/text_translations.json", logger)
    translations = text_translation_db.get("translations", {}) if cache else {}
    key = get_md5(text)
    if not cache or key not in translations:
        logger.info(f"翻译文本: {truncate(text, 64)} 额外信息: {truncate(additional_info, 64)} 目标语言: {dst_lang}")
        try:
            session = ChatSession()
            if additional_info:
                additional_info = f"额外的参考信息:\"{additional_info}\"，"
            else:
                additional_info = ""
            prompt = f"翻译文本到{dst_lang}{additional_info}，请直接输出翻译结果并结束，不要包含其他内容:\n{text}"
            session.append_user_content(prompt)
            response = await asyncio.wait_for(session.get_response(model), timeout=timeout)
            result = response.result.strip()
            logger.info(f"翻译结果: {truncate(result, 64)}")
            translations[key] = result
        except Exception as e:
            logger.print_exc(f"翻译失败: {e}")
            return default
    if cache:
        text_translation_db.set("translations", translations)
    return translations[key]
    
