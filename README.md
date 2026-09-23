# LunaBot

A multi-functional chatbot based on Nonebot2

Note: This project is for reference and learning purposes only, and is **not** a completely deployable application.

- There may be issues with the deployment steps.

- Missing configurations and data will not be provided.

### Deployment Steps

#### 1. Install Dependencies

- Install dependencies using `pip install -r requirements.txt` (Python >=3.10 is recommended)

- Install playwright browsers by running the command: ```playwright install```

- Install system emoji fonts if emojis fail to render for commands such as `/help`

- Replace the line in `{your_python_dir}/site-packages/pilmoji/helpers.py`:

    ```python
    language_pack: Dict[str, str] = unicode_codes.get_emoji_unicode_dict('en')
    ```

    with
    
    ```python
    import emoji
    language_pack: Dict[str, str] = {data['en']: emj for emj, data in emoji.EMOJI_DATA.items() if 'en' in data and data['status'] <= emoji.STATUS['fully_qualified']}
    ```

#### 2. Setup Configurations

- Copy the configuration from the `example_config` directory to the `config` directory and fill in the missing content as needed.
- Example files intentionally leave deployment URLs, API keys, tokens, and account or group IDs blank. Fill these only in the local, ignored `config` directory.

- Rename `.env.example` to `.env`.

- Find and place the missing data yourself


#### 3. Run the Bot

- Start the project using nonebot2 cli command: `nb run`.

- Send a message `@yourbot /enable` to enable the bot in the group.

- (Optional) Start the Sekai Deck Recommendation Service: [README.md](./src/services/deck_recommender/README.md)

### 静默群配置

`silent_groups` 启动时读取，手动修改后须重启 bot 生效。

直接编辑项目内 `data/utils/db.json` 的 `silent_groups` 列表，加上或删去数字群号后重启 bot 即可生效。静默群无需同时加入 `enabled_groups`，仅在后台执行 `/加图`，且完全静默地拦截所有消息发送，不提供聊天指令开关，超级管理员也不能绕过静默限制。`/加图` 仍遵守原有图片权限、冷却和去重规则。移除群号后，将恢复由 `enabled_groups` 决定的原状态。

```json
"silent_groups": []
```

### NapCat on Windows

When the bot and NapCat run on different machines, NapCat cannot read the bot's
local `file://` image paths. Set the following option in `config/global.yaml` to
upload generated images through NapCat's `upload_file_stream` API before sending:

```yaml
msg_send:
  image_transport:
    mode: "napcat_stream"
    napcat_stream:
      chunk_size: 65536
      file_retention_ms: 300000
      verify_sha256: true
```

Keep `mode: "file"` to use the original image delivery method.


## todolist

1. pjsk b30/b50
2. pjsk 查曲/定数/完成表
3. ~~查鸟（lunabot）~~
4. ~~llm chat/summary~~
5. ~~nsy 别名/图库~~
6. ~~gbfr 查输出~~
