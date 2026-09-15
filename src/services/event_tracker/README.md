# Sekai Event Tracker Service for LunaBot

### Usage

1. Use the `bot` conda environment and install dependencies (Python 3.8+ required):

```bash
conda run -n bot python -m pip install -r src/services/event_tracker/requirements.txt
```

2. Run the service from the `lunabot_47` project root:

```bash
conda run -n bot python -m src.services.event_tracker.main
```

3. Configurations are shared with LunaBot:

   - Both `enabled` and `ranking_enabled` in `config/sekai/sekai.yaml` must be `true`. Otherwise the entrypoint exits before loading the database or creating recording tasks. Restart existing processes after changing either switch.

   - `config/sekai/gameapi.yaml` must provide both `ranking_top100_api_url` and `ranking_border_api_url`.
   - The two ranking endpoints are protected Haruki-Sekai-API endpoints. Put the real `gameapi_token` only in local `config/sekai/sekai.yaml`; if it is empty, this service exits before starting any recording request.
   - The service reads master data from the shared local cache `data/sekai/assets/masterdata/{region}/*.json`. It does not download or update master data itself, so populate that cache through the main bot's configured master-data updater before starting the tracker.
