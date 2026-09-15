# Sekai Deck Recommend Service for LunaBot

### Usage

1. Install dependencies (Python 3.8+ required):

```bash
conda run -n bot pip install -r src/services/deck_recommender/requirements.txt
```

2. Configure the service by editing `config/sekai/deck_recommender.yaml`. Both `enabled` and `deck_enabled` in `config/sekai/sekai.yaml` must be `true`; otherwise the service exits before loading the engine or starting workers. Restart existing processes after changing either switch.

3. Run the service:

```bash
conda run -n bot python src/services/deck_recommender/serve.py
```

4. Add the service to LunaBot by updating the bot's configuration file `config/sekai/sekai.yaml`
