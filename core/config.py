import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path="core/config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    @property
    def experiment(self):
        return self.cfg.get("experiment", {})

    @property
    def db(self):
        return self.cfg.get("postgres", {})

    @property
    def model(self):
        return self.cfg.get("model", {})

# 사용 예시
# config = ConfigLoader()
# print(config.experiment)
# print(config.db)
# print(config.model) 