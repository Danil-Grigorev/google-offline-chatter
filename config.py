import json
import os

class Config(dict):
    def __init__(self, file_path):
        config = json.load(open(file_path))
        for k, v in config.copy().items():
            config[k] = os.environ.get(k, v)
        self.update(config)
