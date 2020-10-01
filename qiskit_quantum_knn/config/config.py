import json
import os

LOCAL_CONF = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))


class ConfigException(Exception):
    pass


class Config:
    """Configuration class, load from filename, file or python dictionary.
    Environment variable aware."""

    def __init__(self, filename=None, file=None, config=None):
        if config is not None:
            self.config=config
        elif file is not None:
            self.config = json.load(file)
        elif filename is not None:
            with open(filename) as f:
                self.config = json.load(f)
        else:
            with open(LOCAL_CONF) as f:
                self.config = json.load(f)

    def __getitem__(self, key):
        config = self.config
        for subkey in key.split('.'):
            config = config[subkey]
        if type(config) is str \
                and config.startswith("ENV{") \
                and config.endswith("}"):
            config = os.environ[config[len("ENV{"):-1]]
        return config
