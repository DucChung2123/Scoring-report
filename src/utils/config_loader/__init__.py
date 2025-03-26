from src.utils.config_loader.read_yaml import YamlConfigLoader
from pydantic.dataclasses import dataclass

@dataclass
class ConfigLoaderInstance:
    yaml = YamlConfigLoader()

