from dataclasses import dataclass
from src.api.utils.config_loader.read_yaml import YamlConfigReader

@dataclass
class ConfigReaderInstance:
    """
    A dataclass to hold the instance of the YamlConfigReader.
    This allows for easy instantiation and access to the reader.
    """
    yaml: YamlConfigReader = YamlConfigReader()