import yaml

from src.utils.config_loader.config_interface import ConfigReaderInterface

class YamlConfigLoader(ConfigReaderInterface):
    def __init__(self):
        super().__init__()
        
    def read_config_from_file(self, conf_path: str):
        with open(conf_path) as file:
            config = yaml.safe_load(file)
        return config