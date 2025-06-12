import yaml
from src.api.utils.config_loader.config_interface import ConfigReaderInterface
import os
import re
from dotenv import load_dotenv

class YamlConfigReader(ConfigReaderInterface):
    def __init__(self):
        super().__init__()
        # Load .env file
        load_dotenv()
        
    def read_config_from_file(self, conf_path: str) -> dict:
        """
        Reads configuration from a YAML file.

        :param file_path: Path to the YAML configuration file.
        :return: Parsed configuration data as a dictionary.
        """
        with open(conf_path, 'r') as file:
            content = file.read()
            
        # Thay thế ${VAR_NAME} bằng giá trị từ environment
        content = self._substitute_env_vars(content)
        config_data = yaml.safe_load(content)
        return config_data
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute ${VAR_NAME} with environment variable values"""
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                # print(f"Warning: Environment variable '{var_name}' not found")
                return match.group(0)
            return env_value
        
        return re.sub(pattern, replacer, content)

# if __name__ == "__main__":
#     # Example usage
#     reader = YamlConfigReader()
#     config = reader.read_config_from_file('settings/llm.yaml')
#     print(config)
