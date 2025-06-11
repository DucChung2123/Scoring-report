# Hello, I'm Chung, AI Engineer in MISA JSC!
from pydantic_settings import BaseSettings
from typing import Dict, Any
from src.api.utils.config_loader import ConfigReaderInstance

class GlobalConfig(BaseSettings):
    """
    Global configuration settings for the application.
    This class uses Pydantic's BaseSettings to manage configuration.
    """
    
    # Load configuration files
    APP_CONF: Dict[str, Any] = ConfigReaderInstance.yaml.read_config_from_file(
        "settings/app_config.yaml"
    )
    MODEL_CONF: Dict[str, Any] = ConfigReaderInstance.yaml.read_config_from_file(
        "settings/model.yaml"
    )

# Global settings instance
settings = GlobalConfig()
