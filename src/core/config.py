from src.utils.config_loader import ConfigLoaderInstance
from pydantic import Field, BaseModel
from typing import Any
class AppConfig(BaseModel):
    
    APP_CONFIG_YAML: dict[str, Any] = ConfigLoaderInstance.yaml.read_config_from_file("settings/app_config.yaml")
    
    APP_NAME: str = Field(default=APP_CONFIG_YAML["app"]["app_name"])
    HOST: str = Field(default=APP_CONFIG_YAML["app"]["host"])
    PORT: int = Field(default=APP_CONFIG_YAML["app"]["port"])
    

settings = AppConfig()