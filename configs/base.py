from pydantic import BaseModel
from pathlib import Path
import yaml
import json


class ConfigBase(BaseModel):
    @classmethod
    def from_yaml(cls, file_path: str):
        """Load config from YAML file"""
        return load_config(Path(file_path), cls)
    
    @classmethod 
    def from_json(cls, file_path: str):
        """Load config from JSON file"""
        return load_config(Path(file_path), cls)
    
    def to_yaml(self, file_path: str):
        """Save config to YAML file"""
        save_config(self, Path(file_path))
    
    def to_json(self, file_path: str):
        """Save config to JSON file"""
        save_config(self, Path(file_path))
    

def load_config(config_path: Path, config_class: type[BaseModel]) -> BaseModel:
    """Load and validate configuration from file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        if config_path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config_class(**data)

def save_config(config: BaseModel, config_path: Path):
    """Save configuration to file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
        elif config_path.suffix == '.json':
            f.write(config.model_dump_json(indent=2))
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
