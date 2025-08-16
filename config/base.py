from pydantic import BaseModel
from pathlib import Path
import yaml
import json
from typing import Any, Union, get_origin, get_args


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
    
    @classmethod
    def generate_example_yaml(cls, file_path: str, include_optional: bool = True):
        """
        Generate an example YAML configuration file with default values and comments.
        
        Args:
            file_path: Path where to save the example YAML file
            include_optional: Whether to include optional fields with their default values
        """
        yaml_content = cls._generate_yaml_content(cls, include_optional=include_optional)
        
        with open(file_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Example YAML configuration saved to: {file_path}")
    
    @classmethod
    def _generate_yaml_content(cls, config_class: type, indent: int = 0, include_optional: bool = True) -> str:
        """
        Recursively generate YAML content for a configuration class.
        """
        lines = []
        indent_str = "  " * indent
        
        # Add class header comment if at root level
        if indent == 0:
            lines.append(f"# Example configuration for {config_class.__name__}")
            lines.append(f"# Generated automatically - modify as needed")
            lines.append("")
        
        for field_name, field_info in config_class.model_fields.items():
            # Check if field is required
            is_required = field_info.is_required()
            
            # Skip optional fields if not including them
            if not include_optional and not is_required:
                continue
            
            # Get field description
            description = field_info.description if field_info.description else None
            
            # Add field comment (only if description exists)
            if description:
                lines.append(f"{indent_str}# {description}")
            
            # Get default value
            default_value = field_info.default if hasattr(field_info, 'default') else None
            
            # Check if field has PydanticUndefined (check string representation)
            if str(default_value) == "PydanticUndefined":
                default_value = None
            
            # Check if this is a nested config
            field_type = field_info.annotation
            
            # Handle Optional types
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                if len(args) == 2 and type(None) in args:
                    # This is Optional[SomeType]
                    field_type = args[0] if args[1] is type(None) else args[1]
            
            # Check if field_type is a ConfigBase subclass
            if (isinstance(field_type, type) and 
                issubclass(field_type, ConfigBase) and 
                field_type != ConfigBase):
                
                lines.append(f"{indent_str}{field_name}:")
                nested_content = cls._generate_yaml_content(field_type, indent + 1, include_optional)
                lines.append(nested_content)
            
            else:
                # Handle primitive types and complex defaults
                if is_required and (default_value in (None, ...) or str(default_value) == "PydanticUndefined"):
                    # Required field with no default - use # REQUIRED as value
                    lines.append(f"{indent_str}{field_name}: # REQUIRED")
                elif default_value is not None and default_value != ... and str(default_value) != "PydanticUndefined":
                    # Field has a default value
                    yaml_value = cls._format_yaml_value(default_value)
                    lines.append(f"{indent_str}{field_name}: {yaml_value}")
                else:
                    # Optional field with no default
                    example_value = cls._get_example_value(field_type, field_name)
                    lines.append(f"{indent_str}{field_name}: {example_value}")
            
            lines.append("")  # Add spacing between fields
        
        return "\n".join(lines)
    
    @staticmethod
    def _get_example_value(field_type: type, field_name: str) -> str:
        """Generate example values for optional fields based on type and name."""
        
        # Handle Union types (like Optional)
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Get the non-None type for Optional types
            non_none_types = [arg for arg in args if arg != type(None)]
            if non_none_types:
                field_type = non_none_types[0]
        
        # Field name based examples
        if 'path' in field_name.lower() or 'folder' in field_name.lower():
            return '"/path/to/your/data"'
        elif 'filename' in field_name.lower():
            return '"model_name"'
        elif 'strategy' in field_name.lower():
            return '"standard"'
        elif 'dataset' in field_name.lower() and 'type' in field_name.lower():
            return '"your_dataset_type"'
        elif 'loader' in field_name.lower() and 'type' in field_name.lower():
            return '"your_loader_type"'
        elif 'architecture' in field_name.lower():
            return '"unet"'
        elif 'loss' in field_name.lower():
            return '"l1_loss"'
        elif 'name' in field_name.lower():
            return '"adam"'
        elif field_name.lower() == 'params':
            return "{}"  # For params fields, default to empty dict
        
        # Type based examples
        if field_type == str:
            return '"example_value"'
        elif field_type == int:
            return "1"
        elif field_type == float:
            return "0.001"
        elif field_type == bool:
            return "true"
        elif field_type == dict or str(field_type).startswith('typing.Dict'):
            return "{}"
        elif field_type == list or str(field_type).startswith('typing.List'):
            return "[]"
        else:
            return '"example_value"'
    
    @staticmethod
    def _format_yaml_value(value: Any) -> str:
        """Format a Python value for YAML output."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, dict):
            if not value:
                return "{}"
            return str(value)
        elif isinstance(value, list):
            if not value:
                return "[]"
            return str(value)
        else:
            return f'"{value}"'
    

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


def create_example_config_yaml(config_type: ConfigBase, file_path: str = "example_config.yaml"):
    config_type.generate_example_yaml(file_path, include_optional=True)
    print(f"Example configuration saved to {file_path}")
    print("Please edit the required fields marked with '# REQUIRED' before using.")
