from typing import Dict, Type, Any, Callable, TypeVar, Optional
from abc import ABC, abstractmethod

T = TypeVar('T')

class Component(ABC):
    """
    Base class for all configurable components in DRAGN
    """
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'Component':
        """
        Create component from configuration dictionary
        """
        pass
    
    def to_config(self) -> Dict[str, Any]:
        """
        Convert component to configuration dictionary (optional)
        """
        return {}
