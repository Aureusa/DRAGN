from typing import Dict, Type, Any, Callable, TypeVar, Union
from .component import Component

T = TypeVar('T', bound=Component)

class Registry:
    """Registry for configurable components"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[Component]] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(self, name: str):
        """Decorator for registering components or factory functions"""
        def decorator(cls_or_fn):
            if isinstance(cls_or_fn, type) and issubclass(cls_or_fn, Component):
                self._registry[name] = cls_or_fn
            else:
                self._factories[name] = cls_or_fn
            return cls_or_fn
        return decorator
    
    def build(self, name: str, config: Dict[str, Any], **kwargs) -> Any:
        """Build component from configuration"""
        if name in self._registry:
            return self._registry[name].from_config(config, **kwargs)
        elif name in self._factories:
            return self._factories[name](config, **kwargs)
        else:
            available = list(self._registry.keys()) + list(self._factories.keys())
            raise ValueError(f"Unknown {self.name}: {name}. Available: {available}")
    
    def list_available(self) -> list[str]:
        """List all available components"""
        return list(self._registry.keys()) + list(self._factories.keys())
    
    def get(self, name: str) -> Union[Type[Component], Callable]:
        """Get component class or factory function"""
        if name in self._registry:
            return self._registry[name]
        elif name in self._factories:
            return self._factories[name]
        else:
            raise ValueError(f"Unknown {self.name}: {name}")


# Global registries
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
LOADERS_REGISTRY = Registry("loaders")
TRANSFORM_REGISTRY = Registry("transform")
LOSS_REGISTRY = Registry("loss")
OPTIMIZER_REGISTRY = Registry("optimizer")
TRAINING_STRATEGY_REGISTRY = Registry("training_strategy")
