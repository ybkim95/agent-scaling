from typing import Dict, Type, TypeVar
from .base import AgentEnvironment

T = TypeVar("T", bound=AgentEnvironment)

# Simple registry using a dictionary
_env_registry: Dict[str, Type[AgentEnvironment]] = {}


def register_env(name: str):
    """Decorator to register an environment class."""

    def decorator(cls: Type[T]) -> Type[T]:
        _env_registry[name] = cls
        return cls

    return decorator


def list_envs() -> list[str]:
    """List all registered environment names."""
    return list(_env_registry.keys())


def is_env_registered(name: str) -> bool:
    """Check if an environment is registered."""
    return name in _env_registry


def get_env(name: str, **kwargs) -> AgentEnvironment:
    """Create an environment instance using the factory pattern."""
    if name not in _env_registry:
        raise ValueError(f"Environment '{name}' not found in registry")
    return _env_registry[name](**kwargs)


def get_env_cls(name: str) -> Type[AgentEnvironment]:
    """Get an environment class by name."""
    if name not in _env_registry:
        raise ValueError(f"Environment '{name}' not found in registry")
    return _env_registry[name]
