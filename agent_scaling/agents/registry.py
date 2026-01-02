from typing import Dict, Type

from .base import AgentSystem

# Simple registry using a dictionary
_agent_registry: Dict[str, Type[AgentSystem]] = {}


def register_agent(name: str):
    """Decorator to register an agent class."""

    def decorator(cls: Type[AgentSystem]) -> Type[AgentSystem]:
        _agent_registry[name] = cls
        return cls

    return decorator


def list_agents() -> list[str]:
    """List all registered agent names."""
    return list(_agent_registry.keys())


def is_registered(name: str) -> bool:
    """Check if an agent is registered."""
    return name in _agent_registry


def get_agent(name: str, **kwargs) -> AgentSystem:
    """Create an agent instance using the factory pattern."""
    if name not in _agent_registry:
        raise ValueError(f"Agent '{name}' not found in registry")
    return _agent_registry[name](**kwargs)


def get_agent_cls(name: str) -> Type[AgentSystem]:
    """Get an agent class by name."""
    if name not in _agent_registry:
        raise ValueError(f"Agent '{name}' not found in registry")
    return _agent_registry[name]
