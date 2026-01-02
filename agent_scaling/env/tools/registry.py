from typing import Dict, List, Type

from langchain_core.tools import BaseTool

_tool_registry: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool):
    """Decorator to register a tool class."""
    _tool_registry[tool.name] = tool
    return tool


def list_registered_tools() -> list[str]:
    """List all registered tool names."""
    return list(_tool_registry.keys())


def get_tool(tool_name: str) -> BaseTool:
    if tool_name not in _tool_registry:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    return _tool_registry[tool_name]
