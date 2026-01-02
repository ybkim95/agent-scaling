from agent_scaling.env.tools import cls_tool, get_tool, list_registered_tools

from .base import AgentEnvironmentTools
from .registry import register_env


@register_env("basic")
class BasicEnvironment(AgentEnvironmentTools):

    def __init__(self, **kwargs):
        additional_tools = {name: get_tool(name) for name in list_registered_tools()}
        self.sum = 0
        super().__init__(additional_env_tools=additional_tools, **kwargs)

    @cls_tool
    def add(self, x: int, y: int) -> int:
        """Add two integers."""
        self.sum += x + y
        return x + y
