from typing import Any, Callable, Dict, Literal, Type

from .base import BaseAgent
from .direct_prompt import *

# Import new multi-agent variants
from .multiagent_centralized import *
from .registry import (
    AgentSystem,
    get_agent,
    get_agent_cls,
    is_registered,
    list_agents,
    register_agent,
)
from .single_agent import *

# Import all agent classes after the registry is set up
from .single_agent_cot import *
