from copy import deepcopy
from typing import Callable, Dict

from langchain_core.tools import BaseTool
from agent_scaling.langfuse_client import get_lf_client
from .registry import register_tool, list_registered_tools, get_tool

from .code import python_repl
from .done import done_with_confidence
from .multiply import multiply, multiply_by_max
try:
    from .search import tavily_search_tool, tavily_search_simple_tool
except Exception:  # pragma: no cover
    # search.py instantiates the Tavily client at module load and requires
    # TAVILY_API_KEY. Keep the rest of the tool registry importable when no
    # key is set (e.g. during reviewer code inspection); experiments that
    # need web search will still require the key.
    tavily_search_tool = None  # type: ignore[assignment]
    tavily_search_simple_tool = None  # type: ignore[assignment]
from .utils import enhance_tool, cls_tool
