from langfun.core.coding.python import run
from diskcache import FanoutCache
from langchain.tools import tool
from pyglove.core.coding.errors import CodeError

from agent_scaling.cache import get_function_cache

from .registry import register_tool, list_registered_tools

cache: FanoutCache = get_function_cache()


@register_tool
@tool
def python_repl(code: str) -> str:
    """
    Execute and evaluate Python code.

    The tool returns the result of the last line of the code, unless that line evaluates to `None`.
    In that case, it returns any output captured from stdout during execution (e.g., print statements).
    """
    try:
        res = run(
            code,
            sandbox=False,
            outputs_intermediate=True,
        )
    except CodeError as e:
        return str(e.cause)
    if isinstance(res, dict):
        observation = res.get("__result__", "")
        if observation is None:
            observation = res.get("__stdout__", "")
        return observation
    else:
        return ""
