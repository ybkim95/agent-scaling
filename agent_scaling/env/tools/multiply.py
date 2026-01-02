from typing import Annotated, List

from langchain_core.tools import tool

from .registry import register_tool


@register_tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@register_tool
@tool
def multiply_by_max(
    a: Annotated[int, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)

