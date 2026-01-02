
from langchain.tools import tool
from .registry import register_tool

@register_tool
@tool("done")
def done_with_confidence(answer: str, confidence_score: int) -> str:
    """
    Indicate the best guess answer and a confidence score between 0% to 100%.
    """
    return f"{answer}\t{confidence_score}"