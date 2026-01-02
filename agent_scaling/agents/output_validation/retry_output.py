from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from agent_scaling.llm.litellm_lc import ChatLiteLLMLC
from agent_scaling.logger import logger

ERROR_PROMPT = """
An error occured while parsing the output.
Error message:
{error_message}

Problematic output:
{previous_output}

Please return the output correcting the error.
"""


def run_with_validation(
    llm: ChatLiteLLMLC,
    messages: List[Dict[str, Any]],
    val_func: Callable[[str], Any],
    num_retries: int = 3,
    val_func_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Tuple[List[AIMessage], Any]:
    """
    Run the LLM with validation.
    Args:
        llm: The LLM to use for generation
        messages: List of messages to send to the LLM
        val_func: Validation function that takes the LLM output string and returns parsed result
        num_retries: Number of retries if validation fails
        val_func_kwargs: Additional kwargs to pass to validation function
        **kwargs: Additional kwargs to pass to LLM invoke

    Returns:
        Tuple containing:
        - List of AIMessages from all attempts
        - Parsed result from validation function
    """
    error_msg = ""
    for i in range(num_retries):
        output = llm.invoke(messages, **kwargs)
        outputs = [output]
        try:
            res = val_func(output.text(), **val_func_kwargs)
            return outputs, res
        except Exception as e:
            logger.warning(f"Error parsing output: {e}\tretries left: {num_retries}\n")
            error_msg = str(e)
            messages = messages + [
                convert_to_openai_messages(output),  # type: ignore
                {
                    "role": "user",
                    "content": ERROR_PROMPT.format(
                        error_message=str(e), previous_output=output.text()
                    ),
                },
            ]
    raise RuntimeError(f"Max retries reached. Original error: {error_msg}")
