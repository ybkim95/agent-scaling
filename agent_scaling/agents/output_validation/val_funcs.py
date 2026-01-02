from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser

from agent_scaling.logger import logger


def validate_json(text: str) -> Dict[str, Any]:
    parser = JsonOutputParser()
    try:
        parsed_json = parser.parse(text)
    except Exception as e:
        logger.warning(f"Error parsing JSON from output: {e}")
        raise ValueError(f"Error parsing JSON from output: {e}")
    return parsed_json
