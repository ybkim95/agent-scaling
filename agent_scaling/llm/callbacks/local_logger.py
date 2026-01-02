import datetime
import json
from typing import cast

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse

from agent_scaling.logger import LLM_LEVEL_NAME, PROMPT_LEVEL_NAME, logger


class LocalLogger(CustomLogger):

    def __init__(self, prompt_only: bool = False, **kwargs):
        self.prompt_only = prompt_only
        super().__init__(**kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        response: ModelResponse = cast(ModelResponse, response_obj)
        if not isinstance(end_time, datetime.datetime):
            end_time = datetime.datetime.now().timestamp()
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time.total_seconds()

        if hasattr(response, "usage"):
            usage = response.usage.model_dump()  # type: ignore
        else:
            usage = {"completion_tokens": 0, "total_tokens": 0}
        logger.bind(
            messages=kwargs.get("messages", []),  # type: ignore
        ).log(PROMPT_LEVEL_NAME, "")

        if self.prompt_only:
            message = response.choices[0].message.model_dump()  # type: ignore
            if message.get("content"):
                message = message["content"]
            elif message.get("tool_calls"):
                message = json.dumps(message["tool_calls"], indent=2)
            elif message.get("function_call"):
                message = json.dumps(message["function_call"], indent=2)
            else:
                message = ""
        else:
            message = response.choices[0].message.model_dump_json(indent=2)  # type: ignore

        logger.bind(
            model=response.model,
            message=message,
            elapsed_time=elapsed_time,
            usage=usage,
            from_cache=response._hidden_params.get("cache_hit", False),
        ).log(LLM_LEVEL_NAME, "")
