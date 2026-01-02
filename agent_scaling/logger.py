import json
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import loguru
import yaml
from loguru import logger as loguru_logger

LLM_LEVEL_NAME = "LLM"
PROMPT_LEVEL_NAME = "PROMPT"
CACHE_LEVEL_NAME = "CACHE"
API_LEVEL_NAME = "API"


def write_yaml_str(data: dict) -> str:
    # Configure YAML representers
    def str_presenter(dumper, data):
        if len(data.splitlines()) > 1:  # Check for multiline string
            data = "\n".join([line.rstrip() for line in data.strip().splitlines()])
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        else:
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

    yaml.add_representer(str, str_presenter)

    def truncate_floats_presenter(dumper, data):
        if isinstance(data, float):
            return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data:.3f}")
        return dumper.represent_scalar("tag:yaml.org,2002:float", float(data))

    yaml.add_representer(float, truncate_floats_presenter)

    return yaml.dump(data)


class Formatter:
    def __init__(self):
        self.padding = 0
        self.fmt = "[<green><b>{time:YYYY-MM-DD hh:mm:ss.SS}</b></green>][<cyan><b>{file}:{line}</b></cyan> - <cyan>{name:}:{function}</cyan>][<level>{level}</level>] {message}\n"

    def format(self, record):
        length = len("{file}:{line} - {name:}:{function}".format(**record))
        self.padding = max(self.padding, length)
        record["extra"]["padding"] = " " * (self.padding - length)
        fmt = ""
        if record["level"].name == LLM_LEVEL_NAME and "message" in record["extra"]:
            if record["extra"]["from_cache"]:
                fmt = "<LG>================[[<b> {extra[model]} Response</b> (Cache time={extra[elapsed_time]}  completion tokens={extra[usage][completion_tokens]}  total_tokens={extra[usage][total_tokens]})]]================</LG>\n{extra[message]}\n"
            else:
                fmt = "<LY>================[[<b> {extra[model]} Response</b> (API time={extra[elapsed_time]}  completion tokens={extra[usage][completion_tokens]}  total_tokens={extra[usage][total_tokens]})]]================</LY>\n{extra[message]}\n"
        elif (
            record["level"].name == PROMPT_LEVEL_NAME and "messages" in record["extra"]
        ):
            for i, message in enumerate(record["extra"]["messages"]):
                fmt += (
                    f"<LC>===================[[<b>{message['role']:}</b>]]===================</LC>\n"
                    f"{message['content']}\n"
                )
                if message.get("tool_calls"):
                    fmt += "<W>[[<b>tool call(s)</b>]]</W>\n"
                    for tool_call in message["tool_calls"]:
                        fmt += f"Tool {tool_call['function']['name']} called with arguments:\n"
                        fmt += f"{write_yaml_str(json.loads(tool_call['function']['arguments']))}\n\n"
        ret_fmt = self.fmt

        ret_fmt = ret_fmt.replace("{serialized_short}", "")
        return ret_fmt + fmt


def serialize(record):
    subset = OrderedDict()
    subset["level"] = record["level"].name
    subset["message"] = record["message"]
    subset["time"] = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    subset["file"] = {
        "name": record["file"].name,
        "path": record["file"].path,
        "function": record["function"],
        "line": record["line"],
    }
    subset["extra"] = record["extra"]
    return json.dumps(subset)


def serialize_extras(record):
    return json.dumps(record["extra"])


def patching(record):
    extras = serialize_extras(record)
    record["serialized_short"] = extras[:50] + "..." if len(extras) > 50 else extras
    record["extra"]["serialized"] = serialize(record)

    if record["level"].name == LLM_LEVEL_NAME and "message" in record["extra"]:
        # Escape curly braces and angle brackets in LLM message content
        message_content = record["extra"]["message"]
        message_content = message_content.replace("{", "{{").replace("}", "}}")
        message_content = message_content.replace("<", "\\<").replace(">", "\\>")
        record["extra"]["message"] = message_content
    elif record["level"].name == PROMPT_LEVEL_NAME and "messages" in record["extra"]:
        for i, message in enumerate(record["extra"]["messages"]):
            # Escape curly braces and angle brackets in prompt message content
            content = message["content"]
            content = content.replace("{", "{{").replace("}", "}}")
            content = content.replace("<", "\\<")
            record["extra"]["messages"][i]["content"] = content


def parse_prompt(prompt: List[Dict[str, str]]):
    return "\n\n".join(
        [f"===={m['role'].upper()} ====:\n{m['content']}" for m in prompt]
    )


class LoggerManager:
    """Manages logger configuration following standard Python practices."""

    def __init__(self):
        self._handler_id: Optional[int] = None
        self._is_configured = False
        self._logger: loguru.Logger = loguru_logger.patch(patching)
        self._logger.remove(0)
        self._logger.level(PROMPT_LEVEL_NAME, no=15, color="<white><bold>", icon="📋")
        self._logger.level(CACHE_LEVEL_NAME, no=15, color="<yellow><bold>", icon="💾")
        self._logger.level(API_LEVEL_NAME, no=15, color="<red><bold>", icon="🛜")
        self._logger.level(LLM_LEVEL_NAME, no=15, color="<lm><bold>", icon="🤖")

    def configure(self, level: int | str = 15) -> "loguru.Logger":
        """Configure the logger with the specified settings."""
        if self._is_configured:
            # Remove existing handler if reconfiguring
            if self._handler_id is not None:
                self._logger.remove(self._handler_id)

        formatter = Formatter()
        self._handler_id = self._logger.add(
            sys.stdout, format=formatter.format, level=level
        )
        self._is_configured = True

        return self._logger

    def get_logger(self) -> "loguru.Logger":
        """Get the configured logger, configuring it if necessary."""
        if not self._is_configured:
            return self.configure()
        assert self._logger is not None  # Type guard
        return self._logger


# Global instance following singleton pattern
_logger_manager = LoggerManager()


def configure_logger(level: int | str = 15) -> "loguru.Logger":
    """Configure the logger with the specified settings."""
    return _logger_manager.configure(level)


def get_logger() -> "loguru.Logger":
    """Get the configured logger."""
    return _logger_manager.get_logger()


def add_sink(sink: str) -> int:
    """Add a sink to the logger."""
    return _logger_manager._logger.add(sink, format=Formatter().format)


# Convenience alias for backward compatibility
logger = get_logger()
