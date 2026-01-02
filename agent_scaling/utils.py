import os.path as osp
import time
from typing import List, Optional

import litellm
import orjson
import yaml
from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType

from agent_scaling.llm.callbacks import LocalLogger
from agent_scaling.logger import logger


def get_root_dir():
    return osp.dirname(osp.dirname(__file__))


def get_run_conf_dir():
    return osp.join(get_root_dir(), "run_conf")


def enable_local_logging(prompt_only: bool = False):
    import litellm._logging as litellm_logging

    litellm_logging._disable_debugging()  # type: ignore
    litellm.callbacks.append(LocalLogger(prompt_only=prompt_only))


def clear_callbacks():
    litellm.callbacks = []


def add_local_cache(**kwargs):
    litellm.cache = Cache(type=LiteLLMCacheType.DISK, **kwargs)


def enable_local_cache(**kwargs):
    if litellm.cache is None:
        litellm.cache = Cache(type=LiteLLMCacheType.DISK, **kwargs)
    litellm.enable_cache()


def disable_local_cache():
    litellm.disable_cache()


def read_yaml(path: str, time_it: bool = False):
    if time_it:
        start = time.time()
        file_size = osp.getsize(path)
        if file_size > 100:
            logger.info(
                f"Reading yaml file of size {format_bytes(file_size)} from {osp.basename(path)}"
            )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if time_it:
        end = time.time()
        logger.info(f"Time taken reading yaml: {end - start:.2f} seconds")
    return data


def write_yaml(
    data: dict,
    path: Optional[str] = None,
    time_it: bool = False,
    use_long_str_representer: bool = False,
    truncate_floats: bool = False,
    **kwargs,
) -> Optional[str]:
    if time_it:
        start = time.time()

    # Configure YAML representers
    if use_long_str_representer:

        def str_presenter(dumper, data):
            if len(data.splitlines()) > 1:  # Check for multiline string
                data = "\n".join([line.rstrip() for line in data.strip().splitlines()])
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            else:
                return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

        yaml.add_representer(str, str_presenter)

    if truncate_floats:

        def truncate_floats_presenter(dumper, data):
            if isinstance(data, float):
                return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data:.3f}")
            return dumper.represent_scalar("tag:yaml.org,2002:float", float(data))

        yaml.add_representer(float, truncate_floats_presenter)
    else:

        def truncate_floats_presenter(dumper, data):
            if isinstance(data, float):
                return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data}")
            return dumper.represent_scalar("tag:yaml.org,2002:float", float(data))

        yaml.add_representer(float, truncate_floats_presenter)

    if path is None:
        # Return YAML as string
        return yaml.dump(data, **kwargs)
    else:
        # Write to file
        with open(path, "w") as f:
            yaml.dump(data, f, **kwargs)

    if time_it:
        end = time.time()
        logger.info(f"Time taken writing yaml: {end - start:.2f} seconds")


def write_json(data: dict, path: str, time_it: bool = False, indent: bool = False):
    if time_it:
        start = time.time()
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else None))
    if time_it:
        end = time.time()
        logger.info(f"Time taken writing json: {end - start:.2f} seconds")


def read_json(path: str, time_it: bool = False):
    if time_it:
        start = time.time()
        file_size = osp.getsize(path)
        if file_size > 100:
            logger.info(
                f"Reading json file of size {format_bytes(file_size)} from {osp.basename(path)}"
            )
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    if time_it:
        end = time.time()
        logger.info(f"Time taken reading json: {end - start:.2f} seconds")
    return data


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human readable string with appropriate unit (B, KB, MB, GB, TB).

    Args:
        num_bytes: Number of bytes to format

    Returns:
        Formatted string with appropriate unit
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            if unit == "B":
                return f"{num_bytes} {unit}"
            return f"{num_bytes:.2f} {unit}"
        num_bytes = int(num_bytes / 1024)
    return f"{num_bytes:.2f} TB"


def join_with_leading_dash(items: List[str], dash_prefix: str = "- ") -> str:
    if len(items) == 0:
        return ""
    return dash_prefix + f"\n{dash_prefix}".join(items)
