from diskcache import FanoutCache
from .utils import get_root_dir
import os.path as osp

_cache = None


def get_function_cache() -> FanoutCache:
    global _cache
    if _cache is None:
        _cache = FanoutCache(osp.join(get_root_dir(), ".function_cache"))
    return _cache
