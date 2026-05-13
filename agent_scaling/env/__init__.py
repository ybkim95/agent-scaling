from .base import AgentEnvironment
from .basic import BasicEnvironment
try:
    from .browsecomp import BrowseCompPlusEnvironment
except ImportError:
    pass  # tevatron/torch not installed — browsecomp env unavailable

# from .browsecomp import BrowseCompPlusEnvironment
from .finance_agent import FinanceAgentEnvironment
from .plancraft import PlancraftEnvironment
from .registry import (
    T,
    get_env,
    get_env_cls,
    is_env_registered,
    list_envs,
    register_env,
)
from .swebench import SWEBenchEnvironment
from .terminalbench import TerminalBenchEnvironment
try:
    from .web_search import WebSearchEnvironment
except Exception:  # pragma: no cover
    # web-search relies on Tavily, which validates TAVILY_API_KEY at module
    # import time. Keep the rest of the env registry importable when no key
    # is set (e.g. during reviewer code inspection).
    pass
from .workbench import WorkbenchEnvironment
