import json
import threading
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from tavily import TavilyClient

from agent_scaling.logger import logger

from .registry import register_tool

load_dotenv()


class RateLimiter:
    """Global rate limiter for API calls"""

    def __init__(self, max_calls: int = 40, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.time_window - now + 0.1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                while self.calls and self.calls[0] < time.time() - self.time_window:
                    self.calls.popleft()

            self.calls.append(time.time())


TAVILY_RATE_LIMITER = RateLimiter(max_calls=20, time_window=60)


@register_tool
@tool
def web_search(query: str) -> str:
    """
    Search the web for information related to the query using Tavily.

    Args:
        query: The search query to look up.

    Returns:
        A summary of search results with relevant information as a JSON string.
    """
    TAVILY_RATE_LIMITER.wait_if_needed()
    client = TavilyClient()
    try:
        response = client.search(query)
        # Format the results for the agent
        results = []
        for result in response.get("results", []):
            results.append(
                {
                    "title": result.get("title", ""),
                    "snippet": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                }
            )
        search_response = {
            "query": query,
            "results": results,
            "answer": response.get("answer", ""),
            "summary": response.get(
                "answer", f"Found {len(results)} results for '{query}'."
            ),
        }
        return json.dumps(search_response, indent=2)
    except Exception as e:
        error_response = {
            "query": query,
            "results": [],
            "error": str(e),
            "summary": f"Error searching for '{query}': {str(e)}",
        }
        return json.dumps(error_response, indent=2)


def parse_search_output(output: Any) -> str:
    if isinstance(output, dict):
        if "answer" in output:
            ret = output["answer"]
        else:
            ret = json.dumps(output, indent=2)
    else:
        ret = json.dumps(output, indent=2)
    return ret


tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_raw_content=True,
    include_answer=True,
    search_depth="basic",  # "advanced"
)


tavily_search_simple_tool = TavilySearch(
    name="search_simple",
    topic="general",
    max_results=5,
    include_raw_content=False,
    include_answer=True,
    search_depth="basic",  # "advanced"
)

tavily_search_tool = register_tool(tavily_search_tool)
# tavily_search_simple_tool = enhance_tool(
#     tavily_search_simple_tool, output_parse_func=parse_search_output
# )
tavily_search_simple_tool = register_tool(tavily_search_simple_tool)
