import json

from langchain_core.messages.tool import ToolCall, ToolMessage
from tavily import TavilyClient

from agent_scaling.logger import logger

from .base import AgentEnvironmentTools
from .registry import register_env
from .tools import cls_tool
from .tools.search import RateLimiter


@register_env("web-search")
class WebSearchEnvironment(AgentEnvironmentTools):
    def __init__(self, *args, **kwargs):
        self.rate_limiter = RateLimiter(max_calls=20, time_window=60)
        self.rate_limit_count = 0
        super().__init__(*args, **kwargs)

    @cls_tool
    def web_search(self, query: str) -> str:
        """
        Search the web for information related to the query using Tavily.

        Args:
            query: The search query to look up.

        Returns:
            A summary of search results with relevant information as a JSON string.
        """
        self.rate_limiter.wait_if_needed()
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
            logger.error(f"Tool execution error for web_search: {e}")
            error_response = {
                "query": query,
                "results": [],
                "error": str(e),
                "summary": f"Error searching for '{query}': {str(e)}",
            }
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                self.rate_limit_count += 1
            return json.dumps(error_response, indent=2)

    @cls_tool
    def done(self, answer: str, confidence_score: int) -> str:
        """
        Indicate the best guess answer and a confidence score between 0% to 100%.
        """
        return f"{answer}\t{confidence_score}"

    def execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """
        Execute a single tool.
        """
        if "topic" in tool_call["args"]:
            tool_call["args"]["topic"] = "general"
        return self.tools[tool_call["name"]].invoke(tool_call)

    def should_stop_due_to_rate_limiting(self) -> bool:
        return self.rate_limit_count >= 3
