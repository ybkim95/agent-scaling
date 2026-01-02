from agent_scaling.env.browsecomp_utils.handler import SearchToolHandler

from .base import AgentEnvironmentTools
from .registry import register_env
from .tools import cls_tool

_search_tool_handler = None


@register_env("browsecomp-plus")
class BrowseCompPlusEnvironment(AgentEnvironmentTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _search_tool_handler
        if _search_tool_handler is None:
            _search_tool_handler = (
                SearchToolHandler()
            )  # otherwise need to reload everytime
        self.search_handler = _search_tool_handler

    @cls_tool
    def search_documents(self, query: str) -> str:
        """
        Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits).

        Args:
            query: The search query to look up.

        Returns:
            A summary of search results with relevant information as a JSON string.
        """
        ret = self.search_handler.execute_tool("search", {"query": query})
        return ret

    @cls_tool
    def retrieve_document(self, docid: str) -> str:
        """
        Retrieve the full content of a document by its ID.

        Args:
            docid: The ID of the document to retrieve.

        Returns:
            The full text content of the specified document.
        """
        ret = self.search_handler.execute_tool("get_document", {"docid": docid})
        return ret

    @cls_tool
    def done(self, answer: str, confidence_score: int) -> str:
        """
        Indicate the best guess answer and a confidence score between 0% to 100%.
        """
        return f"Final Answer:{answer}\nConfidence:{confidence_score}"
