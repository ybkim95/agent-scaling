import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from typing import Type

from dotenv import load_dotenv
from transformers import AutoTokenizer

from agent_scaling.env.browsecomp_utils.searchers import SearcherType
from agent_scaling.env.browsecomp_utils.searchers.base import BaseSearcher
from agent_scaling.logger import logger

load_dotenv()


class SearchToolHandler:
    def __init__(
        self,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        parser = argparse.ArgumentParser(
            description="Call OpenAI Responses API with native function calling and local search."
        )
        searcher_class: Type[BaseSearcher] = SearcherType.get_searcher_class("faiss")
        searcher_class.parse_args(parser)
        args = parser.parse_args()
        self.searcher = searcher_class(args)

        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    # def get_tool_definitions(self):
    #     tools = [
    #         {
    #             "type": "function",
    #             "name": "search",
    #             "description": self.searcher.search_description(self.k),
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "query": {
    #                         "type": "string",
    #                         "description": "Search query string",
    #                     }
    #                 },
    #                 "required": ["query"],
    #                 "additionalProperties": False,
    #             },
    #             "strict": True,
    #         }
    #     ]

    #     if self.include_get_document:
    #         tools.append(
    #             {
    #                 "type": "function",
    #                 "name": "get_document",
    #                 "description": self.searcher.get_document_description(),
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "docid": {
    #                             "type": "string",
    #                             "description": "Document ID to retrieve",
    #                         }
    #                     },
    #                     "required": ["docid"],
    #                     "additionalProperties": False,
    #                 },
    #                 "strict": True,
    #             }
    #         )

    #     return tools

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            import time

            start_time = time.time()
            ret = self._search(arguments["query"])
            end_time = time.time() - start_time
            logger.debug(f"Search time: {end_time:.3f} seconds")
            return ret
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)
