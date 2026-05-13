"""Finance-Agent environment.

The upstream Finance-Agent benchmark (https://github.com/vals-ai/finance-agent)
evaluates an LLM agent's ability to perform multi-step quantitative reasoning
and tool use on financial-analyst-style tasks. The upstream framework exposes
five tools: ``web_search`` (Tavily), ``edgar_search`` (SEC EDGAR),
``parse_html_page``, ``retrieve_information``, and ``submit_final_result``.
Grading on the gated Vals platform applies the per-question grading rubric
(see the ``Rubric`` column of ``data/public.csv``) via LLM judges.

This environment mirrors a defensible subset of the upstream tool surface:

- ``web_search`` (Tavily) — the primary upstream tool, required to answer
  most factual questions in ``data/public.csv``. Wired via the existing
  ``tavily_search_simple_tool`` registered in
  ``agent_scaling.env.tools.search``.
- ``python_repl`` — for sandboxed Python (matches upstream's expectation
  that the agent performs arithmetic and computations).
- ``submit`` — terminal action; records the agent's final answer for
  downstream LLM grading.

Scoring uses the dataset's ``get_instance_eval_metrics`` method
(LLM-rubric grader, see ``prompts/eval/finance_agent-grader.yaml``) against
the ``expected_answer`` and (when present) the per-question grading rubric.
This mirrors the spirit of how the Vals platform grades; users who need
bit-for-bit upstream parity should run the upstream framework against the
Vals platform directly.
"""
from typing import Optional

from agent_scaling.env.base import AgentEnvironmentTools
from agent_scaling.logger import logger

from .registry import register_env
from .tools import cls_tool, get_tool


@register_env("finance-agent")
class FinanceAgentEnvironment(AgentEnvironmentTools):
    """Finance-Agent environment with web_search + python_repl + submit."""

    def __init__(self, *args, **kwargs):
        # The base class auto-registers @cls_tool methods (only ``submit``).
        # We also expose the upstream-aligned tools (web_search via Tavily,
        # python_repl) from the global tool registry. ``web_search`` requires
        # TAVILY_API_KEY; if missing, the agent runs with python_repl + submit
        # only and we log a warning so reviewers see the cause.
        additional = dict(kwargs.pop("additional_env_tools", {}) or {})
        additional.setdefault("python_repl", get_tool("python_repl"))
        # web_search is registered in agent_scaling.env.tools.search under the
        # name "web_search" (Tavily-backed). It requires TAVILY_API_KEY; the
        # tools/__init__.py module already guards against missing key so the
        # registry simply won't contain it when key is absent.
        try:
            additional.setdefault("web_search", get_tool("web_search"))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"FinanceAgentEnvironment: web_search unavailable ({exc}); "
                "the agent will see only python_repl + submit. Set "
                "TAVILY_API_KEY to enable upstream-aligned tool use."
            )
        kwargs["additional_env_tools"] = additional
        super().__init__(*args, **kwargs)
        self.is_done: bool = False
        self.success: bool = False
        self.num_steps: int = 0
        self.final_answer: Optional[str] = None

    @cls_tool
    def submit(self, reasoning: str) -> str:
        """Submit the agent's final answer for this Finance-Agent task.

        The full ``reasoning`` content is returned verbatim so that the
        downstream LLM grader receives the actual answer text. The grader
        expects an extractable ``Final Answer: ...`` line, which we wrap
        around the reasoning here.

        Args:
            reasoning: The final answer (and optional supporting reasoning).

        Returns:
            The formatted final-answer string consumed by the grader.
        """
        self.final_answer = reasoning
        self.is_done = True
        # success is determined post-hoc by the LLM grader; we record submission
        # but do not mark success here. The dataset's eval method updates this.
        self.success = True
        return f"Final Answer: {reasoning}"

    def env_done(self) -> bool:
        return self.is_done
