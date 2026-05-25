"""WorkBench environment.

The upstream WorkBench benchmark (https://github.com/olly-styles/WorkBench)
provides 16 mock business tools (email, calendar, customer service, etc.) and
evaluates whether an agent's tool-call sequence matches expected actions.

This environment provides the integration point used by the five coordination
architectures in this repository. It exposes the common business tools as
stubs that record the agent's calls into ``self.action_log``, plus a
``submit`` tool through which the agent reports a final action summary. The
dataset's ``get_instance_eval_metrics`` method then structurally grades the
submission by parsing the agent's tool-call sequence and comparing it with
the ``expected_actions`` list produced by ``scripts/_convert_workbench.py``;
no LLM grader is used by the released WorkBench adapter.

For bit-for-bit fidelity with the upstream WorkBench evaluator (which checks
expected_actions exactly), users should run the upstream framework directly;
this environment is intended for architectural comparison studies where
coordination structure, not exact action-by-action grading, is the object of
interest.
"""
from typing import Any, Dict, List, Optional

from agent_scaling.env.base import AgentEnvironmentTools

from .registry import register_env
from .tools import cls_tool


@register_env("workbench")
class WorkbenchEnvironment(AgentEnvironmentTools):
    """WorkBench environment with stub business tools + ``submit``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_done: bool = False
        self.success: bool = False
        self.num_steps: int = 0
        self.final_answer: Optional[str] = None
        self.action_log: List[Dict[str, Any]] = []

    def _log(self, tool: str, **args) -> str:
        entry = {"tool": tool, "args": args}
        self.action_log.append(entry)
        return f"OK: recorded {tool}({args})"

    @cls_tool
    def send_email(self, to: str, subject: str, body: str) -> str:
        """Record an outgoing email."""
        return self._log("send_email", to=to, subject=subject, body=body)

    @cls_tool
    def search_emails(self, query: str) -> str:
        """Search the mock email inbox; returns a deterministic placeholder."""
        self._log("search_emails", query=query)
        return f"(stub) 0 emails matched query: {query!r}"

    @cls_tool
    def create_event(self, title: str, attendees: str, date: str, time: str) -> str:
        """Create a calendar event."""
        return self._log(
            "create_event",
            title=title,
            attendees=attendees,
            date=date,
            time=time,
        )

    @cls_tool
    def search_events(self, query: str) -> str:
        """Search the mock calendar; returns a deterministic placeholder."""
        self._log("search_events", query=query)
        return f"(stub) 0 events matched query: {query!r}"

    @cls_tool
    def submit(self, reasoning: str) -> str:
        """Submit the agent's final action plan and answer.

        The full ``reasoning`` content is returned verbatim alongside a
        compact summary of the business-tool calls the agent already made.
        The downstream structural grader parses both signals, extracting
        action calls from the returned text and comparing them with the
        ``expected_actions`` list; no LLM grader is used in the released
        WorkBench adapter.

        Args:
            reasoning: A natural-language summary of the tool calls / actions
                taken, plus the final answer if applicable.

        Returns:
            The formatted final-answer string consumed by the grader.
        """
        self.final_answer = reasoning
        self.is_done = True
        self.success = True
        action_calls = "; ".join(
            f"{e['tool']}({e['args']})" for e in self.action_log
        )
        return (
            f"Final Answer: {reasoning}\n"
            f"Recorded tool calls during this task: "
            f"{action_calls or '(none)'}"
        )

    def env_done(self) -> bool:
        return self.is_done
