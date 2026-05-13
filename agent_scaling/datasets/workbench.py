"""WorkBench dataset adapter.

WorkBench is maintained upstream at https://github.com/olly-styles/WorkBench.
Tasks are stored as 6 per-domain CSVs under
``data/processed/queries_and_answers/`` with columns
``query, answer, base_template, chosen_template, domains`` where ``answer``
is a Python-literal list of function-call strings like
``['analytics.create_plot.func(time_min="2023-11-21", ...)']``.

Upstream grading (see ``src/evals/utils.py``) executes both the predicted
and ground-truth actions on the mock business databases and compares the
resulting state. The full state-comparison evaluator requires porting the
upstream's stateful tool implementations (16 tools across 5 domains and 5
mock databases), which is non-trivial.

This adapter implements two grading paths, both honestly described:

- ``is_exact_match``: parses tool calls from the agent's submitted text and
  compares (tool name + key arguments) against the ground-truth action list.
  Mirrors upstream's ``is_exact_match`` function. Robust and reproducible.
- ``is_correct``: not implemented in this adapter (requires state-change
  simulation against the upstream mock databases). Users seeking that
  ground-truth metric should run the upstream framework directly. We report
  ``is_exact_match`` as ``is_correct`` for compatibility with experiment-
  runner accounting, but the metrics dictionary includes both fields so the
  distinction is preserved.
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import Field

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutput,
)
from agent_scaling.datasets.registry import (
    register_dataset,
    register_dataset_instance,
)

DATASET_IDS = ["workbench"]

# Match upstream-style function calls like:
#   analytics.create_plot.func(time_min="2023-11-21", value_to_plot="x")
# We also tolerate variants the agent may emit:
#   create_plot(time_min="2023-11-21", value_to_plot="x")
#   send_email(to="bob@example.com", subject="...", body="...")
_CALL_RE = re.compile(
    r"(?P<name>[a-zA-Z_][\w\.]*)\s*(?:\.func)?\s*\((?P<args>[^)]*)\)"
)


def _parse_calls(text: str) -> List[Tuple[str, Set[str]]]:
    """Extract (function_name_tail, set_of_kwarg_names) from text.

    We compare tool calls by their tail name (last dotted segment) and the
    set of keyword argument names. This is robust to whitespace, ordering,
    and minor formatting differences without simulating execution.
    """
    out: List[Tuple[str, Set[str]]] = []
    for m in _CALL_RE.finditer(text or ""):
        name = m.group("name") or ""
        tail = name.rsplit(".", 1)[-1]
        args_blob = m.group("args") or ""
        kw_names: Set[str] = set()
        for kv in args_blob.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" in kv:
                k = kv.split("=", 1)[0].strip()
                if k:
                    kw_names.add(k)
        out.append((tail, kw_names))
    return out


def _is_exact_match(
    predicted_actions: List[Tuple[str, Set[str]]],
    expected_actions: List[Tuple[str, Set[str]]],
) -> bool:
    """Equivalent of upstream is_exact_match: same actions, same order."""
    if len(predicted_actions) != len(expected_actions):
        return False
    for (pn, pa), (en, ea) in zip(predicted_actions, expected_actions):
        if pn != en or pa != ea:
            return False
    return True


def _is_set_match(
    predicted_actions: List[Tuple[str, Set[str]]],
    expected_actions: List[Tuple[str, Set[str]]],
) -> bool:
    """Order-insensitive variant: same multiset of (name, args) pairs."""
    return sorted(predicted_actions) == sorted(expected_actions)


@register_dataset_instance(DATASET_IDS)
class WorkbenchInstance(DatasetInstance):
    task_id: str
    instructions: str
    expected_actions: List[Dict[str, Any]] = Field(default_factory=list)
    expected_answer: Optional[str] = None
    tools: List[str] = Field(
        default_factory=lambda: [
            "send_email",
            "search_emails",
            "create_event",
            "search_events",
            "submit",
        ]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        # If only expected_actions is given, derive a textual reference that
        # contains the function-call strings for downstream comparison.
        if self.expected_answer:
            self.expected_output = self.expected_answer
        else:
            calls = [a.get("call") for a in self.expected_actions if a.get("call")]
            self.expected_output = "\n".join(calls) if calls else ""

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "task_id": self.task_id,
            "question": self.instructions,
        }


@register_dataset(DATASET_IDS)
class WorkbenchDataset(Dataset):
    instances: List[WorkbenchInstance]
    dataset_id: str = "workbench"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Grading is structural (regex-based action matching). No LLM needed,
    # but the experiment runner still wires eval_llm/eval_prompts through
    # config; we accept them and ignore them in the structural path.
    _required_eval_prompts: List[str] = []
    _require_llm_eval: bool = False

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[WorkbenchInstance]
    ) -> Dict[str, Any]:
        return {
            "answer": instance_output.agent_output,
            "expected_output": instance_output.data_instance.expected_output,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[WorkbenchInstance]
    ) -> Dict[str, Union[int, float, str, bool]]:
        instance = instance_output.data_instance
        response = instance_output.agent_output or ""

        # Build the expected-actions list from the normalised JSON.
        expected_calls_text = "\n".join(
            (a.get("call") or "") for a in instance.expected_actions
        )
        expected_actions = _parse_calls(expected_calls_text)
        predicted_actions = _parse_calls(response)

        exact_match = _is_exact_match(predicted_actions, expected_actions)
        set_match = _is_set_match(predicted_actions, expected_actions)

        return {
            # `is_correct` is wired so the experiment-runner accounting works;
            # we treat exact_match as the most defensible structural proxy for
            # upstream's state-based is_correct.
            "is_correct": exact_match,
            "is_exact_match": exact_match,
            "is_set_match": set_match,
            "n_predicted_calls": len(predicted_actions),
            "n_expected_calls": len(expected_actions),
            "grading_method": "structural_action_match",
            "extracted_answer": response[:500],
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Defensive: exp_runner inserts {"resolved": 0, ...} on agent failure.
        if not eval_outputs:
            return {"avg_accuracy": 0.0, "n": 0}
        exact = sum(1 for e in eval_outputs if bool(e.get("is_exact_match")))
        sset = sum(1 for e in eval_outputs if bool(e.get("is_set_match")))
        correct = sum(1 for e in eval_outputs if bool(e.get("is_correct")))
        return {
            "avg_accuracy": correct / len(eval_outputs),
            "exact_match_rate": exact / len(eval_outputs),
            "set_match_rate": sset / len(eval_outputs),
            "n": len(eval_outputs),
            "n_correct": correct,
        }
