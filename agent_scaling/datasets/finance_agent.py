"""Finance-Agent dataset adapter.

The Finance-Agent benchmark is maintained upstream at
https://github.com/vals-ai/finance-agent. The upstream public test data lives
in ``data/public.csv`` (50 questions); the full Vals platform benchmark and
its grading harness require ``VALS_API_KEY`` access. The public CSV columns
relevant to grading are:

  - ``Question``: the natural-language task
  - ``Answer``: the ground-truth reference answer
  - ``Question Type``: e.g. "Market Analysis", "Calculation"
  - ``Expert time (mins)``: ~30 mins per question typical
  - ``Rubric``: JSON list of grading criteria, each with ``operator``
    (``correctness`` or ``contradiction``) and ``criteria`` (a fact the
    response must include or must not contradict)

This adapter normalises the CSV into instances. Grading uses the per-question
rubric when present (matching the spirit of Vals platform grading: each
criterion is judged by an LLM, score = fraction of correctness criteria
satisfied). When no rubric is present we fall back to a single LLM judge
against ``expected_answer``.
"""
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
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
from agent_scaling.logger import logger

DATASET_IDS = ["finance_agent", "finance-agent"]


@register_dataset_instance(DATASET_IDS)
class FinanceAgentInstance(DatasetInstance):
    task_id: str
    instructions: str
    expected_answer: Optional[str] = None
    tools: List[str] = Field(
        default_factory=lambda: ["web_search", "python_repl", "submit"]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        self.expected_output = self.expected_answer

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "task_id": self.task_id,
            "question": self.instructions,
        }


@register_dataset(DATASET_IDS)
class FinanceAgentDataset(Dataset):
    instances: List[FinanceAgentInstance]
    dataset_id: str = "finance_agent"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    _required_eval_prompts: List[str] = ["grader"]
    _require_llm_eval: bool = True

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[FinanceAgentInstance]
    ) -> Dict[str, Any]:
        return {
            "answer": instance_output.agent_output,
            "expected_output": instance_output.data_instance.expected_output,
        }

    @staticmethod
    def _yes(text: str) -> bool:
        """Lightweight 'yes' detector for LLM judge output."""
        t = text.strip().lower()
        # Accept either an explicit 'correct: yes' line (matches our grader
        # template) or a leading 'yes' answer.
        if "correct: yes" in t:
            return True
        if t.startswith("yes"):
            return True
        return False

    def _grade_one_criterion(
        self,
        question: str,
        response: str,
        criterion: Dict[str, Any],
    ) -> bool:
        """Score a single rubric criterion using the LLM judge.

        ``criterion`` follows the upstream Vals format: ``{"operator": "...",
        "criteria": "..."}``. We treat ``operator == "correctness"`` as
        "response must contain this fact" and ``operator == "contradiction"``
        as "response must not contradict this passage".
        """
        operator = (criterion.get("operator") or "").strip().lower()
        criteria_text = (criterion.get("criteria") or "").strip()
        if not criteria_text:
            return False
        assert self.eval_prompts is not None
        assert self.eval_llm is not None
        # Reuse the same grader prompt but with the criterion as the
        # "correct_answer". This lets a single LLM call decide whether the
        # response satisfies (or does not contradict) the criterion.
        prompt_message = self.eval_prompts["grader"].compile(
            question=question,
            response=response,
            correct_answer=criteria_text,
        )
        try:
            llm_response: BaseMessage = self.eval_llm.invoke(prompt_message)
            grader_text = llm_response.text()
        except Exception as exc:
            logger.warning(
                f"finance-agent rubric grading failed for criterion "
                f"{criterion!r}: {exc}"
            )
            return False
        passed = self._yes(grader_text)
        if operator == "contradiction":
            # For contradiction-style criteria, "yes" from the grader means
            # the response IS contradictory, which we treat as a failure.
            return not passed
        return passed

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[FinanceAgentInstance]
    ) -> Dict[str, Union[int, float, str]]:
        assert self.eval_prompts is not None, "eval_prompts must be set"
        assert self.eval_llm is not None, "eval_llm must be set"
        instance = instance_output.data_instance
        response = instance_output.agent_output or ""

        rubric = (instance.metadata or {}).get("rubric")
        if isinstance(rubric, list) and rubric:
            # Rubric-based grading (matches Vals platform methodology):
            # score = fraction of criteria the response satisfies.
            per_criterion = [
                self._grade_one_criterion(instance.instructions, response, c)
                for c in rubric
            ]
            n_pass = sum(1 for p in per_criterion if p)
            n_total = len(per_criterion)
            rubric_score = n_pass / n_total if n_total else 0.0
            return {
                "is_correct": rubric_score >= 0.5,
                "rubric_score": rubric_score,
                "n_criteria_satisfied": n_pass,
                "n_criteria_total": n_total,
                "grading_method": "rubric",
                "extracted_answer": response[:500],
            }

        # No rubric available — fall back to single LLM judge against the
        # reference answer (still LLM-based, matches the upstream public
        # grader style).
        prompt_message = self.eval_prompts["grader"].compile(
            question=instance.instructions,
            response=response,
            correct_answer=instance.expected_output,
        )
        llm_response: BaseMessage = self.eval_llm.invoke(prompt_message)
        grader_text = llm_response.text().strip().lower()
        is_correct = "correct: yes" in grader_text
        return {
            "is_correct": is_correct,
            "grading_method": "single_llm_judge",
            "extracted_answer": response[:500],
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not eval_outputs:
            return {"avg_accuracy": 0.0, "n": 0}
        correct = sum(1 for e in eval_outputs if bool(e.get("is_correct")))
        rubric_scores = [
            e["rubric_score"] for e in eval_outputs if "rubric_score" in e
        ]
        out: Dict[str, Any] = {
            "avg_accuracy": correct / len(eval_outputs),
            "n": len(eval_outputs),
            "n_correct": correct,
        }
        if rubric_scores:
            out["avg_rubric_score"] = sum(rubric_scores) / len(rubric_scores)
            out["n_rubric_graded"] = len(rubric_scores)
        return out
