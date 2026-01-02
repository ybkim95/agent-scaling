import re
from typing import Any, Dict, List, Union

from pydantic import field_validator

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance


@register_dataset_instance("gaia")
class GAIAInstance(DatasetInstance):
    task_id: str
    question: str
    level: str
    final_answer: str
    file_name: str
    annotator_metadata: Dict[str, str]

    @field_validator("level", mode="before")
    @classmethod
    def validate_level(cls, v):
        """Convert level to string if it's an integer."""
        if isinstance(v, int):
            return str(v)
        return v

    def model_post_init(self, __context):
        # For GAIA, the expected output is the final answer
        self.expected_output = self.final_answer

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "question": self.question,
            "task_id": self.task_id,
            "level": self.level,
        }


@register_dataset("gaia")
class GAIADataset(Dataset):
    dataset_id: str = "gaia"
    instances: List[GAIAInstance]

    def extract_answer(self, llm_output: str) -> str:
        """Extract the final answer from the LLM output."""
        # Look for "Final Answer:" pattern
        final_answer_match = re.search(
            r"Final Answer:\s*(.+)", llm_output, re.IGNORECASE | re.DOTALL
        )
        if final_answer_match:
            return final_answer_match.group(1).strip()

        # Look for "Answer:" pattern
        answer_match = re.search(
            r"Answer:\s*(.+)", llm_output, re.IGNORECASE | re.DOTALL
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Look for "The answer is" pattern
        the_answer_match = re.search(
            r"The answer is\s*([^\.\n]+)", llm_output, re.IGNORECASE
        )
        if the_answer_match:
            return the_answer_match.group(1).strip()

        # If no pattern matches, return the last sentence or phrase
        lines = llm_output.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if last_line:
                return last_line

        return llm_output.strip()

    def extract_reasoning_trace(self, llm_output: str) -> str:
        """Extract the reasoning trace from the LLM output."""
        # Look for reasoning sections
        reasoning_patterns = [
            r"Reasoning:\s*(.+)",
            r"Steps:\s*(.+)",
            r"Let me think through this:\s*(.+)",
            r"Here's my approach:\s*(.+)",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no specific reasoning section, return everything except the final answer
        answer = self.extract_answer(llm_output)
        if answer and answer in llm_output:
            # Remove the answer from the reasoning trace
            reasoning = llm_output.replace(answer, "").strip()
            return reasoning

        return llm_output.strip()

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[GAIAInstance]
    ) -> Dict[str, Any]:
        llm_output = instance_output.agent_output
        return {
            "pred": self.extract_answer(llm_output),
            "reasoning_trace": self.extract_reasoning_trace(llm_output),
            "full_response": llm_output,
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "accuracy": round(
                sum(
                    [
                        e["correct"]
                        for e in eval_outputs
                        if isinstance(e, dict) and "correct" in e
                    ]
                )
                / len(eval_outputs),
                4,
            ),
            "num_instances": len(eval_outputs),
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[GAIAInstance]
    ) -> Dict[str, Union[int, float]]:
        llm_output = instance_output.agent_output
        instance = instance_output.data_instance
        pred_answer = self.extract_answer(llm_output)

        # For GAIA, we need to handle the special case where the answer is "?"
        # which means the model should provide its own answer
        if instance.expected_output == "?":
            # If expected output is "?", we can't determine correctness
            # but we can check if the model provided a non-empty answer
            is_correct = len(pred_answer.strip()) > 0 if pred_answer else False
        else:
            # Simple exact match for now - could be enhanced with fuzzy matching
            is_correct = (
                pred_answer.lower().strip() == instance.expected_output.lower().strip()
                if pred_answer and instance.expected_output
                else False
            )

        return {
            "correct": int(is_correct),
        }

    def format_output_for_evaluation(
        self, task_id: str, llm_output: str
    ) -> Dict[str, str]:
        """Format output according to GAIA evaluation guidelines."""
        return {
            "task_id": task_id,
            "model_answer": self.extract_answer(llm_output),
            "reasoning_trace": self.extract_reasoning_trace(llm_output),
        }
