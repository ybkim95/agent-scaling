import re
from typing import Any, Dict, List, Optional, Union

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance


@register_dataset_instance("healthbench")
class HealthBenchInstance(DatasetInstance):
    prompt_id: str
    prompt: str
    expected_output: Optional[str] = None
    rubric_items: Optional[List[Dict[str, Any]]] = None
    cluster: Optional[str] = None
    axis: Optional[str] = None
    theme: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context):
        # For HealthBench, the expected output is the professional medical response
        # If not provided, we'll use a placeholder for evaluation
        if self.expected_output is None:
            self.expected_output = "Professional medical response"

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "cluster": self.cluster or "",
            "axis": self.axis or "",
            "theme": self.theme or "",
            "difficulty": self.difficulty or "",
        }


@register_dataset("healthbench")
class HealthBenchDataset(Dataset):
    dataset_id: str = "healthbench"
    instances: List[HealthBenchInstance]

    def extract_answer(self, llm_output: str) -> str:
        """Extract the medical response from the LLM output."""
        # For HealthBench, the entire output is the medical response
        # We'll look for any structured sections and extract the main response

        # Look for "Response:" or "Answer:" patterns
        response_patterns = [
            r"Response:\s*(.+)",
            r"Answer:\s*(.+)",
            r"Medical Response:\s*(.+)",
            r"Professional Response:\s*(.+)",
        ]

        for pattern in response_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no specific pattern, return the entire output
        return llm_output.strip()

    def extract_reasoning_trace(self, llm_output: str) -> str:
        """Extract the reasoning trace from the LLM output."""
        # Look for reasoning sections
        reasoning_patterns = [
            r"Reasoning:\s*(.+)",
            r"Analysis:\s*(.+)",
            r"Let me think through this:\s*(.+)",
            r"Here's my approach:\s*(.+)",
            r"Based on the medical information:\s*(.+)",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no specific reasoning section, return everything except the final response
        response = self.extract_answer(llm_output)
        if response and response in llm_output:
            # Remove the response from the reasoning trace
            reasoning = llm_output.replace(response, "").strip()
            return reasoning

        return llm_output.strip()

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[HealthBenchInstance]
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
        self, instance_output: DatasetInstanceOutput[HealthBenchInstance]
    ) -> Dict[str, Union[int, float]]:
        instance = instance_output.data_instance
        llm_output = instance_output.agent_output
        pred_response = self.extract_answer(llm_output)

        # For HealthBench, we evaluate based on rubric criteria if available
        if instance.rubric_items:
            # Calculate score based on rubric items
            total_points = 0
            achieved_points = 0

            for item in instance.rubric_items:
                points = item.get("points", 0)
                if points > 0:
                    total_points += points
                    if item.get("criteria_met", False):
                        achieved_points += points

            if total_points > 0:
                score = achieved_points / total_points
                is_correct = score >= 0.5  # Consider 50% or higher as correct
            else:
                is_correct = len(pred_response.strip()) > 0 if pred_response else False
        else:
            # Fallback: check if response is non-empty
            is_correct = len(pred_response.strip()) > 0 if pred_response else False

        return {
            "correct": int(is_correct),
        }

    def format_output_for_evaluation(
        self, prompt_id: str, llm_output: str
    ) -> Dict[str, str]:
        """Format output according to HealthBench evaluation guidelines."""
        return {
            "prompt_id": prompt_id,
            "medical_response": self.extract_answer(llm_output),
            "reasoning_trace": self.extract_reasoning_trace(llm_output),
        }
