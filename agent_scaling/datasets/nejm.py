import re
from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, field_validator

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance
from agent_scaling.logger import logger


@register_dataset_instance("nejm")
class NEJMInstance(DatasetInstance):
    task_id: str
    title: str
    medical_specialty: str
    primary_specialty: str
    overlap: Optional[str] = None
    final_diagnosis: str
    case_text: str
    year: str
    labs: Optional[str] = None
    case_text_with_labs: Optional[str] = None
    full_article_hyperlink: Optional[str] = None
    full_article_url: Optional[str] = None
    redacted_article_url: Optional[str] = None

    @field_validator("overlap", mode="before")
    @classmethod
    def validate_overlap(cls, v):
        """Convert overlap to string if it's an integer."""
        if isinstance(v, int):
            return str(v)
        return v

    @field_validator("year", mode="before")
    @classmethod
    def validate_year(cls, v):
        """Convert year to string if it's an integer."""
        if isinstance(v, int):
            return str(v)
        return v

    def model_post_init(self, __context):
        # For NEJM, the expected output is the final diagnosis
        self.expected_output = self.final_diagnosis

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "case_text": self.case_text,
            "medical_specialty": self.medical_specialty,
            "primary_specialty": self.primary_specialty,
            "year": self.year,
            "labs": self.labs or "",
            "case_text_with_labs": self.case_text_with_labs or self.case_text,
        }


def _strip_json_code_block(text: str) -> str:
    """
    Remove markdown code block formatting (```json ... ```) from LLM output.
    """
    text = text.strip()
    # Remove triple backtick code block with optional 'json' language
    code_block_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(code_block_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


@register_dataset("nejm")
class NEJMDataset(Dataset):
    """
    NEJM dataset with optional LLM-based grading for evaluation.
    Set use_llm_eval=True to use LLM for grading, otherwise uses string match.
    Optionally set grader_model to specify the LLM model for grading.
    """

    dataset_id: str = "nejm"
    instances: List[NEJMInstance]
    _required_eval_prompts: List[str] = ["grader"]

    def extract_answer(self, llm_output: str) -> str:
        """Extract the final diagnosis from the LLM output."""
        # Look for "Final Diagnosis:" pattern
        final_diagnosis_match = re.search(
            r"Final Diagnosis:\s*(.+)", llm_output, re.IGNORECASE | re.DOTALL
        )
        if final_diagnosis_match:
            return final_diagnosis_match.group(1).strip()

        # Look for "Diagnosis:" pattern
        diagnosis_match = re.search(
            r"Diagnosis:\s*(.+)", llm_output, re.IGNORECASE | re.DOTALL
        )
        if diagnosis_match:
            return diagnosis_match.group(1).strip()

        # Look for "The diagnosis is" pattern
        the_diagnosis_match = re.search(
            r"The diagnosis is\s*([^\.\n]+)", llm_output, re.IGNORECASE
        )
        if the_diagnosis_match:
            return the_diagnosis_match.group(1).strip()

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
            r"Analysis:\s*(.+)",
            r"Let me analyze this case:\s*(.+)",
            r"Here's my approach:\s*(.+)",
            r"Based on the clinical presentation:\s*(.+)",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no specific reasoning section, return everything except the final diagnosis
        diagnosis = self.extract_answer(llm_output)
        if diagnosis and diagnosis in llm_output:
            # Remove the diagnosis from the reasoning trace
            reasoning = llm_output.replace(diagnosis, "").strip()
            return reasoning

        return llm_output.strip()

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[NEJMInstance]
    ) -> Dict[str, Any]:
        return {
            "pred": self.extract_answer(instance_output.agent_output),
            "reasoning_trace": self.extract_reasoning_trace(
                instance_output.agent_output
            ),
            "full_response": instance_output.agent_output,
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
            "rationales": [
                e["rationale"]
                for e in eval_outputs
                if isinstance(e, dict) and e.get("rationale")
            ],
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[NEJMInstance]
    ) -> Dict[str, Union[int, float, str]]:
        """
        Evaluate the model output for a NEJM instance.
        If use_llm_eval is True, use the LLM grader and include the rationale (explanation) in the result.
        Otherwise, use string-matching logic and rationale is empty.
        """
        instance = instance_output.data_instance
        llm_output = instance_output.agent_output
        if self.use_llm_eval:
            # Use LLM grader utility
            reference = (
                instance.expected_output if instance.expected_output is not None else ""
            )
            assert self.eval_prompts is not None, "eval_llm must be set for evaluation"
            assert self.eval_llm is not None, "eval_llm must be set for evaluation"

            prompt_message = self.eval_prompts["grader"].compile(
                model_output=llm_output,
                reference=reference,
            )
            response: BaseMessage = self.eval_llm.invoke(prompt_message)
            response = cast(AIMessage, response)
            text = response.text() or ""
            if not text:
                logger.warning("LLM grader returned empty response!")
                result = {
                    "correct": 0,
                    "score": 0.0,
                    "explanation": "LLM grader returned empty response.",
                }
            try:
                import json

                cleaned = _strip_json_code_block(text)
                result = json.loads(cleaned)
            except Exception as e:
                logger.warning("Failed to parse LLM grader response as JSON:", text)
                result = {
                    "correct": 0,
                    "score": 0.0,
                    "explanation": f"Failed to parse LLM grader response: {text}",
                }

            # Ensure 'correct' is int for compatibility
            result["correct"] = int(result.get("correct", False))
            # Always include rationale/explanation
            result["rationale"] = result.get("explanation", "")
            return result
        else:
            # Original string-matching logic
            pred_diagnosis = self.extract_answer(llm_output)
            if instance.expected_output == "?":
                is_correct = (
                    len(pred_diagnosis.strip()) > 0 if pred_diagnosis else False
                )
            else:
                is_correct = (
                    pred_diagnosis.lower().strip()
                    == instance.expected_output.lower().strip()
                    if pred_diagnosis and instance.expected_output
                    else False
                )
            return {"correct": int(is_correct), "rationale": ""}

    def format_output_for_evaluation(
        self, task_id: str, llm_output: str
    ) -> Dict[str, str]:
        """Format output according to NEJM evaluation guidelines."""
        return {
            "task_id": task_id,
            "model_diagnosis": self.extract_answer(llm_output),
            "reasoning_trace": self.extract_reasoning_trace(llm_output),
        }
