import base64
import hashlib
import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from pydantic import Field

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


DATASET_IDS = [
    "browsecomp",
    "browsecomp_sampled_5_per_category",
    "browsecomp_plus_sampled_100",
]


@register_dataset_instance(DATASET_IDS)
class BrowseCompInstance(DatasetInstance):
    problem: Optional[str] = None
    answer: Optional[str] = None
    canary: Optional[str] = None
    index: Optional[int] = None
    category: Optional[str] = None
    encrypted_problem: Optional[str] = None
    encrypted_answer: Optional[str] = None

    def model_post_init(self, __context):
        if self.problem is None and self.answer is None:
            has_encrypted = (
                (self.encrypted_problem is not None)
                and (self.encrypted_answer is not None)
                and (self.canary is not None)
            )
            if not has_encrypted:
                raise ValueError(
                    "Either problem and answer must be provided, or encrypted_problem, encrypted_answer, and canary."
                )
        self.expected_output = self.answer

    def get_prompt_info(self) -> Dict[str, str]:
        assert self.problem is not None, "Problem must be decrypted"
        assert self.answer is not None, "Answer must be decrypted"
        return {
            "question": self.problem,
        }


@register_dataset(DATASET_IDS)
class BrowseCompDataset(Dataset):
    instances: List[BrowseCompInstance]
    dataset_id: str = "browsecomp"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    _required_eval_prompts: List[str] = ["grader"]
    _require_llm_eval: bool = True

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[BrowseCompInstance]
    ) -> Dict[str, Any]:
        llm_output = instance_output.agent_output
        return {
            "answer": llm_output,
            "expected_output": instance_output.data_instance.expected_output,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[BrowseCompInstance]
    ) -> Dict[str, Union[int, float, str]]:
        assert self.eval_prompts is not None, "eval_llm must be set for evaluation"
        assert self.eval_llm is not None, "eval_llm must be set for evaluation"
        instance = instance_output.data_instance
        prompt_message = self.eval_prompts["grader"].compile(
            question=instance.problem,
            response=instance_output.agent_output,
            correct_answer=instance.expected_output,
        )

        response: BaseMessage = self.eval_llm.invoke(prompt_message)
        grader_text = response.text().strip()
        is_correct = "correct: yes" in grader_text
        extracted_answer = "Not found"
        if "extracted_final_answer:" in grader_text:
            answer_line = (
                grader_text.split("extracted_final_answer:")[1].split("\n")[0].strip()
            )
            if answer_line and answer_line != "none":
                extracted_answer = answer_line

        confidence = 100
        if "confidence:" in grader_text:
            conf_match = re.search(
                r"(\d+)", grader_text.split("confidence:")[1].split("\n")[0]
            )
            if conf_match:
                confidence = float(conf_match.group(1))

        return {
            "is_correct": is_correct,
            "extracted_answer": extracted_answer,
            "confidence": confidence,
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "avg_accuracy": sum(e["is_correct"] for e in eval_outputs)
            / len(eval_outputs),
            "avg_confidence": sum(e["confidence"] for e in eval_outputs)
            / len(eval_outputs),
        }
