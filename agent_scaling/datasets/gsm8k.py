import re
from typing import Any, Dict, List, Union

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance


@register_dataset_instance("gsm8k")
class GSM8KInstance(DatasetInstance):
    question: str
    answer: str

    def model_post_init(self, context: Any) -> None:
        self.expected_output = self.answer.split("#### ")[-1]
        self.question = self.question.strip()

    def get_prompt_info(self) -> Dict[str, str]:
        return {"question": self.question}


@register_dataset("gsm8k")
class GSM8KDataset(Dataset):
    dataset_id: str = "gsm8k"
    instances: List[GSM8KInstance]

    def extract_answer(self, llm_output: str) -> str:
        pred = llm_output.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
        return pred[0] if len(pred) > 0 else ""

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[GSM8KInstance]
    ) -> Dict[str, Any]:
        return {
            "pred": self.extract_answer(instance_output.agent_output),
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
        self, instance_output: DatasetInstanceOutput[GSM8KInstance]
    ) -> Dict[str, Union[int, float]]:
        return {
            "correct": int(
                self.extract_answer(instance_output.agent_output)
                == instance_output.data_instance.answer
            ),
        }
