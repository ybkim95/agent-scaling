import re
from typing import Any, Dict, List, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance


@register_dataset_instance(name=["simpleqa", "simpleqa_sampled"])
class SimpleQAInstance(DatasetInstance):
    answer: str
    domain: str
    question: Optional[str] = None
    problem: Optional[str] = None

    def model_post_init(self, __context):
        if self.question is None and self.problem is not None:
            self.question = self.problem
        self.expected_output = self.answer

    def get_prompt_info(self) -> Dict[str, str]:
        assert self.question is not None, "Question must be set for SimpleQAInstance"
        return {"question": self.question}


@register_dataset(name=["simpleqa", "simpleqa_sampled"])
class SimpleQADataset(Dataset):

    dataset_id: str = "simpleqa"
    instances: List[SimpleQAInstance]
    _required_eval_prompts: List[str] = ["grader"]
    _require_llm_eval: bool = True

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[SimpleQAInstance]
    ) -> Dict[str, Any]:
        agent_output = instance_output.agent_output
        try:
            # Parse the SimpleQA format: Explanation, Exact Answer, Confidence
            lines = agent_output.strip().split('\n')
            answer = ""
            confidence = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith("Exact Answer:"):
                    answer = line.replace("Exact Answer:", "").strip()
                elif line.startswith("Confidence:"):
                    # Extract confidence percentage
                    conf_text = line.replace("Confidence:", "").strip()
                    try:
                        # Remove % sign and convert to float
                        confidence = float(conf_text.replace("%", ""))
                    except ValueError:
                        confidence = 0
            
            # If we couldn't parse the format, fall back to the old tab-separated method
            if not answer:
                try:
                    answer, confidence_str = agent_output.split("\t")
                    confidence = float(confidence_str) if confidence_str.isdigit() else 0
                except Exception:
                    answer = agent_output
                    confidence = 0
                    
        except Exception as e:
            answer = agent_output + f"\n{e}"
            confidence = 0
            
        return {
            "answer": answer,
            "confidence": confidence,
            "expected_output": instance_output.data_instance.expected_output,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[SimpleQAInstance]
    ) -> Dict[str, Any]:
        output = self.get_instance_eval_output(instance_output)
        answer, confidence = output["answer"], output["confidence"]
        assert self.eval_prompts is not None, "eval_llm must be set for evaluation"
        assert self.eval_llm is not None, "eval_llm must be set for evaluation"
        instance = instance_output.data_instance
        prompt_message = self.eval_prompts["grader"].compile(
            question=instance.question,
            predicted_answer=answer,
            target=instance.expected_output,
        )
        response: BaseMessage = self.eval_llm.invoke(prompt_message)
        response = cast(AIMessage, response)
        grade_text = response.text().strip()

        # Extract grade letter
        match = re.search(r"(A|B|C)", grade_text)
        grade_letter = match.group(0) if match else "C"

        # Map to grade string
        grade_map = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
        grade = grade_map.get(grade_letter, "NOT_ATTEMPTED")

        return {"grade": grade, "grade_letter": grade_letter, "confidence": confidence}

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        num_instance = len(eval_outputs)
        return {
            "accuracy": sum(
                e["grade"] == "CORRECT" for e in eval_outputs if isinstance(e, dict)
            )
            / num_instance,
            "pct_incorrect": sum(
                e["grade"] == "INCORRECT" for e in eval_outputs if isinstance(e, dict)
            )
            / num_instance,
            "pct_not_attempted": sum(
                e["grade"] == "NOT_ATTEMPTED"
                for e in eval_outputs
                if isinstance(e, dict)
            )
            / num_instance,
            "avg_confidence": sum(
                float(e["confidence"]) for e in eval_outputs if isinstance(e, dict)
            )
            / num_instance,
            "num_instances": num_instance,
        }
