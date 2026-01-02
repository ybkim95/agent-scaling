from typing import Any, Dict, List

from plancraft.simple import PlancraftExample

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["plancraft-test"]


@register_dataset_instance(DATASET_IDS)
class PlancraftInstance(PlancraftExample, DatasetInstance):
    def model_post_init(self, context: Any) -> None:
        self.expected_output = self.target
        self.slotted_inventory = {int(k): v for k, v in self.slotted_inventory.items()}

    def get_prompt_info(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "target": self.target,
        }


@register_dataset(DATASET_IDS)
class PlancraftDataset(Dataset):
    dataset_id: str = "plancraft-test"
    instances: List[PlancraftInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[PlancraftInstance]
    ) -> Dict[str, Any]:
        return {
            "success": instance_output.final_env_output.success
            if instance_output.final_env_output
            else False,
            "num_steps": instance_output.final_env_output.num_steps
            if instance_output.final_env_output
            else -1,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[PlancraftInstance]
    ) -> Dict[str, Any]:
        return self.get_instance_eval_output(instance_output)

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "avg_success": sum(e["success"] for e in eval_outputs) / len(eval_outputs),
            "avg_num_steps": sum(e["num_steps"] for e in eval_outputs)
            / len(eval_outputs),
            "num_instances": len(eval_outputs),
        }
