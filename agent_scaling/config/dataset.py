import os
from typing import Any, Dict, List, Optional

from langfuse._client.datasets import DatasetClient
from pydantic import BaseModel, Field, model_validator

from agent_scaling.datasets import Dataset, DatasetInstance, DatasetSharedPrompts
from agent_scaling.datasets.registry import get_dataset_cls, get_dataset_instance_cls
from agent_scaling.langfuse_client import get_lf_client

from .env import EnvConfig
from .llm import LLMConfig
from .prompts import Prompt


class DatasetConfig(BaseModel):
    dataset_id: str
    description: Optional[str] = None
    split: Optional[str] = None
    local_path: Optional[str] = None
    templates_local_path: Optional[str] = None
    from_langfuse: bool = True
    use_llm_eval: bool = False
    eval_llm: Optional[LLMConfig] = None
    eval_prompts: Optional[Dict[str, Prompt]] = None
    dataset_filter: Optional[str] = None
    env: EnvConfig = Field(default_factory=EnvConfig)

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        prompts = dict(data.get("eval_prompts", {}))
        for k, prompt in prompts.items():
            prompt = dict(prompt)
            if prompt.get("name") is None:
                prompt["name"] = k
            prompts[k] = prompt
        data["eval_prompts"] = prompts
        return data

    def model_post_init(self, context: Any) -> None:
        dataset_class = get_dataset_cls(self.dataset_id)
        dataset_instance_class = get_dataset_instance_cls(self.dataset_id)
        self._dataset: Dataset | None = None
        self._langfuse_dataset: DatasetClient | None = None
        if self.templates_local_path:
            task_shared_prompts = DatasetSharedPrompts.from_yaml(
                yaml_path=self.templates_local_path,
            )
        else:
            task_shared_prompts = {}
        kwargs = {
            "use_llm_eval": self.use_llm_eval,
            "eval_llm": self.eval_llm.get_llm() if self.eval_llm else None,
            "eval_prompts": self.eval_prompts or {},
            "task_shared_prompts": task_shared_prompts,
        }
        if self.from_langfuse:
            client = get_lf_client()
            if client is None:
                raise ValueError(
                    "Need to have a Langfuse client to load dataset from Langfuse"
                )
            dataset = client.get_dataset(self.dataset_id)
            self._langfuse_dataset = dataset
            if self.local_path is not None:
                if self.local_path.endswith(".yaml"):
                    self._dataset = dataset_class.from_yaml(
                        yaml_path=self.local_path, **kwargs
                    )

                elif self.local_path.endswith(".json"):
                    self._dataset = dataset_class.from_json(
                        json_path=self.local_path, **kwargs
                    )
                else:
                    raise ValueError(
                        f"Expected a .yaml or .json file for local path, got {self.local_path}"
                    )
            else:
                instances: List[DatasetInstance] = []
                for item in dataset.items:
                    instances.append(
                        dataset_instance_class(
                            **item.input, expected_output=item.expected_output
                        )
                    )
                self._dataset = dataset_class(
                    dataset_id=self.dataset_id, instances=instances, **kwargs
                )
                if not os.path.exists(f"datasets/{self.dataset_id}.json"):
                    os.makedirs("datasets", exist_ok=True)
                    self._dataset.to_json(f"datasets/{self.dataset_id}.json")
        else:
            if self.local_path is None:
                raise ValueError(
                    "Need to have a local path to load dataset from local file when from_langfuse is False"
                )
            if self.local_path.endswith(".json"):
                self._dataset = dataset_class.from_json(
                    json_path=self.local_path, **kwargs
                )
            elif self.local_path.endswith(".yaml"):
                self._dataset = dataset_class.from_yaml(
                    yaml_path=self.local_path, **kwargs
                )
            else:
                raise ValueError(
                    f"Expected a .yaml or .json file for local path, got {self.local_path}"
                )

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            raise ValueError("Dataset not loaded")
        return self._dataset

    @property
    def langfuse_dataset(self) -> DatasetClient | None:
        return self._langfuse_dataset
