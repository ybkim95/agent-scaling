from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from langfuse.api.resources.prompts.types import Prompt_Text
from langfuse.model import TextPromptClient
from pydantic import BaseModel, Field

from agent_scaling.config.prompts import Prompt
from agent_scaling.llm import ChatLiteLLMLC
from agent_scaling.utils import read_json, read_yaml, write_json, write_yaml


class DatasetInstance(ABC, BaseModel):
    """A single instance/example in a dataset."""

    expected_output: Optional[Any] = None

    @abstractmethod
    def get_prompt_info(self) -> Dict[str, str]:
        """Get the prompt info for the instance."""
        pass


class DatasetEnvStatus(BaseModel):
    success: bool
    num_steps: int


T = TypeVar("T", bound=DatasetInstance)


class TrajectoryStep(BaseModel):
    action: str
    observation: str
    response: str
    thought: str


class DatasetInstanceOutput(BaseModel, Generic[T]):
    data_instance: T
    agent_output: str | Any


class DatasetInstanceOutputWithTrajectory(DatasetInstanceOutput[T], Generic[T]):
    trajectory: List[TrajectoryStep] = Field(default_factory=list)
    final_env_output: DatasetEnvStatus | None = None


class DatasetSharedPrompts(BaseModel):
    task_description_template: str
    task_description_with_tools_template: str
    task_behavior_template: str
    task_instance_template: str
    task_output_template: Optional[str] = None
    _prompt_templates: Dict[
        Literal[
            "task_description",
            "task_description_with_tools",
            "task_behavior",
            "task_instance",
            "task_output",
        ],
        TextPromptClient,
    ] = {}

    def model_post_init(self, __context: Any) -> None:
        self._prompt_templates = {
            "task_description": self._convert_to_prompt_text(
                "task_description", self.task_description_template
            ),
            "task_description_with_tools": self._convert_to_prompt_text(
                "task_description_with_tools", self.task_description_with_tools_template
            ),
            "task_behavior": self._convert_to_prompt_text(
                "task_behavior", self.task_behavior_template
            ),
            "task_instance": self._convert_to_prompt_text(
                "task_instance", self.task_instance_template
            ),
            "task_output": self._convert_to_prompt_text(
                "task_output", self.task_output_template or ""
            ),
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DatasetSharedPrompts":
        data = read_yaml(yaml_path)
        return cls(**data)

    def _convert_to_prompt_text(
        self, name: str, prompt_template_str: str
    ) -> TextPromptClient:
        prompt_template = Prompt_Text(
            name=name,
            version=0,
            labels=[],
            tags=[],
            config={},
            prompt=prompt_template_str,
        )
        return TextPromptClient(prompt=prompt_template)

    def get_prompt_templates_for_instance(
        self, prompt_info: Dict[str, str]
    ) -> Dict[str, str]:
        return {
            "task_description": self._prompt_templates["task_description"].compile(
                **prompt_info
            ),
            "task_description_with_tools": self._prompt_templates[
                "task_description_with_tools"
            ].compile(**prompt_info),
            "task_behavior": self._prompt_templates["task_behavior"].compile(
                **prompt_info
            ),
            "task_instance": self._prompt_templates["task_instance"].compile(
                **prompt_info
            ),
            "task_output": self._prompt_templates["task_output"].compile(**prompt_info),
        }


class Dataset(BaseModel, ABC):
    """Abstract base class for datasets."""

    dataset_id: str
    instances: List[DatasetInstance]
    use_llm_eval: bool = Field(default=False, exclude=True)
    eval_llm: Optional[ChatLiteLLMLC] = Field(default=None, exclude=True)
    eval_prompts: Optional[Dict[str, Prompt]] = Field(default=None, exclude=True)
    task_shared_prompts: Optional[DatasetSharedPrompts] = None
    _required_eval_prompts: List[str] = []
    _require_llm_eval: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self._require_llm_eval or self.use_llm_eval:
            if self.eval_prompts is not None and self.eval_llm is None:
                raise ValueError(
                    "If eval_prompts are provided, eval_llm must also be provided."
                )
            if self.use_llm_eval and self.eval_llm is None:
                raise ValueError("If use_llm_eval is True, eval_llm must be provided.")
        if self.eval_prompts is not None:
            self.check_required_prompts(
                self.eval_prompts or {}, self._required_eval_prompts
            )

    @classmethod
    def check_required_prompts(
        cls, prompts: Dict[str, Prompt], required_eval_prompts: List[str]
    ) -> None:
        required_but_not_found = []
        for prompt_name in required_eval_prompts:
            if prompt_name not in prompts:
                required_but_not_found.append(prompt_name)
        if required_but_not_found:
            raise ValueError(
                f"Prompt(s) **{', '.join(required_but_not_found)}** are required for {cls.__name__} but not found."
            )

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs) -> "Dataset":
        """Load a dataset from a YAML file."""
        data = read_yaml(yaml_path)
        return cls(**data, **kwargs)

    def to_yaml(self, yaml_path: str) -> None:
        """Save the dataset to a YAML file."""
        data = self.model_dump(exclude_none=False)
        write_yaml(data, yaml_path)

    @classmethod
    def from_json(cls, json_path: str, **kwargs) -> "Dataset":
        """Load a dataset from a JSON file."""
        data = read_json(json_path)
        return cls(**data, **kwargs)

    def to_json(self, json_path: str) -> None:
        """Save the dataset to a JSON file."""
        data = self.model_dump(exclude_none=True, mode="json")
        write_json(data, json_path, indent=True)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> DatasetInstance:
        return self.instances[idx]

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset_id={self.dataset_id}, num_instances={len(self.instances)})"

    @cached_property
    def num_instances(self) -> int:
        return len(self.instances)

    @abstractmethod
    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput
    ) -> Dict[str, Any]:
        """Get the evaluation output for the instance."""
        pass

    @abstractmethod
    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput
    ) -> Dict[str, Union[int, float]]:
        """Get the evaluation metrics for the instance."""
        pass

    @abstractmethod
    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        """Get the metrics for the dataset."""
        pass
