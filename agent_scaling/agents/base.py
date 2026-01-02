from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from agent_scaling.config.dataset import DatasetConfig
from agent_scaling.config.llm import LLMConfig, LLMParams
from agent_scaling.config.prompts import Prompt
from agent_scaling.datasets import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.env import AgentEnvironment, get_env_cls
from agent_scaling.llm import ChatLiteLLMLC

AgentEnvType = TypeVar("AgentEnvType", bound=AgentEnvironment)


class BaseAgent:
    """Abstract base class for agents."""

    required_prompts: List[str] = []

    def __init__(
        self,
        llm: ChatLiteLLMLC,
        dataset: Dataset,
        prompts: Dict[str, Prompt],
        **kwargs,
    ):
        self.llm = llm
        self.dataset = dataset
        assert dataset.task_shared_prompts is not None, (
            "task_shared_prompts must be provided"
        )
        self.dataset_prompt_templates = dataset.task_shared_prompts
        self.check_required_prompts(prompts)
        self.prompts: Dict[str, Prompt] = {k: prompts[k] for k in self.required_prompts}

    def _get_llm_params_dict(
        self, llm_params: Optional[LLMParams] = None
    ) -> Dict[str, Any]:
        if not llm_params:
            ret = {}
        else:
            ret = llm_params.model_dump()
        return ret

    @classmethod
    def from_config(
        cls,
        llm_config: LLMConfig,
        dataset_config: DatasetConfig,
        prompts: Dict[str, Prompt],
        **kwargs,
    ) -> "BaseAgent":
        llm = llm_config.get_llm()
        dataset = dataset_config.dataset
        return cls(llm=llm, dataset=dataset, prompts=prompts, **kwargs)

    @classmethod
    def check_required_prompts(cls, prompts: Dict[str, Prompt]) -> None:
        required_but_not_found = []
        for prompt_name in cls.required_prompts:
            if prompt_name not in prompts:
                required_but_not_found.append(prompt_name)
        if required_but_not_found:
            raise ValueError(
                f"Prompt(s) **{', '.join(required_but_not_found)}** are required for {cls.__name__} but not found."
            )


class BaseAgentSystem(ABC):
    @abstractmethod
    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutput:
        """
        Run the agent on a single instance and return the llm output.
        Log anything to instance_dir if needed.
        """
        pass


class BaseAgentWithTools(BaseAgent, Generic[AgentEnvType]):
    """Abstract base class for agents that use tools."""

    def __init__(
        self,
        *args,
        tools: Optional[List[str]] = None,
        env: Optional[str] = None,
        env_prompts: Dict[str, Prompt] | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        assert self.dataset is not None, "dataset must be provided"
        assert tools is not None, "tools must be provided"
        assert env_prompts is not None, "env_prompts must be provided"
        self.env_name = env
        self.env_prompts = env_prompts
        self.tools = tools
        self.env, self.llm_w_tools = self.init_environment()

    @classmethod
    def from_config(
        cls,
        llm_config: LLMConfig,
        dataset_config: DatasetConfig,
        prompts: Dict[str, Prompt],
        **kwargs,
    ) -> "BaseAgentWithTools":
        llm = llm_config.get_llm()
        dataset = dataset_config.dataset
        env_config = dataset_config.env
        env = env_config.name
        env_prompts = env_config.prompts
        tools = env_config.tools
        return cls(
            llm=llm,
            dataset=dataset,
            prompts=prompts,
            env=env,
            env_prompts=env_prompts,
            tools=tools,
            **kwargs,
        )

    def init_env_only(
        self,
        dataset_instance: DatasetInstance | None = None,
        agent_id: Optional[str] = None,
    ) -> AgentEnvironment:
        env = self.env_name or "basic"
        env = get_env_cls(env)(
            dataset=self.dataset,
            dataset_instance=dataset_instance,
            tools=self.tools,
            log_langfuse=self.llm.log_langfuse,
            env_prompts=self.env_prompts,
            agent_id=agent_id,
        )
        return env

    def init_environment(
        self,
        dataset_instance: DatasetInstance | None = None,
        agent_id: Optional[str] = None,
    ) -> Tuple[AgentEnvType, Runnable[LanguageModelInput, BaseMessage]]:
        env = self.init_env_only(dataset_instance, agent_id)
        llm_w_tools = self.llm.bind_tools(env.tools_list)
        return env, llm_w_tools  # type: ignore

    def get_dataset_prompt_templates(
        self,
        env: Optional[AgentEnvironment] = None,
        dataset_instance: DatasetInstance | None = None,
    ) -> Dict[str, Any]:
        assert env is not None or dataset_instance is not None, (
            "env or dataset_instance must be provided"
        )
        if env is None:
            env = self.init_env_only(dataset_instance)
        return self.dataset_prompt_templates.get_prompt_templates_for_instance(
            env.get_instance_prompt_info()
        )


class AgentSystem(BaseAgentSystem, BaseAgent, ABC):
    """Abstract base class for agents that use a language model."""

    @classmethod
    def from_config(
        cls,
        llm_config: LLMConfig,
        dataset_config: DatasetConfig,
        prompts: Dict[str, Prompt],
        **kwargs,
    ) -> "AgentSystem":
        llm = llm_config.get_llm()
        dataset = dataset_config.dataset
        return cls(llm=llm, dataset=dataset, prompts=prompts, **kwargs)


class AgentSystemWithTools(AgentSystem, BaseAgentWithTools[AgentEnvType], ABC):
    """Abstract base class for agents that use tools."""

    @classmethod
    def from_config(
        cls,
        llm_config: LLMConfig,
        dataset_config: DatasetConfig,
        prompts: Dict[str, Prompt],
        **kwargs,
    ) -> "AgentSystemWithTools":
        llm = llm_config.get_llm()
        dataset = dataset_config.dataset
        env_config = dataset_config.env
        env = env_config.name
        env_prompts = env_config.prompts
        tools = env_config.tools
        return cls(
            llm=llm,
            dataset=dataset,
            prompts=prompts,
            env=env,
            env_prompts=env_prompts,
            tools=tools,
            **kwargs,
        )
