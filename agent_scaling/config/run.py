from typing import Any, Dict, Optional, Self

from litellm import _logging as litellm_logging
from pydantic import BaseModel, Field, model_validator

from agent_scaling.agents import AgentSystem, get_agent_cls
from agent_scaling.langfuse_client import get_lf_client
from agent_scaling.utils import (
    disable_local_cache,
    enable_local_cache,
    enable_local_logging,
)

from .dataset import DatasetConfig
from .llm import LLMConfig
from .prompts import Prompt


class MultiAgentResearchConfig(BaseModel):
    n_base_agents: int = 3
    max_orchestrator_turns: int = 2
    min_searches_per_agent: int = 3
    max_iterations_per_agent: int = 7


class AgentConfig(BaseModel):
    name: str
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    agent_specific_config: Optional[MultiAgentResearchConfig] = None

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        agent_cls = get_agent_cls(self.name)
        agent_cls.check_required_prompts(self.prompts)
        return self

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        prompts = dict(data.get("prompts", {}))
        for k, prompt in prompts.items():
            prompt = dict(prompt)
            if prompt.get("name") is None:
                prompt["name"] = k
            prompts[k] = prompt
        data["prompts"] = prompts
        return data

    def get_run_metadata(self) -> Dict[str, Any]:
        prompts = {}
        assert self.prompts is not None, "Prompts must be defined in AgentConfig"
        for k, prompt in self.prompts.items():
            prompts[k] = {
                "name": prompt.name,
            }
        return {
            "name": self.name,
            "prompts": prompts,
        }


class RunConfig(BaseModel):
    agent: AgentConfig
    dataset: DatasetConfig
    llm: LLMConfig
    run_name: str
    save_dir: Optional[str] = None
    log_langfuse: bool = True
    use_disk_cache: bool = False
    debug: bool = False
    max_instances: Optional[int] = None
    num_workers: int = 1

    @property
    def run_parallel(self) -> bool:
        return self.num_workers > 1

    def model_post_init(self, context: Any) -> None:
        client = get_lf_client()
        self.log_langfuse = (
            self.log_langfuse and client is not None
            # and self.dataset.langfuse_dataset is not None
        )
        if not self.log_langfuse and not self.run_parallel:
            enable_local_logging(prompt_only=True)
        litellm_logging._disable_debugging()  # type: ignore
        # Disable litellm debugging logs
        if self.use_disk_cache:
            enable_local_cache()
        else:
            disable_local_cache()

    def get_agent(self) -> AgentSystem:
        return get_agent_cls(self.agent.name).from_config(
            llm_config=self.llm,
            dataset_config=self.dataset,
            prompts=self.agent.prompts,
            **(
                self.agent.agent_specific_config.model_dump(exclude_none=True)
                if self.agent.agent_specific_config is not None
                else {}
            ),
        )

    def get_run_metadata(self) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "agent": self.agent.get_run_metadata(),
            "llm": self.llm.model_dump(exclude_none=True),
            "dataset": self.dataset.model_dump(exclude_none=True),
        }
        if self.save_dir is not None:
            ret["save_dir"] = self.save_dir
        ret["run_name"] = self.run_name
        ret["num_workers"] = self.num_workers
        return ret
