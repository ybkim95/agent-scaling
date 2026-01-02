from typing import Dict, List, Self

from pydantic import BaseModel, Field, model_validator

from agent_scaling.env.registry import get_env_cls

from .prompts import Prompt


class EnvConfig(BaseModel):
    name: str = "basic"
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        env_cls = get_env_cls(self.name)
        if self.prompts is None:
            self.prompts = {}
        env_cls.check_required_prompts(self.prompts)
        return self
