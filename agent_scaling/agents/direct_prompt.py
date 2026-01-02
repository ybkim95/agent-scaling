import os.path as osp
from typing import Optional, cast

from langchain_core.messages import AIMessage, BaseMessage  # type: ignore
from langchain_core.messages.utils import convert_to_openai_messages  # type: ignore

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutput
from agent_scaling.utils import write_yaml

from .registry import register_agent


@register_agent("direct-prompt")
class DirectPrompt(AgentSystemWithTools):
    """
    A direct prompt agent that can interact with tools.
    """

    required_prompts = ["main"]

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutput:
        llm_params_dict = llm_params.model_dump() if llm_params else {}
        env, _ = self.init_environment(instance)

        shared_prompt_templates = self.get_dataset_prompt_templates(env)

        messages = cast(
            list,
            self.prompts["main"].compile(**shared_prompt_templates),
        )

        response: BaseMessage = self.llm.invoke(messages, **llm_params_dict)  # type: ignore
        response = cast(AIMessage, response)

        messages.append(convert_to_openai_messages(response))

        if instance_dir is not None:
            write_yaml(
                messages,  # type: ignore
                osp.join(instance_dir, "agent_messages.yaml"),
                use_long_str_representer=True,
            )
        return DatasetInstanceOutput(
            data_instance=instance,
            agent_output=response.content,
        )
