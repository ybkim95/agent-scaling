import os.path as osp
from typing import Optional, cast

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from agent_scaling.agents import register_agent
from agent_scaling.agents.base import AgentSystem, AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance
from agent_scaling.datasets.base import DatasetInstanceOutput
from agent_scaling.env import BasicEnvironment
from agent_scaling.utils import write_yaml


@register_agent("single-agent-zero-shot-cot")
class SingleAgentZeroShotCoT(AgentSystem):
    """
    A single agent that runs on a single instance using zero-shot CoT.
    See https://arxiv.org/abs/2205.11916 Fig. 2 for details.
    """

    required_prompts = ["reasoning_extraction", "answer_generation"]

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutput:
        llm_params_dict = self._get_llm_params_dict(llm_params)
        prompt_info_reasoning = instance.get_prompt_info()
        prompt_reasoning = self.prompts["reasoning_extraction"].compile(
            **prompt_info_reasoning
        )
        response_reasoning: AIMessage = cast(
            AIMessage, self.llm.invoke(prompt_reasoning, **llm_params_dict)
        )
        prompt_info_answer = {
            **prompt_info_reasoning,
            "reasoning": response_reasoning.text(),
        }

        prompt_answer = self.prompts["answer_generation"].compile(**prompt_info_answer)
        response_answer: AIMessage = cast(
            AIMessage, self.llm.invoke(prompt_answer, **llm_params_dict)
        )
        if instance_dir is not None:
            out = {
                "reasoning": response_reasoning.text(),
                "agent_answer": response_answer.text(),
            }
            write_yaml(
                out,
                osp.join(instance_dir, "agent_output.yaml"),
                use_long_str_representer=True,
            )
        return DatasetInstanceOutput(
            data_instance=instance, agent_output=response_answer.text()
        )


@register_agent("single-agent-zero-shot-cot-with-tools")
class SingleAgentZeroShotCoTWithTools(AgentSystemWithTools[BasicEnvironment]):
    """
    A single agent that runs on a single instance using zero-shot CoT with tools.
    """

    required_prompts = ["reasoning_extraction", "final_answer"]

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
    ) -> DatasetInstanceOutput:
        llm_params_dict = self._get_llm_params_dict(llm_params)
        prompt_info_reasoning = instance.get_prompt_info()
        chat_msgs = self.prompts["reasoning_extraction"].compile(
            **prompt_info_reasoning
        )
        assert self.llm_w_tools is not None, "LLM with tools is not initialized."
        response_reasoning = cast(
            AIMessage,
            self.llm_w_tools.invoke(chat_msgs, **llm_params_dict),  # type: ignore
        )
        chat_msgs.append(convert_to_openai_messages(response_reasoning))  # type: ignore

        tools_called = []
        if response_reasoning.tool_calls:
            tool_calls = response_reasoning.tool_calls
            tool_messages = self.env.execute_tools(tool_calls)
            for tool_message, tool_call in zip(tool_messages, tool_calls):
                chat_msgs.append(convert_to_openai_messages(tool_message))  # type: ignore
                tools_called.append(
                    {
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                    }
                )
        chat_msgs.extend(self.prompts["final_answer"].compile())

        response_answer: AIMessage = self.llm.invoke(chat_msgs, **llm_params_dict)
        chat_msgs.append(convert_to_openai_messages(response_answer))  # type: ignore
        if instance_dir is not None:
            out = {
                "chat_msgs": chat_msgs,
                "agent_answer": response_answer.text(),
                "tools_called": tools_called,
            }
            write_yaml(
                out,
                osp.join(instance_dir, "agent_output.yaml"),
                use_long_str_representer=True,
            )
        return DatasetInstanceOutput(
            data_instance=instance, agent_output=response_answer.text()
        )
