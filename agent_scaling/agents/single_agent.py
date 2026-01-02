import os.path as osp
import traceback
from typing import List, Optional, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,  # type: ignore
    ToolMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages  # type: ignore

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import (
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
    TrajectoryStep,
)
from agent_scaling.env import AgentEnvironment
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml

from .registry import register_agent


@register_agent("single-agent")
class SingleAgent(AgentSystemWithTools[AgentEnvironment]):
    """
    A single agent that can interact with tools. Developed in SWE-Agent framework.
    """

    required_prompts = ["main"]

    def __init__(self, max_steps: int = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        llm_params_dict = llm_params.model_dump() if llm_params else {}
        env, llm_w_tools = self.init_environment(instance)
        shared_prompt_templates = self.get_dataset_prompt_templates(env)

        messages = cast(
            list,
            self.prompts["main"].compile(**shared_prompt_templates),
        )
        trajectory: List[TrajectoryStep] = []
        final_answer = ""
        final_env_output = {}
        is_done = False
        for step in range(self.max_steps):
            response: BaseMessage = llm_w_tools.invoke(messages, **llm_params_dict)  # type: ignore
            response = cast(AIMessage, response)
            if response.tool_calls:
                response.tool_calls = [response.tool_calls[0]]

            messages.append(convert_to_openai_messages(response))
            tool_resp: ToolMessage | None = None
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = ""
                try:
                    tool_resp = env.execute_tool(tool_call)
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    action = f"{tool_name}({', '.join([f'{k}={v}' for k, v in tool_input.items()])})"
                    messages.append(convert_to_openai_messages(tool_resp))
                    is_done = tool_name == "done"
                except Exception as e:
                    action = ""
                    messages.append(
                        {
                            "role": "user",
                            "content": f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call.",
                        }
                    )
                    logger.warning(
                        f"Tool **{tool_name}** failed with error: {str(e)}\n{traceback.format_exc()}"
                    )
            else:
                action = ""
                messages.append(
                    {
                        "role": "user",
                        "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                    }
                )
                logger.warning("No tool calls found in the response.")
            trajectory.append(
                TrajectoryStep(
                    action=action,
                    observation=str(tool_resp.content) if tool_resp else "",
                    response=str(response.content),
                    thought=str(response.content),
                )
            )
            if is_done or env.env_done():
                final_answer = trajectory[-1].observation
                break
        final_env_output = env.env_status()
        if instance_dir is not None:
            out = {
                "trajectory": [t.model_dump() for t in trajectory],
                "final_answer": final_answer,
            }
            write_yaml(
                out,
                osp.join(instance_dir, "agent_output.yaml"),
                use_long_str_representer=True,
            )
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=trajectory,
            final_env_output=final_env_output,
        )
