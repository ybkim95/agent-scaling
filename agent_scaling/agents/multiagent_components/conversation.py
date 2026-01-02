from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, convert_to_openai_messages
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field, computed_field

from agent_scaling.datasets.base import DatasetEnvStatus

from .plan import OrchestrationPlan

"""
from litellm import completion_cost
completion_cost(response_obj)

# Use https://artificialanalysis.ai/models/gpt-4 to track 
"""


class MessageTurn(BaseModel):
    round_num: int
    timestamp: str


class MessageTurnExternal(MessageTurn):
    role: Literal["subagent", "lead_agent"]
    message: str


class MessageTurnInternal(MessageTurn):
    role: Literal["user", "assistant", "tool"]
    iteration_num: int
    message: Dict[str, Any]
    litellm_message: Optional[ModelResponse] = None

    @property
    def cost(self) -> Optional[float]:
        return completion_cost(self.litellm_message) if self.litellm_message else None


class LLMResponseMessage(BaseModel):
    tag: Optional[str] = None
    litellm_message: ModelResponse

    @computed_field
    @cached_property
    def cost(self) -> Optional[float]:
        return completion_cost(self.litellm_message) if self.litellm_message else None


class AgentConversationHistory(BaseModel):
    agent_id: str
    status: str = "active"
    messages: List[LLMResponseMessage] = Field(default_factory=list)

    def add_response(self, llm_response: AIMessage, tag: Optional[str] = None):
        self.add_response_helper(
            llm_response.response_metadata["litellm_response"],
            tag,
        )

    def add_response_helper(
        self,
        litellm_message: ModelResponse,
        tag: Optional[str] = None,
    ):
        self.messages.append(
            LLMResponseMessage(
                tag=tag,
                litellm_message=litellm_message,
            )
        )


class SubAgentConversationHistory(BaseModel):
    agent_id: str
    status: str = "active"
    external_comms: List[MessageTurnExternal] = Field(default_factory=list)
    internal_comms: List[List[MessageTurnInternal]] = Field(default_factory=list)

    @property
    def current_round(self) -> int:
        return len(self.internal_comms)

    @property
    def total_iterations(self) -> int:
        num_iterations = 0
        for msgs in self.internal_comms:
            if len(msgs) > 0:
                num_iterations += msgs[-1].iteration_num
        return num_iterations

    @property
    def curr_iteration(self) -> int:
        if len(self.internal_comms) == 0:
            return 1
        if len(self.internal_comms[-1]) == 0:
            return 1
        return self.internal_comms[-1][-1].iteration_num

    @property
    def last_outgoing_external_message(self) -> str | None:
        ret = None
        for msg in reversed(self.external_comms):
            if msg.role == "subagent":
                ret = msg.message
                break
        return ret

    def last_n_iterations_messages(self, n: Optional[int] = 20) -> List[Dict[str, Any]]:
        ret = []
        for msgs in reversed(
            self.internal_comms
        ):  # start from the most recent iteration
            for msg in reversed(
                msgs
            ):  # start from the most recent message in the iteration
                ret.append(msg)
        if n is None or len(ret) == 0:
            return [convert_to_openai_messages(msg.message) for msg in reversed(ret)]  # type: ignore
        count_iterations = 0
        cur_iteration = ret[0].iteration_num
        new_ret = []
        for msg in ret:
            if msg.iteration_num != cur_iteration:
                count_iterations += 1
                cur_iteration = msg.iteration_num
                if count_iterations >= n:
                    break
            new_ret.append(msg)
        return [convert_to_openai_messages(msg.message) for msg in reversed(new_ret)]  # type: ignore

    def start_new_round(self):
        self.internal_comms.append([])

    def add_internal_message(
        self,
        message: Dict[str, Any],
        iteration_num: int,
        litellm_message: Optional[ModelResponse] = None,
    ):
        self.internal_comms[-1].append(
            MessageTurnInternal(
                round_num=self.current_round,
                iteration_num=iteration_num,
                role=message["role"],
                message=message,
                timestamp=datetime.now().isoformat(),
                litellm_message=litellm_message,
            )
        )

    def add_external_message(
        self, role: Literal["subagent", "lead_agent"], message: str
    ):
        self.external_comms.append(
            MessageTurnExternal(
                round_num=self.current_round,
                role=role,
                message=message,
                timestamp=datetime.now().isoformat(),
            )
        )


class SubAgentRoundResult(BaseModel):
    agent_id: str
    findings: str
    env_status: DatasetEnvStatus
    error_msg: str | None = None

    @property
    def error(self) -> bool:
        return self.error_msg is not None


class OrchestrationResult(BaseModel):
    plan: OrchestrationPlan
    subagent_conversations: Dict[str, SubAgentConversationHistory]
    subagent_env_status: Dict[str, DatasetEnvStatus]
    subagent_findings: Dict[str, List[str]]
    total_findings: int
    lead_agent_conversation: AgentConversationHistory
    synthesized_answer: Optional[str] = None

    @computed_field
    @cached_property
    def total_agent_iterations(self) -> int:
        return sum(
            conversation.total_iterations
            for conversation in self.subagent_conversations.values()
        )

    @property
    def combined_env_status(self) -> DatasetEnvStatus:
        return DatasetEnvStatus(
            success=any(status.success for status in self.subagent_env_status.values()),
            num_steps=sum(
                status.num_steps for status in self.subagent_env_status.values()
            ),
        )
