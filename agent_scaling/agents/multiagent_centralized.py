import asyncio
import os.path as osp
import time
from typing import Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml

from .multiagent_components.conversation import OrchestrationResult
from .multiagent_components.mas_lead_agent import LeadAgent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent


@register_agent("multi-agent-centralized")
class CentralizedMultiAgentSystem(AgentSystemWithTools):
    """Centralized multi-agent system with orchestrator coordinating workers"""

    required_prompts = ["lead_agent", "subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()

        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent

        self.lead_agent = LeadAgent(
            *args,
            memory=self.memory,
            min_iterations_per_agent=min_iterations_per_agent,
            num_base_agents=n_base_agents,
            domain_config={"task_blurb": kwargs.get("task_blurb", "task coordinator")},
            **kwargs,
        )

        logger.info(
            f"CentralizedMultiAgentSystem initialized with: {n_base_agents} agents, {min_iterations_per_agent} min iterations per agent (adaptive orchestration)"
        )
        logger.info(
            "Using prompt compilation with dataset-shared templates like single-agent system"
        )

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        """Synchronous wrapper for backward compatibility"""
        return asyncio.run(
            self.run_agent_async(instance, instance_dir, llm_params, instance_idx)
        )

    async def run_agent_async(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        """Async version of run_agent for better performance"""
        start_time = time.time()
        logger.info(f"Starting multi-agent processing for instance {instance_idx}")
        logger.info(
            f"Configuration: {self.n_base_agents} agents, {self.min_iterations_per_agent} min iterations per agent"
        )
        # Process llm_params like single_agent.py
        llm_params_dict = llm_params.model_dump() if llm_params else {}

        logger.info("Starting lead agent orchestration...")
        processing_result: OrchestrationResult = await self.lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params_dict,
        )

        final_answer = processing_result.synthesized_answer
        if final_answer is not None:
            logger.info(
                f"Final answer extracted: {final_answer[:200]}..."
                if len(str(final_answer)) > 200
                else f"Final answer extracted: {final_answer}"
            )

        execution_time = time.time() - start_time

        logger.info(
            f"Processing completed in {execution_time:.2f}s with {processing_result.total_agent_iterations} total iterations across {len(processing_result.subagent_conversations)} agents"
        )

        if instance_dir is not None:
            write_yaml(
                processing_result.model_dump(),
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )

        # Return DatasetInstanceOutputWithTrajectory like single_agent.py
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],  # Multi-agent doesn't have a single trajectory
            final_env_output=processing_result.combined_env_status,
        )
