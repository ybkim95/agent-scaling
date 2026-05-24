"""Independent multi-agent system: N agents work in parallel with synthesis_only aggregation.

Matches the Methods section of the revised manuscript (Section 6.1):
    A = {a_1, ..., a_n}, C = {(a_i, a_agg)},  Omega = synthesis_only.

The synthesis_only aggregator concatenates sub-agent outputs without
cross-validation or majority voting; the aggregator performs no analytical
comparison of responses, so any performance differences arise purely from
parallel exploration rather than error correction.

This is a parallel ensemble without coordination. The aggregator is
intentionally the weakest in the architectural ablation, isolating
"parallelism" from "coordination". All N sub-agents run to completion and
their outputs are concatenated in insertion order; the system does not
perform any first-to-succeed race, voting, or cross-validation.
"""
import asyncio
import os.path as osp
import time
from typing import Dict, List, Optional, Tuple

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml

from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent


@register_agent("multi-agent-independent")
class IndependentMultiAgentSystem(AgentSystemWithTools):
    """N agents run in parallel with no inter-agent communication.

    Aggregator policy: synthesis_only - concatenates each sub-agent's final
    answer into a single response. No voting, no cross-validation, no
    analytic comparison.
    """

    required_prompts = ["subagent"]

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
        self.subagents: Dict[str, WorkerSubagent] = {}

        logger.info(
            f"IndependentMultiAgentSystem: {n_base_agents} agents in parallel, "
            f"synthesis_only aggregation (concatenation, no voting)"
        )

    def _create_subagents(self, task_instance: DatasetInstance):
        """Create N fully independent agents, each working on the full task."""
        self.subagents = {}
        for i in range(self.n_base_agents):
            agent_id = f"agent_{i + 1}"
            subagent = WorkerSubagent.init_from_agent(
                agent=self,
                agent_id=agent_id,
                objective="Solve the task completely on your own",
                original_query=self.memory.original_task,
                strategy=f"Independent agent {i + 1}",
                task_instance=task_instance,
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
            )
            self.subagents[agent_id] = subagent
        logger.info(
            f"Created {len(self.subagents)} independent agents (no coordination)"
        )

    def _synthesize_only(self) -> Tuple[str, List[str]]:
        """Concatenate each sub-agent's final answer without analysis or voting.

        Returns (synthesized_answer, contributing_agent_ids).
        """
        parts: List[str] = []
        contributing: List[str] = []
        for agent_id, agent in self.subagents.items():
            answer = agent.conv_history.last_outgoing_external_message or ""
            if not answer:
                continue
            parts.append(f"=== {agent_id} ===\n{answer}")
            contributing.append(agent_id)
        return ("\n\n".join(parts), contributing)

    def _auto_submit(self, synthesized_answer: str) -> str:
        """Submit the synthesized answer via the first available agent's env."""
        for agent_id, agent in self.subagents.items():
            env = agent.env
            if env.env_done():
                continue
            submit_tool_name = None
            if "submit_patch" in env.tools:
                submit_tool_name = "submit_patch"
            elif "submit" in env.tools:
                submit_tool_name = "submit"
            if submit_tool_name:
                logger.info(
                    f"Submitting synthesis via {agent_id} using {submit_tool_name}"
                )
                try:
                    tool_call = {
                        "name": submit_tool_name,
                        "args": {"reasoning": synthesized_answer[:500]},
                        "id": "synthesis_submit_independent",
                        "type": "tool_call",
                    }
                    tool_msg = env.execute_tool(tool_call)
                    return str(tool_msg.content)
                except Exception as e:
                    logger.warning(f"Synthesis auto-submit failed: {e}")
            break
        return ""

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
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
        start_time = time.time()

        shared_prompt_templates = self.get_dataset_prompt_templates(
            dataset_instance=instance
        )
        self.memory.original_task = shared_prompt_templates.get(
            "task_instance", str(instance.get_prompt_info())
        )

        self._create_subagents(instance)

        # Single message: each agent works fully independently on the task.
        message = (
            "Solve the task completely on your own. Work systematically and submit "
            "when done.\n\n"
            "CRITICAL: You MUST make changes and submit within your iteration budget. "
            "Do NOT spend more than half your iterations on exploration/analysis. "
            "After understanding the problem, immediately start implementing the fix. "
            "If you reach 50% of your budget without making changes, STOP exploring "
            "and START implementing your best solution immediately."
        )

        tasks: List[Tuple[str, asyncio.Task]] = []
        for agent_id, subagent in self.subagents.items():
            task = asyncio.create_task(
                asyncio.to_thread(subagent.process_orchestrator_message, message)
            )
            tasks.append((agent_id, task))

        # Per-task time budget: only enforced if the dataset instance sets a
        # positive `time_limit`. Defaults to no limit so all N agents run to
        # completion before synthesis_only aggregates their answers.
        time_limit = getattr(instance, "time_limit", None)
        if time_limit is not None and time_limit > 0:
            done, pending = await asyncio.wait(
                [t for _, t in tasks], timeout=time_limit
            )
        else:
            done, pending = await asyncio.wait(
                [t for _, t in tasks], return_when=asyncio.ALL_COMPLETED
            )
        for t in pending:
            t.cancel()

        # Drain each task (logs failures but does not stop synthesis).
        for agent_id, task in tasks:
            if task in done:
                try:
                    await task
                except Exception as e:
                    logger.warning(f"Agent {agent_id} failed: {e}")

        # synthesis_only aggregator: concatenate every agent's last answer.
        # The synthesized text is the canonical agent output returned to the
        # caller and used for grading; individual agent env statuses are
        # secondary and reported only as infrastructure metadata.
        synthesized_answer, contributing_ids = self._synthesize_only()

        # Always attempt to submit the synthesized answer to an env so that
        # the grader sees the aggregated output rather than any single
        # agent's intermediate submission. _auto_submit is a no-op on
        # already-done envs, so this never double-submits.
        submission_response = ""
        if synthesized_answer:
            submission_response = self._auto_submit(synthesized_answer)

        # Canonical env status. For text-output benchmarks (Finance-Agent,
        # WorkBench, BrowseComp-Plus, PlanCraft) the grader reads
        # agent_output, so env_status is only infrastructure metadata. For
        # environment-state benchmarks evaluated against a Docker container's
        # terminal state (SWE-bench Verified, Terminal-Bench), independent
        # container states cannot be physically merged across sub-agent runs,
        # so we deterministically report the first sub-agent's env status in
        # insertion order. This is not first-to-succeed: there is no
        # success-based selection, consistent with the synthesis-only
        # contract that prohibits voting or cross-validation.
        final_env_status = None
        for agent in self.subagents.values():
            if agent.env.env_done():
                final_env_status = agent.env.env_status()
                break
        if final_env_status is None:
            for agent in self.subagents.values():
                final_env_status = agent.env.env_status()
                break

        final_answer = synthesized_answer or submission_response

        execution_time = time.time() - start_time
        total_iterations = sum(
            a.conv_history.total_iterations for a in self.subagents.values()
        )
        logger.info(
            f"Independent (synthesis_only) processing completed in {execution_time:.2f}s "
            f"with {total_iterations} total iterations across {len(self.subagents)} agents. "
            f"Contributing agents: {contributing_ids}"
        )

        if instance_dir is not None:
            output_data = {
                "architecture": "independent",
                "aggregator": "synthesis_only",
                "n_agents": self.n_base_agents,
                "contributing_agents": contributing_ids,
                "total_iterations": total_iterations,
                "execution_time": execution_time,
                "synthesized_answer": synthesized_answer,
            }
            write_yaml(
                output_data,
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],
            final_env_output=final_env_status,
        )
