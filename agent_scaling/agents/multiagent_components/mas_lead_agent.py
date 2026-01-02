import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from agent_scaling.agents.base import BaseAgentWithTools
from agent_scaling.agents.output_validation import run_with_validation, validate_json
from agent_scaling.config.prompts import NamedPrompt
from agent_scaling.datasets import DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.utils import join_with_leading_dash

from .conversation import (
    AgentConversationHistory,
    OrchestrationResult,
    SubAgentRoundResult,
)
from .mas_subagent import WorkerSubagent
from .memory import EnhancedMemory
from .plan import OrchestrationPlan, Subtask


class LeadAgent(BaseAgentWithTools):
    """Generic lead agent for coordinating multiple worker agents"""

    required_prompts = ["lead_agent", "subagent"]  # subagent prompt passed to subagent

    def __init__(
        self,
        *args,
        memory: EnhancedMemory,
        min_iterations_per_agent: int = 3,
        num_base_agents: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory
        self.min_iterations_per_agent = min_iterations_per_agent
        self.num_base_agents = num_base_agents
        self.subagents: Dict[str, WorkerSubagent] = {}
        self.subagent_kwargs = kwargs
        self.conv_history = AgentConversationHistory(agent_id="lead_agent")

    def analyze_query_and_plan(
        self, shared_prompt_templates: Dict[str, Any]
    ) -> OrchestrationPlan:
        """Analyze query and create research plan using dataset-shared prompt templates"""
        logger.info(
            f"Analyzing query with shared templates: {list(shared_prompt_templates.keys())}"
        )

        # Use named template from lead_agent.yaml
        planning_template = self.prompts["lead_agent"].get_template("planning")
        planning_messages = planning_template.compile(
            **shared_prompt_templates, num_agents=self.num_base_agents
        )
        try:
            outputs, plan = run_with_validation(
                self.llm, planning_messages, self._validate_plan
            )
            for i, output in enumerate(outputs):
                postfix = f"_{i}" if i > 0 else ""
                self.conv_history.add_response(
                    llm_response=output,
                    tag=f"planning{postfix}",
                )
        except Exception as e:
            logger.warning(f"Prompt compilation failed: {e}, using simple fallback")
            plan = self._create_simple_plan()
        return plan

    def _validate_plan(self, llm_response: str) -> OrchestrationPlan:
        """Validate that the plan has the required structure"""
        plan = validate_json(llm_response)
        # Check required top-level keys
        if not all(key in plan for key in ["subtasks"]):
            missing_keys = [key for key in ["subtasks"] if key not in plan]
            raise ValueError(f"Plan missing required keys: {missing_keys}")
        # Check subtasks structure
        subtasks = plan.get("subtasks", [])
        if not isinstance(subtasks, list) or len(subtasks) == 0:
            raise ValueError(
                "Invalid subtasks structure. Subtasks must be a non-empty list."
            )
        # Check each subtask has required fields
        for i, subtask in enumerate(subtasks):
            error_msg = ""
            if not all(key in subtask for key in ["agent_id", "objective"]):
                missing_keys = [
                    key for key in ["agent_id", "objective"] if key not in subtask
                ]
                error_msg += (
                    f"subtask in index {i} missing required fields: {missing_keys}\n"
                )
            if error_msg:
                raise ValueError(error_msg)
        return OrchestrationPlan(**plan)

    def _create_simple_plan(self) -> OrchestrationPlan:
        """Create a simple plan structure without LLM call when templates are missing"""
        # Create simple but distinct subtasks
        subtasks = []

        for i in range(self.num_base_agents):
            subtasks.append(
                {
                    "agent_id": f"agent_{i + 1}",
                    "objective": f"Work on aspect {i + 1} of the main task",
                    "focus": f"Aspect {i + 1}",
                }
            )
        return OrchestrationPlan(
            subtasks=subtasks,
            reasoning="Simple plan created due to missing templates",
        )

    def _create_subagents(
        self, plan: OrchestrationPlan, task_instance: DatasetInstance
    ):
        """Create subagents based on the plan with proper LLM instance isolation"""
        self.subagents = {}
        # Filter out conflicting parameters from subagent_kwargs
        filtered_subagent_kwargs = {
            k: v
            for k, v in self.subagent_kwargs.items()
            if k
            not in [
                "llm_w_tools",
                "env",
                "llm",
                "dataset",
                "prompts",
                "env_prompts",
                "tools",
            ]
        }

        # Create subagents based on the plan
        for subtask in plan.subtasks:
            logger.info(
                f"Creating agent {subtask.agent_id} with focus: {subtask.focus}"
            )
            # Create subagent with proper parameters
            subagent = WorkerSubagent.init_from_agent(
                # Required parameters for ResearchSubagent
                agent=self,
                agent_id=subtask.agent_id,
                objective=subtask.objective,
                original_query=self.memory.original_task,
                strategy=subtask.focus,
                task_instance=task_instance,
                min_iterations_per_agent=self.min_iterations_per_agent,
                **filtered_subagent_kwargs,
            )
            self.subagents[subtask.agent_id] = subagent

        logger.info(
            f"Created {len(self.subagents)} subagents with isolated LLM instances"
        )

    async def orchestrate_work(
        self,
        task_instance: DatasetInstance,
        llm_params_dict: Optional[Dict] = None,
    ) -> OrchestrationResult:
        """Main orchestration loop for coordinating multiple agents"""
        start_time = time.time()
        max_execution_time = 300
        # Store original query in memory
        shared_prompt_templates = self.get_dataset_prompt_templates(
            dataset_instance=task_instance
        )
        self.shared_prompt_templates = shared_prompt_templates
        self.memory.original_task = shared_prompt_templates["task_instance"]

        # Step 1: Analyze and plan
        plan = self.analyze_query_and_plan(shared_prompt_templates)
        logger.info(f"Created plan with {len(plan.subtasks)} agents")

        # Store the plan in memory
        self.memory.execution_plan = plan
        self._create_subagents(plan, task_instance)

        # Step 3: Orchestrate work in rounds
        round_num = 0
        max_rounds = 5
        start_time = time.time()
        max_execution_time = 300

        round_results = None
        while round_num < max_rounds:
            # Check timeout
            if time.time() - start_time > max_execution_time:
                logger.warning("Execution timeout reached, stopping early")
                break

            round_num += 1
            logger.info(f"\n=== Orchestration Round {round_num} ===")

            round_results = await self._coordinate_and_run_subagents(plan, round_num)

            if not round_results:
                logger.warning(f"No results from round {round_num}, stopping")
                break

            # Update memory with round results
            self._update_memory_with_turn_results(round_results)

            # Check if we should stop orchestration
            if self._should_stop_orchestration(round_num, round_results):
                logger.info(f"Orchestrator decided to stop after round {round_num}")
                break

        # Step 5: Synthesize answer
        synthesis = None
        if round_results and all(
            not result.env_status.success for result in round_results.values()
        ):
            synthesis = self._synthesize_findings()

        return OrchestrationResult(
            plan=plan,
            synthesized_answer=synthesis,
            subagent_conversations={
                agent_id: agent.conv_history
                for agent_id, agent in self.subagents.items()
            },
            subagent_env_status={
                agent_id: agent.env.env_status()
                for agent_id, agent in self.subagents.items()
            },
            subagent_findings={
                agent_id: self.memory.agent_findings[agent_id]
                for agent_id in self.subagents.keys()
                if agent_id in self.memory.agent_findings
            },
            total_findings=len(self.memory.all_findings),
            lead_agent_conversation=self.conv_history,
        )

    async def _coordinate_and_run_subagents(
        self, plan: OrchestrationPlan, round_num: int
    ) -> Dict[str, SubAgentRoundResult]:
        """Coordinate agents for a single round with proper error handling and validation"""
        if not self.subagents:
            logger.error("No subagents available for coordination")
            raise ValueError("No subagents available for coordination")

        # Validate plan structure
        if not plan.subtasks:
            logger.error("Invalid plan: no subtasks found")
            raise ValueError("Invalid plan: no subtasks found")

        # Get active agents
        active_agents = [
            agent_id
            for agent_id, agent in self.subagents.items()
            if agent.conv_history.status == "active"
            and not agent.should_stop_due_to_rate_limiting()
        ]

        if not active_agents:
            logger.info("No active agents for coordination")
            return {}

        logger.info(
            f"Coordinating {len(active_agents)} active agents for round {round_num}"
        )

        def construct_tasks(
            active_agents: List[str],
            plan: OrchestrationPlan,
            round_num: int,
        ) -> List[Tuple[str, asyncio.Task]]:
            tasks = []
            for agent_id in active_agents:
                subagent = self.subagents[agent_id]
                subtask = next((t for t in plan.subtasks if t.agent_id == agent_id))

                # Validate agent and subtask
                if not subagent or not subtask:
                    logger.warning(f"Skipping invalid agent/subtask pair: {agent_id}")
                    continue
                # Create message using orchestrator's LLM-driven feedback
                message = self._create_message_for_agent(subagent, subtask, round_num)
                # Create task
                task = asyncio.create_task(
                    asyncio.to_thread(subagent.process_orchestrator_message, message)
                )
                tasks.append((agent_id, task))
            return tasks

        tasks = construct_tasks(active_agents, plan, round_num)

        # Wait for all agents with timeout
        try:
            done, pending = await asyncio.wait(
                [task for _, task in tasks],
                timeout=120,  # 2 minutes timeout per round
            )
            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Collect results with proper error handling
            results = {}
            for agent_id, task in tasks:
                if task in done:
                    result: SubAgentRoundResult = await task
                    # Validate result structure
                    results[agent_id] = result
                else:
                    logger.warning(f"Orchestrator task for agent {agent_id} timed out")
                    results[agent_id] = SubAgentRoundResult(
                        agent_id=agent_id,
                        findings="",
                        env_status=self.subagents[agent_id].env.env_status(),
                        error_msg="Timeout while waiting for agent to respond",
                    )

        except asyncio.TimeoutError:
            logger.warning("Orchestrator processing timed out")
            results = {
                agent_id: SubAgentRoundResult(
                    agent_id=agent_id,
                    findings="",
                    env_status=self.subagents[agent_id].env.env_status(),
                    error_msg="Timeout while waiting for agent to respond",
                )
                for agent_id, _ in tasks
            }

        return results

    def _create_message_for_agent(
        self, agent: WorkerSubagent, subtask: Subtask, round_num: int
    ) -> str:
        """Create LLM-driven messages for agent coordination with full team visibility"""

        def prepare_prompt_messages(template: NamedPrompt) -> List[Dict[str, Any]]:
            objective = subtask.objective
            strategy = subtask.focus

            # Get THIS agent's recent progress and findings
            agent_findings = agent.conv_history.last_outgoing_external_message
            if agent_findings:
                agent_findings_summary = (
                    f"Agent {agent.agent_id} recent findings:\n{agent_findings}"
                )
            else:
                agent_findings_summary = (
                    f"Agent {agent.agent_id}: No significant findings yet."
                )

            # Get ALL findings from ALL agents for full orchestrator visibility
            all_team_findings = []
            for other_agent_id, other_agent in self.subagents.items():
                if other_agent_id != agent.agent_id:
                    other_agent_findings = (
                        other_agent.conv_history.last_outgoing_external_message
                    )
                    if other_agent_findings:
                        all_team_findings.append(
                            f"Agent {other_agent_id}: {other_agent_findings}"
                        )
                # Create comprehensive coordination prompt using named template system
            team_context = join_with_leading_dash(all_team_findings)
            coordination_messages = template.compile(
                **self.shared_prompt_templates,
                original_query=self.memory.original_task,
                round_num=round_num + 1,
                agent_id=agent.agent_id,
                agent_objective=objective,
                agent_strategy=strategy,
                agent_iterations=agent.conv_history.total_iterations,
                min_iterations=self.min_iterations_per_agent,
                agent_status=agent.conv_history.status,
                agent_findings_summary=agent_findings_summary,
                team_context=team_context,
            )
            return coordination_messages

        coordination_messages = prepare_prompt_messages(
            self.prompts["lead_agent"].get_template("coordination")
        )

        response = self.llm.invoke(coordination_messages)
        self.conv_history.add_response(
            llm_response=response,
            tag=f"coordination_{round_num}",
        )

        message = response.text()

        if message and len(message.strip()) > 10:
            logger.info(
                f"Orchestrator coordination for {agent.agent_id}: {message[:100]}..."
            )
            return message
        else:
            logger.warning(
                f"Orchestrator coordination too short for {agent.agent_id}, using fallback"
            )
        # Fallback to coordination_fallback template from YAML
        try:
            coordination_fallback_template = self.prompts["lead_agent"].get_template(
                "coordination_fallback"
            )
            fallback_messages = prepare_prompt_messages(coordination_fallback_template)

            response = self.llm.invoke(fallback_messages)
            self.conv_history.add_response(
                llm_response=response,
                tag=f"coordination_fallback_{round_num}",
            )
            message = response.text()

            if not message or len(message.strip()) < 10:
                return self._get_fallback_message(agent, subtask.objective, round_num)

            return message

        except Exception as e:
            logger.warning(f"Failed to generate coordination fallback: {e}")
            return self._get_fallback_message(agent, subtask.objective, round_num)

    def _get_fallback_message(
        self, agent: WorkerSubagent, objective: str, round_num: int
    ) -> str:
        """Fallback message when LLM feedback fails"""
        if round_num == 0:
            return f"Work on: {objective}"
        elif agent.conv_history.total_iterations < self.min_iterations_per_agent:
            remaining = (
                self.min_iterations_per_agent - agent.conv_history.total_iterations
            )
            return f"Continue working on: {objective}. You need {remaining} more iterations."
        else:
            return f"Provide final summary for: {objective}"

    def _should_stop_orchestration(
        self, round_num: int, round_results: Dict[str, SubAgentRoundResult]
    ) -> bool:
        """LLM-driven orchestrator decision on whether to stop with FULL team visibility"""
        # We have found a solution
        if any(result.env_status.success for result in round_results.values()):
            return True
        elif round_num == 1:
            return False

        # Collect ALL findings from ALL agents across ALL rounds (not just current round)
        all_agent_findings = []

        # Get findings from all subagents
        for agent_id, agent in self.subagents.items():
            for finding in self.memory.agent_findings[agent_id]:
                if finding and len(finding.strip()) > 20:
                    all_agent_findings.append(f"Agent {agent_id}: {finding[:200]}...")
        # Also include current round results for immediate context
        current_round_summary = []
        for agent_id, result in round_results.items():
            if result.error:
                current_round_summary.append(
                    f"Agent {agent_id} encountered an error: {result.error_msg}"
                )
            else:
                iterations = self.subagents[agent_id].conv_history.curr_iteration
                current_round_summary.append(
                    f"Agent {agent_id}: {iterations} iterations"
                )

        # If no findings yet, continue (up to round 3)
        if not all_agent_findings and round_num <= 3:
            return False

        stopping_template = self.prompts["lead_agent"].get_template("stopping_decision")
        decision_messages = stopping_template.compile(
            **self.shared_prompt_templates,
            round_num=round_num,
            total_findings=len(all_agent_findings),
            total_agents=len(self.subagents),
            current_round_summary=chr(10).join(current_round_summary)
            if current_round_summary
            else "No current round activity",
            all_findings=chr(10).join(all_agent_findings)
            if all_agent_findings
            else "No significant findings collected yet",
        )

        response = self.llm.invoke(decision_messages)
        self.conv_history.add_response(
            llm_response=response,
            tag=f"stopping_decision_{round_num}",
        )
        decision = response.text().strip()

        if "STOP" in decision.upper():
            logger.info(f"Orchestrator decided to STOP: {response.text().strip()}")
            return True
        else:
            logger.info(f"Orchestrator decided to CONTINUE: {response.text().strip()}")
            return False

    def _synthesize_findings(self) -> str:
        """Synthesize findings into a final answer"""
        # Collect all findings from all agents
        all_findings = []
        for agent_id, findings in self.memory.agent_findings.items():
            for finding in findings:
                all_findings.append(f"Agent {agent_id}: {finding}")

        logger.info(f"Synthesizing findings for the task: {self.memory.original_task}")
        logger.info(f"Total findings: {len(all_findings)}")

        synthesis_template = self.prompts["lead_agent"].get_template("synthesis")
        synthesis_messages = synthesis_template.compile(
            **self.shared_prompt_templates,
            all_findings=join_with_leading_dash(all_findings),
        )

        response = self.llm.invoke(synthesis_messages)
        self.conv_history.add_response(
            llm_response=response,
            tag="synthesis",
        )
        synthesis = response.text().strip()
        logger.info(f"Synthesis completed: {synthesis[:150]}...")
        return synthesis

    def _update_memory_with_turn_results(
        self, turn_results: Dict[str, SubAgentRoundResult]
    ) -> None:
        """Update memory with turn results for better orchestration"""
        for agent_id, result in turn_results.items():
            if not result.error:
                # Add new findings to memory
                new_findings = result.findings
                if new_findings:
                    self.memory.add_findings(agent_id, new_findings)
