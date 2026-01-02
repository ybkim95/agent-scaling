import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from agent_scaling.agents.base import BaseAgentWithTools
from agent_scaling.logger import logger

from .memory import Memory
from .worker import Worker


def parse_json_output(text: str) -> dict:
    """Parse JSON output with better error handling"""
    # Remove common markdown/code block wrappers
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        raise e

class Orchestrator(BaseAgentWithTools):
    """Generic lead agent for coordinating multiple worker agents"""
    required_prompts = [
        "lead_agent"
    ]

    def __init__(
        self,
        *args,
        memory: Memory,
        min_iterations: int = 3,
        num_workers: int = 3,
        max_rounds: int = 5,
        max_execution_time: int = 300,
        worker_timeout: int = 120,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory
        self.min_iterations = min_iterations
        self.num_workers = num_workers
        self.max_rounds = max_rounds
        self.max_execution_time = max_execution_time
        self.worker_timeout = worker_timeout
        self.workers: Dict[str, Worker] = {}
        self.worker_kwargs = kwargs
        
        self.agent_progress: Dict[str, Dict[str, Any]] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self.round_summaries: List[Dict[str, Any]] = []
        
        self.metrics = {
            "orchestrator_llm_calls": 0,
            "worker_llm_calls": {},
            "total_tool_calls": 0,
            "communication_count": 0,
            "start_time": None,
            "end_time": None
        }

    def _invoke_llm_for_planning(self, messages):
        """Invoke LLM for planning using LiteLLM's built-in retry"""
        self.metrics["orchestrator_llm_calls"] += 1
        if self.llm_w_tools:
            return self.llm_w_tools.invoke(messages, num_retries=2, **getattr(self, 'llm_params_dict', {}))
        else:
            return self.llm.invoke(messages)

    def _invoke_llm_for_synthesis(self, messages):
        """Invoke LLM for synthesis using LiteLLM's built-in retry"""
        self.metrics["orchestrator_llm_calls"] += 1
        if self.llm_w_tools:
            return self.llm_w_tools.invoke(messages, num_retries=2, **getattr(self, 'llm_params_dict', {}))
        else:
            return self.llm.invoke(messages)

    def _invoke_llm_for_orchestration_decision(self, content):
        """Invoke LLM for orchestration decisions using LiteLLM's built-in retry"""
        self.metrics["orchestrator_llm_calls"] += 1
        if isinstance(content, list):
            messages = content
        else:
            if hasattr(self, 'prompts') and 'lead_agent' in self.prompts and hasattr(self, 'shared_prompt_templates') and self.shared_prompt_templates:
                messages = self.prompts["lead_agent"].compile(**self.shared_prompt_templates)
                messages.append({"role": "user", "content": content})
            else:
                messages = [{"role": "user", "content": content}]
        
        if self.llm_w_tools:
            return self.llm_w_tools.invoke(messages, num_retries=2, **getattr(self, 'llm_params_dict', {}))
        else:
            return self.llm.invoke(messages[-1]["content"] if isinstance(messages, list) else content)

    def _invoke_llm_for_agent_feedback(self, content):
        """Invoke LLM for creating agent feedback using LiteLLM's built-in retry"""
        self.metrics["orchestrator_llm_calls"] += 1
        self.metrics["communication_count"] += 1  

        if isinstance(content, list):
            messages = content
        else:
            if hasattr(self, 'prompts') and 'lead_agent' in self.prompts and hasattr(self, 'shared_prompt_templates') and self.shared_prompt_templates:
                messages = self.prompts["lead_agent"].compile(**self.shared_prompt_templates)
                messages.append({"role": "user", "content": content})
            else:
                messages = [{"role": "user", "content": content}]
        
        if self.llm_w_tools:
            return self.llm_w_tools.invoke(messages, num_retries=2, **getattr(self, 'llm_params_dict', {}))
        else:
            return self.llm.invoke(messages[-1]["content"] if isinstance(messages, list) else content)

    def _invoke_llm(self, messages):
        """Generic LLM invocation method for various purposes"""
        self.metrics["orchestrator_llm_calls"] += 1
        if self.llm_w_tools:
            return self.llm_w_tools.invoke(messages, num_retries=2, **getattr(self, 'llm_params_dict', {}))
        else:
            return self.llm.invoke(messages)

    def generate_plan(self, query: str, shared_prompt_templates: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze query and create execution plan"""
        if not shared_prompt_templates:
            shared_prompt_templates = getattr(self, 'shared_prompt_templates', {})
            
        logger.info(f"Analyzing query with shared templates: {list(shared_prompt_templates.keys())}")
        
        try:
            planning_template = self.prompts["lead_agent"].get_template("planning")
            planning_messages = planning_template.compile(
                **shared_prompt_templates,
                query=query,
                num_workers=self.num_workers
            )
            
            response = self._invoke_llm_for_planning(planning_messages)
            
            response_text = response.content.strip()
            logger.info(f"LLM planning response: {response_text}...")
            
            try:
                # Use improved JSON parsing with error handling
                plan = parse_json_output(response_text)
                
                if plan and "subtasks" in plan:
                    logger.info(f"Successfully created plan with {len(plan['subtasks'])} subtasks")
                    return plan
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return
                
        except Exception as e:
            logger.warning(f"Prompt compilation failed: {e}, using simple fallback")
            return

    def _create_workers(self, plan: Dict[str, Any]):
        """Create workers based on the plan with proper LLM instance isolation"""
        self.workers = {}
        dataset_templates = plan.get("dataset_templates", {})
        self.agent_progress = {}

        if not self.llm_w_tools and not self.llm:
            logger.error("No LLM instance available for workers")
            raise ValueError("LLM instance must be provided for worker creation")

        # Create workers based on the plan
        for subtask in plan["subtasks"]:
            objective = subtask.get("objective", "")
            focus = subtask.get("focus", "General work")

            logger.info(f"Creating agent {subtask['agent_id']} with focus: {focus}")

            isolated_llm_w_tools = None
            if self.llm_w_tools:
                try:
                    isolated_llm_w_tools = self.llm_w_tools
                    logger.info(f"Using shared LLM instance for {subtask['agent_id']} (LLM instances are typically thread-safe)")
                except Exception as e:
                    logger.warning(f"Failed to create isolated LLM for {subtask['agent_id']}, using shared instance: {e}")
                    isolated_llm_w_tools = self.llm_w_tools

            # Create worker with proper parameters
            try:
                agent = Worker(
                    agent_id=subtask["agent_id"],
                    objective=objective,
                    original_query=self.memory.original_query,
                    strategy=focus,
                    memory=self.memory,
                    min_iterations=self.min_iterations,
                    shared_prompt_templates=dataset_templates,
                    env=self.env,
                    llm_w_tools=isolated_llm_w_tools,  # Use isolated instance
                    llm_params_dict=self.llm_params_dict, # Pass llm_params_dict
                    llm=self.llm,
                    dataset=self.dataset,
                    prompts=self.prompts,
                )
                
                self.workers[subtask["agent_id"]] = agent
                
                # Initialize progress tracking for this agent
                self.agent_progress[subtask["agent_id"]] = {
                    "objective": objective,
                    "focus": focus,
                    "total_iterations": 0,
                    "findings_count": 0,
                    "last_round_findings": 0,
                    "progress_summary": "",
                    "status": "active",
                    "round_history": []
                }
                
                logger.info(f"Successfully created worker {subtask['agent_id']}")
                
            except Exception as e:
                logger.error(f"Failed to create worker {subtask['agent_id']}: {e}")
                raise RuntimeError(f"Subagent creation failed for {subtask['agent_id']}: {e}")
            
        logger.info(f"Created {len(self.workers)} workers with LLM instance isolation and progress tracking")

    async def orchestrate(self, query: str, shared_prompt_templates: Optional[Dict] = None, env=None, llm_w_tools=None, llm_params_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Main orchestration loop for coordinating multiple agents"""
        start_time = time.time()
        self.metrics["start_time"] = start_time

        self.env = env
        self.llm_w_tools = llm_w_tools
        self.shared_prompt_templates = shared_prompt_templates
        self.llm_params_dict = llm_params_dict or {}

        if not self.env:
            logger.error("No environment provided for workers")
            raise ValueError("Environment must be provided for worker execution")
        
        if not self.llm_w_tools and not self.llm:
            logger.error("No LLM instance provided for workers")
            raise ValueError("LLM instance must be provided for worker execution")
        
        if not self.shared_prompt_templates:
            logger.warning("No shared prompt templates provided, using fallback prompts")

        self.memory.original_query = query

        # Step 1: Analyze and generate plan
        plan = self.generate_plan(query, shared_prompt_templates)
        logger.info(f"Created plan with {len(plan['subtasks'])} agents")
        self.memory.execution_plan = plan
        plan["dataset_templates"] = shared_prompt_templates

        # Step 2: Create workers
        try:
            self._create_workers(plan) # agent init with subtasks
            logger.info(f"Successfully created {len(self.workers)} workers")
        
        except Exception as e:
            logger.error(f"Failed to create workers: {e}")
            raise RuntimeError(f"Subagent creation failed: {e}")

        # Step 3: Orchestrate work in rounds
        round_num = 0
        start_time = time.time()
        
        # Start main while loop
        while round_num < self.max_rounds:
            if time.time() - start_time > self.max_execution_time:
                logger.warning("Execution timeout reached, stopping early")
                break

            round_num += 1
            logger.info(f"\n=== Orchestration Round {round_num} ===")

            round_results = await self.run_orchestrate(plan, round_num)
            
            if not round_results:
                break
            
            # Dynamically update plan and progress based on the findings
            plan = self.update_plan(plan, round_results, round_num)
            self._cleanup()
            
            # Check stopping orchestration
            if self.should_stop(query, round_num, round_results):
                logger.info(f"Orchestrator decided to stop after round {round_num}")
                break
        
        # Step 4: Collect all findings and worker metrics
        for agent in self.workers.values():
            self.memory.add_findings(agent.agent_id, agent.conversation.findings)
            self.metrics["worker_llm_calls"][agent.agent_id] = agent.conversation.llm_call_count
            self.metrics["total_tool_calls"] += len(agent.conversation.findings)

        # Step 5: Synthesize answer
        answer = self.aggregate_findings(query)
        
        self.metrics["end_time"] = time.time()
        execution_time = self.metrics["end_time"] - self.metrics["start_time"]
        total_llm_calls = self.metrics["orchestrator_llm_calls"] + sum(self.metrics["worker_llm_calls"].values())

        return {
            "plan": plan,
            "answer": answer,
            "worker_summaries": {
                agent_id: agent.get_conversation_summary()
                for agent_id, agent in self.workers.items()
            },
            "num_findings": len(self.memory.all_findings),
            "coordination_history": self.coordination_history,
            "round_summaries": self.round_summaries,
            "agent_progress": self.agent_progress,
            "metrics": {
                "architecture": "centralized",
                "num_workers": self.num_workers,
                "rounds_completed": round_num,
                "execution_time_s": round(execution_time, 2),
                "orchestrator_llm_calls": self.metrics["orchestrator_llm_calls"],
                "worker_llm_calls": self.metrics["worker_llm_calls"],
                "total_llm_calls": total_llm_calls,
                "total_tool_calls": self.metrics["total_tool_calls"],
                "communication_count": self.metrics["communication_count"],
                "findings_per_agent": {
                    agent_id: len(agent.conversation.findings)
                    for agent_id, agent in self.workers.items()
                }
            }
        }

    # Communication: (Orchestrator) -> (Sub-Agent)
    async def run_orchestrate(self, plan: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        if not self.workers:
            logger.error("No workers available for coordination")
            return {}
        
        if not plan.get("subtasks"):
            logger.error("Invalid plan: no subtasks found")
            return {}
        
        active_agents = [
            agent_id for agent_id, agent in self.workers.items() if agent.conversation.status == "active" and not agent.is_rate_limited()
        ]
        
        if not active_agents:
            logger.info("No active agents for coordination")
            return {}
        
        logger.info(f"Coordinating {len(active_agents)} active agents for round {round_num}")
        
        # Create tasks for each agent with staggered execution to prevent rate limiting
        sub_tasks = []
        for idx, agent_id in enumerate(active_agents):
            agent = self.workers[agent_id]
            subtask = next((t for t in plan["subtasks"] if t["agent_id"] == agent_id), {})
            
            if not agent or not subtask:
                continue
            
            try:
                # Strategic feedback based on the progress and findings
                feedback = self.get_feedback(agent, subtask, round_num)
            
            except Exception as e:
                logger.error(f"Error creating feedback for {agent_id}: {e}")
                feedback = f"Work on: {subtask.get('objective', 'your objective')}"
            
            if idx > 0:
                await asyncio.sleep(0.5)
            
            # Communication: (Orchestrator) -> (Sub-Agent)
            task = asyncio.create_task(
                asyncio.to_thread(agent.process_orchestrator_request, feedback)
            )
            sub_tasks.append((agent_id, task))

        # Wait for all agents to complete
        try:
            done, pending = await asyncio.wait(
                [task for _, task in sub_tasks],
                timeout=self.worker_timeout,
            )

            # Cancel any pending tasks that didn't complete
            for task in pending:
                task.cancel()

            # Collect results from all completed agents
            round_results = {}
            for agent_id, task in sub_tasks:
                if task in done:
                    try:
                        result = await task
                        
                        if not isinstance(result, dict):
                            logger.error(f"Invalid result from {agent_id}: {type(result)}")
                            result = {"error": "Invalid result structure"}
                        
                        round_results[agent_id] = result
                        logger.info(f"Agent {agent_id}: {result.get('iterations', 0)} iterations, status: {result.get('status', 'unknown')}")
                    
                    except Exception as e:
                        logger.error(f"Error in orchestrator processing for agent {agent_id}: {e}")
                        round_results[agent_id] = {"error": str(e), "status": "error"}
                else:
                    logger.warning(f"Orchestrator task for agent {agent_id} timed out")
                    round_results[agent_id] = {"error": "Timeout", "status": "timeout"}

        except asyncio.TimeoutError:
            logger.warning("Orchestrator processing timed out")
            round_results = {agent_id: {"error": "Round timeout", "status": "timeout"} for agent_id, _ in sub_tasks}

        # Check for early consensus termination
        if round_num > 1:
            impossible_count = 0
            for agent_id, result in round_results.items():
                findings = result.get("new_findings", [])
                for finding in findings:
                    if "impossible" in str(finding.get("tool", "")).lower():
                        impossible_count += 1
                        break
            
            if impossible_count >= len(active_agents):
                logger.info(f"All {len(active_agents)} agents agree task is impossible - early termination")
                for agent_id in active_agents:
                    if agent_id in self.workers:
                        self.workers[agent_id].conversation_state["status"] = "completed"
        
        return round_results

    def get_feedback(self, agent: Worker, subtask: Dict[str, Any], round_num: int) -> str:
        objective = subtask.get("objective", "")
        strategy = subtask.get("focus", "general approach")
        
        # Get agent's progress from memory and progress tracking
        agent_id = agent.agent_id
        agent_progress = self.agent_progress.get(agent_id, {})
        
        # Get agent's recent findings from memory
        agent_findings = self.memory.get_findings_for_agent(agent_id)
        recent_findings = agent_findings[-3:] if agent_findings else []
        
        # Create agent progress summary
        agent_findings_summary = ""
        if recent_findings:
            agent_findings_summary = f"Agent {agent_id} recent findings:\n"
            for i, finding in enumerate(recent_findings, 1):
                content = finding.get("content", "")
                if content:
                    agent_findings_summary += f"- {content}\n"
        else:
            agent_findings_summary = f"Agent {agent_id}: No significant findings yet."
        
        # Create comprehensive progress analysis using agent_progress data
        progress_analysis = ""
        if agent_progress:
            total_iterations = agent_progress.get("total_iterations", 0)
            findings_count = agent_progress.get("findings_count", 0)
            last_round_findings = agent_progress.get("last_round_findings", 0)
            round_history = agent_progress.get("round_history", [])
            agent_status = agent_progress.get("status", "active")
            recent_performance = '; '.join(round_history[-2:]) if round_history else 'No recent rounds'
            
            try:
                # Use the progress_analysis template from YAML
                progress_template = self.prompts["lead_agent"].get_template("progress_analysis")
                progress_analysis = progress_template.compile(
                    **self.shared_prompt_templates,
                    total_iterations=total_iterations,
                    findings_count=findings_count,
                    last_round_findings=last_round_findings,
                    agent_status=agent_status,
                    recent_performance=recent_performance,
                    recent_performance_analysis=f"Recent performance shows: {recent_performance}"
                )
            except Exception as e:
                logger.warning(f"Failed to use progress_analysis template: {e}")
                # Fallback to simple text
                progress_analysis = f"Progress: {findings_count} findings, {total_iterations} iterations, status: {agent_status}"
        
        # Get all findings from all agents for full orchestrator visibility
        all_team_findings = []
        for other_agent_id, _ in self.workers.items():
            if other_agent_id != agent_id:
                other_findings = self.memory.get_findings_for_agent(other_agent_id)
                for finding in other_findings[-3:]:
                    content = finding.get("content", "")
                    if content and len(content.strip()) > 20:
                        all_team_findings.append(f"Agent {other_agent_id}: {content}...")
        
        # Add memory findings for complete picture
        memory_findings = []
        if hasattr(self.memory, 'all_findings') and self.memory.all_findings:
            for finding in self.memory.all_findings[-3:]:  
                content = finding.get("content", "")
                if content and len(content.strip()) > 20:
                    memory_findings.append(f"Memory: {content}...")
        
        team_context = ""
        if all_team_findings or memory_findings:
            team_context = "\nTeam's findings for context:\n"
            for finding in (all_team_findings + memory_findings)[-5:]:  # Limit to 5 most recent
                team_context += f"- {finding}\n"
        else:
            team_context = "\nTeam's findings: No significant findings from other agents yet."
        
        # Get strategic direction from LLM
        strategies = self.get_stratigies(agent_id, agent_findings or [], all_team_findings, round_num)
        
        # Use YAML template system with enhanced progress data and strategic direction
        try:
            coordination_template = self.prompts["lead_agent"].get_template("coordination")
            coordination_messages = coordination_template.compile(
                **self.shared_prompt_templates,
                original_query=self.memory.original_query,
                round_num=round_num + 1,
                agent_id=agent_id,
                agent_objective=objective,
                agent_strategy=strategy,
                agent_iterations=agent.conversation.total_iterations,
                min_iterations=self.min_iterations,
                agent_status=agent.conversation.status,
                agent_findings_summary=agent_findings_summary,
                team_context=team_context,
                progress_analysis=progress_analysis,
                strategic_direction=strategies
            )
                
            response = self._invoke_llm_for_agent_feedback(coordination_messages)
            message = response.content.strip()
            
            if message:
                logger.info(f"Orchestrator strategic feedback for {agent_id}: {message}...")
                
                # Store coordination message in history for context continuity
                self.coordination_history.append({
                    "round": round_num,
                    "agent_id": agent_id,
                    "message": message,
                    "timestamp": time.time()
                })
                
                return message
            else:
                logger.warning(f"Orchestrator coordination too short for {agent_id}, using fallback")
            
        except Exception as e:
            logger.warning(f"Failed to use named template system for coordination: {e}")
        
        try:
            coordination_fallback_template = self.prompts["lead_agent"].get_template("coordination_fallback")
            fallback_messages = coordination_fallback_template.compile(
                **self.shared_prompt_templates,
                original_query=self.memory.original_query,
                agent_id=agent_id,
                agent_objective=objective,
                agent_strategy=strategy,
                round_num=round_num + 1,
                agent_status=agent.conversation.status,
                agent_findings_summary=agent_findings_summary,
                team_context=team_context,
                progress_analysis=progress_analysis,
                strategic_direction=strategies
            )
            
            response = self._invoke_llm_for_agent_feedback(fallback_messages)
            message = response.content.strip()
            
            if not message:
                logger.warning(f"YAML fallback coordination too short for agent {agent_id}, using simple fallback")
                return self._get_fallback_message(agent, objective, round_num)
            
            logger.info(f"YAML fallback coordination for agent {agent_id}: {message[:100]}...")
            
            self.coordination_history.append({
                "round": round_num,
                "agent_id": agent_id,
                "message": message,
                "timestamp": time.time()
            })
            
            return message
            
        except Exception as e:
            logger.warning(f"Failed to generate YAML fallback coordination for agent {agent_id}: {e}")
            return self._get_fallback_message(agent, objective, round_num)

    def _get_fallback_message(self, agent: Worker, objective: str, round_num: int) -> str:
        """Fallback message when LLM feedback fails"""
        if round_num == 0:
            return f"Work on: {objective}"
        elif agent.conversation.total_iterations < self.min_iterations:
            remaining = self.min_iterations - agent.conversation.total_iterations
            return f"Continue working on: {objective}. You need {remaining} more iterations."
        else:
            return f"Provide final summary for: {objective}"

    def should_stop(self, query: str, round_num: int, round_results: Dict[str, Any]) -> bool:
        """LLM-driven orchestrator decision on whether to stop or continue"""
        try:
            # Collect all findings from all agents across all rounds
            all_agent_findings = []
            
            # Get findings from all workers
            for agent_id, agent in self.workers.items():
                if hasattr(agent.conversation, 'findings') and agent.conversation.findings:
                    for finding in agent.conversation.findings:
                        content = finding.get("content", "")
                        if content and len(content.strip()) > 20:
                            all_agent_findings.append(f"Agent {agent_id}: {content[:200]}...")
            
            # Get findings from memory
            memory_findings = []
            if hasattr(self.memory, 'all_findings') and self.memory.all_findings:
                for finding in self.memory.all_findings:
                    content = finding.get("content", "")
                    if content and len(content.strip()) > 20:
                        memory_findings.append(f"Memory: {content[:200]}...")
            
            # Combine all findings
            all_findings = all_agent_findings + memory_findings
            
            # Also include current round results for immediate context
            current_round_summary = []
            for agent_id, result in round_results.items():
                if result.get("error"):
                    current_round_summary.append(f"Agent {agent_id} encountered an error: {result['error']}")
                else:
                    iterations = result.get("iterations", 0)
                    findings_count = len(result.get("new_findings", []))
                    current_round_summary.append(f"Agent {agent_id}: {iterations} iterations, {findings_count} new findings this round")
            
            # Configuration-based stopping conditions
            min_rounds = min(2, self.max_rounds)  # At least 2 rounds or max_rounds if less
            
            # Always continue for minimum rounds
            if round_num < min_rounds:
                return False
                
            # If no findings yet, continue up to round 3
            if not all_findings and round_num <= min(3, self.max_rounds):
                return False
            
            # Create comprehensive orchestrator decision prompt using named template system
            try:
                stopping_template = self.prompts["lead_agent"].get_template("stopping_decision")
                decision_messages = stopping_template.compile(
                    **self.shared_prompt_templates,
                    query=query,
                    round_num=round_num,
                    total_findings=len(all_findings),
                    total_agents=len(self.workers),
                    current_round_summary=chr(10).join(current_round_summary) if current_round_summary else "No current round activity",
                    all_findings=chr(10).join(all_findings[-10:]) if all_findings else "No significant findings collected yet"
                )
                
                response = self._invoke_llm_for_orchestration_decision(decision_messages)
                decision = response.content.strip().upper()
                
                if "stop" in decision.lower():
                    logger.info(f"Orchestrator decided to STOP (from YAML template): {response.content.strip()}")
                    return True
                else:
                    logger.info(f"Orchestrator decided to CONTINUE (from YAML template): {response.content.strip()}")
                    return False
                    
            except Exception as e:
                logger.warning(f"Error in named template orchestrator decision: {e}")
            
            try:
                stopping_fallback_template = self.prompts["lead_agent"].get_template("stopping_decision_fallback")
                fallback_messages = stopping_fallback_template.compile(
                    **self.shared_prompt_templates,
                    query=query,
                    round_num=round_num,
                    total_findings=len(all_findings),
                    all_findings=chr(10).join(all_findings[-5:]) if all_findings else "No significant findings yet"
                )
                
                response = self._invoke_llm_for_orchestration_decision(fallback_messages)
                decision = response.content.strip().upper()
                
                if "stop" in decision.lower():
                    logger.info(f"Orchestrator decided to STOP (YAML fallback): {response.content.strip()}")
                    return True
                else:
                    logger.info(f"Orchestrator decided to CONTINUE (YAML fallback): {response.content.strip()}")
                    return False
                    
            except Exception as e:
                logger.warning(f"Error in YAML fallback orchestrator decision: {e}")
            
            # Safety fallback
            all_agents_completed_minimum = all(
                agent.conversation.total_iterations >= self.min_iterations
                for agent in self.workers.values()
            )
            
            # Stop if all agents completed minimum and we have some findings
            if all_agents_completed_minimum and len(all_findings) > 0:
                logger.info("Safety fallback stopping condition: all agents completed minimum with findings")
                return True
                
            if round_num >= self.max_rounds - 1:
                logger.info(f"Safety limit reached - stopping after round {round_num}")
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Error in orchestration stopping condition: {e}")
            return round_num >= 3

    def aggregate_findings(self, query: str) -> str:
        """Synthesize findings into a final answer using prompt templates"""
        try:
            # Collect unique findings from all sources
            all_findings = []
            seen_content = set()
            
            # Collect from workers first
            for agent in self.workers.values():
                for finding in agent.conversation.findings:
                    content = finding.get("content", "").strip()
                    content_key = content
                    if content and content_key not in seen_content:
                        all_findings.append(finding)
                        seen_content.add(content_key)
            
            logger.info(f"Synthesizing {len(all_findings)} unique findings for: {query}")
            
            if not all_findings:
                return f"No information was collected to answer: {query}"
            
            # Build findings content for LLM
            findings_content = ""
            for i, finding in enumerate(all_findings[:20], 1):
                content = finding.get("content", "")
                if content:
                    agent_id = finding.get("agent_id", "unknown")
                    findings_content += f"Finding {i} (from {agent_id}): {content.strip()}\n\n"
            
            # Try LLM synthesis first
            try:
                if hasattr(self, 'prompts') and 'lead_agent' in self.prompts:
                    synthesis_template = self.prompts["lead_agent"].get_template("synthesis")
                    synthesis_messages = synthesis_template.compile(
                        **self.shared_prompt_templates,
                        query=query,
                        all_findings=findings_content
                    )
                    
                    response = self._invoke_llm_for_synthesis(synthesis_messages)
                    synthesis = response.content.strip()
                    
                    # Validate synthesis is a direct answer
                    if synthesis and len(synthesis) > 20:
                        # Remove any JSON or code block formatting
                        if synthesis.startswith('{') or synthesis.startswith('```'):
                            logger.warning("Synthesis returned structured format, using fallback")
                            return self._create_simple_synthesis(all_findings, query)
                        return synthesis
                        
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}, using fallback")
            
            # Fallback: Create simple synthesis from findings
            return self._create_simple_synthesis(all_findings, query)
                
        except Exception as e:
            logger.error(f"Critical error in synthesis: {e}")
            return f"Error occurred while processing the question: {query}"
    
    def _create_simple_synthesis(self, findings: list, query: str) -> str:
        """Create a simple synthesis when LLM fails"""
        if not findings:
            return f"Unable to find information to answer: {query}"
        
        # Prioritize findings with substantive content
        substantive_findings = [
            f for f in findings 
            if len(f.get("content", "")) > 50 and 
            "error" not in f.get("content", "").lower()
        ]
        
        if substantive_findings:
            # Use the most substantive finding
            best_finding = max(substantive_findings, key=lambda f: len(f.get("content", "")))
            content = best_finding.get("content", "").strip()
            
            # Clean up the content for presentation
            if len(content) > 500:
                content = content[:500] + "..."
            
            return f"Based on the investigation: {content}"
        
        # Last resort: use any available finding
        for finding in findings:
            content = finding.get("content", "").strip()
            if content and len(content) > 20:
                return f"Available information: {content[:300]}..."
        
        return f"Unable to find specific information to answer: {query}"
    
    def update_plan(self, plan: Dict[str, Any], round_results: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        """Update the plan objectives directly based on LLM decision"""
        try:
            total_new_findings = 0
            agent_progress_summary = []
            
            for agent_id, result in round_results.items():
                if not result.get("error"):
                    new_findings = result.get("new_findings", [])
                    if new_findings:
                        self.memory.add_findings(agent_id, new_findings)
                        total_new_findings += len(new_findings)
                        logger.info(f"Added {len(new_findings)} findings from {agent_id} to memory")
                    
                    # Update progress tracking
                    if agent_id in self.agent_progress:
                        self.agent_progress[agent_id]["total_iterations"] = result.get("total_iterations", 0)
                        self.agent_progress[agent_id]["findings_count"] += len(new_findings)
                        self.agent_progress[agent_id]["last_round_findings"] = len(new_findings)
                        self.agent_progress[agent_id]["status"] = result.get("status", "active")
                        
                        # Create progress summary for this agent
                        progress_summary = f"Round {round_num}: {len(new_findings)} new findings, {result.get('iterations', 0)} iterations"
                        self.agent_progress[agent_id]["round_history"].append(progress_summary)
                        
                        # Add to agent progress summary for LLM
                        agent_progress_summary.append(f"{agent_id}: {len(new_findings)} findings, status={result.get('status', 'active')}")
                    
                    # Update conversation state if available
                    conversation_state = result.get("conversation_state", {})
                    if conversation_state and agent_id in self.workers:
                        self.workers[agent_id].conversation_state = conversation_state
                else:
                    # Handle error case
                    if agent_id in self.agent_progress:
                        self.agent_progress[agent_id]["status"] = "error"
                        self.agent_progress[agent_id]["last_error"] = result.get("error", "Unknown error")
                        agent_progress_summary.append(f"{agent_id}: ERROR")
            
            # LLM decides if we need to update the plan
            try:
                team_summary = f"Round {round_num}: {total_new_findings} new findings"
                if agent_progress_summary:
                    team_summary += f"\nAgents: {'; '.join(agent_progress_summary)}"
                
                plan_update_template = self.prompts["lead_agent"].get_template("plan_update")
                plan_update_messages = plan_update_template.compile(
                    **self.shared_prompt_templates,
                    original_query=self.memory.original_query,
                    round_num=round_num,
                    total_findings_so_far=len(self.memory.all_findings),
                    total_iterations_so_far=sum(p.get("total_iterations", 0) for p in self.agent_progress.values()),
                    current_subtasks_count=len(plan.get("subtasks", [])),
                    new_findings_this_round=total_new_findings,
                    active_agents_count=len([a for a in agent_progress_summary if "ERROR" not in a]),
                    agent_progress_summary="\n".join(agent_progress_summary) if agent_progress_summary else "No progress yet",
                    team_performance_analysis=team_summary
                )
                
                response = self._invoke_llm(plan_update_messages)
                
                # Use improved JSON parsing with error handling
                response_text = response.content.strip()
                plan_update_decision = parse_json_output(response_text)
                
                if 'update' in plan_update_decision.get("plan_update_decision", "").lower():
                    logger.info(f"Plan update: {plan_update_decision.get('reasoning', 'Update needed')}")
                    
                    # Update strategy/focus, not the core objective
                    modified_subtasks = plan_update_decision.get("modified_subtasks", [])
                    for modification in modified_subtasks:
                        agent_id = modification["agent_id"]
                        
                        # Find the index and update strategy/focus but preserve original objective
                        for i, subtask in enumerate(plan["subtasks"]):
                            if subtask["agent_id"] == agent_id:
                                if "modification" in modification:
                                    if "original_objective" not in plan["subtasks"][i]:
                                        plan["subtasks"][i]["original_objective"] = plan["subtasks"][i]["objective"]
                                    
                                    # Update focus/strategy, not objective
                                    plan["subtasks"][i]["focus"] = modification.get("focus", plan["subtasks"][i].get("focus", "general"))
                                    logger.info(f"Updated focus for {agent_id}: {plan['subtasks'][i]['focus']}, keeping original objective")
                else:
                    logger.info(f"Plan maintained: {plan_update_decision.get('reasoning', 'Current plan working well')}")
                    
            except Exception as e:
                logger.warning(f"Error in LLM plan update: {e}. Keeping current plan.")
            
            logger.info(f"Round {round_num} completed with {total_new_findings} new findings")
            
            self.memory.execution_plan = plan
            return plan
            
        except Exception as e:
            logger.warning(f"Error updating execution plan: {e}")
            return plan

    def _cleanup(self) -> None:
        """Smart cleanup: preserve important context while preventing memory bloat"""
        for agent_id, agent in self.workers.items():
            if hasattr(agent, 'conversation_state') and agent.conversation_state:
                messages = agent.conversation_state.get("messages", [])
                
                if len(messages) > 20:
                    # Keep system, user, and error messages + sample of assistant messages
                    important_messages = []
                    
                    # Always keep first few messages for context
                    important_messages.extend(messages[:3])
                    
                    # Keep all user/system messages and errors
                    for msg in messages[3:]:
                        role = msg.get("role", "")
                        content = str(msg.get("content", "")).lower()
                        
                        if role in ["user", "system"] or "error" in content:
                            important_messages.append(msg)
                        elif role == "assistant" and len(important_messages) % 3 == 0:
                            important_messages.append(msg)
                    
                    if messages[-2:] not in important_messages:
                        important_messages.extend(messages[-2:])
                    
                    old_count = len(messages)
                    agent.conversation_state["messages"] = important_messages[-15:]  
                    logger.info(f"Smart cleanup for {agent_id}: {old_count} → {len(agent.conversation_state['messages'])} messages")
                
                # Preserve other important state
                agent.conversation_state["current_round"] = agent.conversation_state.get("current_round", 0)
                agent.conversation_state["total_iterations"] = agent.conversation_state.get("total_iterations", 0)
                agent.conversation_state["status"] = agent.conversation_state.get("status", "active")
            
            # Cleanup conversation turns more conservatively
            if hasattr(agent, 'conversation') and agent.conversation.turns:
                if len(agent.conversation.turns) > 20:  
                    old_turns = len(agent.conversation.turns)
                    agent.conversation.turns = agent.conversation.turns[-10:]
                    logger.info(f"Cleaned conversation turns for {agent_id}: {old_turns} → 10 turns")

    def get_stratigies(self, agent_id: str, agent_findings: List, team_findings: List[str], round_num: int) -> str:
        """Ask LLM for strategic direction for this agent"""
        try:
            agent_progress = self.agent_progress.get(agent_id, {})
            total_iterations = agent_progress.get("total_iterations", 0)
            findings_count = agent_progress.get("findings_count", 0)
            
            strategic_template = self.prompts["lead_agent"].get_template("strategic_direction")
            strategic_messages = strategic_template.compile(
                **self.shared_prompt_templates,
                agent_id=agent_id,
                round_num=round_num,
                findings_count=findings_count,
                total_iterations=total_iterations,
                team_findings_count=len(team_findings)
            )
            
            response = self._invoke_llm(strategic_messages)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Error getting strategic direction: {e}")
            return "Continue current approach and report progress."