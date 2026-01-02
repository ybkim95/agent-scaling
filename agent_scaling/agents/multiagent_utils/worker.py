import time
import threading
from typing import Any, Dict, List, Optional

from langchain_core.messages.utils import convert_to_openai_messages

from agent_scaling.logger import logger

from .conversation import WorkerConversation
from .memory import Memory


class Worker:
    def __init__(
        self,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        memory: Memory,
        min_iterations: int = 3,
        max_iterations: int = 10,
        shared_prompt_templates: Optional[Dict] = None,
        env=None,
        llm_w_tools=None,
        llm_params_dict: Optional[Dict] = None,
        llm=None,
        dataset=None,
        prompts=None,
    ):
        self.agent_id = agent_id
        
        # Core agent attributes
        self.objective = objective
        self.original_query = original_query
        self.strategy = strategy
        self.memory = memory
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        
        # LLM and environment setup
        self.agent_llm_w_tools = llm_w_tools
        self.agent_env = env
        self.llm_params_dict = llm_params_dict or {}
        
        # Prompt templates
        self.shared_prompt_templates = shared_prompt_templates
        self.prompts = prompts
        
        # Base agent attributes
        self.llm = llm
        self.dataset = dataset
        
        # Conversation management
        self.conversation = WorkerConversation(agent_id)
        
        # Conversation state initialization with memory integration
        self.conversation_state = {
            "messages": [],
            "current_round": 0,
            "total_iterations": 0,
            "status": "active",
            "accumulated_findings": [],
            "orchestrator_feedback": [],
            "memory_context": [],
            "llm_call_count": 0
        }
        
        # Thread safety for environment access
        self._execution_lock = threading.Lock()
        
        # Rate limiting to prevent API overload
        self.last_tool_call_time = 0
        self.min_tool_call_interval = 1.0  
        
        logger.info(f"Worker {agent_id} initialized with objective: {objective[:100]}...")

    def _execute_tool_with_retry(self, tool_call, max_retries=2):
        """Execute tool with retry mechanism for transient failures and rate limiting"""
        if not self.agent_env:
            raise Exception(f"Agent {self.agent_id} has no environment to execute tools")
        
        tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
        last_error = None
        
        # Rate limiting
        time_since_last = time.time() - self.last_tool_call_time
        if time_since_last < self.min_tool_call_interval:
            sleep_time = self.min_tool_call_interval - time_since_last
            logger.debug(f"Agent {self.agent_id}: Rate limiting, sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        for attempt in range(max_retries):
            try:
                # Use thread lock to prevent concurrent environment access
                with self._execution_lock:
                    self.last_tool_call_time = time.time()
                    result = self.agent_env.execute_tool(tool_call)
                    if attempt > 0:
                        logger.info(f"Agent {self.agent_id}: Tool {tool_name} succeeded on attempt {attempt + 1}")
                    return result
                    
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if error is retryable
                retryable_errors = [
                    "timeout",
                    "connection",
                    "rate limit",
                    "temporarily",
                    "try again"
                ]
                
                is_retryable = any(err in error_str for err in retryable_errors)
                
                if is_retryable and attempt < max_retries - 1:
                    logger.warning(f"Agent {self.agent_id}: Tool {tool_name} failed (attempt {attempt + 1}/{max_retries}), retrying: {str(e)}")
                    backoff_time = (2 ** attempt) + (0.1 * attempt)  
                    if "excessive requests" in error_str or "rate limit" in error_str:
                        backoff_time *= 2  
                    time.sleep(backoff_time)
                else:
                    logger.warning(f"Agent {self.agent_id}: Tool {tool_name} failed after {attempt + 1} attempts: {str(e)}")
                    break
        
        raise last_error

    def _invoke_llm_with_recovery(self, messages):
        """Simplified LLM invocation with automatic recovery"""
        self.conversation_state["llm_call_count"] += 1  # Track LLM calls
        try:
            if self.agent_llm_w_tools:
                return self.agent_llm_w_tools.invoke(
                    messages, 
                    num_retries=2,
                    **self.llm_params_dict
                )
            else:
                return self.llm.invoke(messages[-1]["content"])
                
        except Exception as e:
            # Check for conversation state errors
            if self._is_conversation_error(e):
                logger.warning(f"Agent {self.agent_id}: Conversation error detected, using clean state")
                
                # Use minimal clean state for recovery
                clean_messages = self._get_minimal_context(messages)
                
                if self.agent_llm_w_tools:
                    return self.agent_llm_w_tools.invoke(
                        clean_messages, 
                        num_retries=1,
                        **self.llm_params_dict
                    )
                else:
                    return self.llm.invoke(clean_messages[-1]["content"])
            
            raise e
    
    def _is_conversation_error(self, error: Exception) -> bool:
        """Check if error is related to conversation state"""
        error_msg = str(error).lower()
        error_patterns = [
            "function response parts",
            "function call parts",
            "missing corresponding tool call",
            "tool response message"
        ]
        return any(pattern in error_msg for pattern in error_patterns)
    
    def _get_minimal_context(self, messages: list) -> list:
        """Get minimal valid context for recovery"""
        minimal = []
        
        for msg in messages[:3]:
            if msg.get("role") in ["system", "user"]:
                minimal.append(msg)
        
        # Add most recent user message if different
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg not in minimal:
                minimal.append(msg)
                break
        
        return minimal if minimal else [{"role": "user", "content": self.objective}]


    def execute_task(self, orchestrator_feedback: str = None):
        """Execute the task with orchestrator feedback and proper conversation state management"""
        new_findings = []
        iterations_made = 0

        logger.info(f"Agent {self.agent_id} starting process")
        logger.info(f"Agent {self.agent_id} objective: {self.objective}")

        # Initialize messages with orchestrator feedback
        messages = self._init_messages(orchestrator_feedback)

        # Continue from previous conversation state if exists and valid
        if self.conversation_state["messages"] and self.conversation_state["current_round"] > 1:
            logger.info(f"Agent {self.agent_id} continuing from round {self.conversation_state['current_round']} with {len(self.conversation_state['messages'])} messages")
            # Keep only essential context to prevent bloat
            validated_messages = self._validate_conversation_state(self.conversation_state["messages"])
            # Take last 5 messages for context continuity
            if len(validated_messages) > 5:
                validated_messages = validated_messages[-5:]
            messages.extend(validated_messages)

        # Process task through multiple iterations with conversation persistence
        for iteration in range(self.max_iterations):
            try:
                iterations_made += 1
                logger.info(f"Agent {self.agent_id} iteration {iteration}/{self.max_iterations}")

                if self.agent_llm_w_tools:
                    logger.debug(f"Agent {self.agent_id}: Using LLM with tools: {type(self.agent_llm_w_tools)}")
                    if hasattr(self.agent_llm_w_tools, 'tools'):
                        logger.debug(f"Agent {self.agent_id}: Available tools: {len(getattr(self.agent_llm_w_tools, 'tools', []))}")
                    elif hasattr(self.agent_llm_w_tools, 'bound_tools'):
                        logger.debug(f"Agent {self.agent_id}: Available tools: {len(getattr(self.agent_llm_w_tools, 'bound_tools', []))}")
                    else:
                        logger.debug(f"Agent {self.agent_id}: No tools attribute found on LLM")
                else:
                    logger.debug(f"Agent {self.agent_id}: No agent_llm_w_tools available")

                # Invoke LLM with tools using retry logic and conversation recovery
                response = self._invoke_llm_with_recovery(messages)
                # Sync LLM call count to conversation object
                self.conversation.llm_call_count = self.conversation_state["llm_call_count"]

                logger.debug(f"Agent {self.agent_id}: LLM response type: {type(response)}")
                logger.debug(f"Agent {self.agent_id}: Has tool_calls attr: {hasattr(response, 'tool_calls')}")
                if hasattr(response, "tool_calls"):
                    logger.debug(f"Agent {self.agent_id}: tool_calls value: {response.tool_calls}")
                if hasattr(response, "content"):
                    logger.debug(f"Agent {self.agent_id}: response content: {response.content[:200]}...")

                # Limit to single tool call for consistency and VertexAI compatibility
                if hasattr(response, "tool_calls") and response.tool_calls:
                    response.tool_calls = [response.tool_calls[0]]

                # Add LLM response to messages and update conversation state
                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)
                
                # Update conversation state with proper memory management
                self._update_conversation_state(response_msg)
                
                # Process tool calls if any are present
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call.get("name", "")
                    
                    try:
                        # Execute tool with automatic retry for transient failures
                        tool_resp = self._execute_tool_with_retry(tool_call, max_retries=2)
                        
                        # Add tool response to messages and state
                        tool_msg = convert_to_openai_messages(tool_resp)
                        messages.append(tool_msg)
                        
                        finding = {
                            "content": str(tool_resp.content),
                            "tool": tool_name,
                            "agent_id": self.agent_id,
                            "iteration": iteration,
                            "round": self.conversation_state["current_round"],
                            "timestamp": time.time(),
                            "status": "success"
                        }
                        new_findings.append(finding)
                        
                        # Add to memory immediately for real-time context sharing
                        self.memory.add_findings(self.agent_id, [finding])
                        
                        # Update conversation state for continuity
                        self.conversation_state["accumulated_findings"].append(finding)
                        
                        # Check if agent decided to finish with 'done' tool
                        if tool_name == "done":
                            logger.info(f"Agent {self.agent_id} decided to finish with 'done' tool")
                            break
                            
                    except Exception as e:
                        error_finding = {
                            "content": f"Tool {tool_name} failed: {str(e)}",
                            "tool": tool_name,
                            "agent_id": self.agent_id,
                            "iteration": iteration,
                            "round": self.conversation_state["current_round"],
                            "timestamp": time.time(),
                            "status": "error"
                        }
                        new_findings.append(error_finding)
                        self.memory.add_findings(self.agent_id, [error_finding])
                        self.conversation_state["accumulated_findings"].append(error_finding)
                        
                        error_msg = {
                            "role": "user",
                            "content": f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call."
                        }
                        messages.append(error_msg)
                        logger.warning(f"Tool **{tool_name}** failed with error: {str(e)}")
                        
                else:
                    # Capture reasoning or analysis as finding
                    if hasattr(response, 'content') and response.content:
                        reasoning_finding = {
                            "content": str(response.content),
                            "tool": "reasoning",
                            "agent_id": self.agent_id,
                            "iteration": iteration,
                            "round": self.conversation_state["current_round"],
                            "timestamp": time.time(),
                            "status": "analysis"
                        }
                        new_findings.append(reasoning_finding)
                        self.memory.add_findings(self.agent_id, [reasoning_finding])
                        self.conversation_state["accumulated_findings"].append(reasoning_finding)
                    
                    # Check for stuck behavior
                    no_tool_errors = sum(1 for msg in messages[-6:] 
                                        if isinstance(msg, dict) and 
                                        "ERROR: No tool calls found" in msg.get("content", ""))
                    
                    if no_tool_errors >= 2:
                        # Agent is stuck, force completion
                        logger.warning(f"Agent {self.agent_id}: Stuck in no-tool-call loop after {no_tool_errors} attempts, forcing completion")
                        finding = {
                            "content": f"Unable to proceed - {response.content if hasattr(response, 'content') else 'Task cannot be completed with available tools'}",
                            "tool": "forced_completion",
                            "agent_id": self.agent_id,
                            "iteration": iteration,
                            "round": self.conversation_state["current_round"],
                            "timestamp": time.time(),
                            "status": "incomplete"
                        }
                        new_findings.append(finding)
                        self.memory.add_findings(self.agent_id, [finding])
                        break
                    else:
                        error_msg = {
                            "role": "user",
                            "content": "ERROR: No tool calls found. Please use the tools to solve the task, or use 'impossible' tool if the task cannot be completed."
                        }
                        messages.append(error_msg)
                        logger.warning(f"Agent {self.agent_id}: No tool calls found in iteration {iteration}")

                self.conversation.add_turn("agent", str(response.content), iteration)

            except Exception as e:
                logger.error(f"Agent {self.agent_id} iteration {iteration} failed: {e}")
                self.conversation.add_turn("agent", f"Error in iteration {iteration}: {str(e)}", iteration)
                
                # Simple error recovery: reset if conversation error
                if self._is_conversation_error(e):
                    logger.warning(f"Agent {self.agent_id}: Resetting conversation due to error")
                    self._reset_conversation_state()
                    messages = self._init_messages(orchestrator_feedback)
                    continue

        # Update agent's conversation and return results
        self.conversation.findings.extend(new_findings)
        self.conversation.total_iterations = iterations_made
        logger.info(f"Agent {self.agent_id} completed with {len(new_findings)} findings")

        # Return comprehensive response structure
        if new_findings:
            best_finding = max(new_findings, key=lambda f: len(f.get("content", "")))
            main_response = best_finding.get("content", "No substantial response generated")
        else:
            main_response = "No findings generated during processing"
        
        return {
            "main_response": main_response,
            "findings": new_findings,
            "iterations": iterations_made,
            "status": "completed" if iterations_made > 0 else "no_progress",
            "conversation_state": self.conversation_state
        }
    
    def _init_messages(self, orchestrator_feedback: str = None):
        """Initialize conversation with memory context and orchestrator guidance"""
        messages = []
        
        if self.prompts and 'subagent' in self.prompts and self.shared_prompt_templates:
            try:
                # Use the subagent prompt template properly
                subagent_prompt = self.prompts["subagent"]
                compiled_messages = subagent_prompt.compile(**self.shared_prompt_templates)
                messages = list(compiled_messages)
                logger.info(f"Agent {self.agent_id}: prompt compilation successful with {len(messages)} messages")
                
                # Add memory context for continuity and team awareness
                memory_context = self._get_memory_context()
                if memory_context:
                    messages.append({
                        "role": "system",
                        "content": f"Memory Context: {memory_context}"
                    })
                    logger.info(f"Agent {self.agent_id}: Added memory context for continuity")
                
                # Add orchestrator feedback if provided as part of the task
                if orchestrator_feedback and len(orchestrator_feedback.strip()) > 10:
                    messages.append({
                        "role": "user",
                        "content": f"Based on team coordination, focus on: {orchestrator_feedback}\n\nSpecifically for your objective: {self.objective}"
                    })
                    logger.info(f"Agent {self.agent_id}: Added orchestrator guidance to LLM context")
                
            except Exception as e:
                logger.warning(f"Failed to compile subagent prompt: {e}, using fallback")
                messages = self._get_fallback_messages()
        else:
            logger.warning(f"Agent {self.agent_id}: Missing prompts or templates, using fallback")
            messages = self._get_fallback_messages()
            
        return messages
    
    def _get_memory_context(self) -> str:
        """Get relevant memory context for this agent to provide continuity across rounds"""
        try:
            # Get agent's own findings from shared memory
            agent_findings = self.memory.get_findings_for_agent(self.agent_id)
            
            # Get recent findings from other agents for collaboration awareness
            all_findings = self.memory.get_all_findings()
            team_findings = [f for f in all_findings if f.get("agent_id") != self.agent_id]
            
            context_parts = []
            
            # Add agent's own recent progress for continuity
            if agent_findings:
                recent_findings = agent_findings[-2:]  # Last 2 findings for context
                context_parts.append(f"Your recent findings: {len(recent_findings)} items")
                
                # Add brief content preview of most recent finding
                if recent_findings:
                    latest_content = recent_findings[-1].get("content", "")[:100]
                    if latest_content:
                        context_parts.append(f"Latest finding: {latest_content}...")
            
            # Add team context for collaborative decision making
            if team_findings:
                recent_team_findings = team_findings[-3:]  # Last 3 team findings
                context_parts.append(f"Team has {len(recent_team_findings)} recent findings")
                
                # Add brief team progress summary
                if recent_team_findings:
                    team_summary = "; ".join([f"Agent {f.get('agent_id', 'unknown')}: {f.get('content', '')[:50]}..." 
                                            for f in recent_team_findings if f.get('content')])
                    if team_summary:
                        context_parts.append(f"Team progress: {team_summary}")
            
            # Add strategic focus context from the original plan
            if hasattr(self.memory, 'execution_plan') and self.memory.execution_plan:
                plan = self.memory.execution_plan
                if 'subtasks' in plan:
                    agent_subtask = next((s for s in plan['subtasks'] if s.get('agent_id') == self.agent_id), None)
                    if agent_subtask:
                        context_parts.append(f"Your strategic focus: {agent_subtask.get('focus', 'general')}")
                        
                        # Add objective reminder for context
                        objective = agent_subtask.get('objective', self.objective)
                        if objective and len(objective) < 100:
                            context_parts.append(f"Current objective: {objective}")
            
            # Combine all context parts into coherent summary
            if context_parts:
                return "; ".join(context_parts)
            else:
                return "Starting fresh work on this objective"
                
        except Exception as e:
            logger.warning(f"Error getting memory context: {e}")
            return "Starting fresh work on this objective"
    
    def _update_conversation_state(self, message):
        """Update conversation state while preserving important context across rounds"""
        try:
            # Maintain rolling window of messages
            self.conversation_state["messages"].append(message)
            
            # Preserve orchestrator feedback and key findings
            if len(self.conversation_state["messages"]) > 15:
                preserved_messages = []
                
                # Keep initial system/user messages for context
                for msg in self.conversation_state["messages"][:2]:
                    preserved_messages.append(msg)
                
                # Keep orchestrator feedback messages
                for msg in self.conversation_state["messages"]:
                    if msg.get("role") == "user" and "ORCHESTRATOR" in msg.get("content", ""):
                        if msg not in preserved_messages:
                            preserved_messages.append(msg)
                
                preserved_messages.extend(self.conversation_state["messages"][-8:])
                
                seen = set()
                unique_messages = []
                for msg in preserved_messages:
                    msg_key = (msg.get("role", ""), msg.get("content", "")[:100])
                    if msg_key not in seen:
                        seen.add(msg_key)
                        unique_messages.append(msg)
                
                self.conversation_state["messages"] = unique_messages[-15:]  
                        
        except Exception as e:
            logger.warning(f"Agent {self.agent_id}: Error updating state: {e}")

    def _validate_conversation_state(self, messages: List[Dict]) -> List[Dict]:
        """Validate and clean conversation state to prevent VertexAI errors"""
        try:
            tool_calls_map = {}  
            tool_responses_map = {}  
            
            for i, msg in enumerate(messages):
                msg_role = msg.get("role", "")
                
                if msg_role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            tool_calls_map[tool_call_id] = i
                
                elif msg_role == "tool":
                    tool_call_id = msg.get("tool_call_id")
                    if tool_call_id:
                        tool_responses_map[tool_call_id] = i
            
            # Only include valid message pairs and standalone messages
            validated = []
            skip_indices = set()
            
            for i, msg in enumerate(messages):
                if i in skip_indices:
                    continue
                    
                msg_role = msg.get("role", "")
                
                if msg_role == "assistant":
                    if "tool_calls" in msg and msg["tool_calls"]:
                        all_tool_calls_matched = True
                        for tool_call in msg["tool_calls"]:
                            tool_call_id = tool_call.get("id")
                            if tool_call_id not in tool_responses_map:
                                all_tool_calls_matched = False
                                break
                        
                        if all_tool_calls_matched:
                            validated.append(msg)
                        else:
                            logger.warning(f"Agent {self.agent_id}: Skipping assistant message with unmatched tool calls")
                            for tool_call in msg["tool_calls"]:
                                tool_call_id = tool_call.get("id")
                                if tool_call_id in tool_responses_map:
                                    skip_indices.add(tool_responses_map[tool_call_id])
                    else:
                        validated.append(msg)
                
                elif msg_role == "tool":
                    tool_call_id = msg.get("tool_call_id")
                    if tool_call_id in tool_calls_map:
                        assistant_index = tool_calls_map[tool_call_id]
                        if assistant_index < len(messages):
                            assistant_msg = messages[assistant_index]
                            all_matched = True
                            if "tool_calls" in assistant_msg and assistant_msg["tool_calls"]:
                                for tc in assistant_msg["tool_calls"]:
                                    if tc.get("id") not in tool_responses_map:
                                        all_matched = False
                                        break
                            
                            if all_matched:
                                validated.append(msg)
                            else:
                                logger.warning(f"Agent {self.agent_id}: Skipping orphaned tool response")
                        else:
                            logger.warning(f"Agent {self.agent_id}: Skipping tool response with invalid assistant reference")
                    else:
                        logger.warning(f"Agent {self.agent_id}: Skipping tool response without matching assistant message")
                
                elif msg_role in ["user", "system"]:
                    validated.append(msg)
                
            logger.info(f"Agent {self.agent_id}: Validated {len(validated)}/{len(messages)} conversation messages")
            return validated
            
        except Exception as e:
            logger.warning(f"Agent {self.agent_id}: Error validating conversation state: {e}")
            safe_messages = []
            for msg in messages:
                if msg.get("role") in ["user", "system"]:
                    safe_messages.append(msg)
                elif msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    safe_messages.append(msg)
            return safe_messages

    def _get_fallback_messages(self):
        """Fallback messages when prompt compilation fails"""
        return [
            {
                "role": "system",
                "content": f"You are an agent. Your objective: {self.objective}"
            },
            {
                "role": "user", 
                "content": f"Work on: {self.objective}"
            }
        ]

    def process_orchestrator_request(self, feedback: str) -> Dict[str, Any]:
        """Process orchestrator's request and execute task"""
        try:
            self.conversation_state["current_round"] += 1
            
            if not feedback or not isinstance(feedback, str):
                logger.warning(f"Agent {self.agent_id}: Invalid orchestrator feedback: {type(feedback)}")
                feedback = f"Work on: {self.objective}"
            
            # Store orchestrator feedback in conversation state for context preservation
            self.conversation_state["orchestrator_feedback"].append({
                "round": self.conversation_state["current_round"],
                "feedback": feedback,
                "timestamp": time.time()
            })
            
            # Keep only recent feedback to prevent bloat
            if len(self.conversation_state["orchestrator_feedback"]) > 5:
                self.conversation_state["orchestrator_feedback"] = self.conversation_state["orchestrator_feedback"][-5:]
            
            # Add orchestrator feedback to conversation history
            self.conversation.add_turn("orchestrator", feedback)
            logger.info(f"Agent {self.agent_id} ({self.strategy}) processing feedback for round {self.conversation_state['current_round']}")

            # Execute task with the feedback
            result = self.execute_task(feedback)

            # Extract main response for conversation tracking
            main_response = result.get("main_response", "No response generated")
            self.conversation.add_turn("worker", main_response, self.conversation.total_iterations) 

            # Update agent status based on completion criteria
            if self.conversation.total_iterations >= self.min_iterations:
                self.conversation.status = "completed"
            elif self.is_rate_limited():
                self.conversation.status = "rate_limited"

            return {
                "agent_id": self.agent_id,
                "response": main_response,
                "new_findings": result.get("findings", []),
                "iterations": result.get("iterations", 0),
                "total_iterations": self.conversation.total_iterations,
                "status": self.conversation.status,
                "conversation_state": self.conversation_state,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error processing orchestrator feedback: {e}", exc_info=True)
            return {
                "agent_id": self.agent_id,
                "response": f"Error processing feedback: {str(e)}",
                "new_findings": [],
                "iterations": 0,
                "total_iterations": self.conversation.total_iterations,
                "status": "error",
                "main_response": f"Error: {str(e)}",
                "conversation_state": self.conversation_state,  
                "error": str(e)
            }

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        return {
            "agent_id": self.agent_id,
            "objective": self.objective,
            "strategy": self.strategy,
            "turns": len(self.conversation.turns) // 2,
            "total_iterations": self.conversation.total_iterations,
            "findings_count": len(self.conversation.findings),
            "status": self.conversation.status,
            "rate_limit_count": getattr(self.agent_env, 'rate_limit_count', 0),
            "conversation_rounds": self.conversation_state.get("current_round", 0),
            "accumulated_findings_count": len(self.conversation_state.get("accumulated_findings", [])),
            "memory_context_count": len(self.conversation_state.get("memory_context", [])),
            "orchestrator_feedback_count": len(self.conversation_state.get("orchestrator_feedback", []))
        }


    def is_rate_limited(self) -> bool:
        """Check if agent is rate limited"""
        if hasattr(self.agent_env, 'stop_rate_limited'):
            return self.agent_env.stop_rate_limited()
        return False

    def _reset_conversation_state(self):
        """Simple reset preserving only essential state"""
        logger.info(f"Agent {self.agent_id}: Resetting conversation state")
        
        preserved = {
            "current_round": self.conversation_state.get("current_round", 0),
            "total_iterations": self.conversation_state.get("total_iterations", 0),
            "accumulated_findings": self.conversation_state.get("accumulated_findings", [])[-10:],  # Keep recent
        }
        
        self.conversation_state = {
            "messages": [],
            "current_round": preserved["current_round"],
            "total_iterations": preserved["total_iterations"],
            "status": "active",
            "accumulated_findings": preserved["accumulated_findings"],
            "orchestrator_feedback": [],
            "memory_context": []
        }