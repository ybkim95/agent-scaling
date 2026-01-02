"""
Enhanced Metrics Collection for Multi-Agent Systems
Collects comprehensive metrics for analysis and paper writing
"""

import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class CommunicationMetrics:
    """Track all agent communication"""
    message_id: str
    timestamp: float
    sender: str
    recipients: List[str]  # Can be multiple for broadcast
    message_type: str  # orchestrator_guidance, peer_message, finding_share, vote
    content_length: int
    round: int
    iteration: int
    latency_ms: Optional[float] = None  # Time from send to receive


@dataclass
class LLMMetrics:
    """Track LLM usage and costs"""
    agent_id: str
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float  # Based on model pricing
    latency_ms: float
    success: bool
    rate_limited: bool = False
    retry_count: int = 0


@dataclass
class ToolMetrics:
    """Track tool usage patterns"""
    agent_id: str
    tool_name: str
    timestamp: float
    arguments: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    execution_time_ms: float
    round: int
    iteration: int


@dataclass
class AgentMetrics:
    """Per-agent performance metrics"""
    agent_id: str
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    total_findings: int = 0
    unique_findings: int = 0
    execution_time_s: float = 0.0
    idle_time_s: float = 0.0
    communication_sent: int = 0
    communication_received: int = 0


@dataclass
class SystemMetrics:
    """Overall system performance"""
    architecture: str  # centralized, decentralized, hybrid
    total_agents: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Execution metrics
    total_execution_time_s: float = 0.0
    parallel_efficiency: float = 0.0  # Ratio of parallel to sequential time
    
    # Communication metrics
    total_messages: int = 0
    avg_message_latency_ms: float = 0.0
    communication_overhead_percent: float = 0.0
    
    # LLM metrics
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    
    # Quality metrics
    task_success: bool = False
    accuracy: float = 0.0
    duplicate_work_ratio: float = 0.0
    consensus_rate: float = 0.0  # For voting-based systems
    
    # Error handling
    total_errors: int = 0
    error_recovery_success_rate: float = 0.0


class MetricsCollector:
    """Centralized metrics collection for multi-agent systems"""
    
    # Model pricing (per 1K tokens) - UPDATE WITH ACTUAL PRICING
    MODEL_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "gemini-pro": {"input": 0.00025, "output": 0.00125},
        "gemini-2.0-flash": {"input": 0.00025, "output": 0.00125},
    }
    
    def __init__(self, architecture: str, num_agents: int):
        self.system_metrics = SystemMetrics(
            architecture=architecture,
            total_agents=num_agents
        )
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.communication_log: List[CommunicationMetrics] = []
        self.llm_log: List[LLMMetrics] = []
        self.tool_log: List[ToolMetrics] = []
        
    def log_communication(
        self,
        sender: str,
        recipients: List[str],
        message_type: str,
        content: str,
        round: int,
        iteration: int,
        latency_ms: Optional[float] = None
    ):
        """Log agent communication"""
        metric = CommunicationMetrics(
            message_id=f"{sender}_{time.time()}",
            timestamp=time.time(),
            sender=sender,
            recipients=recipients,
            message_type=message_type,
            content_length=len(content),
            round=round,
            iteration=iteration,
            latency_ms=latency_ms
        )
        self.communication_log.append(metric)
        
        # Update agent metrics
        if sender not in self.agent_metrics:
            self.agent_metrics[sender] = AgentMetrics(agent_id=sender)
        self.agent_metrics[sender].communication_sent += 1
        
        for recipient in recipients:
            if recipient not in self.agent_metrics:
                self.agent_metrics[recipient] = AgentMetrics(agent_id=recipient)
            self.agent_metrics[recipient].communication_received += 1
    
    def log_llm_call(
        self,
        agent_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        rate_limited: bool = False,
        retry_count: int = 0
    ):
        """Log LLM API call"""
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        model_key = model.split("/")[-1]  # Handle provider/model format
        if model_key in self.MODEL_PRICING:
            pricing = self.MODEL_PRICING[model_key]
            cost = (input_tokens * pricing["input"] + 
                   output_tokens * pricing["output"]) / 1000
        else:
            cost = 0.0  # Unknown model pricing
        
        metric = LLMMetrics(
            agent_id=agent_id,
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            latency_ms=latency_ms,
            success=success,
            rate_limited=rate_limited,
            retry_count=retry_count
        )
        self.llm_log.append(metric)
        
        # Update agent metrics
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        
        agent = self.agent_metrics[agent_id]
        agent.total_llm_calls += 1
        agent.total_tokens += total_tokens
        agent.total_cost += cost
        
        # Update system metrics
        self.system_metrics.total_llm_calls += 1
        self.system_metrics.total_tokens_used += total_tokens
        self.system_metrics.total_cost_usd += cost
    
    def log_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        execution_time_ms: float,
        round: int,
        iteration: int,
        error_message: Optional[str] = None
    ):
        """Log tool usage"""
        metric = ToolMetrics(
            agent_id=agent_id,
            tool_name=tool_name,
            timestamp=time.time(),
            arguments=arguments,
            success=success,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            round=round,
            iteration=iteration
        )
        self.tool_log.append(metric)
        
        # Update agent metrics
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        
        agent = self.agent_metrics[agent_id]
        agent.total_tool_calls += 1
        if success:
            agent.successful_tool_calls += 1
        else:
            self.system_metrics.total_errors += 1
    
    def calculate_final_metrics(self):
        """Calculate derived metrics at end of execution"""
        self.system_metrics.end_time = time.time()
        self.system_metrics.total_execution_time_s = (
            self.system_metrics.end_time - self.system_metrics.start_time
        )
        
        # Communication metrics
        if self.communication_log:
            self.system_metrics.total_messages = len(self.communication_log)
            latencies = [m.latency_ms for m in self.communication_log 
                        if m.latency_ms is not None]
            if latencies:
                self.system_metrics.avg_message_latency_ms = sum(latencies) / len(latencies)
        
        # Calculate communication overhead
        comm_time = sum(m.latency_ms or 0 for m in self.communication_log) / 1000
        if self.system_metrics.total_execution_time_s > 0:
            self.system_metrics.communication_overhead_percent = (
                comm_time / self.system_metrics.total_execution_time_s * 100
            )
        
        # Calculate duplicate work ratio
        all_tool_calls = {}
        for tool_metric in self.tool_log:
            key = f"{tool_metric.tool_name}_{tool_metric.arguments}"
            if key not in all_tool_calls:
                all_tool_calls[key] = []
            all_tool_calls[key].append(tool_metric.agent_id)
        
        duplicate_calls = sum(1 for agents in all_tool_calls.values() 
                             if len(set(agents)) > 1)
        if all_tool_calls:
            self.system_metrics.duplicate_work_ratio = (
                duplicate_calls / len(all_tool_calls)
            )
        
        # Error recovery rate
        if self.system_metrics.total_errors > 0:
            recovered = sum(1 for t in self.tool_log 
                          if not t.success and any(
                              t2.agent_id == t.agent_id and 
                              t2.tool_name == t.tool_name and 
                              t2.success and t2.timestamp > t.timestamp
                              for t2 in self.tool_log
                          ))
            self.system_metrics.error_recovery_success_rate = (
                recovered / self.system_metrics.total_errors
            )
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics as dictionary"""
        self.calculate_final_metrics()
        
        return {
            "system_metrics": asdict(self.system_metrics),
            "agent_metrics": {
                agent_id: asdict(metrics) 
                for agent_id, metrics in self.agent_metrics.items()
            },
            "communication_log": [asdict(m) for m in self.communication_log[-100:]],  # Last 100
            "llm_usage": {
                "total_calls": len(self.llm_log),
                "by_agent": {
                    agent_id: {
                        "calls": sum(1 for l in self.llm_log if l.agent_id == agent_id),
                        "tokens": sum(l.total_tokens for l in self.llm_log 
                                    if l.agent_id == agent_id),
                        "cost": sum(l.estimated_cost for l in self.llm_log 
                                  if l.agent_id == agent_id)
                    }
                    for agent_id in self.agent_metrics.keys()
                }
            },
            "tool_usage": {
                "total_calls": len(self.tool_log),
                "success_rate": (
                    sum(1 for t in self.tool_log if t.success) / len(self.tool_log)
                    if self.tool_log else 0
                ),
                "by_tool": {
                    tool: {
                        "calls": sum(1 for t in self.tool_log if t.tool_name == tool),
                        "success_rate": (
                            sum(1 for t in self.tool_log 
                              if t.tool_name == tool and t.success) /
                            sum(1 for t in self.tool_log if t.tool_name == tool)
                        )
                    }
                    for tool in set(t.tool_name for t in self.tool_log)
                }
            },
            "summary": {
                "architecture": self.system_metrics.architecture,
                "agents": self.system_metrics.total_agents,
                "execution_time_s": self.system_metrics.total_execution_time_s,
                "total_llm_calls": self.system_metrics.total_llm_calls,
                "total_tokens": self.system_metrics.total_tokens_used,
                "total_cost_usd": round(self.system_metrics.total_cost_usd, 4),
                "communication_overhead_percent": round(
                    self.system_metrics.communication_overhead_percent, 2
                ),
                "duplicate_work_ratio": round(
                    self.system_metrics.duplicate_work_ratio, 3
                ),
                "task_success": self.system_metrics.task_success
            }
        }