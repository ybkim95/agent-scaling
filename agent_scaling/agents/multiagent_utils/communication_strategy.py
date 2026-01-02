"""
Communication Strategy Interface for Multi-Agent Systems
Enables easy switching between different communication patterns for ablation studies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class Message:
    """Standard message format for agent communication"""
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    content: Any
    message_type: str  # 'finding', 'query', 'response', 'coordination'
    timestamp: float
    metadata: Dict[str, Any] = None


class CommunicationStrategy(ABC):
    """Abstract base class for different communication strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_history: List[Message] = []
    
    @abstractmethod
    def send_message(self, sender_id: str, message: Any, recipient_id: Optional[str] = None) -> None:
        """Send a message from one agent"""
        pass
    
    @abstractmethod
    def receive_messages(self, agent_id: str) -> List[Message]:
        """Receive messages for a specific agent"""
        pass
    
    @abstractmethod
    def share_finding(self, agent_id: str, finding: Dict[str, Any]) -> None:
        """Share a finding with other agents"""
        pass
    
    @abstractmethod
    def get_shared_context(self, agent_id: str) -> Dict[str, Any]:
        """Get the shared context visible to an agent"""
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronization point for round-based strategies"""
        pass


class BlackboardStrategy(CommunicationStrategy):
    """Shared blackboard that all agents can read/write"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.blackboard = {
            "findings": [],
            "agent_states": {},
            "shared_knowledge": {}
        }
        self.read_only_mode = config.get("read_only_workers", False)
    
    def send_message(self, sender_id: str, message: Any, recipient_id: Optional[str] = None) -> None:
        """Write message to blackboard"""
        if self.read_only_mode and sender_id != "orchestrator":
            return  # Workers can't write in read-only mode
        
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=message,
            message_type="blackboard_update",
            timestamp=time.time()
        )
        self.message_history.append(msg)
        self.blackboard["shared_knowledge"][sender_id] = message
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """Read relevant messages from blackboard"""
        return [msg for msg in self.message_history 
                if msg.recipient_id is None or msg.recipient_id == agent_id]
    
    def share_finding(self, agent_id: str, finding: Dict[str, Any]) -> None:
        """Add finding to shared blackboard"""
        if not self.read_only_mode or agent_id == "orchestrator":
            finding["shared_by"] = agent_id
            finding["shared_at"] = time.time()
            self.blackboard["findings"].append(finding)
    
    def get_shared_context(self, agent_id: str) -> Dict[str, Any]:
        """All agents see the entire blackboard"""
        return self.blackboard.copy()
    
    def synchronize(self) -> None:
        """No explicit synchronization needed for blackboard"""
        pass


class BroadcastStrategy(CommunicationStrategy):
    """Every agent broadcasts to all others"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_inboxes: Dict[str, List[Message]] = {}
        self.broadcast_delay = config.get("broadcast_delay", 0.0)
    
    def send_message(self, sender_id: str, message: Any, recipient_id: Optional[str] = None) -> None:
        """Broadcast message to all agents"""
        msg = Message(
            sender_id=sender_id,
            recipient_id=None,  # Broadcast to all
            content=message,
            message_type="broadcast",
            timestamp=time.time()
        )
        
        # Add to all agent inboxes except sender
        for agent_id in self.agent_inboxes:
            if agent_id != sender_id:
                self.agent_inboxes[agent_id].append(msg)
        
        self.message_history.append(msg)
        
        # Optional delay to simulate network latency
        if self.broadcast_delay > 0:
            time.sleep(self.broadcast_delay)
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """Get all messages for this agent"""
        if agent_id not in self.agent_inboxes:
            self.agent_inboxes[agent_id] = []
        
        messages = self.agent_inboxes[agent_id].copy()
        self.agent_inboxes[agent_id] = []  # Clear inbox after reading
        return messages
    
    def share_finding(self, agent_id: str, finding: Dict[str, Any]) -> None:
        """Broadcast finding to all agents"""
        self.send_message(agent_id, {"type": "finding", "data": finding})
    
    def get_shared_context(self, agent_id: str) -> Dict[str, Any]:
        """Get messages and findings visible to this agent"""
        return {
            "messages": self.receive_messages(agent_id),
            "broadcast_history": self.message_history[-10:]  # Last 10 broadcasts
        }
    
    def synchronize(self) -> None:
        """No synchronization in pure broadcast"""
        pass


class PipelineStrategy(CommunicationStrategy):
    """Sequential pipeline where each agent passes results to the next"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipeline_order: List[str] = config.get("pipeline_order", [])
        self.current_stage = 0
        self.stage_outputs: Dict[str, Any] = {}
    
    def send_message(self, sender_id: str, message: Any, recipient_id: Optional[str] = None) -> None:
        """Pass message to next agent in pipeline"""
        current_idx = self.pipeline_order.index(sender_id)
        if current_idx < len(self.pipeline_order) - 1:
            next_agent = self.pipeline_order[current_idx + 1]
            msg = Message(
                sender_id=sender_id,
                recipient_id=next_agent,
                content=message,
                message_type="pipeline_handoff",
                timestamp=time.time()
            )
            self.message_history.append(msg)
            self.stage_outputs[sender_id] = message
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """Get messages from previous agent in pipeline"""
        agent_idx = self.pipeline_order.index(agent_id)
        if agent_idx > 0:
            prev_agent = self.pipeline_order[agent_idx - 1]
            if prev_agent in self.stage_outputs:
                return [msg for msg in self.message_history 
                       if msg.recipient_id == agent_id]
        return []
    
    def share_finding(self, agent_id: str, finding: Dict[str, Any]) -> None:
        """Add finding to stage output"""
        if agent_id not in self.stage_outputs:
            self.stage_outputs[agent_id] = {"findings": []}
        self.stage_outputs[agent_id]["findings"].append(finding)
    
    def get_shared_context(self, agent_id: str) -> Dict[str, Any]:
        """Get outputs from all previous stages"""
        agent_idx = self.pipeline_order.index(agent_id)
        context = {}
        for i in range(agent_idx):
            prev_agent = self.pipeline_order[i]
            if prev_agent in self.stage_outputs:
                context[prev_agent] = self.stage_outputs[prev_agent]
        return context
    
    def synchronize(self) -> None:
        """Move to next stage in pipeline"""
        self.current_stage = (self.current_stage + 1) % len(self.pipeline_order)


class ConsensusStrategy(CommunicationStrategy):
    """Agents vote on findings before they become shared knowledge"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.consensus_threshold = config.get("consensus_threshold", 0.66)
        self.pending_findings: List[Dict[str, Any]] = []
        self.votes: Dict[int, Dict[str, bool]] = {}  # finding_id -> agent_id -> vote
        self.approved_findings: List[Dict[str, Any]] = []
    
    def send_message(self, sender_id: str, message: Any, recipient_id: Optional[str] = None) -> None:
        """Send voting request or vote"""
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=message,
            message_type="consensus_message",
            timestamp=time.time()
        )
        self.message_history.append(msg)
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """Get voting requests and results"""
        return [msg for msg in self.message_history 
                if msg.recipient_id is None or msg.recipient_id == agent_id]
    
    def share_finding(self, agent_id: str, finding: Dict[str, Any]) -> None:
        """Submit finding for consensus voting"""
        finding_id = len(self.pending_findings)
        finding["id"] = finding_id
        finding["proposed_by"] = agent_id
        self.pending_findings.append(finding)
        self.votes[finding_id] = {agent_id: True}  # Proposer auto-votes yes
    
    def vote_on_finding(self, agent_id: str, finding_id: int, vote: bool) -> None:
        """Cast vote on a pending finding"""
        if finding_id in self.votes:
            self.votes[finding_id][agent_id] = vote
    
    def get_shared_context(self, agent_id: str) -> Dict[str, Any]:
        """Get approved findings and pending votes"""
        return {
            "approved_findings": self.approved_findings,
            "pending_votes": [f for f in self.pending_findings 
                            if f["id"] not in self.votes.get(f["id"], {}).get(agent_id, False)]
        }
    
    def synchronize(self) -> None:
        """Tally votes and approve findings that meet threshold"""
        for finding in self.pending_findings:
            finding_id = finding["id"]
            if finding_id in self.votes:
                total_votes = len(self.votes[finding_id])
                yes_votes = sum(1 for v in self.votes[finding_id].values() if v)
                if yes_votes / total_votes >= self.consensus_threshold:
                    self.approved_findings.append(finding)
        
        # Clear processed findings
        self.pending_findings = [f for f in self.pending_findings 
                                if f not in self.approved_findings]


# Factory function to create strategies
def create_communication_strategy(strategy_type: str, config: Dict[str, Any]) -> CommunicationStrategy:
    """Factory to create communication strategies based on config"""
    
    strategies = {
        "blackboard": BlackboardStrategy,
        "broadcast": BroadcastStrategy,
        "pipeline": PipelineStrategy,
        "consensus": ConsensusStrategy,
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown communication strategy: {strategy_type}")
    
    return strategies[strategy_type](config)