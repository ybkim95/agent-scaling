from datetime import datetime
from typing import Any, Dict, List, Optional


class ConversationStateManager:
    """Manages conversation state with message compression and rotation"""
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
        self.compressed_history: List[str] = []
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message and manage history size"""
        self.messages.append(message)
        
        if len(self.messages) >= self.max_messages:
            self._compress_history()
    
    def _compress_history(self):
        """Compress old messages into summaries to manage memory"""
        # Keep the most recent half of messages
        keep_count = self.max_messages // 2
        messages_to_compress = self.messages[:-keep_count]
        
        # Create a summary of the compressed messages
        if messages_to_compress:
            summary = f"[Compressed {len(messages_to_compress)} messages from conversation history]"
            self.compressed_history.append(summary)
            
        # Keep only recent messages
        self.messages = self.messages[-keep_count:]
    
    def get_full_context(self) -> List[Any]:
        """Get full conversation context including compressed history"""
        context = []
        if self.compressed_history:
            context.extend([{"role": "system", "content": "\n".join(self.compressed_history)}])
        context.extend(self.messages)
        return context


class WorkerConversation:
    """Tracks multi-turn conversation between orchestrator and worker"""

    def __init__(self, agent_id: str, max_messages: int = 50):
        self.agent_id = agent_id
        self.turns = []
        self.status = "active"
        self.total_iterations = 0  # Tool-agnostic variable name
        self.findings = []
        self.llm_call_count = 0  # Track LLM calls for metrics
        
        # Enhanced conversation state management
        self.state_manager = ConversationStateManager(max_messages)

    def add_turn(self, role: str, message: str, iterations: int = 0):
        """Add a conversation turn with enhanced state management"""
        turn_data = {
            "turn_number": len(self.turns),
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "iterations": iterations,  # Tool-agnostic variable name
        }
        
        self.turns.append(turn_data)
        self.state_manager.add_message(turn_data)
        
        if role == "worker":
            self.total_iterations += iterations  # Tool-agnostic variable name
    
    def get_conversation_context(self) -> List[Any]:
        """Get conversation context with compression management"""
        return self.state_manager.get_full_context()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation state"""
        return {
            "agent_id": self.agent_id,
            "total_turns": len(self.turns),
            "total_iterations": self.total_iterations,
            "findings_count": len(self.findings),
            "status": self.status,
            "compressed_messages": len(self.state_manager.compressed_history),
            "active_messages": len(self.state_manager.messages),
        }
