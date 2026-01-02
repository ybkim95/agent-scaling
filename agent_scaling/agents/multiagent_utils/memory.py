import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_scaling.logger import logger


class Memory:
    """Enhanced memory system for multi-agent coordination with thread safety"""
    
    def __init__(self):
        self.all_findings: List[Dict[str, Any]] = []
        self.agent_findings: Dict[str, List[Dict[str, Any]]] = {}
        self.execution_plan: Optional[Dict[str, Any]] = None
        self.original_query: str = ""
        self.conversation_history: List[Dict[str, Any]] = []
        
        # CRITICAL FIX: Enhanced memory for better multi-agent coordination
        self.agent_progress: Dict[str, Dict[str, Any]] = {}
        self.coordination_messages: List[Dict[str, Any]] = []
        self.round_summaries: List[Dict[str, Any]] = []
        self.team_context: Dict[str, Any] = {}
        
        # Thread safety for parallel agent access
        self._lock = threading.Lock()
        
    def add_findings(self, agent_id: str, findings: List[Dict[str, Any]]):
        """Thread-safe method to add findings from an agent"""
        with self._lock:
            if agent_id not in self.agent_findings:
                self.agent_findings[agent_id] = []
            
            # Deduplicate findings before adding
            unique_findings = self._deduplicate_findings(findings)
            
            self.agent_findings[agent_id].extend(unique_findings)
            self.all_findings.extend(unique_findings)
            
            # CRITICAL FIX: Update agent progress tracking
            self._update_agent_progress(agent_id, len(unique_findings))
            
            # CRITICAL FIX: Update team context with new findings
            self._update_team_context(agent_id, unique_findings)
    
    def _update_agent_progress(self, agent_id: str, new_findings_count: int):
        """Update agent progress tracking in memory"""
        if agent_id not in self.agent_progress:
            self.agent_progress[agent_id] = {
                "total_findings": 0,
                "last_update": datetime.now().isoformat(),
                "findings_history": []
            }
        
        self.agent_progress[agent_id]["total_findings"] += new_findings_count
        self.agent_progress[agent_id]["last_update"] = datetime.now().isoformat()
        
        # Add to findings history
        self.agent_progress[agent_id]["findings_history"].append({
            "timestamp": datetime.now().isoformat(),
            "count": new_findings_count
        })
        
        # Keep only last 10 updates
        if len(self.agent_progress[agent_id]["findings_history"]) > 10:
            self.agent_progress[agent_id]["findings_history"] = self.agent_progress[agent_id]["findings_history"][-10:]
    
    def _update_team_context(self, agent_id: str, new_findings: List[Dict[str, Any]]):
        """Update team context with new findings for better coordination"""
        if "team_findings" not in self.team_context:
            self.team_context["team_findings"] = {}
        
        if agent_id not in self.team_context["team_findings"]:
            self.team_context["team_findings"][agent_id] = []
        
        # Add new findings to team context
        for finding in new_findings:
            team_finding = {
                "content": finding.get("content", "")[:200],  # Limit content length
                "tool": finding.get("tool", "unknown"),
                "timestamp": finding.get("timestamp", datetime.now().isoformat()),
                "iteration": finding.get("iteration", 0),
                "round": finding.get("round", 0)
            }
            self.team_context["team_findings"][agent_id].append(team_finding)
            
            # Keep only recent findings per agent
            if len(self.team_context["team_findings"][agent_id]) > 5:
                self.team_context["team_findings"][agent_id] = self.team_context["team_findings"][agent_id][-5:]
    
    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple deduplication based on content similarity"""
        seen_content = set()
        unique_findings = []
        
        # Build set of existing content (first 100 chars normalized)
        for existing in self.all_findings:
            content = existing.get("content", "").strip().lower()[:100]
            if content:
                seen_content.add(content)
        
        # Check new findings
        for finding in findings:
            content = finding.get("content", "").strip()
            
            # Skip too short or low quality
            if len(content) < 20 or self._is_low_quality_finding(content):
                continue
            
            # Check for duplicate (normalized first 100 chars)
            content_key = content.lower()[:100]
            if content_key in seen_content:
                continue
                
            seen_content.add(content_key)
            unique_findings.append(finding)
                
        return unique_findings
    
    def _is_low_quality_finding(self, content: str) -> bool:
        """Check if a finding is low quality and should be filtered out"""
        content_lower = content.lower().strip()
        
        # Filter out only truly low-quality patterns
        low_quality_patterns = [
            "no tool calls found",
            "please check the tool call",
            "something went wrong",
            "try again",
        ]
        
        # NOTE: "error:", "failed", "unable to", "could not" might be legitimate findings
        # when agents determine something is impossible or encounter real issues
        
        # Check if content is mostly error messages
        error_count = sum(1 for pattern in low_quality_patterns if pattern in content_lower)
        if error_count > 0 and len(content) < 50:
            return True
        
        # Filter out very short generic responses
        if len(content) < 20 and any(word in content_lower for word in ["ok", "done", "yes", "no"]):
            return True
            
        return False
    
    def get_findings_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Thread-safe method to get findings for a specific agent"""
        with self._lock:
            return self.agent_findings.get(agent_id, []).copy()
    
    def get_all_findings(self) -> List[Dict[str, Any]]:
        """Thread-safe method to get all findings"""
        with self._lock:
            return self.all_findings.copy()
    
    def get_team_context_summary(self, agent_id: str) -> str:
        """Get team context summary for a specific agent"""
        with self._lock:
            if "team_findings" not in self.team_context:
                return "No team context available"
            
            context_parts = []
            
            # Get other agents' recent findings
            for other_agent_id, findings in self.team_context["team_findings"].items():
                if other_agent_id != agent_id and findings:
                    recent_findings = findings[-2:]  # Last 2 findings
                    context_parts.append(f"Agent {other_agent_id}: {len(recent_findings)} recent findings")
            
            # Get overall team progress
            total_agents = len(self.agent_progress)
            active_agents = sum(1 for progress in self.agent_progress.values() if progress.get("total_findings", 0) > 0)
            
            if context_parts:
                context_parts.append(f"Team: {active_agents}/{total_agents} agents active")
                return "; ".join(context_parts)
            else:
                return f"Team: {active_agents}/{total_agents} agents active"
    
    def get_agent_progress_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get progress summary for a specific agent"""
        with self._lock:
            if agent_id not in self.agent_progress:
                return {"total_findings": 0, "last_update": None, "findings_history": []}
            
            return self.agent_progress[agent_id].copy()
    
    def add_coordination_message(self, round_num: int, agent_id: str, message: str):
        """Add coordination message to memory for context"""
        with self._lock:
            coordination_msg = {
                "round": round_num,
                "agent_id": agent_id,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            self.coordination_messages.append(coordination_msg)
            
            # Keep only recent coordination messages
            if len(self.coordination_messages) > 20:
                self.coordination_messages = self.coordination_messages[-20:]
    
    def get_recent_coordination_messages(self, agent_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent coordination messages for a specific agent"""
        with self._lock:
            agent_messages = [msg for msg in self.coordination_messages if msg.get("agent_id") == agent_id]
            return agent_messages[-limit:] if agent_messages else []
    
    def add_conversation_turn(self, turn_data: Dict[str, Any]):
        """Thread-safe method to add conversation turn"""
        with self._lock:
            self.conversation_history.append(turn_data)
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 100:
                # Keep most recent 50 turns
                self.conversation_history = self.conversation_history[-50:]
    
    def add_round_summary(self, round_summary: Dict[str, Any]):
        """Add round summary to memory"""
        with self._lock:
            self.round_summaries.append(round_summary)
            
            # Keep only recent round summaries
            if len(self.round_summaries) > 10:
                self.round_summaries = self.round_summaries[-10:]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Thread-safe method to get memory summary"""
        with self._lock:
            return {
                "total_findings": len(self.all_findings),
                "agents_with_findings": list(self.agent_findings.keys()),
                "conversation_turns": len(self.conversation_history),
                "has_research_plan": self.execution_plan is not None,
                "original_query": self.original_query,
                "total_coordination_messages": len(self.coordination_messages),
                "total_round_summaries": len(self.round_summaries),
                "agent_progress": self.agent_progress.copy()
            }
    
    def get_agent_context_for_orchestration(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive context for orchestrator decision making"""
        with self._lock:
            # Get agent's own findings
            agent_findings = self.agent_findings.get(agent_id, [])
            recent_agent_findings = agent_findings[-3:] if agent_findings else []
            
            # Get other agents' findings for team context
            other_agents_findings = {}
            for other_id, findings in self.agent_findings.items():
                if other_id != agent_id and findings:
                    other_agents_findings[other_id] = findings[-2:]  # Last 2 findings
            
            # Get recent coordination messages
            recent_coordination = self.get_recent_coordination_messages(agent_id, 2)
            
            # Get team progress summary
            team_progress = {
                "total_agents": len(self.agent_progress),
                "active_agents": sum(1 for progress in self.agent_progress.values() if progress.get("total_findings", 0) > 0),
                "total_findings": len(self.all_findings)
            }
            
            return {
                "agent_id": agent_id,
                "agent_findings": recent_agent_findings,
                "other_agents_findings": other_agents_findings,
                "recent_coordination": recent_coordination,
                "team_progress": team_progress,
                "execution_plan": self.execution_plan,
                "original_query": self.original_query
            }
    
    def clear(self, max_findings: int = 50):
        """Automatic memory management to prevent bloat"""
        with self._lock:
            # Auto-trim when too many findings
            if len(self.all_findings) > max_findings:
                # Keep most recent findings
                self.all_findings = self.all_findings[-max_findings:]
                
                # Update agent findings to match
                for agent_id in self.agent_findings:
                    agent_findings_in_all = [
                        f for f in self.all_findings 
                        if f.get("agent_id") == agent_id
                    ]
                    self.agent_findings[agent_id] = agent_findings_in_all
                
                logger.info(f"Memory auto-trimmed to {max_findings} findings")
            
            # Auto-trim other collections
            if len(self.coordination_messages) > 20:
                self.coordination_messages = self.coordination_messages[-20:]
            
            if len(self.round_summaries) > 10:
                self.round_summaries = self.round_summaries[-10:]
