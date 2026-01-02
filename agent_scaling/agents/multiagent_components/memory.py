import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .plan import OrchestrationPlan


class EnhancedMemory(BaseModel):
    """Enhanced memory system for multi-agent coordination with thread safety"""

    all_findings: List[str] = []
    agent_findings: Dict[str, List[str]] = {}
    execution_plan: Optional[OrchestrationPlan] = None
    original_task: str = ""

    def model_post_init(self, __context: Any):
        # Thread safety for parallel agent access
        self._lock = threading.Lock()

    def add_findings(self, agent_id: str, findings: str):
        """Thread-safe method to add findings from an agent"""
        with self._lock:
            if agent_id not in self.agent_findings:
                self.agent_findings[agent_id] = []

            # Deduplicate findings before adding

            self.agent_findings[agent_id].append(findings)
            self.all_findings.append(findings)

    def get_findings_for_agent(self, agent_id: str) -> List[str]:
        """Thread-safe method to get findings for a specific agent"""
        with self._lock:
            return self.agent_findings.get(agent_id, []).copy()

    def get_all_findings(self) -> List[str]:
        """Thread-safe method to get all findings"""
        with self._lock:
            return self.all_findings.copy()

    def get_memory_summary(self) -> Dict[str, Any]:
        """Thread-safe method to get memory summary"""
        with self._lock:
            return self.model_dump()
