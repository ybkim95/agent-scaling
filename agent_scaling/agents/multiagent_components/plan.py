from typing import List

from pydantic import BaseModel


class Subtask(BaseModel):
    agent_id: str
    objective: str
    focus: str


class OrchestrationPlan(BaseModel):
    subtasks: List[Subtask]
    reasoning: str
