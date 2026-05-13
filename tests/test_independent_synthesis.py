"""Unit tests for Independent's synthesis_only aggregator (concatenation, no voting)."""
from types import SimpleNamespace

from agent_scaling.agents.multiagent_independent import (
    IndependentMultiAgentSystem,
)


def _make_subagent_stub(agent_id: str, last_answer: str | None):
    conv = SimpleNamespace(
        last_outgoing_external_message=last_answer,
        status="active",
        total_iterations=0,
    )
    env = SimpleNamespace(
        env_done=lambda: False,
        env_status=lambda: None,
        tools={},
    )
    return SimpleNamespace(
        agent_id=agent_id,
        conv_history=conv,
        env=env,
        should_stop_due_to_rate_limiting=lambda: False,
    )


def _new_system():
    cls = IndependentMultiAgentSystem
    sys_ = cls.__new__(cls)
    from agent_scaling.agents.multiagent_components.memory import EnhancedMemory

    sys_.memory = EnhancedMemory()
    sys_.n_base_agents = 3
    sys_.min_iterations_per_agent = 3
    sys_.max_iterations_per_agent = 10
    sys_.subagents = {}
    return sys_


def test_synthesize_only_concatenates_all_agent_answers():
    sys_ = _new_system()
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", "first answer"),
        "agent_2": _make_subagent_stub("agent_2", "second answer"),
        "agent_3": _make_subagent_stub("agent_3", "third answer"),
    }
    synthesized, contributing = sys_._synthesize_only()
    assert "first answer" in synthesized
    assert "second answer" in synthesized
    assert "third answer" in synthesized
    assert contributing == ["agent_1", "agent_2", "agent_3"]


def test_synthesize_only_skips_empty_answers():
    sys_ = _new_system()
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", "first answer"),
        "agent_2": _make_subagent_stub("agent_2", ""),
        "agent_3": _make_subagent_stub("agent_3", None),
    }
    synthesized, contributing = sys_._synthesize_only()
    assert "first answer" in synthesized
    assert contributing == ["agent_1"]


def test_synthesize_only_does_not_vote_or_compare():
    """synthesis_only must NOT collapse repeats or pick a winner - just concatenate."""
    sys_ = _new_system()
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", "same answer"),
        "agent_2": _make_subagent_stub("agent_2", "same answer"),
        "agent_3": _make_subagent_stub("agent_3", "different"),
    }
    synthesized, contributing = sys_._synthesize_only()
    # Both copies of "same answer" must appear: no deduplication, no voting.
    assert synthesized.count("same answer") == 2
    assert "different" in synthesized
    assert contributing == ["agent_1", "agent_2", "agent_3"]


def test_synthesize_only_with_all_empty_returns_empty_string():
    sys_ = _new_system()
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", None),
        "agent_2": _make_subagent_stub("agent_2", ""),
    }
    synthesized, contributing = sys_._synthesize_only()
    assert synthesized == ""
    assert contributing == []


def test_synthesize_only_preserves_insertion_order():
    sys_ = _new_system()
    sys_.subagents = {
        "agent_3": _make_subagent_stub("agent_3", "C"),
        "agent_1": _make_subagent_stub("agent_1", "A"),
        "agent_2": _make_subagent_stub("agent_2", "B"),
    }
    synthesized, contributing = sys_._synthesize_only()
    assert contributing == ["agent_3", "agent_1", "agent_2"]
    assert synthesized.index("C") < synthesized.index("A") < synthesized.index("B")
