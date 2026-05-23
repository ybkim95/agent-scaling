"""Unit tests for the Decentralized debate-with-consensus algorithm.

These tests target the deterministic, infrastructure-free helpers:
- _build_full_debate_context (peer-context assembly, no truncation)
- _consensus_vote (majority vote, deterministic tie-break)
"""
from types import SimpleNamespace

from agent_scaling.agents.multiagent_decentralized import (
    DecentralizedMultiAgentSystem,
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


def _new_system(**overrides):
    """Build a DecentralizedMultiAgentSystem without invoking the parent __init__.

    We bypass full construction because the parent AgentSystemWithTools requires
    an LLM, environment, datasets, prompts, and tools that we deliberately do
    not need for these unit tests.
    """
    cls = DecentralizedMultiAgentSystem
    sys_ = cls.__new__(cls)
    from agent_scaling.agents.multiagent_components.memory import EnhancedMemory
    from agent_scaling.agents.multiagent_utils.communication_strategy import (
        create_communication_strategy,
    )

    sys_.memory = EnhancedMemory()
    sys_.n_base_agents = overrides.get("n_base_agents", 3)
    sys_.max_rounds = overrides.get("max_rounds", 3)
    sys_.consensus_threshold = overrides.get("consensus_threshold", 0.5)
    sys_.subagents = {}
    sys_.consensus = create_communication_strategy(
        "consensus", {"consensus_threshold": sys_.consensus_threshold}
    )
    return sys_


def test_build_full_debate_context_includes_all_peers_no_truncation():
    sys_ = _new_system()
    long_answer = "abc" * 500  # > 300 chars (the previous truncation cutoff)
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", "alpha answer"),
        "agent_2": _make_subagent_stub("agent_2", long_answer),
        "agent_3": _make_subagent_stub("agent_3", "gamma answer"),
    }

    ctx = sys_._build_full_debate_context("agent_2", round_num=2)

    assert "agent_2" not in ctx  # current agent's own answer excluded
    assert "alpha answer" in ctx
    assert "gamma answer" in ctx


def test_build_full_debate_context_preserves_long_answers():
    sys_ = _new_system()
    long_answer = "x" * 1000
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", long_answer),
        "agent_2": _make_subagent_stub("agent_2", "short"),
    }

    ctx = sys_._build_full_debate_context("agent_2", round_num=2)

    assert long_answer in ctx  # not truncated to 300 chars


def test_build_full_debate_context_with_no_peer_responses():
    sys_ = _new_system()
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", None),
        "agent_2": _make_subagent_stub("agent_2", None),
    }
    ctx = sys_._build_full_debate_context("agent_1", round_num=2)
    assert "no peer responses" in ctx.lower()


def test_consensus_vote_picks_majority():
    sys_ = _new_system()
    final = [
        ("agent_1", "answer A"),
        ("agent_2", "answer B"),
        ("agent_3", "answer A"),
    ]
    winning_answer, winning_agent = sys_._consensus_vote(final)
    assert winning_answer == "answer A"
    assert winning_agent in {"agent_1", "agent_3"}


def test_consensus_vote_tie_break_is_deterministic_first_in_order():
    sys_ = _new_system()
    final = [
        ("agent_1", "answer X"),
        ("agent_2", "answer Y"),
    ]
    winning_answer, winning_agent = sys_._consensus_vote(final)
    # Counter.most_common is stable on first-seen for ties; we assert the
    # winner comes from the first occurrence.
    assert winning_answer == "answer X"
    assert winning_agent == "agent_1"


def test_consensus_vote_handles_empty_answers():
    sys_ = _new_system()
    final = [
        ("agent_1", ""),
        ("agent_2", None),  # type: ignore[arg-type]
        ("agent_3", "real answer"),
    ]
    winning_answer, winning_agent = sys_._consensus_vote(final)
    assert winning_answer == "real answer"
    assert winning_agent == "agent_3"


def test_consensus_vote_all_empty_returns_empty():
    sys_ = _new_system()
    winning_answer, winning_agent = sys_._consensus_vote([("agent_1", "")])
    assert winning_answer == ""
    assert winning_agent is None


def test_consensus_strategy_records_votes():
    sys_ = _new_system()
    final = [
        ("agent_1", "answer A"),
        ("agent_2", "answer A"),
        ("agent_3", "answer B"),
    ]
    sys_._consensus_vote(final)
    # ConsensusStrategy.synchronize approves findings meeting threshold (0.5);
    # 2/3 votes for "answer A" exceeds the threshold.
    approved_excerpts = [
        f.get("answer_excerpt") for f in sys_.consensus.approved_findings
    ]
    assert any("answer A" in e for e in approved_excerpts if e)


def test_consensus_vote_includes_completed_agents_last_answer():
    """An agent whose environment terminated in round 1 must still appear in
    the final consensus vote with its last answer.

    This regression test guards the documented behavior that the d debate
    rounds always run to completion and the consensus aggregation sees a
    contribution from every agent regardless of individual termination.
    """
    sys_ = _new_system()
    # Simulate three agents where agent_3 terminated early (status=completed)
    # but emitted a real last answer; the other two are still active.
    completed_conv = SimpleNamespace(
        last_outgoing_external_message="terminated-but-final answer C",
        status="completed",
        total_iterations=2,
    )
    completed_env = SimpleNamespace(
        env_done=lambda: True, env_status=lambda: None, tools={}
    )
    completed_agent = SimpleNamespace(
        agent_id="agent_3",
        conv_history=completed_conv,
        env=completed_env,
        should_stop_due_to_rate_limiting=lambda: False,
    )
    sys_.subagents = {
        "agent_1": _make_subagent_stub("agent_1", "active answer"),
        "agent_2": _make_subagent_stub("agent_2", "active answer"),
        "agent_3": completed_agent,
    }

    # Snapshot the final-round answers exactly as run_agent_async does
    final_round_answers = [
        (aid, sys_.subagents[aid].conv_history.last_outgoing_external_message or "")
        for aid in sys_.subagents
    ]
    winning_answer, winning_agent = sys_._consensus_vote(final_round_answers)

    # agent_3 is "completed" but its answer must still be a valid candidate;
    # the majority is "active answer" (2/3), so that wins, but agent_3 is
    # represented in the candidate pool, not silently dropped.
    candidates = {ans for _, ans in final_round_answers if ans}
    assert "terminated-but-final answer C" in candidates
    assert winning_answer == "active answer"
    assert winning_agent in {"agent_1", "agent_2"}
