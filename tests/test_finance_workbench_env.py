"""Smoke tests for the Finance-Agent and WorkBench environment + dataset adapters.

Verifies that:
- the environment class is registered under the same name used by the YAML config
- the dataset loader is registered and can be instantiated from normalized JSON
- the tool list specified in the YAML config resolves through the env
- ``submit`` transitions ``env_done`` to True and records the final answer
"""
from agent_scaling.datasets import FinanceAgentDataset, FinanceAgentInstance, WorkbenchDataset, WorkbenchInstance
from agent_scaling.datasets.registry import get_dataset_cls, get_dataset_instance_cls
from agent_scaling.env.finance_agent import FinanceAgentEnvironment
from agent_scaling.env.registry import get_env_cls, is_env_registered
from agent_scaling.env.workbench import WorkbenchEnvironment


def test_env_names_registered():
    assert is_env_registered("finance-agent")
    assert is_env_registered("workbench")
    assert get_env_cls("finance-agent") is FinanceAgentEnvironment
    assert get_env_cls("workbench") is WorkbenchEnvironment


def test_dataset_names_registered():
    assert get_dataset_cls("finance-agent") is FinanceAgentDataset
    assert get_dataset_cls("finance_agent") is FinanceAgentDataset
    assert get_dataset_cls("workbench") is WorkbenchDataset
    assert get_dataset_instance_cls("finance-agent") is FinanceAgentInstance
    assert get_dataset_instance_cls("workbench") is WorkbenchInstance


def test_finance_agent_env_resolves_yaml_tools():
    inst = FinanceAgentInstance(
        task_id="fin-001",
        instructions="What is 2 * 3?",
        expected_answer="6",
    )
    env = FinanceAgentEnvironment(
        dataset_instance=inst, tools=["python_repl", "submit"]
    )
    assert set(env.tools.keys()) == {"python_repl", "submit"}
    assert env.env_done() is False
    ret = env.submit.invoke({"reasoning": "The answer is 6"})
    assert env.env_done() is True
    assert env.success is True
    assert env.final_answer is not None and "6" in env.final_answer
    # The submit return value must surface the actual answer text — single-agent
    # uses trajectory[-1].observation as agent_output, so the LLM grader sees
    # this string, not a confirmation placeholder.
    assert "Final Answer" in ret and "6" in ret


def test_workbench_env_resolves_yaml_tools_and_logs_actions():
    inst = WorkbenchInstance(
        task_id="wb-001",
        instructions="Email Bob about Monday standup",
        expected_actions=[{"tool": "send_email", "args": {"to": "bob@example.com"}}],
    )
    env = WorkbenchEnvironment(
        dataset_instance=inst,
        tools=["send_email", "search_emails", "create_event", "search_events", "submit"],
    )
    expected_tools = {
        "send_email",
        "search_emails",
        "create_event",
        "search_events",
        "submit",
    }
    assert set(env.tools.keys()) == expected_tools

    env.send_email.invoke({"to": "bob@example.com", "subject": "Standup", "body": "Mon 9am"})
    env.create_event.invoke({
        "title": "Standup",
        "attendees": "bob",
        "date": "Mon",
        "time": "9am",
    })
    assert len(env.action_log) == 2
    assert env.action_log[0]["tool"] == "send_email"
    assert env.action_log[1]["tool"] == "create_event"

    ret = env.submit.invoke({"reasoning": "Sent email + created calendar event"})
    assert env.env_done() is True
    assert env.success is True
    # submit must surface the answer text and the recorded tool calls so the
    # grader can evaluate both the reasoning and the action sequence.
    assert "Final Answer" in ret and "Sent email" in ret
    assert "send_email" in ret and "create_event" in ret


def test_finance_agent_instance_round_trip_through_normalized_json():
    """The shape produced by scripts/_convert_finance_agent.py must load cleanly."""
    raw = {
        "task_id": "fin-test",
        "instructions": "Compute the P/E ratio for price=100 and EPS=5.",
        "expected_answer": "20",
        "tools": ["python_repl", "submit"],
        "metadata": {"source": "finance-agent", "upstream_path": "tasks/task001.json"},
    }
    inst = FinanceAgentInstance(**raw)
    info = inst.get_prompt_info()
    assert info["task_id"] == "fin-test"
    assert "P/E" in info["question"]
    assert inst.expected_output == "20"


def test_workbench_instance_round_trip_through_normalized_json():
    raw = {
        "task_id": "wb-test",
        "instructions": "Send a follow-up email to bob@example.com",
        "expected_actions": [
            {"call": 'send_email(to="bob@example.com", subject="Follow-up", body="...")'}
        ],
        "expected_answer": None,
        "tools": ["send_email", "search_emails", "create_event", "search_events", "submit"],
        "metadata": {"source": "workbench", "domain": "email"},
    }
    inst = WorkbenchInstance(**raw)
    info = inst.get_prompt_info()
    assert info["task_id"] == "wb-test"
    assert "bob@example.com" in info["question"]
    # expected_output derives from the function-call strings inside
    # expected_actions when expected_answer is None.
    assert "send_email" in inst.expected_output


def test_finance_agent_dataset_loads_normalized_json_payload(tmp_path):
    """End-to-end: the {dataset_id, instances:[...]} payload produced by the
    converter loads cleanly through Dataset.from_json with real shared prompts."""
    import json
    from agent_scaling.datasets.base import DatasetSharedPrompts

    payload = {
        "dataset_id": "finance_agent",
        "instances": [
            {
                "task_id": "fin-1",
                "instructions": "Compute the P/E ratio.",
                "expected_answer": "20",
                "tools": ["python_repl", "submit"],
                "metadata": {"source": "finance-agent"},
            }
        ],
    }
    fp = tmp_path / "fa.json"
    fp.write_text(json.dumps(payload))
    sp = DatasetSharedPrompts.from_yaml("prompts/dataset-shared/finance_agent.yaml")

    ds = FinanceAgentDataset.from_json(
        json_path=str(fp),
        eval_prompts=None,
        eval_llm=None,
        use_llm_eval=False,
        task_shared_prompts=sp,
    )
    assert len(ds.instances) == 1
    assert ds.instances[0].task_id == "fin-1"

    # And the shared prompts compile cleanly for this instance
    compiled = ds.task_shared_prompts.get_prompt_templates_for_instance(
        ds.instances[0].get_prompt_info()
    )
    assert "fin-1" in compiled["task_instance"]
    assert "P/E ratio" in compiled["task_instance"]


def test_workbench_dataset_loads_normalized_json_payload(tmp_path):
    import json
    from agent_scaling.datasets.base import DatasetSharedPrompts

    payload = {
        "dataset_id": "workbench",
        "instances": [
            {
                "task_id": "wb-1",
                "instructions": "Email Bob about Monday standup.",
                "expected_actions": [{"tool": "send_email", "args": {"to": "bob@example.com"}}],
                "expected_answer": None,
                "tools": ["send_email", "search_emails", "create_event", "search_events", "submit"],
                "metadata": {"source": "workbench", "domain": "email"},
            }
        ],
    }
    fp = tmp_path / "wb.json"
    fp.write_text(json.dumps(payload))
    sp = DatasetSharedPrompts.from_yaml("prompts/dataset-shared/workbench.yaml")

    ds = WorkbenchDataset.from_json(
        json_path=str(fp),
        eval_prompts=None,
        eval_llm=None,
        use_llm_eval=False,
        task_shared_prompts=sp,
    )
    assert len(ds.instances) == 1
    assert ds.instances[0].task_id == "wb-1"

    compiled = ds.task_shared_prompts.get_prompt_templates_for_instance(
        ds.instances[0].get_prompt_info()
    )
    assert "wb-1" in compiled["task_instance"]
    assert "Monday standup" in compiled["task_instance"]


def test_finance_agent_converter_emits_dataset_envelope(tmp_path):
    """The converter must consume the real upstream ``data/public.csv`` layout
    and wrap instances in ``{dataset_id, instances:[...]}`` — a flat list
    would crash ``Dataset.from_json``."""
    import json
    import subprocess

    upstream = tmp_path / "upstream"
    (upstream / "data").mkdir(parents=True)
    (upstream / "data" / "public.csv").write_text(
        'Question,Answer,Question Type,Expert time (mins),Rubric\n'
        '"What is 2 plus 2?","4","Arithmetic","1",""\n'
        '"Capital of France?","Paris","Geography","1",""\n'
    )
    out = tmp_path / "out.json"
    subprocess.run(
        ["python", "scripts/_convert_finance_agent.py",
         "--upstream-dir", str(upstream), "--out", str(out)],
        check=True,
    )
    payload = json.loads(out.read_text())
    assert isinstance(payload, dict), "Converter must emit a dict, not a bare list"
    assert payload["dataset_id"] == "finance_agent"
    assert isinstance(payload["instances"], list)
    assert len(payload["instances"]) == 2
    assert payload["instances"][0]["task_id"] == "fa-0000"
    assert "Capital of France" in payload["instances"][1]["instructions"]
    assert payload["instances"][1]["expected_answer"] == "Paris"


def test_workbench_structural_grader_exact_match(tmp_path):
    """The structural grader mirrors upstream's is_exact_match: same tool calls
    in the same order, with matching kwarg name sets."""
    from agent_scaling.datasets.base import DatasetInstanceOutput
    from agent_scaling.datasets.workbench import WorkbenchDataset, WorkbenchInstance

    inst = WorkbenchInstance(
        task_id="wb-1",
        instructions="Email Bob and create event",
        expected_actions=[
            {"call": 'send_email(to="bob@example.com", subject="x", body="y")'},
            {"call": 'create_event(title="standup", attendees="bob", date="Mon", time="9am")'},
        ],
    )
    ds = WorkbenchDataset(
        dataset_id="workbench",
        instances=[inst],
        task_shared_prompts=None,
        use_llm_eval=False,
    )

    # Predicted text contains both calls in the same order with matching args
    predicted = (
        'send_email(to="bob@example.com", subject="hi", body="see you")\n'
        'create_event(title="standup", attendees="bob", date="Mon", time="9am")'
    )
    output = DatasetInstanceOutput(data_instance=inst, agent_output=predicted)
    metrics = ds.get_instance_eval_metrics(output)
    assert metrics["is_exact_match"] is True
    assert metrics["is_correct"] is True
    assert metrics["grading_method"] == "structural_action_match"


def test_workbench_structural_grader_wrong_actions(tmp_path):
    """Wrong actions should fail exact-match and set-match."""
    from agent_scaling.datasets.base import DatasetInstanceOutput
    from agent_scaling.datasets.workbench import WorkbenchDataset, WorkbenchInstance

    inst = WorkbenchInstance(
        task_id="wb-2",
        instructions="Email Bob",
        expected_actions=[
            {"call": 'send_email(to="bob@example.com", subject="x", body="y")'},
        ],
    )
    ds = WorkbenchDataset(
        dataset_id="workbench", instances=[inst],
        task_shared_prompts=None, use_llm_eval=False,
    )
    output = DatasetInstanceOutput(
        data_instance=inst,
        agent_output='search_emails(query="anything")',
    )
    metrics = ds.get_instance_eval_metrics(output)
    assert metrics["is_exact_match"] is False
    assert metrics["is_set_match"] is False
    assert metrics["is_correct"] is False


def test_finance_agent_rubric_grader_path(monkeypatch):
    """Rubric-based grading routes to _grade_one_criterion and aggregates."""
    from agent_scaling.datasets.base import DatasetInstanceOutput
    from agent_scaling.datasets.finance_agent import (
        FinanceAgentDataset, FinanceAgentInstance,
    )

    inst = FinanceAgentInstance(
        task_id="fa-1",
        instructions="What happened?",
        expected_answer="Detail A. Detail B.",
        metadata={"rubric": [
            {"operator": "correctness", "criteria": "Mentions detail A"},
            {"operator": "correctness", "criteria": "Mentions detail B"},
            {"operator": "correctness", "criteria": "Mentions detail C"},
        ]},
    )

    # Build the dataset without LLM-eval prerequisites, then patch the
    # eval_prompts/eval_llm sentinels and the per-criterion grader.
    ds = FinanceAgentDataset(
        dataset_id="finance_agent", instances=[inst],
        task_shared_prompts=None, use_llm_eval=False,
    )
    ds.eval_prompts = {"grader": object()}  # not used by stub
    ds.eval_llm = object()                  # not used by stub
    decisions = iter([True, True, False])
    monkeypatch.setattr(
        ds, "_grade_one_criterion",
        lambda question, response, criterion: next(decisions),
    )

    output = DatasetInstanceOutput(
        data_instance=inst, agent_output="My response includes detail A and detail B."
    )
    metrics = ds.get_instance_eval_metrics(output)
    assert metrics["grading_method"] == "rubric"
    assert metrics["n_criteria_satisfied"] == 2
    assert metrics["n_criteria_total"] == 3
    assert abs(metrics["rubric_score"] - 2 / 3) < 1e-9
    assert metrics["is_correct"] is True  # >= 0.5


def test_workbench_converter_emits_dataset_envelope(tmp_path):
    """The converter must consume the real upstream
    ``data/processed/queries_and_answers/*.csv`` layout."""
    import json
    import subprocess

    import csv as _csv
    upstream = tmp_path / "upstream"
    qa = upstream / "data" / "processed" / "queries_and_answers"
    qa.mkdir(parents=True)
    with (qa / "analytics_queries_and_answers.csv").open("w", newline="") as fh:
        w = _csv.writer(fh, quoting=_csv.QUOTE_ALL)
        w.writerow(["query", "answer", "base_template", "chosen_template", "domains"])
        w.writerow([
            "Plot total visits",
            "['analytics.create_plot.func(value=\"visits\")']",
            "T", "T", "['analytics']",
        ])
    with (qa / "email_queries_and_answers.csv").open("w", newline="") as fh:
        w = _csv.writer(fh, quoting=_csv.QUOTE_ALL)
        w.writerow(["query", "answer", "base_template", "chosen_template", "domains"])
        w.writerow([
            "Email Bob",
            "['send_email(to=\"bob\")']",
            "T", "T", "['email']",
        ])
    out = tmp_path / "out.json"
    subprocess.run(
        ["python", "scripts/_convert_workbench.py",
         "--upstream-dir", str(upstream), "--out", str(out)],
        check=True,
    )
    payload = json.loads(out.read_text())
    assert isinstance(payload, dict), "Converter must emit a dict, not a bare list"
    assert payload["dataset_id"] == "workbench"
    assert isinstance(payload["instances"], list)
    assert len(payload["instances"]) == 2
    task_ids = {i["task_id"] for i in payload["instances"]}
    assert task_ids == {"analytics-0000", "email-0000"}
    analytics = next(i for i in payload["instances"] if i["task_id"] == "analytics-0000")
    assert "expected_actions" in analytics
    assert analytics["expected_actions"][0]["call"].startswith("analytics.create_plot")


def test_workbench_converter_stratified_subset_is_deterministic():
    """The converter must produce a deterministic stratified 100-instance subset
    matching the paper: exactly 100 tasks, balanced across the 6 domains, seed=42.

    Uses the in-memory ``stratified_subset`` helper so the test runs without
    cloning the upstream repository.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_convert_workbench",
        Path(__file__).resolve().parent.parent / "scripts" / "_convert_workbench.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    domains = [
        "analytics",
        "calendar",
        "customer_relationship_manager",
        "email",
        "multi_domain",
        "project_management",
    ]
    records = [
        {"task_id": f"{d}-{i:04d}", "instructions": f"task {d} {i}"}
        for d in domains
        for i in range(115)
    ]
    a = mod.stratified_subset(records, sample_size=100, seed=42)
    b = mod.stratified_subset(records, sample_size=100, seed=42)
    assert len(a) == 100
    assert a == b, "Stratified subset must be deterministic across calls"

    from collections import Counter
    per_domain = Counter(r["task_id"].split("-", 1)[0] for r in a)
    assert sum(per_domain.values()) == 100
    counts = sorted(per_domain.values())
    assert counts == [16, 16, 17, 17, 17, 17], (
        f"Expected 17+17+17+17+16+16 balance across 6 domains, got {counts}"
    )
    assert set(per_domain.keys()) == set(domains), (
        f"All six domains must be represented, got {set(per_domain.keys())}"
    )

    c = mod.stratified_subset(records, sample_size=100, seed=43)
    assert c != a, "Different seed must yield a different subset"
