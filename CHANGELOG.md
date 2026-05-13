# Changelog

## Unreleased

### Added

- Multi-agent debate with consensus voting for the Decentralized architecture. Each agent sees the full set of peer responses from the previous round; after `d` rounds, a majority vote over agents' final answers selects the system output (Du et al. 2023, adapted to tool-using agents).
- `synthesis_only` aggregator for the Independent architecture. Concatenates sub-agent outputs in insertion order without voting or cross-validation.
- Finance-Agent integration:
  - `agent_scaling/env/finance_agent.py` — env class exposing `web_search` (Tavily), `python_repl`, and `submit`.
  - `agent_scaling/datasets/finance_agent.py` — dataset loader and instance class.
  - `scripts/setup_finance_agent.sh` — clones the upstream repo and writes a normalized JSON file (50 tasks from `data/public.csv`).
  - `prompts/dataset-shared/finance_agent.yaml` and `prompts/eval/finance_agent-grader.yaml`.
  - Per-criterion rubric grading driven by the `Rubric` column of the upstream public test set.
- WorkBench integration:
  - `agent_scaling/env/workbench.py` — env class with stub business tools (`send_email`, `search_emails`, `create_event`, `search_events`) plus `submit`.
  - `agent_scaling/datasets/workbench.py` — dataset loader.
  - `scripts/setup_workbench.sh` — clones the upstream repo and writes two normalized JSON files: `datasets/workbench.json` (a deterministic stratified 100-instance subset, 16–17 per query split with seed 42) and `datasets/workbench_full_690.json` (the complete upstream conversion of 690 tasks across analytics, calendar, CRM, email, multi-domain, and project-management).
  - `prompts/dataset-shared/workbench.yaml` and `prompts/eval/workbench-grader.yaml`.
  - Structural action-call matching for scoring (parses tool calls from the agent's submission and compares them to the expected action list by tool name + keyword-argument set).
- Unit tests under `tests/` covering the new helpers and end-to-end dataset loading, including a determinism test for the WorkBench stratified subset (27 tests total).
- `.env.example` at the repo root for documenting expected API keys.

### Changed

- Per-task timeouts in the Decentralized and Independent runners are now opt-in (default: no limit). The timeout is enabled only when the dataset instance sets a positive `time_limit` attribute.
- Both datasets' `get_metrics` are defensive against the experiment runner's exception-fallback dict (uses `e.get("is_correct", False)` instead of `e["is_correct"]`).
- `agent_scaling/env/__init__.py` and `agent_scaling/env/tools/__init__.py` now wrap the Tavily-dependent imports in `try/except` so `import agent_scaling.env` succeeds without `TAVILY_API_KEY`. Experiments needing web search still require the key.

### Fixed

- Converter scripts now emit a `{dataset_id, instances: [...]}` envelope JSON loadable by `Dataset.from_json`. Earlier output was a flat list and would have crashed dataset loading.
- Tool-name mismatch in the Finance-Agent dataset config: changed `python` to `python_repl` (the actually-registered tool name) in the YAML template, the `FinanceAgentInstance.tools` default, and the converter output.
- Removed a stale empty `agents/multiagent_hybrid.py` at the repository root (the canonical implementation lives under `agent_scaling/agents/multiagent_hybrid.py`).
