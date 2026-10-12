# Changelog

## 2.2.3

### Changed

- `REPRODUCTION.md`: the set-up instructions check out the latest release instead of v2.1.3. The experiment code is the same in v2.1.3 and v2.2.x.

## 2.2.2

### Changed

- `scripts/architecture_selection_eval.py`: output headings and docstrings reworded. No computed value changes, and `--decisions` writes the same file.

## 2.2.1

### Fixed

- `scripts/error_amplification_analysis.py` built 270 rows, including the ten `gemini-3-flash-preview` configurations, for its task-level error-amplification tables. It now uses the 260 configurations analysed in the article. No value reported in the article is computed by this script.
- `scripts/alternative_capability_metric.py` and `scripts/extended_mixed_effects_6benchmarks.py` printed stale configuration counts in their notes. The printed counts now match the 260 configurations analysed.

## 2.2.0

Maintenance release. Corrects the default analysis set of the regression scripts, adds the analysis scripts and data for the regression, the complete-choice-set evaluation of architecture selection and the threshold sign check, and includes the analysis notebooks.

### Fixed

- `scripts/extended_mixed_effects_6benchmarks.py` and `scripts/alternative_capability_metric.py` built and asserted 270 rows by default, including the ten `gemini-3-flash-preview` configurations that Supplementary Table S11 identifies as out of sample. Both now exclude those rows and analyse the 260 configurations reported in the article; the published regression values (R2_train = 0.463, R2_CV = 0.373, ACI R2_CV = 0.413) reproduce with this release and did not with v2.1.3.
- `INTELLIGENCE_SCORES["gemini-3-flash-preview"]` corrected from 71 to 75 to match Supplementary Table S1 (Gemini-3.0 Flash). The entry is not used by the N=260 analysis.

### Added

- `scripts/architecture_selection_eval.py`: evaluates how often Equation 1 selects the best-performing architecture when a complete (benchmark, model) choice set is held out, alongside chance, always-single-agent and benchmark-mean baselines. Runs from `data/Source_Data_Fig1_Fig2.csv`; `--check` asserts the reported values and `--decisions PATH` writes the per-choice-set output.
- `scripts/reproduce_published_regression.py`: reproduces the N=260 regression in a single run, including Table 2, Supplementary Tables S8 to S10 and Extended Data Table 1.
- `scripts/threshold_sign_match.py`: computes the sign-match rate for the ~45% capability-saturation threshold on the 16 SWE-bench Verified and Terminal-Bench pairs and reports its sensitivity to the cut-off and gain conventions. `--sensitivity` reports the rate on both gain endpoints, the best of the four multi-agent architectures (15 of 16 over 0.41-0.45, 14 of 16 at 0.40) and their mean (13 of 16 over the same range), and asserts each.
- `data/Source_Data_Fig1_Fig2.csv`: the 260 configurations deposited with the article.
- `data/architecture_selection_decisions.csv`: the decision-level output of `architecture_selection_eval.py`, one row per choice set, giving each rule's selected architecture, its score, the observed optimum and the regret.
- `colab_analysis/`: the two analysis notebooks (`mixed_effect_model.ipynb`, `agent_scaling_final.ipynb`) previously excluded from the release.

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
- Unit tests under `tests/` covering the new helpers and end-to-end dataset loading, including a determinism test for the WorkBench stratified subset and a regression test for completed-agent participation in the Decentralized consensus vote (28 tests total).
- `.env.example` at the repo root for documenting expected API keys.

### Changed

- Per-task timeouts in the Decentralized and Independent runners are now opt-in (default: no limit). The timeout is enabled only when the dataset instance sets a positive `time_limit` attribute.
- Both datasets' `get_metrics` are defensive against the experiment runner's exception-fallback dict (uses `e.get("is_correct", False)` instead of `e["is_correct"]`).
- `agent_scaling/env/__init__.py` and `agent_scaling/env/tools/__init__.py` now wrap the Tavily-dependent imports in `try/except` so `import agent_scaling.env` succeeds without `TAVILY_API_KEY`. Experiments needing web search still require the key.

### Fixed

- Converter scripts now emit a `{dataset_id, instances: [...]}` envelope JSON loadable by `Dataset.from_json`. Earlier output was a flat list and would have crashed dataset loading.
- Tool-name mismatch in the Finance-Agent dataset config: changed `python` to `python_repl` (the actually-registered tool name) in the YAML template, the `FinanceAgentInstance.tools` default, and the converter output.
- Removed a stale empty `agents/multiagent_hybrid.py` at the repository root (the canonical implementation lives under `agent_scaling/agents/multiagent_hybrid.py`).
