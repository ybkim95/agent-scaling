# Reproduction Guide

This document provides step-by-step instructions for reproducing all experiments in the paper.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [API Keys Required](#api-keys-required)
3. [Running Experiments](#running-experiments)
4. [Prompt Templates](#prompt-templates)
5. [Agent Configurations](#agent-configurations)
6. [Dataset Files](#dataset-files)
7. [Expected Runtime and Cost](#expected-runtime-and-cost)

---

## Environment Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), Docker (required for SWE-bench and Terminal-Bench)

```bash
# 1. Clone the repository and check out the exact release tag cited in the manuscript.
# The Nature Machine Intelligence manuscript cites release v2.1.3 (Zenodo version-specific
# DOI 10.5281/zenodo.20388843). The concept DOI 10.5281/zenodo.20144433 always resolves to
# the latest archived release; for reproducing the manuscript-reported numbers use v2.1.3.
git clone https://github.com/ybkim95/agent-scaling.git
cd agent-scaling
git checkout v2.1.3

# 2. Install dependencies
uv sync --prerelease=allow

# 3. Install flash-attn (required for BrowseComp-Plus environment)
uv pip install --no-build-isolation flash-attn

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Create a .env file with your API keys (see section below)
cp .env.example .env   # then fill in your keys
```

### Optional: LangFuse Tracing

If you wish to enable LLM call tracing via LangFuse, add the following to your `.env`:

```bash
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
```

Then pass `log_langfuse=true` when running experiments.

---

## API Keys Required

Add the following to a `.env` file in the repository root. At minimum, one LLM provider key is required per experiment.

```bash
# LLM providers (add keys for the models you intend to use)
OPENAI_API_KEY="your-openai-key"          # for gpt-5, gpt-5-mini, gpt-5-nano
GEMINI_API_KEY="your-gemini-key"          # for gemini-2.0-flash, gemini-2.5-pro
ANTHROPIC_API_KEY="your-anthropic-key"    # for claude-sonnet-4-5, claude-sonnet-4, claude-sonnet-3-7

# Required for BrowseComp-Plus and Finance-Agent web-search environments
TAVILY_API_KEY="your-tavily-key"

# Optional: LangFuse tracing
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

**Do NOT commit your `.env` file.** It is listed in `.gitignore`.

---

## Running Experiments

The framework uses [Hydra](https://hydra.cc/docs/intro/) for configuration. The main entry point is:

```bash
python scripts/run_experiment.py [overrides...]
```

### Core parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `agent` | Agent configuration name | `multi-agent-centralized` |
| `dataset` | Dataset configuration name | `plancraft-test` |
| `llm.model` | LiteLLM model identifier | `gemini/gemini-2.0-flash` |
| `llm.params.temperature` | Sampling temperature | `0.0` |
| `num_workers` | Parallel instance workers | `1` |
| `max_instances` | Cap on instances to process | unlimited |
| `debug` | Debug mode (3 instances) | `false` |

### Paper experiments

All experiments in the paper use `temperature=0.0` and `n_base_agents=3` (for multi-agent configs). The commands below show one representative model per (architecture, benchmark) cell, drawn from the paper's model pool (GPT-5, GPT-5-mini, GPT-5-nano, Gemini-2.0-Flash, Gemini-2.5-Flash, Gemini-2.5-Pro, Claude Sonnet 3.7, Claude Sonnet 4, Claude Sonnet 4.5; Supplementary Information Table S1 lists the full pool). To reproduce any specific cell of the 260-configuration grid, swap `llm.model=` to the corresponding model identifier; the canonical per-cell mapping is encoded in the run configs and the analysis scripts.

#### Single-agent baseline

```bash
# PlanCraft
python scripts/run_experiment.py agent=single-agent dataset=plancraft-test llm.model=gemini/gemini-2.0-flash

# BrowseComp-Plus
python scripts/run_experiment.py agent=single-agent dataset=browsecomp-plus llm.model=openai/gpt-5-mini

# SWE-bench Verified (20-instance subset)
python scripts/run_experiment.py agent=single-agent dataset=swebench-verified llm.model=openai/gpt-5-mini

# Terminal-Bench (20-instance subset)
python scripts/run_experiment.py agent=single-agent dataset=terminalbench llm.model=openai/gpt-5-mini

# Finance-Agent (after `bash scripts/setup_finance_agent.sh`)
python scripts/run_experiment.py agent=single-agent dataset=finance-agent llm.model=openai/gpt-5-mini

# Workbench (after `bash scripts/setup_workbench.sh`)
python scripts/run_experiment.py agent=single-agent dataset=workbench llm.model=openai/gpt-5-mini
```

#### Multi-agent centralized (lead + subagents)

```bash
python scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
python scripts/run_experiment.py agent=multi-agent-centralized dataset=browsecomp-plus llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-centralized dataset=swebench-verified llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-centralized dataset=terminalbench llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-centralized dataset=finance-agent llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-centralized dataset=workbench llm.model=openai/gpt-5-mini
```

#### Multi-agent decentralized (Du-et-al-2023-style debate with consensus voting)

```bash
python scripts/run_experiment.py agent=multi-agent-decentralized dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
python scripts/run_experiment.py agent=multi-agent-decentralized dataset=browsecomp-plus llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-decentralized dataset=swebench-verified llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-decentralized dataset=terminalbench llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-decentralized dataset=finance-agent llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-decentralized dataset=workbench llm.model=openai/gpt-5-mini
```

#### Multi-agent hybrid

```bash
python scripts/run_experiment.py agent=multi-agent-hybrid dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
python scripts/run_experiment.py agent=multi-agent-hybrid dataset=browsecomp-plus llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-hybrid dataset=swebench-verified llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-hybrid dataset=terminalbench llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-hybrid dataset=finance-agent llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-hybrid dataset=workbench llm.model=openai/gpt-5-mini
```

#### Multi-agent independent (synthesis_only concatenation)

```bash
python scripts/run_experiment.py agent=multi-agent-independent dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
python scripts/run_experiment.py agent=multi-agent-independent dataset=browsecomp-plus llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-independent dataset=swebench-verified llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-independent dataset=terminalbench llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-independent dataset=finance-agent llm.model=openai/gpt-5-mini
python scripts/run_experiment.py agent=multi-agent-independent dataset=workbench llm.model=openai/gpt-5-mini
```

The full reproduction matrix above covers all 30 (architecture, dataset) combinations: 5 architectures (single-agent, multi-agent-centralized, multi-agent-decentralized, multi-agent-hybrid, multi-agent-independent) times 6 benchmarks (plancraft-test, browsecomp-plus, swebench-verified, terminalbench, finance-agent, workbench). Swap the `llm.model=` override to any model in the paper's pool (see Supplementary Information Table S1 for the full list) to reproduce a different cell of the experiment grid. Models are matched per-benchmark in the canonical runs to control for per-system compute as described in the Methods.

### Output location

Results are saved to:

```
exp_outputs/{dataset_id}/{agent_name}/{llm_provider}/{llm_model}/{date}/{time}/
├── run_config.yaml          # Full resolved configuration
├── run.log                  # Execution log
├── dataset_eval_metrics.json  # Aggregated metrics
└── instance_runs/
    ├── 0000/
    │   └── instance_save.yaml  # Per-instance result
    ├── 0001/
    ...
```

---

## Prompt Templates

All prompt templates are under `prompts/`. They use `{{variable}}` placeholder syntax resolved at runtime.

### Agent prompts

| File | Description |
|------|-------------|
| `prompts/single-agent/single-agent.yaml` | System + user prompt for single-agent runs |
| `prompts/multi-agent/lead_agent.yaml` | Lead agent prompt (centralized / hybrid) |
| `prompts/multi-agent/subagent.yaml` | Subagent prompt (centralized / hybrid / independent) |
| `prompts/multi-agent/agent_decision.yaml` | Agent action-decision prompt |
| `prompts/multi-agent/agent_feedback.yaml` | Inter-agent feedback prompt |
| `prompts/multi-agent/orchestration_decision.yaml` | Orchestration routing prompt |
| `prompts/direct-prompt/direct-prompt.yaml` | Zero-shot direct-prompt baseline |

### Dataset-specific task templates

| File | Dataset |
|------|---------|
| `prompts/dataset-shared/plancraft.yaml` | PlanCraft |
| `prompts/dataset-shared/browsecomp.yaml` | BrowseComp-Plus |
| `prompts/dataset-shared/swebench.yaml` | SWE-bench Verified |
| `prompts/dataset-shared/terminalbench.yaml` | Terminal-Bench |
| `prompts/dataset-shared/finance_agent.yaml` | Finance-Agent |
| `prompts/dataset-shared/workbench.yaml` | Workbench |

### Evaluation / grading prompts

| File | Description |
|------|-------------|
| `prompts/eval/grader.yaml` | General LLM-as-judge grader |
| `prompts/eval/qa-grader.yaml` | QA-specific grader |
| `prompts/eval/browsecomp-grader.yaml` | BrowseComp-Plus answer judge |
| `prompts/eval/finance_agent-grader.yaml` | Finance-Agent rubric judge (per-criterion against the `Rubric` column of `data/public.csv`) |
| `prompts/eval/workbench-grader.yaml` | (Reserved.) Workbench scoring is structural (regex-based action matching against `expected_actions`); see `WorkbenchDataset.get_instance_eval_metrics`. |

---

## Agent Configurations

All agent configs are under `run_conf/agent/`.

| File | Agent type | `n_base_agents` | Notes |
|------|-----------|-----------------|-------|
| `run_conf/agent/single-agent.yaml` | Single agent | 1 | Baseline |
| `run_conf/agent/multi-agent-centralized.yaml` | Centralized MAS | 3 | Lead + subagents, orchestrated communication |
| `run_conf/agent/multi-agent-decentralized.yaml` | Decentralized MAS | 3 | Peer consensus, 70% agreement threshold |
| `run_conf/agent/multi-agent-hybrid.yaml` | Hybrid MAS | 3 | Lead + peer communication enabled |
| `run_conf/agent/multi-agent-independent.yaml` | Independent MAS | 3 | No inter-agent coordination |

Key shared parameters (centralized / decentralized / hybrid / independent):

```yaml
n_base_agents: 3
min_iterations_per_agent: 3
max_iterations_per_agent: 25
max_rounds: 10
```

---

## Dataset Files

The paper evaluates on six benchmarks. All six are runnable from this repository. Dataset acquisition depends on the benchmark's upstream licensing: `datasets/browsecomp_plus_sampled_100.json` (BrowseComp-Plus, Apache 2.0) and `datasets/plancraft-test.json` (PlanCraft, MIT) are shipped directly in this repository under their permissive upstream licenses. Finance-Agent, WorkBench, SWE-bench Verified, and Terminal-Bench raw or derived dataset files are not redistributed; users obtain them from upstream sources, either via the documented setup scripts (Finance-Agent and WorkBench) or by manual download (SWE-bench Verified and Terminal-Bench). Dataset configuration files (`run_conf/dataset/*.yaml`) are provided for benchmarks integrated directly, and a `.yaml.template` plus `scripts/setup_*.sh` adapter is provided for benchmarks downloaded from upstream.

### Integrated benchmarks (run directly from this repository)

| Dataset | Config file | Expected local path | Instances used | Selection method |
|---------|-------------|---------------------|----------------|-----------------|
| PlanCraft | `run_conf/dataset/plancraft-test.yaml` | `datasets/plancraft-test.json` | 100 | Full test set |
| BrowseComp-Plus | `run_conf/dataset/browsecomp-plus.yaml` | `datasets/browsecomp_plus_sampled_100.json` | 100 | 100-instance fixed random sample |
| SWE-bench Verified | `run_conf/dataset/swebench-verified.yaml` | `datasets/swebench-verified.json` | 20 | Deterministic shuffle (seed 42), first 20 |
| Terminal-Bench | `run_conf/dataset/terminalbench.yaml` | `datasets/terminalbench.json` | 20 | First 20 in canonical order |

### Upstream benchmarks (downloaded via setup scripts, then run from this repository)

| Dataset | Setup script | Config template | Upstream repository | Instances used |
|---------|--------------|-----------------|---------------------|----------------|
| Finance-Agent | `bash scripts/setup_finance_agent.sh` | `run_conf/dataset/finance-agent.yaml.template` | https://github.com/vals-ai/finance-agent | 50 (full evaluation set) |
| WorkBench | `bash scripts/setup_workbench.sh` | `run_conf/dataset/workbench.yaml.template` | https://github.com/olly-styles/WorkBench | 100 (stratified per-domain subset of 690 upstream tasks, seed 42; full set also written to `datasets/workbench_full_690.json`) |

Each `setup_*.sh` script:
1. Clones the upstream repository into `third_party/<benchmark>/` (skippable via `--upstream-dir`).
2. Runs the adapter `scripts/_convert_<benchmark>.py` to convert upstream task definitions into the normalized JSON schema this repository consumes.
3. Copies the config template to `run_conf/dataset/<benchmark>.yaml` if not already present.

Once setup completes, the same `python scripts/run_experiment.py dataset=finance-agent ...` (and `dataset=workbench ...`) commands run the experiments using our five coordination architectures.

If the upstream layout changes, edit the field-mapping helpers in `scripts/_convert_finance_agent.py` / `scripts/_convert_workbench.py` (the relevant `extract_*` functions are deliberately small and tolerant of schema drift). The adapter loaders themselves (`agent_scaling/datasets/finance_agent.py`, `workbench.py`) consume the normalized JSON schema and are stable.

See `DATA_AVAILABILITY.md` for full benchmark sources and licensing.

---

## Expected Runtime and Cost

All estimates assume `num_workers=1`, `temperature=0.0`, `n_base_agents=3`.

| Dataset | Agent type | Model | Est. wall time | Est. API cost |
|---------|-----------|-------|---------------|---------------|
| PlanCraft (100 inst.) | Single-agent | gemini-2.0-flash | ~1 hr | ~$2 |
| PlanCraft (100 inst.) | Multi-agent-centralized | gemini-2.0-flash | ~3 hr | ~$8 |
| BrowseComp-Plus (100 inst.) | Single-agent | gpt-5-mini | ~2 hr | ~$5 |
| BrowseComp-Plus (100 inst.) | Multi-agent-centralized | gpt-5-mini | ~6 hr | ~$18 |
| SWE-bench Verified (20 inst.) | Single-agent | gpt-5-mini | ~1 hr | ~$4 |
| SWE-bench Verified (20 inst.) | Multi-agent-centralized | gpt-5-mini | ~3 hr | ~$14 |
| Terminal-Bench (20 inst.) | Single-agent | gpt-5-mini | ~1 hr | ~$4 |
| Terminal-Bench (20 inst.) | Multi-agent-centralized | gpt-5-mini | ~3 hr | ~$14 |
| Finance-Agent (50 inst.) | Single-agent | gpt-5-mini | ~1.5 hr | ~$3 |
| Finance-Agent (50 inst.) | Multi-agent-centralized | gpt-5-mini | ~4 hr | ~$10 |
| Workbench (100 inst., stratified subset of 690, seed 42) | Single-agent | gpt-5-mini | ~1.5 hr | ~$3 |
| Workbench (100 inst., stratified subset of 690, seed 42) | Multi-agent-centralized | gpt-5-mini | ~4 hr | ~$10 |

**Notes:**
- Estimates are approximate; actual cost depends on task difficulty and model verbosity.
- Using `num_workers=4` reduces wall time by ~3-4x with proportional cost.
- SWE-bench and Terminal-Bench require Docker and pull benchmark container images on first run (~5–10 min overhead).
- Finance-Agent: `web_search` (Tavily) requires `TAVILY_API_KEY`; without it the agent falls back to `python_repl + submit` and many factual questions will be unanswerable. This is expected and documented in `agent_scaling/env/finance_agent.py`.
- Workbench: grading uses structural action-match (mirrors upstream `is_exact_match`); the harder upstream `is_correct` (state-change simulation) is documented in CHANGELOG.md as an out-of-scope item.
- Cost estimates use pricing at time of submission; check current provider pricing before large runs.

---

## Quick Start (fresh clone)

If you are a reviewer or first-time user, the minimum end-to-end sequence is:

```bash
# 1. Clone + install
git clone https://github.com/ybkim95/agent-scaling.git
cd agent-scaling
uv sync --prerelease=allow
source .venv/bin/activate

# 2. Set at least one LLM API key
cp .env.example .env  # then fill in OPENAI_API_KEY or GEMINI_API_KEY

# 3. Set up the two upstream-downloaded benchmarks (5 min each, no API key needed for setup)
bash scripts/setup_finance_agent.sh
bash scripts/setup_workbench.sh

# 4. (Optional) Download the four directly-integrated benchmarks
#     - PlanCraft:           install `plancraft` upstream and place test JSON at datasets/plancraft-test.json
#     - BrowseComp-Plus:     follow upstream README; place sampled JSON at datasets/browsecomp_plus_sampled_100.json
#     - SWE-bench Verified:  download from https://www.swebench.com/; produces datasets/swebench-verified.json
#     - Terminal-Bench:      install `terminalbench` upstream; place at datasets/terminalbench.json

# 5. Run a smoke test (1 instance, cheap model, no Docker required)
python scripts/run_experiment.py agent=single-agent dataset=finance-agent \
    llm.model=openai/gpt-5-mini max_instances=1 num_workers=1

# 6. Inspect the output
ls exp_outputs/finance_agent/single-agent/openai/gpt-5-mini/<date>/<time>/
#     run_config.yaml   |  dataset_eval_metrics.json   |  instance_runs/0000/instance_save.yaml

# 7. Run unit tests to verify the algorithm implementations
PYTHONPATH="" PYTHONNOUSERSITE=1 .venv/bin/python -m pytest \
    tests/test_decentralized_debate.py \
    tests/test_independent_synthesis.py \
    tests/test_finance_workbench_env.py -v
```

Expected result of step 7: **28 passed**.
