# Data Availability Statement

This document lists all benchmarks used in the paper, their sources, and the subsets selected for evaluation.

---

## Benchmarks

### 1. PlanCraft

- **Reference:** Dagan et al., "PlanCraft: An Open-Ended Multi-Modal Planning Benchmark in Minecraft" (2024), arXiv:2412.21033
- **URL:** https://github.com/gautierdag/plancraft
- **Task type:** Tool-use planning (Minecraft crafting)
- **Total instances:** 100 (test split)
- **Instances used:** 100
- **Subset selection:** Full test split; no sub-sampling
- **Expected local path:** `datasets/plancraft-test.json`
- **Dataset config:** `run_conf/dataset/plancraft-test.yaml`

### 2. BrowseComp-Plus

- **Reference:** Chen et al., "BrowseComp-Plus" (2025), arXiv:2508.06600
- **URL:** https://github.com/wbbbbbz/BrowseComp-Plus
- **Task type:** Multi-hop web-search question answering
- **Total instances:** 830 (full dataset)
- **Instances used:** 100
- **Subset selection:** Fixed 100-instance random sample; sample indices are in the dataset config
- **Expected local path:** `datasets/browsecomp_plus_sampled_100.json`
- **Dataset config:** `run_conf/dataset/browsecomp-plus.yaml`

### 3. Finance Agent

- **Reference:** Bigeard et al., "Finance-Agent" (2025), arXiv:2508.00828
- **URL:** https://github.com/vals-ai/finance-agent
- **Task type:** Multi-step financial reasoning and tool-use
- **Total instances:** 50
- **Instances used:** 50 (full evaluation set)
- **Subset selection:** All instances
- **Expected local path:** `datasets/finance_agent.json`
- **Dataset config template:** `run_conf/dataset/finance-agent.yaml.template`
- **Adapter loader:** `agent_scaling/datasets/finance_agent.py`
- **Setup script:** `bash scripts/setup_finance_agent.sh`
- **Integration:** The repository ships a loader (`FinanceAgentDataset`) plus a setup script that (i) clones the upstream repo, (ii) converts upstream task definitions into the normalized JSON consumed by the loader, and (iii) writes the dataset config. Upstream raw data is not redistributed.

### 4. Workbench

- **Reference:** Styles et al., "WorkBench: A Benchmark Dataset for Agents in a Realistic Workplace Setting" (2024), arXiv:2405.00823
- **URL:** https://github.com/olly-styles/WorkBench
- **Task type:** Common business tool-use tasks (16 tools)
- **Upstream total instances:** 690 (across 6 domain CSVs: analytics, calendar, customer relationship management, email, multi-domain, project management)
- **Instances used in study:** 100
- **Subset selection:** Stratified per-domain sample (16–17 instances per domain) selected by a deterministic shuffle with seed 42 in `scripts/_convert_workbench.py`. Default `setup_workbench.sh` writes both `datasets/workbench.json` (the 100-instance subset, matching the paper) and `datasets/workbench_full_690.json` (the complete upstream set).
- **Expected local path:** `datasets/workbench.json`
- **Dataset config template:** `run_conf/dataset/workbench.yaml.template`
- **Adapter loader:** `agent_scaling/datasets/workbench.py`
- **Setup script:** `bash scripts/setup_workbench.sh`
- **Integration:** The repository ships a loader (`WorkbenchDataset`) plus a setup script that (i) clones the upstream repo, (ii) converts upstream task definitions into the normalized JSON consumed by the loader, applying a deterministic stratified 100-instance subsample by default, and (iii) writes the dataset config. Upstream raw data is not redistributed.

### 5. SWE-bench Verified

- **Reference:** Jimenez et al., "SWE-bench: Can Language Models Resolve Real-world GitHub Issues?" (2024), arXiv:2310.06770
- **URL:** https://www.swebench.com/
- **Task type:** Software engineering; real GitHub issue resolution in Docker sandboxes
- **Total instances:** 500 (verified split)
- **Instances used:** 20
- **Subset selection:** Deterministic shuffle with `seed=42`, first 20 instances taken
- **Expected local path:** `datasets/swebench-verified.json`
- **Dataset config:** `run_conf/dataset/swebench-verified.yaml`
- **Environment:** Requires Docker; benchmark containers pulled automatically on first run

### 6. Terminal-Bench

- **Reference:** Merrill et al., "Terminal-Bench" (2026)
- **URL:** https://www.tbench.ai/
- **Task type:** Terminal / CLI task completion (system administration, security, ML training) in sandboxed Docker environments
- **Total instances:** 86
- **Instances used:** 20
- **Subset selection:** First 20 instances in canonical order
- **Expected local path:** `datasets/terminalbench.json`
- **Dataset config:** `run_conf/dataset/terminalbench.yaml`
- **Environment:** Requires Docker

---

## Licensing

- **PlanCraft:** MIT License (see upstream repository)
- **BrowseComp-Plus:** Apache 2.0 (see upstream repository)
- **Finance Agent:** See upstream repository for license terms
- **Workbench:** See upstream repository for license terms
- **SWE-bench Verified:** MIT License (see upstream repository)
- **Terminal-Bench:** See https://www.tbench.ai/ for license terms

All benchmark data remain the property of their respective authors. This repository does not redistribute benchmark data.

---

## How to Obtain the Data

What is shipped in this repository, and what must be obtained from upstream, depends on the benchmark's licensing and on whether the upstream artifact is a single small JSON file or a larger derived workload:

- **Shipped in the repository under their upstream permissive licenses (no download required):** `datasets/browsecomp_plus_sampled_100.json` (BrowseComp-Plus, Apache 2.0; the fixed 100-instance sample used in the paper) and `datasets/plancraft-test.json` (PlanCraft, MIT; the 100-instance test split). These are the exact files used for the canonical cluster runs reported in the paper.

- **Finance-Agent and WorkBench: acquisition is automated.** Run the documented setup scripts (`scripts/setup_finance_agent.py` and `scripts/setup_workbench.py`) once; each script clones the upstream repository, converts upstream task definitions into the normalized JSON consumed by the loader (applying the deterministic stratified 100-instance subsample for WorkBench), and writes the dataset config. Upstream raw data are not redistributed; users obtain them through their upstream sources, mediated by the setup scripts. No manual JSON placement is required.

- **SWE-bench Verified and Terminal-Bench:** download each benchmark from its original source (URLs above) and place the resulting JSON files under `datasets/` at the expected local paths listed in the sections above. Docker images for these benchmarks are pulled automatically from public registries on first run; users need a working Docker installation. Upstream raw data are not redistributed.
