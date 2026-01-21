# Agent Scaling

A framework for studying scaling behaviors of LLM-based single-agent and multi-agent systems on complex reasoning tasks.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ybkim95/agent-scaling.git
cd agent-scaling

# Install dependencies
uv sync --prerelease=allow

# Install flash-attn (needed for BrowseComp+ environment)
uv pip install --no-build-isolation flash-attn

# Activate the virtual environment
source .venv/bin/activate
```

### Setting Environment Variables

Create a `.env` file with your LLM API keys. See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for supported providers.

```bash
# Required: At least one LLM provider API key
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-gemini-key"
ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: LangFuse for LLM call tracing
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_SECRET_KEY="your-secret-key"
LANGFUSE_PUBLIC_KEY="your-public-key"
```

## Running Experiments

### Basic Usage

Run an experiment with default configuration:

```bash
python run_scripts/run_experiment.py
```

Run in debug mode (processes fewer instances):

```bash
python run_scripts/run_experiment.py debug=true
```

### Configuring Experiments

The framework uses [Hydra](https://hydra.cc/docs/intro/) for configuration management. Override parameters via command line:

```bash
# Run single-agent on PlanCraft dataset
python run_scripts/run_experiment.py agent=single-agent dataset=plancraft-test

# Run multi-agent centralized system
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test

# Run with different LLM
python run_scripts/run_experiment.py llm.model=gpt-4o-mini

# Run with parallel workers
python run_scripts/run_experiment.py num_workers=4

# Process more instances
python run_scripts/run_experiment.py max_instances=10
```

### Available Configurations

#### Agent Types

| Agent | Config Name | Description |
|-------|-------------|-------------|
| Single Agent | `single-agent` | Single LLM agent with tool use |
| Multi-Agent Centralized | `multi-agent-centralized` | Orchestrated multi-agent system with lead agent |
| Multi-Agent Decentralized | `multi-agent-decentralized` | Peer-to-peer multi-agent coordination |
| Multi-Agent Hybrid | `multi-agent-hybrid` | Hybrid coordination approach |
| Multi-Agent Independent | `multi-agent-independent` | Independent parallel agents |

#### Datasets

| Dataset | Config Name | Description |
|---------|-------------|-------------|
| BrowseComp+ | `browsecomp-plus` | Web browsing comprehension tasks |
| PlanCraft | `plancraft` | Minecraft crafting planning tasks |
| Workbench | `workbench` | Tool use benchmark tasks |
| FinanceAgent | `finance-agent` | Financial reasoning tasks |

## Example Experiments

### Single-Agent on PlanCraft

```bash
python run_scripts/run_experiment.py \
    agent=single-agent \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Multi-Agent Centralized on PlanCraft

```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Multi-Agent Centralized on BrowseComp+

```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=browsecomp-plus \
    llm.model=gpt-4o-mini \
    max_instances=5
```

### Scaling Number of Agents

The multi-agent centralized system supports configuring the number of agents:

```bash
# Run with 5 agents
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    agent.n_base_agents=5 \
    dataset=plancraft-test

# Run with 10 agents
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    agent.n_base_agents=10 \
    dataset=plancraft-test
```

## Output Structure

Experiment outputs are saved to `exp_outputs/{dataset}/{agent}/{model}/{date}/{time}/`:

```
exp_outputs/
└── plancraft-test/
    └── multi-agent-centralized/
        └── gemini/
            └── gemini-2.0-flash/
                └── 2025-01-21/
                    └── 12-30-45/
                        ├── run_config.yaml        # Experiment configuration
                        ├── run.log                # Detailed execution logs
                        ├── dataset_eval_metrics.json  # Aggregated metrics
                        └── instance_runs/         # Per-instance outputs
                            ├── 0000/
                            ├── 0001/
                            └── ...
```

### Output Files

- **`run_config.yaml`**: Full configuration used for the experiment
- **`run.log`**: Detailed logs including prompts, LLM responses, and tool calls
- **`dataset_eval_metrics.json`**: Aggregated evaluation metrics
  ```json
  {
    "avg_success": 0.85,
    "avg_num_steps": 7.2,
    "num_instances": 100
  }
  ```

## Example Output

See `example_outputs/` directory for sample experiment outputs demonstrating:
- Single-agent execution traces
- Multi-agent coordination logs
- Evaluation metrics

## Project Structure

```
agent-scaling/
├── agent_scaling/           # Main Python package
│   ├── agents/              # Agent implementations
│   │   ├── single_agent.py
│   │   ├── multiagent_centralized.py
│   │   └── ...
│   ├── datasets/            # Dataset loaders
│   ├── env/                 # Environment & tools
│   ├── llm/                 # LLM integration
│   └── config/              # Configuration classes
├── run_scripts/             # Entry points
│   └── run_experiment.py
├── run_conf/                # Hydra configurations
│   ├── agent/               # Agent configs
│   ├── dataset/             # Dataset configs
│   └── run_exp.yaml         # Master config
├── datasets/                # Dataset files
├── prompts/                 # Prompt templates
└── example_outputs/         # Sample outputs
```

## Configuration Reference

### Master Config (`run_conf/run_exp.yaml`)

```yaml
defaults:
  - agent: multi-agent-centralized  # Agent type
  - dataset: plancraft-test         # Dataset

llm:
  model: gemini/gemini-2.0-flash    # LLM model
  params:
    temperature: 0.0                # Generation temperature

log_langfuse: false                 # Enable LangFuse tracing
use_disk_cache: true                # Cache LLM calls
num_workers: 1                      # Parallel workers
debug: true                         # Debug mode
max_instances: 3                    # Max instances to process
```

### Multi-Agent Config (`run_conf/agent/multi-agent-centralized.yaml`)

```yaml
name: multi-agent-centralized
n_base_agents: 3                    # Number of agents
min_iterations_per_agent: 3         # Min iterations per agent
max_iterations_per_agent: 10        # Max iterations per agent
max_rounds: 5                       # Max orchestration rounds
communication:
  strategy: orchestrated            # Communication strategy
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kim2025towards,
  title={Towards a science of scaling agent systems},
  author={Kim, Yubin and Gu, Ken and Park, Chanwoo and Park, Chunjong and Schmidgall, Samuel and Heydari, A Ali and Yan, Yao and Zhang, Zhihan and Zhuang, Yuchen and Malhotra, Mark and others},
  journal={arXiv preprint arXiv:2512.08296},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
