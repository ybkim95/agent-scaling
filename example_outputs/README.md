# Example Outputs

This directory contains sample experiment configurations, results, and execution traces demonstrating the framework's capabilities.

## Directory Structure

```
example_outputs/
├── plancraft-test/
│   └── multi-agent-centralized/    # Multi-agent on PlanCraft task
│       ├── run_config.yaml         # Experiment configuration
│       ├── run_trace.txt           # Execution trace with prompts and responses
│       └── dataset_eval_metrics.json  # Results
└── browsecomp-plus/
    ├── single-agent/               # Single agent on BrowseComp+ task
    │   ├── run_config.yaml
    │   └── dataset_eval_metrics.json
    └── multi-agent-centralized/    # Multi-agent on BrowseComp+ task
        ├── run_config.yaml
        └── dataset_eval_metrics.json
```

## Output Files

### `run_config.yaml`
Complete configuration used for the experiment, including agent type, dataset, LLM model, and all hyperparameters.

### `run_trace.txt`
Detailed execution trace including:
- System and user prompts sent to the LLM
- LLM responses with token counts
- Tool calls and their results
- Agent coordination messages (for multi-agent systems)

### `dataset_eval_metrics.json`
Aggregated evaluation metrics:
```json
{
  "avg_success": 0.82,      // Average task success rate (PlanCraft)
  "avg_accuracy": 0.60,     // Average accuracy (BrowseComp+)
  "num_instances": 100      // Number of instances evaluated
}
```

## Reproducing These Results

### PlanCraft Multi-Agent
```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash
```

### BrowseComp+ Single-Agent
```bash
python run_scripts/run_experiment.py \
    agent=single-agent \
    dataset=browsecomp-plus \
    llm.model=openai/gpt-5-mini
```

### BrowseComp+ Multi-Agent
```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=browsecomp-plus \
    llm.model=openai/gpt-5
```

Note: Exact results may vary due to LLM non-determinism even with temperature=0.
