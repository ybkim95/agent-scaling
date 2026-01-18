# Agent Scaling

## 🚀 Quick Start

Agent scaling uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies (e.g. python version, libraries, etc). After installing uv, run the following command to install dependencies.
```bash
# install dependencies (creates virtual environment) and sync to latest environment
uv sync --prerelease=allow

uv pip install --no-build-isolation flash-attn # Needed for faiss (browsecomp-plus environment)
# activate venv environment
source .venv/bin/activate
```

In particular, we use [langchain](https://python.langchain.com/docs/introduction/) for using LLMs with tools, [litellm](https://docs.litellm.ai/docs/), and [langfuse](https://langfuse.com/docs) for tracking llm calls.

### Setting the envrionment variables
Next, set the relevant LLM api keys in `.env ` environment variable file to use LLMs.
See [litellm documentation](https://docs.litellm.ai/docs/providers) for the environment variable for the corresponding provider.

``` .env
DEEPSEEK_API_KEY = ""
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""
ANTHROPIC_API_KEY = ""
```

To use langfuse for LLM tracing, we can also set the following variables
```
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_SECRET_KEY=""
LANGFUSE_PUBLIC_KEY=""
```

### Benchmarks

For the evaluations, we have used the following state-of-the-art benchmarks:

1. BrowseComp-Plus (https://github.com/texttron/BrowseComp-Plus)
2. Finance-Agent (https://www.vals.ai/benchmarks/finance_agent)
3. Plancraft (https://github.com/gautierdag/plancraft)
4. WorkBench (https://github.com/olly-styles/WorkBench)

### Quickstart
To run a LLM with tracing, we use a custom langchain BaseChatModel class.
```python
from agent_scaling.llm import ChatLiteLLMLC

llm = ChatLiteLLMLC(
        model="gemini/gemini-2.0-flash",
        temperature=0.1,
        log_langfuse=False, # logs to langfuse if set to True
    )
response = llm.invoke([{"role": "user", "content": "Hello, what LLM are you?"}])

# print the response
response.pretty_print()
print(response.model_dump_json(indent=2))
```


We use [hydra](https://hydra.cc/docs/intro/) to manage our experiment parameters. Try running a basic experiment (after activating the environment) with:
```bash
python run_scripts/run_experiment.py debug=true
```

## TODO
- [ ] function to specify which samples to run (should just preprocess this)
- [ ] a way to adapt the input arguments to a tool
- [ ] a shared output for the tool
- [ ] ablation runs
