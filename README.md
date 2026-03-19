# Evo-Memory + Ganglion: Biological Memory for LLM Agent Benchmarking

A fork of [Evo-Memory](https://github.com/zhaosnw/evo_mem) that adds:

1. **Ganglion Agent** — plugs [ganglion-memory](https://github.com/tensorlink-dev/ganglion-memory)'s biological learning (Hebbian strengthening, apoptosis, lateral inhibition, salience gating, consolidation) into the Evo-Memory evaluation harness
2. **Chutes AI Backend** — run evals cheaply on decentralised GPU via [Chutes](https://chutes.ai) (DeepSeek-R1, Llama, Qwen, etc.)
3. **Head-to-head comparison tooling** — scripts to run Ganglion vs ExpRAG vs History baseline and plot cumulative accuracy curves

## What This Fork Adds

| Component | File | Description |
|-----------|------|-------------|
| Chutes LLM | `evo_memory/llm/chutes_llm.py` | OpenAI-compatible client pointed at `api.chutes.ai/v1` |
| Ganglion Agent | `evo_memory/agents/ganglion_agent.py` | Hybrid agent: ganglion beliefs + Evo-Memory embedding retrieval |
| Configs | `configs/*.yaml` | Ready-to-run configs for Ganglion, ExpRAG, History on MMLU-Pro |
| Comparison runner | `scripts/run_comparison.sh` | Runs all 3 agents head-to-head |
| Result analyser | `scripts/compare_results.py` | Comparison tables + cumulative accuracy curves |

## Installation

```bash
pip install -e .
pip install ganglion-memory openai>=1.0 sentence-transformers
```

## Quick Start

```bash
# Set your Chutes API key
export CHUTES_API_KEY="cpk_..."

# Smoke test: 10 tasks, 1 stream
python -m evo_memory.main run \
    --agent ganglion \
    --dataset mmlu_pro \
    --backend chutes \
    --model "deepseek-ai/DeepSeek-R1" \
    --task-limit 10

# Full comparison: Ganglion vs ExpRAG vs History
bash scripts/run_comparison.sh

# Analyse results
python scripts/compare_results.py results/comparison_*/
```

## Architecture

### How Ganglion Maps to the (F, U, R, C) Tuple

Evo-Memory formalises memory-augmented agents as a tuple **(F, U, R, C)**:

```
Standard Evo-Memory Agent:
  F = LLM (e.g. DeepSeek-R1 via Chutes)
  U = append experience to vector store
  R = top-k cosine similarity retrieval
  C = concatenate retrieved experiences into prompt

GanglionAgent (hybrid mode):
  F = LLM (same)
  U = DUAL update:
      ├─ ganglion.assimilate(task, response, success)
      │   → Hebbian strengthening (success ↑ confidence)
      │   → Contradiction apoptosis (conflicting beliefs die)
      │   → Lateral inhibition (one strategy wins, rivals weaken)
      │   → Salience gating (surprising results encode stronger)
      └─ evo_memory.evolve(state) → standard embedding store
  R = DUAL retrieval:
      ├─ ganglion.recall(query) → strategy-level beliefs
      │   ("batch 32 works", "avoid greedy decoding")
      └─ embedding_retriever.retrieve(query) → example-level memories
          ("here's a similar MMLU question I solved")
  C = combined context:
      ├─ "STRATEGIES THAT HAVE WORKED: ..."
      ├─ "STRATEGIES THAT HAVE FAILED: ..."
      └─ "RELEVANT EXPERIENCE FROM SIMILAR TASKS: ..."
```

Between evaluation streams, `ganglion.forget()` runs consolidation (merge similar beliefs, evict weak ones).

### In-Memory SQLite

Each eval stream gets `db_path=":memory:"` so there's no cross-contamination. This correctly measures learning *within* a stream.

## Interpreting Results

The key chart is the **cumulative accuracy curve** — accuracy at task N across the stream:

- A **rising curve** means the agent is learning from experience
- **Ganglion rising faster** than ExpRAG means biological memory helps the agent learn strategies more efficiently
- **Ganglion-only** (hybrid_mode=False) vs **hybrid** shows whether embedding retrieval adds value on top of Hebbian beliefs

The comparison script outputs:
- `comparison_summary.json` with per-agent accuracy mean/std
- `cumulative_accuracy.png` (if matplotlib installed) with the learning curves
- ASCII table printed to stdout

## Ganglion-Specific Metrics

The GanglionAgent captures additional signals per task step (saved in stream results):

| Metric | Description |
|--------|-------------|
| `num_beliefs` | Total beliefs in ganglion memory at this step |
| `num_contradictions` | Contradictions detected (triggers apoptosis) |
| `num_apoptosis` | Beliefs that died from contradiction |
| `confidence_distribution` | Distribution of belief confidences |
| `beliefs_injected` | Whether ganglion context was injected (vs empty) |

## Original Evo-Memory

### Agents

| Agent | Description |
|-------|-------------|
| **GanglionAgent** | **Biological memory with Hebbian learning (NEW)** |
| ExpRAG | Experience retrieval-augmented generation |
| ExpRecent | Recency-based experience retrieval |
| ReMem | Think-Act-Refine loop |
| ReAct | Reasoning and acting with memory |
| A-mem | Experience accumulation |
| Self-RAG | Self-reflection with retrieval |
| Mem0 | Hierarchical memory |
| LangMem | Language-based memory management |
| DynamicCheatsheet | Dynamic knowledge aggregation |
| AWM | Agent workflow memory |

### Datasets

**Single-Turn:** MMLU-Pro, GPQA, AIME 2024/2025, ToolBench
**Multi-Turn:** AlfWorld, BabyAI, PDDL/Blocksworld, ScienceWorld

### Evaluation Metrics

- **Answer Accuracy**: Correctness for single-turn tasks
- **Success Rate**: Task completion for multi-turn tasks
- **Progress Rate**: Partial progress measurement
- **Step Efficiency**: Steps taken for successful multi-turn tasks
- **Learning Improvement**: Late accuracy minus early accuracy

## Configuration

### YAML Config

```yaml
name: ganglion_chutes_mmlu
agent_type: ganglion
dataset_type: mmlu_pro
llm_backend: chutes
model_name: "deepseek-ai/DeepSeek-R1"
num_streams: 3
task_limit: 100
agent_kwargs:
  hybrid_mode: true
  db_path: ":memory:"
  capability: "answering MMLU-Pro multiple choice questions"
  store_successful_only: false
```

### Python API

```python
from evo_memory import ExperimentConfig, ExperimentRunner, AgentType, DatasetType, LLMBackend

config = ExperimentConfig(
    name="ganglion_test",
    agent_type=AgentType.GANGLION,
    dataset_type=DatasetType.MMLU_PRO,
    llm_backend=LLMBackend.CHUTES,
    model_name="deepseek-ai/DeepSeek-R1",
    num_streams=3,
    agent_kwargs={"hybrid_mode": True, "db_path": ":memory:"},
)

runner = ExperimentRunner(config)
results = runner.run()
print(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
```

## License

MIT License
