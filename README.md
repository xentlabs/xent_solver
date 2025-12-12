# Xentlabs Solver

A generic token-optimization solver designed for "hacking" models using Gradient Coordinate Gradient (GCG) strategies.

## 1. Install & Setup

- Install Python 3.13 (managed automatically by `uv`).
- Clone the repo and install dependencies:
  ```bash
  uv sync
  ```
- Run scripts through `uv run …` to use the project environment.

## 2. Quick Start

An example is provided in `example.py`. It demonstrates how to optimize a prefix to satisfy a specific objective (in this case, a "condense" task).

Run it with:
```bash
uv run example.py
```

This will:
1. Load the configuration from `configs/gcg/test.json`.
2. Load the target data from `data/condense/test.json`.
3. Optimize a prefix using the GCG strategy.
4. Save results to `outputs/condense_program`.

## 3. How it Works

The solver optimizes a set of variables to improve a loss built from **compositions** (weighted cross-entropy objectives).

### The Workflow
1. **Define Data**: A JSON file containing the text targets and constraints (see `data/condense/test.json`).
2. **Build a Spec**: A Python function that translates the data into a `ProgramSpecTemplate`. This defines the variables (e.g., the prefix) and the objective function (e.g., maximize cross-entropy).
3. **Run**: The `run_generic` function handles the optimization loop, distributing work across GPUs if available.

### Building a Spec

The `ProgramSpecTemplate` defines the optimization problem using symbols and compositions.

#### 1. Define Symbols
First, define the parts of your sequence. Symbols can be **Fixed** (text/ids) or **Variable** (optimized tokens).
```python
s = Symbols({
    "prefix": Variable(20),                # 20 learnable tokens
    "prompt": Fixed("Tell me a joke: "),   # Constant text
    "target": Fixed("Why did the..."),     # Constant text
})
```

#### 2. Define Objectives (Compositions)
Objectives are built using helper functions that represent cross-entropy terms.
- `xent(seq, ctx)`: **Minimize** Cross-Entropy of `seq` given `ctx`. (Standard language modeling loss).
- `nex(seq, ctx)`: **Maximize** Cross-Entropy (Negative XEnt). Used to make text *less* likely.
- `xed(seq, ctx)`: Difference between `xent(seq)` (prior) and `xent(seq | ctx)`. Maximizing this maximizes the **Mutual Information** or "condensing" power.
- `dex(seq, ctx)`: The reverse of `xed` (difference between conditional and prior).

**Examples:**

*Standard "Jailbreak" (Maximize probability of target given prompt+prefix):*
```python
# We want to minimize Loss(target | prompt + prefix)
# Which is equivalent to Maximizing nex(target, prompt + prefix)
objective = nex(s["target"], s["prompt"] + s["prefix"])
```

*Condense (Find a prefix that summarizes the text):*
```python
# Maximize Mutual Information: xent(text) - xent(text | prefix)
# This uses the 'xed' helper which returns [xent(text), nex(text, prefix)]
objective = xed(s["text"], s["prefix"])
```

#### 3. Combine and Return
You can combine multiple terms linearly.
```python
# Minimize loss on target1 but also keep target2 unlikely
objective = nex(s["target1"], s["prefix"]) + 0.5 * xent(s["target2"], s["prefix"])
return ProgramSpecTemplate(objective=objective, goal="maximize")
```

## 4. Configuration

The optimization behavior is controlled by JSON config files (e.g., `configs/gcg/test.json`). Key parameters include:

- `model_names`: List of models to optimize against.
- `top_k`: Number of candidate tokens to consider per position based on gradient information.
- `batch_size`: (Implicitly handled) affects memory usage and search quality.
- `selection`: Parameters for the selection strategy (e.g., epsilon-greedy).

## 5. Directory Structure

- `src/`: Core solver logic.
  - `dsl/`: Domain Specific Language for defining programs (Spec, Symbols, Builder).
  - `engine/`: Optimization engine and task management.
  - `strategies/`: Optimization algorithms (e.g. GCG).
  - `data/`: Data loading and constraint policies.
  - `utils/`: Utilities for models, logging, and hardware.
