# Integrating Background Knowledge for Scalable Causal Discovery

Paper: https://arxiv.org/abs/2607.10456 \
Website: https://matyasch.github.io/bk

This repository contains the code to reproduce all results presented in the paper "Integrating Background Knowledge for Scalable Causal Discovery", accepted to NeurIPS 2026.

> [!NOTE]
> This branch contains all code to reproduce the results presented in the paper. Check out the [main branch](https://github.com/Matyasch/bk/tree/main) for a minimal and portable implementation.

## Repository Structure

The codebase is organized as follows:

- **`main.py`**: Main entry point for running experiments. Handles experiment generation, algorithm execution, and result logging.
- **`generate_experiment.py`**: Functions for generating synthetic experiments and realistic experiments from various causal graph models.
- **`evaluate.py`**: Evaluation utilities for analyzing experimental results, including computing metrics, adjustment sets, and intervention distances.
- **`algorithms/`**: Directory containing implementations of various causal discovery algorithms.
- **`experiments/`**: Cache directory where generated experimental data (graphs, samples, background knowledge) is stored to enable reuse across runs.
- **`results/`**: Output directory where experiment results are saved in timestamped subdirectories.
- **`requirements.txt`**: Python dependencies required to run the experiments.

When you run experiments, results are automatically saved in the `results/` directory with the following structure:

```
results/
  run_YYYYMMDD_HHMMSS/
    config.json          # Experiment configuration parameters
    results.jsonl        # Experiment results (one JSON object per line)
```

### Algorithms

Each algorithm that uses background knowledge (BK) during its search is implemented in the same file as its post-processing baseline (suffix `_pp`), which runs the same search without BK and only applies orientation BK to the output. The names below are the values accepted by `--algorithm`.

- [`algorithms/pc.py`](algorithms/pc.py): Algorithms related to PC
  - `pc`: PC-BK algorithm
  - `pc_pp`: PC with post-processing
  - `cl_pc`, `cl_pc_stable`: PC and PC-stable as implemented in [causal-learn](https://github.com/py-why/causal-learn), with BK passed through its interface. causal-learn cannot express adjacencies without an orientation, so these only receive gaps and orientations.
  - `pcalg_pc`, `pcalg_pc_stable`: PC and PC-stable as implemented in the R package [pcalg](https://cran.r-project.org/package=pcalg), with BK passed as fixed edges, fixed gaps and added orientations. The R side is in [`algorithms/pcalg_pc.R`](algorithms/pcalg_pc.R) and requires the optional R dependencies (see [Installation](#installation)).
- [`algorithms/snap.py`](algorithms/snap.py): Algorithms related to SNAP
  - `snap`: SNAP($\infty$)-BK
  - `snap_pp`: SNAP($\infty$) with post-processing
- [`algorithms/load.py`](algorithms/load.py): Algorithms related to LOAD
  - `load`: LOAD-BK
  - `load_pp`: LOAD with post-processing
- [`algorithms/mb_by_mb_plus.py`](algorithms/mb_by_mb_plus.py): Algorithms related to MB-by-MB, extended with the checks needed to return adjustment sets for a pair of targets
  - `mb_by_mb_plus`: MB-by-MB-BK
  - `mb_by_mb_pp`: MB-by-MB with post-processing
- [`algorithms/ldecc_plus.py`](algorithms/ldecc_plus.py): Algorithms related to LDECC, extended with the checks needed to return adjustment sets for a pair of targets
  - `ldecc_plus`: LDECC-BK
  - `ldecc_pp`: LDECC with post-processing
- [`algorithms/orient.py`](algorithms/orient.py): Shared orientation steps (v-structures, Meek's rules, and applying BK to a CPDAG), not an algorithm itself

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The `pcalg_pc` and `pcalg_pc_stable` baselines additionally require R with the packages `pcalg`, `graph` and `RBGL`, and the Python package `rpy2`. All other algorithms run without them.

```bash
Rscript -e 'install.packages("BiocManager"); BiocManager::install(c("graph", "RBGL")); install.packages("pcalg")'
pip install rpy2
```

## Usage

The general command structure for running experiments is:

```bash
python3 main.py [OPTIONS]
```

### Key Parameters

- **`--exp`**: Number of experiments to run (default: 100)
- **`--model`**: Model type - `synthetic` for random graphs or specific model names like `ecoli70` (default: `synthetic`)
- **`--nodes`**: Number of nodes in the graph (default: 100)
- **`--samples`**: Number of samples to generate (default: 0)
- **`--citest`**: Conditional independence test. Choices: `d_separation` (oracle), `fisherz` (linear Gaussian data), `gsq` (binary data)
- **`--algorithm`**: Causal discovery algorithm (see [Algorithms](#algorithms)). Choices: `pc`, `snap`, `mb_by_mb_plus`, `ldecc_plus`, `load`, their post-processing baselines `pc_pp`, `snap_pp`, `mb_by_mb_pp`, `ldecc_pp`, `load_pp`, and the external PC implementations `cl_pc`, `cl_pc_stable`, `pcalg_pc`, `pcalg_pc_stable`
- **`--alpha`**: Significance level for CI tests (default: 0.01)
- **Background Knowledge Parameters**:
  - **`--bk-rate`**: Overall background knowledge rate (overrides specific rates if > 0)
  - **`--adj-rate`**: Adjacency background knowledge rate
  - **`--ori-rate`**: Orientation background knowledge rate
  - **`--gap-rate`**: Gap (non-edge) background knowledge rate
- **Error Parameters**:
  - **`--bk-error`**: Overall error rate (overrides specific error rates if > 0)
  - **`--adj-error`**: Error rate for adjacency background knowledge
  - **`--ori-error`**: Error rate for orientation background knowledge
  - **`--gap-error`**: Error rate for gap background knowledge

## Experiment Types

### 1. Synthetic Data Experiments

Randomly generated synthetic causal graphs with varying amounts of background knowledge.

**Configuration:**
- Model: `synthetic` (default)
- Nodes: 100 (default)
- BK-rate: varies in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- All other parameters: default values

**Example command** to run the PC-BK algorithm:
```bash
python3 main.py --nodes 100 --samples 10000 --bk-rate 0.2 --citest fisherz --algorithm pc
```

**Example variations:**
- Different rate of BK: `--bk-rate 0.6`, `--bk-rate 1.0`
- Different BK algorithms: `--algorithm pc`, `--algorithm snap`, `--algorithm load`
- Post-processing baselines: `--algorithm pc_pp`, `--algorithm snap_pp`, `--algorithm load_pp`
- Different CI tests: `--citest fisherz`, `--citest gsq --binary`

### 2. Background Knowledge Type Ablation

Isolates the effect of different types of background knowledge (adjacency, orientation, gap) by varying them independently.

**Configuration:**
- Model: `synthetic` (default)
- Nodes: 100
- BK-rate: 0.0 (disabled)
- Adj-rate / Ori-rate / Gap-rate: varies independently in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- All other parameters: default values

**Example commands:**

Testing adjacency background knowledge:
```bash
python3 main.py --nodes 100 --samples 10000 --adj-rate 0.2 --citest fisherz --algorithm pc
```

Testing orientation background knowledge:
```bash
python3 main.py --nodes 100 --samples 10000 --ori-rate 0.4 --citest fisherz --algorithm pc
```

Testing gap background knowledge:
```bash
python3 main.py --nodes 100 --samples 10000 --gap-rate 0.6 --citest fisherz --algorithm pc
```

### 3. Background Knowledge Error Ablation

Evaluate the robustness of algorithms to errors in the provided background knowledge.

**Configuration:**
- Model: `synthetic` (default)
- Nodes: 100
- BK-rate: varies in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- Fixed errors: `--adj-error 0.1`, `--ori-error 0.1`, `--gap-error 0.05`
- All other parameters: default values

**Example command:**
```bash
python3 main.py --nodes 100 --samples 10000 --bk-rate 0.2 --adj-error 0.1 --ori-error 0.1 --gap-error 0.05 --citest fisherz --algorithm pc
```

**Variations:**
- Run with different `--bk-rate` values while keeping error rates fixed
- Experiment with different error rate combinations

### 4. Realistic Data

Realistic causal graph structures from `bnlearn`.

**Configuration:**
- Model: Specific model name (e.g., `ecoli70`, or other available models)
- CI test: Determined by the model (have to be specified!)
- Nodes: Determined by the model (not specified)
- BK-rate: varies in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- All other parameters: default values

**Example command:**
```bash
python3 main.py --samples 10000 --bk-rate 0.2 --model ecoli70 --citest fisherz --algorithm pc
```

## Analyzing Results

Results are saved in JSON format in the `results/` directory. Each run creates a timestamped directory containing:

1. **`config.json`**: All experiment parameters used for the run
2. **`results.jsonl`**: One line per experiment containing results such as:
   - `id`: Unique experiment identifier
   - `targets`: Target variables
   - `amat`: Learned adjacency matrix
   - `tests`: Number of conditional independence tests performed (by order)
   - `time`: Execution time

You can use the utilities in `evaluate.py` to load and analyze results. Here is how to compute the number of CI tests, execution time, and intervention distance, consistent with the methodology used in the paper (removing the best and worst 5 results).

Imports and run path:

```python
import numpy as np
from evaluate import (
    get_test_nums,
    get_times,
    generate_run_data,
    true_causal_effects,
    generate_samples_from_graphs,
    estimate_ates,
    intervention_distance
)

# Path to your experiment run directory
run_path = "results/run_YYYYMMDD_HHMMSS"
```

### CI Tests and Execution Time

```python
# Get number of CI tests and filter top/bottom 5 outliers
tests = np.sort(get_test_nums(run_path))[5:95]
print(f"CI Tests: {np.mean(tests):.2f} ± {np.std(tests):.2f}")

# Get execution times and filter top/bottom 5 outliers
times = np.sort(get_times(run_path))[5:95]
print(f"Time: {np.mean(times):.2f} ± {np.std(times):.2f}")
```

### Intervention Distance

Calculating intervention distance involves generating samples from the true models and estimating effects:

```python
# 1. Load experimental data
experiments = generate_run_data(run_path)

# 2. Get true effects
true_effects = true_causal_effects(experiments)

# 3. Generate new samples
samples = generate_samples_from_graphs(experiments)

# 4. Estimate effects from experiment with samples
est_effects = estimate_ates(run_path, samples)

# 5. Get intervention distance (filtering outliers)
dist = np.sort(intervention_distance(est_effects, true_effects, "abs"))[5:95]
print(f"Intervention Distance: {np.mean(dist):.2f} ± {np.std(dist):.2f}")
```

## Citation

```bibtex
@inproceedings{schubert2026integrating,
  title={Integrating Background Knowledge for Scalable Causal Discovery},
  author={M{\'a}ty{\'a}s Schubert and Theofanis Aslanidis and Tom Claassen and Sara Magliacane},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```
