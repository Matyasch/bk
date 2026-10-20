from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from causallearn.utils.cit import CIT
import networkx as nx
import numpy as np
from tqdm import tqdm

from algorithms import ALGORITHMS
from generate_experiment import generate_experiment


class CountingTest:
    """
    Wrapper for CI tests that counts the number of tests done.
    """

    def __init__(
        self,
        data: np.ndarray,
        ci_test: str,
        true_dag: nx.DiGraph | None = None,
        **kwargs,
    ):
        self.cit = CIT(data, ci_test, true_dag=true_dag)
        self.method = self.cit.method
        self.cit.true_dag = true_dag
        self.tests_done = defaultdict(set)

    def __call__(
        self, X: int, Y: int, condition_set: list[int] | None = [], *args, **kwargs
    ) -> float:
        """
        Run the conditional independence test.

        Args:
            X (int): The first variable.
            Y (int): The second variable.
            condition_set (list[int] | None): The conditioning set.

        Returns:
            float: The p-value of the CI test.
        """
        condition_set = condition_set or []
        self.tests_done[frozenset((X, Y))] |= {tuple(condition_set)}
        return self.cit(X, Y, condition_set)

    def get_tests_per_order(self) -> np.ndarray:
        """
        Get the number of tests done per order.

        Returns:
            np.ndarray: The number of tests done per order.
        """
        num_nodes = self.cit.data.shape[1]
        cond_sets = self.tests_done.values()
        if not cond_sets:
            return np.zeros(num_nodes, dtype=int)
        orders, test_num = np.unique(
            [len(cond) for conds in cond_sets for cond in conds],
            return_counts=True,
        )
        tests_per_order = np.zeros(num_nodes, dtype=int)
        tests_per_order[orders] = test_num
        return tests_per_order


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp", type=int, default=100, help="Number of experiments.")
    parser.add_argument("--model", type=str, default="synthetic", help="Model name.")
    parser.add_argument("--nodes", type=int, default=100, help="Number of nodes.")
    parser.add_argument(
        "--exp-degree", type=float, default=3.0, help="Expected degree."
    )
    parser.add_argument("--max-degree", type=int, default=10, help="Maximum degree.")
    parser.add_argument("--targets-num", type=int, default=2, help="Number of targets.")
    parser.add_argument(
        "--targets-type",
        type=str,
        default="ancestral",
        help="Type of target selection.",
        choices=["random", "ancestral", "identifiable", "explicit"],
    )
    parser.add_argument(
        "--citest",
        type=str,
        help="Conditional independence test.",
        choices=["d_separation", "fisherz", "gsq"],
    )
    parser.add_argument("--samples", type=int, default=0, help="Number of samples.")
    parser.add_argument("--binary", action="store_true", help="Generate binary data.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance level.")
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS.keys(),
        help="Causal discovery algorithm.",
    )
    parser.add_argument(
        "--bk-rate",
        type=float,
        help="Background knowledge rate, overrides other rates if > 0.",
        default=0.0,
    )
    parser.add_argument(
        "--adj-rate",
        type=float,
        help="Adjacency background knowledge rate.",
        default=0.0,
    )
    parser.add_argument(
        "--gap-rate",
        type=float,
        help="Gap background knowledge rate.",
        default=0.0,
    )
    parser.add_argument(
        "--ori-rate",
        type=float,
        help="Orientation background knowledge rate.",
        default=0.0,
    )
    parser.add_argument(
        "--bk-error",
        type=float,
        help="Error rate of background knowledge.",
        default=0.0,
    )
    parser.add_argument(
        "--adj-error",
        type=float,
        help="Error rate of adjacency background knowledge.",
        default=0.0,
    )
    parser.add_argument(
        "--gap-error",
        type=float,
        help="Error rate of gap background knowledge.",
        default=0.0,
    )
    parser.add_argument(
        "--ori-error",
        type=float,
        help="Error rate of orientation background knowledge.",
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed.")
    args = parser.parse_args(sys.argv[1:])
    return args


def run_algorithm(
    id: str,
    data: np.ndarray,
    targets: list[int],
    algorithm: str,
    ci_test: str,
    alpha: float,
    **kwargs,
):
    """
    Runs the specified algorithm on the provided experimental data.

    Args:
        id (str): The experiment ID.
        data (np.ndarray): The experimental data.
        targets (list[int]): The target variables.
        algorithm (str): The algorithm to run.
        ci_test (str): Name of the conditional independence test to use.
        alpha (float): The significance level for the CI tests.
        filter (bool): Whether to pre-filter with SNAP(0).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: The result of the algorithm and other statistics.
    """
    ci_test = CountingTest(data, ci_test, **kwargs)
    start = perf_counter()
    # Run algorithm
    result = ALGORITHMS[algorithm](
        data=data,
        ci_test=ci_test,
        alpha=alpha,
        targets=targets,
        **kwargs,
    )
    # Log statistics
    result["time"] = perf_counter() - start
    result["id"] = id
    result["targets"] = np.asarray(targets).tolist()
    result["tests"] = ci_test.get_tests_per_order().tolist()
    return result


def create_run_directory(base_dir: str = "results") -> tuple[str, str]:
    """
    Create a new run directory with timestamp-based ID.

    Args:
        base_dir (str): Base directory for all runs.

    Returns:
        tuple[str, str]: Run directory path and run ID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(base_dir, run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def save_config(run_dir: str, config: dict) -> None:
    """
    Save experiment configuration to config.json.

    Args:
        run_dir (str): The run directory.
        config (dict): Configuration dictionary.
    """
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def log_result(run_dir: str, result: dict) -> None:
    """
    Append a result to results.jsonl (newline-delimited JSON).

    Args:
        run_dir (str): The run directory.
        result (dict): Result dictionary to log.
    """
    results_path = os.path.join(run_dir, "results.jsonl")
    with open(results_path, "a") as f:
        json.dump(result, f, default=str)
        f.write("\n")


def experiment(args, s: int) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """
    exp = generate_experiment(
        model=args.model,
        nodes=args.nodes,
        exp_degree=float(args.exp_degree),
        max_degree=args.max_degree,
        binary=args.binary,
        targets_num=args.targets_num,
        targets_type=args.targets_type,
        samples_num=args.samples,
        bk_rate=args.bk_rate,
        adj_rate=args.adj_rate,
        gap_rate=args.gap_rate,
        ori_rate=args.ori_rate,
        bk_error=args.bk_error,
        adj_error=args.adj_error,
        gap_error=args.gap_error,
        ori_error=args.ori_error,
        seed=args.seed + s,
    )
    if args.algorithm is None:
        return {}
    return run_algorithm(
        algorithm=args.algorithm,
        ci_test=args.citest,
        alpha=args.alpha,
        **exp,
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    algorithm = args.algorithm

    # Create run directory and save configuration
    run_dir, run_id = create_run_directory()
    save_config(run_dir, vars(args))
    print(f"Logging results to: {run_dir}")

    # Run experiments
    workers = 1 if args.citest in ["fisherz", "kci"] else 16
    processes = []
    with ProcessPoolExecutor(workers) as exec:
        for s in range(args.exp):
            process = exec.submit(experiment, args, s)
            processes.append(process)

        if algorithm is None:  # Only generating experiments
            for p in tqdm(
                as_completed(processes),
                total=len(processes),
                desc="Generated experiment",
            ):
                pass
        else:  # Also running algorithms
            for p in tqdm(
                as_completed(processes),
                total=len(processes),
                desc="Completed algorithm",
            ):
                log_result(run_dir, p.result())

