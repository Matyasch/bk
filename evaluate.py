from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import combinations, permutations
import json
import logging
import os

import networkx as nx
import numpy as np
from pgmpy.base import DAG, PDAG
from pgmpy.global_vars import logger
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from tqdm import tqdm

from generate_experiment import generate_experiment, generate_samples_from_model
from algorithms.orient import cpdag2mpdag

logger.setLevel(logging.ERROR)


# HELPER FUNCTIONS FOR JSON FILE OPERATIONS
def load_run_config(run_path: str) -> dict:
    """
    Load run configuration from config.json file.

    Args:
        run_path (str): Path to the run directory.

    Returns:
        dict: Configuration dictionary.
    """
    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_run_results(run_path: str, keys: list[str] | None = None) -> list[dict]:
    """
    Load run results from results.jsonl file.

    Args:
        run_path (str): Path to the run directory.
        keys (list[str] | None): Optional list of keys to filter from each result.

    Returns:
        list[dict]: List of result dictionaries.
    """
    results_path = os.path.join(run_path, "results.jsonl")
    results = []
    with open(results_path, "r") as f:
        for line in f:
            result = json.loads(line.strip())
            if keys is not None:
                # Filter to only requested keys (and always include 'id')
                filtered = {k: result.get(k) for k in keys}
                if "id" in result:
                    filtered["id"] = result["id"]
                results.append(filtered)
            else:
                results.append(result)
    return results


# UTILS
def generate_run_data(run_path: str) -> dict:
    """
    Generate experimental data corresponding to a run.

    Args:
        run_path (str): Path to the run directory.
    Returns:
        dict: A dictionary of experiment data with experiment IDs as keys.
    """
    config = load_run_config(run_path)
    proc = []
    with ProcessPoolExecutor() as e:
        for s in range(config["exp"]):
            # Generate data
            p = e.submit(
                generate_experiment,
                model=config["model"],
                nodes=config["nodes"],
                exp_degree=float(config["exp_degree"]),
                max_degree=config["max_degree"],
                binary=config["binary"],
                targets_num=config["targets_num"],
                targets_type=config["targets_type"],
                samples_num=config["samples"],
                bk_rate=float(config["bk_rate"]),
                adj_rate=float(config["adj_rate"]),
                ori_rate=float(config["ori_rate"]),
                gap_rate=float(config["gap_rate"]),
                bk_error=float(config["bk_error"]),
                adj_error=float(config["adj_error"]),
                gap_error=float(config["gap_error"]),
                ori_error=float(config["ori_error"]),
                seed=config["seed"] + s,
            )
            proc.append(p)
    data = {}
    for p in tqdm(as_completed(proc), desc="Getting run data", total=len(proc)):
        data[p.result()["id"]] = p.result()
    return data


def generate_samples_from_graphs(
    experiments: dict, samples_num: int = int(1e4), seed: int = 0
) -> dict:
    """
    Generate new samples from the true DAGs of all experiments.
    Args:
        experiments (dict): Dictionary with experimental data with their IDs as keys.
        samples_num (int): The number of samples to generate for each experiment.
        seed (int): The random seed for reproducibility.
    Returns:
        dict: A dictionary of samples with experiment IDs as keys.
    """
    samples = {}

    proc = []
    with ProcessPoolExecutor() as e:
        for exp_id, exp in experiments.items():
            p = e.submit(
                generate_samples_from_model, exp["true_dag"], samples_num, seed
            )
            p.exp_id = exp_id
            proc.append(p)
        for p in tqdm(as_completed(proc), desc="Generating samples", total=len(proc)):
            samples[p.exp_id] = p.result()
    return samples


def dag2cpdag(dag: DAG) -> PDAG:
    """
    Convert a DAG to its corresponding CPDAG with same order of nodes.

    Args:
        dag (DAG): The input DAG.
    Returns:
        PDAG: The corresponding CPDAG.
    """
    pdag = PDAG()
    pdag.add_nodes_from(dag.nodes())
    pdag.add_edges_from(dag.to_pdag().edges())
    return pdag


def dag2mpdag(dag: DAG, bk: np.ndarray | None = None) -> PDAG:
    """
    Convert a DAG to its corresponding MPDAG with same order of nodes
    using the given background knowledge.

    Args:
        dag (DAG): The input DAG.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        PDAG: The corresponding CPDAG.
    """
    cpdag = dag2cpdag(dag)
    if bk is None:
        return cpdag
    # Convert CPDAG to causal-learn format
    cpdag_m = -nx.to_numpy_array(cpdag)
    cpdag_m[(cpdag_m == 0) & (cpdag_m.T == -1)] = 1
    mpdag_m = cpdag2mpdag(cpdag_m, bk)
    # Convert back to betworkx format
    amat = np.zeros_like(mpdag_m, dtype=int)
    amat[mpdag_m == -1] = 1
    mpdag = PDAG()
    mpdag.add_nodes_from(dag.nodes())
    mpdag.add_edges_from(nx.DiGraph(amat).edges())
    return mpdag


def get_true_cpdags(experiments: dict) -> dict:
    """
    Get the true CPDAGs for all experiments.
    Args:
        experiments (dict): Dictionary with experimental data with their IDs as keys.
    Returns:
        dict: A dictionary of CPDAGs with experiment IDs as keys.
    """
    cpdags = {}
    proc = []
    with ProcessPoolExecutor() as e:
        for exp_id in experiments:
            dag = nx.to_numpy_array(experiments[exp_id]["true_dag"])
            p = e.submit(dag2cpdag, dag)
            p.exp_id = exp_id
            proc.append(p)
        for p in tqdm(as_completed(proc), desc="Generating CPDAGs", total=len(proc)):
            cpdags[p.exp_id] = p.result()
    return cpdags


# TESTS & TIME
def get_test_nums(run_path: str) -> np.ndarray:
    """
    Get the number of CI tests performed for each experiment in a run.

    Args:
        run_path (str): Path to the run directory.
    Returns:
        np.ndarray: An array of CI test counts for each experiment.
    """
    results = load_run_results(run_path, keys=["tests"])
    test_sum = [np.sum(h["tests"]) for h in results]
    return np.array(test_sum)


def get_times(run_path: str) -> np.ndarray:
    """
    Get the execution times for each experiment in a run.
    Args:
        run_path (str): Path to the run directory.
    Returns:
        np.ndarray: An array of execution times for each experiment.
    """
    results = load_run_results(run_path, keys=["time"])
    return np.array([h["time"] for h in results])


# ADJ-SETS
def is_amenable(cpdag: np.ndarray, treatment: int, outcome: int) -> bool:
    """Returns if the CPDAG is amenable relative to (treatment, outcome),
    by checking whether all possibly directed paths from treatment to outcome
    start with a directed edge out of treatment.

    Args:
        cpdag (PDAG): The input CPDAG.
        treatment (int): The treatment node index.
        outcome (int): The outcome node index.

    Returns:
        bool: True if the CPDAG is amenable relative to (treatment, outcome),
        False otherwise.
    """
    cpdag = cpdag.copy()
    # Remove directed edges out of treatment
    cpdag[treatment, (cpdag[treatment, :] == 1) & (cpdag[:, treatment] == 0)] = 0
    # Check if treatment can still reach outcome
    reach = cpdag.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    # If treatment cannot reach outcome, the graph is amenable
    return not reach[treatment, outcome]


def get_optimal_adj_set(
    amat: np.ndarray, treatment: int, outcome: int
) -> set[int] | None:
    """
    Get the optimal adjustment set (Oset) for estimating the causal effect
    of treatment on outcome. If the Oset does not exist, or the treatment
    is not a (possible) ancestor of the outcome, return None.

    Args:
        amat (np.ndarray): The adjacency matrix of the (CP)DAG.
        treatment (int): The treatment node index.
        outcome (int): The outcome node index.

    Returns:
        set[int] | None: The optimal adjustment set, or None if it does not exist.
    """
    if not is_ancestor(treatment, outcome, amat):
        return None
    if not is_amenable(amat, treatment, outcome):
        return None
    dir_amat = (amat != 0) & (amat.T == 0)
    dir_graph = nx.DiGraph(dir_amat)
    mediators = nx.descendants(dir_graph, treatment).intersection(
        nx.ancestors(dir_graph, outcome)
    )
    mediators.add(outcome)
    oset = set(list(np.sum(dir_amat[:, list(mediators)], axis=1).nonzero()[0]))
    oset = set(list(oset)) - mediators - {treatment}
    return oset


def get_true_osets(experiments: dict) -> dict:
    """
    Get the true optimal adjustment sets (Osets) for all experiments.
    Args:
        experiments (dict): Dictionary with experimental data with their IDs as keys.
    Returns:
        dict: A dictionary of true Osets with experiment IDs as keys.
    """
    true_osets = {}

    cpdag_procs = []
    with ProcessPoolExecutor() as cdapg_e:
        for exp_id, exp in experiments.items():
            p = cdapg_e.submit(dag2cpdag, nx.to_numpy_array(exp["true_dag"]))
            p.exp_id = exp_id
            cpdag_procs.append(p)
        oset_procs = []
        with ProcessPoolExecutor() as oset_e:
            for cpdag_p in tqdm(
                as_completed(cpdag_procs), desc="Getting CPDAGs", total=len(cpdag_procs)
            ):
                cpdag = nx.to_numpy_array(cpdag_p.result())
                exp_id = cpdag_p.exp_id
                exp = experiments[exp_id]
                targets = exp["targets"]
                if targets[0] in nx.ancestors(exp["true_dag"], targets[1]):
                    treatment, outcome = targets[0], targets[1]
                else:
                    treatment, outcome = targets[1], targets[0]
                p = oset_e.submit(get_optimal_adj_set, cpdag, treatment, outcome)
                p.exp_id, p.treatment, p.outcome = exp_id, treatment, outcome
                oset_procs.append(p)

            for p in tqdm(
                as_completed(oset_procs),
                desc=f"Getting true oset",
                total=len(oset_procs),
            ):
                true_osets[p.exp_id] = {
                    "treatment": p.treatment,
                    "outcome": p.outcome,
                    "oset": p.result(),
                }
    return true_osets


def get_locally_valid_sets(target: int, amat: np.ndarray) -> list:
    """
    Get all locally valid parent adjustment sets for a target node
    in a given adjacency matrix.

    Args:
        target (int): The target node index.
        amat (np.ndarray): The adjacency matrix of the (CP)DAG.
    Returns:
        list: A list of locally valid parent adjustment sets.
    """
    parents = np.logical_and(amat[:, target] != 0, amat[target, :] == 0)
    parents = np.where(parents)[0].tolist()
    unoriented = np.logical_and(amat[:, target] != 0, amat[target, :] != 0)
    unoriented = np.where(unoriented)[0]
    skeleton = (amat + amat.T) != 0
    np.fill_diagonal(skeleton, True)

    valid_sets = [parents]
    for l in range(1, len(unoriented) + 1):
        for comb in combinations(unoriented, l):
            candidate_set = list(comb) + parents
            if np.all(skeleton[comb, :][:, candidate_set]):
                valid_sets.append(candidate_set)
    return valid_sets


# INTERVENTION DISTANCE
def is_ancestor(t1: int, t2: int, amat: np.ndarray) -> bool:
    """
    Check if t1 is a (possible) ancestor of t2 in the graph represented by amat.

    Args:
        t1 (int): The potential ancestor node index.
        t2 (int): The potential descendant node index.
        amat (np.ndarray): The adjacency matrix of the (CP)DAG.
    Returns:
        bool: True if t1 is a (possible) ancestor of t2, False otherwise.
    """
    reach = amat.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    return reach[t1, t2]


def true_linear_gaussian_effect(model: DAG, treatment: int, outcome: int) -> float:
    """
    Calculate the true causal effect of a treatment on an outcome in a
    linear Gaussian Bayesian Network.

    Args:
        model (DAG): The pgmpy Bayesian Network model.
        treatment (int): The treatment variable (node index).
        outcome (int): The outcome variable (node index).
    Returns:
        float: The true causal effect of the treatment on the outcome.
    """
    if treatment not in nx.ancestors(model, outcome):
        return 0.0
    amat = nx.to_numpy_array(model)
    return sum(
        np.prod([amat[path[i], path[i + 1]] for i in range(len(path) - 1)])
        for path in nx.all_simple_paths(model, treatment, outcome)
    )


def true_binary_effect(model: DAG, treatment: int, outcome: int) -> float:
    """
    Estimate the true average treatment effect (ATE) of a treatment on an outcome
    in a given pgmpy binary Bayesian Network model using using synthetic data generated from
    interventions as do(treatment=1) - do(treatment=0).

    Args:
        model (DAG): The pgmpy Bayesian Network model.
        treatment (int): The treatment variable (node index).
        outcome (int): The outcome variable (node index).
    Returns:
        float: The estimated average treatment effect (ATE).
    """
    if treatment not in nx.ancestors(model, outcome):
        return 0.0
    args = {"n_samples": int(1e4), "seed": 0, "show_progress": False}
    do0 = model.simulate(do={treatment: 0}, **args)[outcome].astype(int)
    do1 = model.simulate(do={treatment: 1}, **args)[outcome].astype(int)
    return do1.mean() - do0.mean()


def true_causal_effects(experiments: dict) -> dict:
    """
    Get the true causal effects for all experiments.
    Args:
        experiments (dict): Dictionary with experimental data with their IDs as keys.
    Returns:
        dict: A dictionary of true causal effects with experiment IDs as keys.
    """
    model_type = type(next(iter(experiments.values()))["true_dag"])
    if model_type is LinearGaussianBayesianNetwork:
        get_effect = true_linear_gaussian_effect
    elif model_type is DiscreteBayesianNetwork:
        get_effect = true_binary_effect  # Have to estimate for binary
    else:
        raise ValueError("Invalid model type")

    effects = defaultdict(dict)
    procs = []
    with ProcessPoolExecutor() as e:
        for exp_id, exp in experiments.items():
            for t1, t2 in permutations(exp["targets"], 2):
                p = e.submit(
                    get_effect, model=exp["true_dag"], treatment=t1, outcome=t2
                )
                p.exp_id, p.t1, p.t2 = exp_id, t1, t2
                procs.append(p)
        for p in tqdm(
            as_completed(procs), desc="Getting true effects", total=len(procs)
        ):
            exp_id, t1, t2 = p.exp_id, p.t1, p.t2
            effects[exp_id][(t1, t2)] = p.result()
    return effects


def estimate_ate(
    treatment: int,
    outcome: int,
    adj_sets: list[list[int]],
    samples: np.ndarray,
) -> list[float]:
    """
    Estimate the average treatment effect (ATE) of a treatment
    on an outcome using given adjustment sets and samples.

    Args:
        treatment (int): The treatment variable (node index).
        outcome (int): The outcome variable (node index).
        adj_sets (list[list[int]]): The list of adjustment sets.
        samples (np.ndarray): The input samples.
        family (str): The type of data ("gaussian" or "binary").
    Returns:
        list[float]: The estimated average treatment effects for each adjustment set.
    """
    effects = []
    for adj_set in adj_sets:
        adj_set = list(adj_set)
        x = samples[:, [treatment] + adj_set]
        y = samples[:, outcome]
        A = np.vstack((x.T, np.ones(len(x))))
        effect = np.linalg.lstsq(A.T, y, rcond=None)[0][0]
        effects.append(effect)
    return effects


def get_adj_sets(
    algorithm: str, result: dict, t1: int, t2: int
) -> list[list[int]] | None:
    """
    Get the adjustment sets for estimating the causal effect
    of t1 on t2 from an experimental result.
    For global discovery methods PC, FGES, MARVEL, and SNAP,
    we use the Oset if it exists, otherwise
    we fall back to locally valid parent adjustment sets.
    For local causal discovery methods MB-by-MB and LDECC,
    we use the locally valid parent adjustment sets.
    For LOAD, we use the returned adjustment sets.
    The same holds for the post-processing baselines (suffix _pp)
    and the causal-learn and pcalg implementations of PC.

    Args:
        algorithm (str): The name of the algorithm.
        result (dict): The result of the run.
        t1 (int): The treatment node index.
        t2 (int): The outcome node index.
    Returns:
        list[list[int]] | None: The adjustment sets, or None if not found.
    """
    if "failed" in result and result["failed"]:
        return [[]]
    if algorithm in [
        "pc",
        "fges",
        "marvel",
        "snap",
        "pc_pp",
        "snap_pp",
        "cl_pc",
        "cl_pc_stable",
        "pcalg_pc",
        "pcalg_pc_stable",
    ]:
        amat = np.array(result["amat"], dtype=np.int8)
        if is_ancestor(t1, t2, amat):
            oset = get_optimal_adj_set(amat, t1, t2)
            if oset != None:  # identifiable
                return [oset]
            else:  # unidentifiable, fall back to local IDA
                return get_locally_valid_sets(t1, amat)
        else:
            return None
    elif algorithm in ["mb_by_mb_plus", "ldecc_plus", "mb_by_mb_pp", "ldecc_pp"]:
        adj_sets: dict = eval(result["adj_sets"])
        return adj_sets.get((t1, t2), None)
    elif algorithm in ["load", "load_oracle", "load_pp"]:
        adj_sets: dict = eval(result["adj_sets"])
        return adj_sets.get((t1, t2), None)
    else:
        raise ValueError("Invalid algorithm")


def estimate_ates(run_path: str, samples: dict) -> dict:
    """
    Estimate multiple average treatment effects (ATEs) according to all
    experimental results in a run.
    Args:
        run_path (str): Path to the run directory.
        samples (dict): Dictionary of samples with experiment IDs as keys.
    Returns:
        dict: A dictionary of estimated ATEs with experiment IDs as keys.
    """
    config = load_run_config(run_path)
    algorithm = config["algorithm"]
    results = load_run_results(run_path)
    ates = {}
    adj_set_proc = []
    with ProcessPoolExecutor() as adj_set_e:
        for result in results:
            ates[result["id"]] = {}
            for t1, t2 in permutations(result["targets"], 2):
                p = adj_set_e.submit(get_adj_sets, algorithm, result, t1, t2)
                p.exp_id = result["id"]
                p.t1 = t1
                p.t2 = t2
                adj_set_proc.append(p)
        ate_procs = []
        with ProcessPoolExecutor() as ate_e:
            for adj_sets_p in tqdm(
                as_completed(adj_set_proc),
                desc=f"Getting adjsets for {algorithm}/{run_path}",
                total=len(adj_set_proc),
            ):
                adj_sets = adj_sets_p.result()
                if adj_sets == None:
                    ates[adj_sets_p.exp_id][(adj_sets_p.t1, adj_sets_p.t2)] = [0.0]
                else:
                    p = ate_e.submit(
                        estimate_ate,
                        treatment=adj_sets_p.t1,
                        outcome=adj_sets_p.t2,
                        adj_sets=adj_sets,
                        samples=samples[adj_sets_p.exp_id].astype(np.float64),
                    )
                    p.exp_id = adj_sets_p.exp_id
                    p.t1 = adj_sets_p.t1
                    p.t2 = adj_sets_p.t2
                    ate_procs.append(p)
            for p in tqdm(
                as_completed(ate_procs),
                desc=f"Estimating ATEs for {algorithm}/{run_path}",
                total=len(ate_procs),
            ):
                ates[p.exp_id][(p.t1, p.t2)] = p.result()
        return ates


def intervention_distance(
    est_ates: dict, true_ates: dict, aggr: str = "abs"
) -> np.ndarray:
    """
    Calculate the intervention distance between estimated and true average treatment effects (ATEs).
    Args:
        est_ates (dict): Dictionary of estimated ATEs with experiment IDs as keys.
        true_ates (dict): Dictionary of true ATEs with experiment IDs as keys.
        aggr (str): The aggregation method to use ("mse" or "abs").
    Returns:
        np.ndarray: An array of intervention distances for each experiment.
    """
    distances = []
    for exp in tqdm(est_ates, desc="Calculating intervention distances"):
        exp_dist = []
        for pair in est_ates[exp]:
            true_ate = true_ates[exp][pair]
            if aggr == "mse":
                dist = [(true_ate - est_ate) ** 2 for est_ate in est_ates[exp][pair]]
            elif aggr == "abs":
                dist = [np.abs(true_ate - est_ate) for est_ate in est_ates[exp][pair]]
            else:
                raise ValueError("Invalid aggregation")
            exp_dist.append(np.mean(dist))
        distances.append(np.mean(exp_dist))
    return np.array(distances)
