from hashlib import sha256
import json
from pathlib import Path
import pickle
import random

import igraph as ig
import networkx as nx
import numpy as np
from pgmpy.base import DAG, PDAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.utils import get_example_model


def generate_degseq(
    nodes: int, exp_degree: float, max_degree: int, seed: int
) -> list[int]:
    """
    Generate a degree sequence for a DAG with the given number of nodes,
    expected degree, and maximum degree using a binomial distribution.

    Args:
        nodes (int): Number of nodes in the DAG.
        exp_degree (float): Expected degree of each node.
        max_degree (int): Maximum degree of each node.
        seed (int): Random seed for reproducibility.
    Returns:
        list[int]: A list representing the degree sequence.
    """
    np.random.seed(seed)
    p = exp_degree / nodes
    degseq = []
    for _ in range(nodes):
        deg = np.random.binomial(nodes - 1, p)
        while max_degree < deg or deg == 0:
            deg = np.random.binomial(nodes - 1, p)
        degseq.append(deg)
    return degseq


def generate_dag(nodes: int, exp_degree: float, max_degree: int, seed: int) -> DAG:
    """
    Generate a random DAG with the given number of nodes, expected degree, and maximum degree.
    Args:
        nodes (int): Number of nodes in the DAG.
        exp_degree (float): Expected degree of each node.
        max_degree (int): Maximum degree of each node.
        seed (int): Random seed for reproducibility.
    Returns:
        DAG: A randomly generated pgmpy DAG.
    """
    random.seed(seed)
    np.random.seed(seed)
    while True:
        degseq = generate_degseq(
            nodes, exp_degree, max_degree, np.random.randint(0, 1e9)
        )
        if ig.is_graphical(degseq):
            try:
                g = ig.Graph.Degree_Sequence(degseq, method="vl")
                g.to_directed(mode="acyclic")
                g = g.permute_vertices(np.random.permutation(nodes))
                return DAG(g.to_networkx())
            except Exception:
                pass


def sample_ancestral_pairs(dag: DAG) -> list[list[int]]:
    """
    Find treatment-outcome pairs (T, O) such that T is an explicit ancestor of O
    in the CPDAG corresponding to the input DAG, using Lemma 4.3 from LOAD.

    Args:
        dag (DAG): The input DAG.
    Returns:
        list[list[int]]: A list of explicit ancestral (treatment, outcome) pairs.
    """
    pairs = []
    for treatment in dag.nodes():
        for outcome in nx.descendants(dag, treatment):
            pairs.append([treatment, outcome])
    return pairs


def sample_explicit_ancestral_pairs(dag: DAG) -> list[list[int]]:
    """
    Find treatment-outcome pairs (T, O) such that T is an explicit ancestor of O
    in the CPDAG corresponding to the input DAG, using Lemma 4.3 from LOAD.

    Args:
        dag (DAG): The input DAG.
    Returns:
        list[list[int]]: A list of explicit ancestral (treatment, outcome) pairs.
    """

    dir = dag.to_pdag()._directed_graph()
    pairs = []
    for treatment in dir.nodes():
        for outcome in nx.descendants(dir, treatment):
            pairs.append([treatment, outcome])
    return pairs


def sample_identifiable_pairs(dag: DAG) -> list[list[int]]:
    """
    Find treatment-outcome pairs (T, O) such that the causal effect of T on O is identifiable
    in the CPDAG corresponding to the input DAG.

    Args:
        dag (DAG): The input DAG.
    Returns:
        list[list[int]]: A list of identifiable (treatment, outcome) pairs.
    """
    cpdag = dag.to_pdag()
    dir = cpdag._directed_graph()
    pairs = []
    for treatment in dir.nodes():
        descendants = list(nx.descendants(dir, treatment))
        if len(descendants) == 0:
            continue
        siblings = cpdag.undirected_neighbors(treatment)
        for outcome in nx.descendants(dir, treatment):
            identifiable = True
            for sibling in siblings:
                cond = cpdag.directed_parents(sibling) | {treatment}
                if not nx.is_d_separator(dag, outcome, sibling, cond):
                    identifiable = False
                    break
            if identifiable:
                pairs.append([treatment, outcome])
    return pairs


def generate_bk_from_dag(
    model: DAG,
    bk_rate: float = 0.0,
    adj_rate: float = 0.0,
    ori_rate: float = 0.0,
    gap_rate: float = 0.0,
    bk_error: float = 0.0,
    adj_error: float = 0.0,
    ori_error: float = 0.0,
    gap_error: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate background knowledge at the given rates for a given DAG.
    The background knowledge is represented as a matrix B where:
    - B[i, j] = -1 and B[j, i] = -1 indicates adjacency: i - j.
    - B[i, j] = -1 and B[j, i] =  1 indicates orientation: i -> j.
    - B[i, j] =  1 and B[j, i] =  1 indicates gap: i -/- j.
    - B[i, j] =  0 and B[j, i] =  0 indicates no knowledge about i and j.

    Args:
        model (DAG): The true DAG model.
        bk_rate (float): The overall rate of background knowledge. If > 0, it overrides other rates.
        adj_rate (float): The rate of adjacency background knowledge.
        ori_rate (float): The rate of orientation background knowledge.
        gap_rate (float): The rate of gap background knowledge.
        bk_error (float): The error rate of background knowledge. If > 0, it overrides other error rates.
        adj_error (float): The error rate of adjacency background knowledge.
        ori_error (float): The error rate of orientation background knowledge.
        gap_error (float): The error rate of gap background knowledge.
        seed (int): The random seed.
    Returns:
        np.ndarray: The background knowledge matrix.
    """
    random.seed(seed)
    np.random.seed(seed)

    m = nx.to_numpy_array(model) != 0
    num_nodes = len(model.nodes)
    bk = np.zeros((num_nodes, num_nodes), dtype=int)
    if bk_rate > 0.0:
        # Sample indices for background knowledge
        rows, cols = np.triu_indices(num_nodes, k=1)
        n_bk = np.round(len(rows) * bk_rate).astype(int)
        bk_inds = np.random.choice(len(rows), size=n_bk, replace=False)
        # Exract true values for sampled indices
        r = rows[bk_inds]
        c = cols[bk_inds]
        m_rc = m[r, c]
        m_cr = m[c, r]
        # Set gaps background knowledge
        gaps = ~m_rc & ~m_cr
        bk[r[gaps], c[gaps]] = 1
        bk[c[gaps], r[gaps]] = 1
        # Set edges background knowledge
        edges = m_rc | m_cr
        # Extract row and column indices for edges
        r_edge = r[edges]
        c_edge = c[edges]
        # Sample adjacency or orientation background knowledge
        adj_not_ori = np.random.rand(np.sum(edges)) < 0.5
        # Set adjacency background knowledge
        bk[r_edge[adj_not_ori], c_edge[adj_not_ori]] = -1
        bk[c_edge[adj_not_ori], r_edge[adj_not_ori]] = -1
        # Extract edge directions
        m_rc_edge = m_rc[edges]
        m_cr_edge = m_cr[edges]
        # Set orientation background knowledge
        ori_not_adj = ~adj_not_ori
        r_ori = r_edge[ori_not_adj]
        c_ori = c_edge[ori_not_adj]
        m_rc_ori = m_rc_edge[ori_not_adj]
        m_cr_ori = m_cr_edge[ori_not_adj]
        # Edge direction: r -> c
        bk[r_ori, c_ori] = np.where(m_rc_ori == 1, -1, 1)
        # Edge direction: c -> r
        bk[c_ori, r_ori] = np.where(m_cr_ori == 1, -1, 1)
    else:
        edges = list(model.edges)
        n_edges = len(edges)
        # Select edges for orientation background knowledge
        if ori_rate > 0.0:
            n_ori_bk = np.round(n_edges * ori_rate).astype(int)
            ori_bk_edges = random.sample(edges, n_ori_bk)
            row_ori, col_ori = tuple(zip(*ori_bk_edges))
            bk[(row_ori, col_ori)] = -1
            bk[(col_ori, row_ori)] = 1
        else:
            ori_bk_edges = []

        # Select remaining edges for adjacency background knowledge
        if adj_rate > 0.0:
            n_adj_bk = np.round(n_edges * adj_rate).astype(int)
            remaining = list(set(edges) - set(ori_bk_edges))
            adj_bk_edges = random.sample(remaining, n_adj_bk)
            row_adj, col_adj = tuple(zip(*adj_bk_edges))
            bk[(row_adj, col_adj)] = -1
            bk[(col_adj, row_adj)] = -1

        # Select gaps for gaps background knowledge
        if gap_rate > 0.0:
            gaps_m = (m == 0) & (m.T == 0)
            gaps_m[np.tril_indices(gaps_m.shape[0])] = False  # Ignore symmetry
            gaps_row, gaps_col = np.where(gaps_m)
            n_gaps = len(gaps_row)
            n_gaps_bk = np.round(n_gaps * gap_rate).astype(int)
            gap_bk_inds = np.random.choice(n_gaps, size=n_gaps_bk, replace=False)
            gap_bk_row = gaps_row[gap_bk_inds]
            gap_bk_col = gaps_col[gap_bk_inds]
            bk[gap_bk_row, gap_bk_col] = 1
            bk[gap_bk_col, gap_bk_row] = 1

    # Introduce errors
    if bk_error > 0.0:
        adj_error = bk_error
        ori_error = bk_error
        gap_error = bk_error

    if adj_error > 0.0 or ori_error > 0.0 or gap_error > 0.0:
        # Find all pairs with background knowledge at upper triangle to avoid double counting
        rows, cols = np.triu_indices(num_nodes, k=1)
        bk_present = bk[rows, cols] != 0
        bk_indices = np.where(bk_present)[0]

        # Iterate through all present BK and check if they should be corrupted
        r_present = rows[bk_indices]
        c_present = cols[bk_indices]

        for r, c in zip(r_present, c_present):
            val_r = bk[r, c]
            val_c = bk[c, r]

            # 1. Adjacency (-1, -1)
            if val_r == -1 and val_c == -1:
                if np.random.rand() < adj_error:
                    if np.random.rand() < 0.5:  # Become Gap (1, 1)
                        bk[r, c] = 1
                        bk[c, r] = 1
                    else:  # Become Orientation in Wrong Direction
                        is_r_to_c = m[r, c]

                        if is_r_to_c:  # Force c->r (1, -1)
                            bk[r, c] = 1
                            bk[c, r] = -1
                        else:  # Force r->c (-1, 1)
                            bk[r, c] = -1
                            bk[c, r] = 1

            # 2. Orientation (-1, 1) or (1, -1)
            elif (val_r == -1 and val_c == 1) or (val_r == 1 and val_c == -1):
                if np.random.rand() < ori_error:
                    if np.random.rand() < 0.5:  # Flip
                        bk[r, c] = -val_r  # become 1 if -1, or -1 if 1
                        bk[c, r] = -val_c  # become -1 if 1, or 1 if -1
                    else:  # Gap (1, 1)
                        bk[r, c] = 1
                        bk[c, r] = 1

            # 3. Gap (1, 1)
            elif val_r == 1 and val_c == 1:
                if np.random.rand() < gap_error:
                    choice = np.random.randint(3)
                    if choice == 0:  # Adjacency
                        bk[r, c] = -1
                        bk[c, r] = -1
                    elif choice == 1:  # Orientation r -> c (-1, 1)
                        bk[r, c] = -1
                        bk[c, r] = 1
                    else:  # Orientation c -> r (1, -1)
                        bk[r, c] = 1
                        bk[c, r] = -1

    return bk


def generate_bk_from_cpdag(
    model: DAG, adj_rate: float, ori_rate: float, gap_rate: float, seed: int
) -> np.ndarray:
    """
    Generate background knowledge at the given rates from the CPDAG
    of the given DAG model.
    The background knowledge is represented as a matrix B where:
    - B[i, j] = -1 and B[j, i] = -1 indicates adjacency: i - j.
    - B[i, j] = -1 and B[j, i] =  1 indicates orientation: i -> j.
    - B[i, j] =  1 and B[j, i] =  1 indicates gap: i -/- j.
    - B[i, j] =  0 and B[j, i] =  0 indicates no knowledge about i and j.

    Args:
        model (DAG): The true DAG model.
        adj_rate (float): The rate of adjacency background knowledge.
        ori_rate (float): The rate of orientation background knowledge.
        gap_rate (float): The rate of gap background knowledge.
        seed (int): The random seed.
    Returns:
        np.ndarray: The background knowledge matrix.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Convert to CPDAG
    pdag_ = model.to_pdag()
    pdag = PDAG()
    pdag.add_nodes_from(model.nodes())
    pdag.add_edges_from(pdag_.edges())
    pdag.directed_edges = pdag_.directed_edges
    pdag.undirected_edges = pdag_.undirected_edges

    bk = np.zeros((len(pdag.nodes), len(pdag.nodes)), dtype=int)

    # Select oriented edges for orientation background knowledge
    if ori_rate > 0.0:
        n_directed = len(pdag.directed_edges)
        n_ori_bk = np.round(n_directed * ori_rate).astype(int)
        ori_bk_edges = random.sample(list(pdag.directed_edges), n_ori_bk)
        row_ori, col_ori = tuple(zip(*ori_bk_edges))
        bk[(row_ori, col_ori)] = -1
        bk[(col_ori, row_ori)] = 1
    else:
        ori_bk_edges = []

    # Select remaining edges for adjacency background knowledge
    if adj_rate > 0.0:
        n_edges = len(pdag.directed_edges) + len(pdag.undirected_edges)
        n_adj_bk = np.round(n_edges * adj_rate).astype(int)
        remaining = list(
            pdag.undirected_edges | pdag.undirected_edges - set(ori_bk_edges)
        )
        adj_bk_edges = random.sample(remaining, n_adj_bk)
        row_adj, col_adj = tuple(zip(*adj_bk_edges))
        bk[(row_adj, col_adj)] = -1
        bk[(col_adj, row_adj)] = -1

    # Select gaps for gaps background knowledge
    if gap_rate > 0.0:
        amat = nx.to_numpy_array(pdag) != 0
        gaps_m = ~amat & ~amat.T
        gaps_m[np.tril_indices(gaps_m.shape[0])] = False  # Ignore symmetry
        gaps_row, gaps_col = np.where(gaps_m)
        n_gaps = len(gaps_row)
        gap_bk_inds = np.random.choice(
            n_gaps, size=np.round(n_gaps * gap_rate).astype(int), replace=False
        )
        gap_bk_row = gaps_row[gap_bk_inds]
        gap_bk_col = gaps_col[gap_bk_inds]
        bk[gap_bk_row, gap_bk_col] = 1
        bk[gap_bk_col, gap_bk_row] = 1

    return bk


def generate_samples_from_model(model: DAG, samples_num: int, seed: int) -> np.ndarray:
    """
    Generate samples from a given Bayesian Network model.
    Args:
        model (DAG): The Bayesian Network model.
        samples_num (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
    Returns:
        np.ndarray: Generated samples as a NumPy array.
    """
    if samples_num == 0:
        return np.empty((0, len(model.nodes())))
    args = {"n_samples": samples_num, "seed": seed}
    if type(model) is DiscreteBayesianNetwork:
        args["show_progress"] = False
    samples = model.simulate(**args)
    return samples[sorted(samples.columns, key=int)].to_numpy()


def generate_synthetic_model(
    nodes: int, exp_degree: float, max_degree: int, binary: bool = False, seed: int = 0
) -> DAG:
    """
    Generate synthetic Bayesian Network.

    Args:
        nodes (int): Number of nodes in the DAG.
        exp_degree (float): Expected degree of each node.
        max_degree (int): Maximum degree of each node.
        binary (bool): Whether to generate binary or linear Gaussian BN.
        seed (int): Random seed for reproducibility.

    Returns:
        DAG: The generated synthetic Bayesian Network.
    """
    np.random.seed(seed)
    # Generate DAG
    dag = generate_dag(nodes, exp_degree, max_degree, np.random.randint(0, 1e9))
    # Generate BN
    if binary:
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(dag.nodes())
        model.add_edges_from(dag.edges(), weights=[1] * len(dag.edges()))
        model.get_random_cpds(n_states=2, inplace=True)
    else:
        model = LinearGaussianBayesianNetwork()
        model.add_nodes_from(dag.nodes())
        model.add_edges_from(dag.edges())
        cpds = []
        for var in model.nodes():
            parents = model.get_parents(var)
            beta = np.random.uniform(0.5, 3.0, size=len(parents) + 1)
            beta *= (-1) ** np.random.randint(2, size=len(parents) + 1)
            cpds.append(LinearGaussianCPD(var, beta, 1, parents))
            for p in parents:
                model[p][var]["weight"] = beta[parents.index(p) + 1]
        model.add_cpds(*cpds)
    assert model.check_model()
    return model


def to_int_model(model: DAG) -> DAG:
    """
    Convert a pgmpy Bayesian Network with any node names to int node names.

    Args:
        model (DAG): The input Bayesian Network model.
    Returns:
        DAG: A new Bayesian Network model with int node names.
    """
    if type(model) is DiscreteBayesianNetwork:
        out_model = DiscreteBayesianNetwork()
    elif type(model) is LinearGaussianBayesianNetwork:
        out_model = LinearGaussianBayesianNetwork()
    else:
        raise ValueError("Unsupported model type")

    mapping = {n: i for i, n in enumerate(model.nodes())}
    out_model.add_nodes_from([mapping[n] for n in model.nodes()])
    out_model.add_edges_from([(mapping[u], mapping[v]) for u, v in model.edges()])
    cpds = []
    for cpd in model.get_cpds():
        var_index = mapping[cpd.variable]
        if type(model) is DiscreteBayesianNetwork:
            out_cpd = TabularCPD(
                variable=var_index,
                variable_card=cpd.variable_card,
                values=cpd.get_values(),
                evidence=[mapping[ev] for ev in cpd.get_evidence()],
                evidence_card=cpd.cardinality[1:] if cpd.get_evidence() else [],
            )
        elif type(model) is LinearGaussianBayesianNetwork:
            out_cpd = LinearGaussianCPD(
                variable=var_index,
                beta=cpd.beta,
                std=cpd.std,
                evidence=[mapping[ev] for ev in cpd.evidence],
            )
            for p in model.get_parents(cpd.variable):
                p_index = mapping[p]
                weight = out_cpd.beta[cpd.evidence.index(p) + 1]
                out_model.edges[(p_index, var_index)]["weight"] = weight
        cpds.append(out_cpd)
    out_model.add_cpds(*cpds)
    return out_model


def generate_experiment_from_model(
    model: DAG,
    targets_num: int = 2,
    targets_type: str = "random",
    samples_num: int = 0,
    bk_rate: float = 0.0,
    adj_rate: float = 0.0,
    ori_rate: float = 0.0,
    gap_rate: float = 0.0,
    bk_error: float = 0.0,
    adj_error: float = 0.0,
    ori_error: float = 0.0,
    gap_error: float = 0.0,
    seed: int = 0,
):
    """
    Generate an experiment from a given Bayesian Network model.

    Args:
        model (DAG): The Bayesian Network model.
        targets_num (int): Number of target nodes.
        targets_type (str): Type of target relations ("random", "identifiable", "explicit
        samples_num (int): Number of samples to generate.
        bk_rate (float): The overall rate of background knowledge. If > 0, it overrides other rates.
        adj_rate (float): The rate of adjacency background knowledge.
        ori_rate (float): The rate of orientation background knowledge.
        gap_rate (float): The rate of gap background knowledge.
        bk_error (float): The error rate of background knowledge.
        adj_error (float): The error rate of adjacency background knowledge.
        ori_error (float): The error rate of orientation background knowledge.
        gap_error (float): The error rate of gap background knowledge.
        seed (int): Random seed for reproducibility.
    Returns:
        dict: A dictionary containing the experimental data.
    """
    np.random.seed(seed)
    if targets_type == "random":
        targets = np.random.choice(model.nodes(), size=targets_num, replace=False)
    elif targets_type in ["ancestral", "explicit", "identifiable"]:
        assert targets_num == 2, f"{targets_type} targets can only be 2."
        if targets_type == "ancestral":
            pairs = sample_ancestral_pairs(model)
        elif targets_type == "explicit":
            pairs = sample_explicit_ancestral_pairs(model)
        else:  # identifiable
            pairs = sample_identifiable_pairs(model)
        if len(pairs) == 0:
            raise ValueError(f"No {targets_type} targets found.")
        targets = np.array(random.sample(pairs, 1)[0])
    else:
        raise ValueError(f"Unknown targets_type: {targets_type}")

    bk = generate_bk_from_dag(
        model,
        bk_rate,
        adj_rate,
        ori_rate,
        gap_rate,
        bk_error,
        adj_error,
        ori_error,
        gap_error,
        np.random.randint(0, 1e9),
    )

    objects = {
        "true_dag": model,
        "data": generate_samples_from_model(model, samples_num, seed),
        "targets": targets,
        "treatment": targets[0] if targets_num > 1 else None,
        "outcome": targets[1] if targets_num > 1 else None,
        "bk": bk,
    }
    objects["id"] = sha256(pickle.dumps(objects)).hexdigest()
    return objects


def generate_synthetic_experiment(
    nodes: int,
    exp_degree: float,
    max_degree: int,
    binary: bool = False,
    targets_num: int = 2,
    targets_type: str = "random",
    samples_num: int = 0,
    bk_rate: float = 0.0,
    adj_rate: float = 0.0,
    ori_rate: float = 0.0,
    gap_rate: float = 0.0,
    bk_error: float = 0.0,
    adj_error: float = 0.0,
    ori_error: float = 0.0,
    gap_error: float = 0.0,
    seed: int = 0,
) -> dict:
    """
    Generate a synthetic experiment by sampling a random Bayesian Network.

    Args:
        nodes (int): Number of nodes in the DAG.
        exp_degree (float): Expected degree of each node.
        max_degree (int): Maximum degree of each node.
        binary (bool): Whether to generate binary or linear Gaussian BN.
        targets_num (int): Number of target nodes.
        targets_type (str): Type of target relations ("random", "identifiable", "explicit").
        samples_num (int): Number of samples to generate.
        bk_rate (float): The overall rate of background knowledge. If > 0, it overrides other rates.
        adj_rate (float): The rate of adjacency background knowledge.
        ori_rate (float): The rate of orientation background knowledge.
        gap_rate (float): The rate of gap background knowledge.
        bk_error (float): The error rate of background knowledge.
        adj_error (float): The error rate of adjacency background knowledge.
        ori_error (float): The error rate of orientation background knowledge.
        gap_error (float): The error rate of gap background knowledge.
        seed (int): Random seed for reproducibility.
    Returns:
        dict: A dictionary containing the generated model, data, and targets.
    """
    # Generate model
    np.random.seed(seed)
    while True:
        s = np.random.randint(0, 1e9)
        try:
            model = generate_synthetic_model(nodes, exp_degree, max_degree, binary, s)
            return generate_experiment_from_model(
                model=model,
                targets_num=targets_num,
                targets_type=targets_type,
                samples_num=samples_num,
                bk_rate=bk_rate,
                adj_rate=adj_rate,
                ori_rate=ori_rate,
                gap_rate=gap_rate,
                bk_error=bk_error,
                adj_error=adj_error,
                ori_error=ori_error,
                gap_error=gap_error,
                seed=s,
            )
        except ValueError as e:
            print(e)
            pass


def save_experiment(data: dict, seed: int, **kwargs):
    """
    Save experimental setting to a file identified by the seed and the kwargs.

    Args:
        data (dict): The experimental data to save.
        seed (int): The seed used to generate the data.
        **kwargs: Additional arguments used to generate the data.
    """
    exp_id = sha256(pickle.dumps(kwargs)).hexdigest()
    file_path = Path(f"experiments/{exp_id}/{seed}.pkl")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as file:
        pickle.dump(data, file)

    metadata_path = Path(f"experiments/{exp_id}/metadata.json")
    if not metadata_path.exists():
        with metadata_path.open("w") as meta_file:
            json.dump(kwargs, meta_file, indent=4)


def load_experiment(seed: int, **kwargs) -> dict:
    """
    Load experimental setting from a file identified by the seed and the kwargs.

    Args:
        seed (int): The seed used to generate the data.
        **kwargs: Additional arguments used to generate the data.

    Returns:
        dict: The loaded experimental data.
    """
    exp_id = sha256(pickle.dumps(kwargs)).hexdigest()
    file_path = Path(f"experiments/{exp_id}/{seed}.pkl")
    with file_path.open("rb") as file:
        return pickle.load(file)


def generate_experiment(
    model: str = "synthetic",
    nodes: int | None = None,
    exp_degree: float | None = None,
    max_degree: int | None = None,
    binary: bool = False,
    targets_num: int = 2,
    targets_type: str = "random",
    samples_num: int = 0,
    bk_rate: float = 0.0,
    adj_rate: float = 0.0,
    ori_rate: float = 0.0,
    gap_rate: float = 0.0,
    bk_error: float = 0.0,
    adj_error: float = 0.0,
    ori_error: float = 0.0,
    gap_error: float = 0.0,
    seed: int = 0,
    save: bool = True,
) -> dict:
    """
    Generate experimental setting given the parameters.

    Args:
        nodes (int): Number of nodes in the DAG.
        exp_degree (float): Expected degree of each node.
        max_degree (int): Maximum degree of each node.
        model (str): The model to use for data generation. If "synthetic", a random BN is generated.
        binary (bool): Whether to generate binary or linear Gaussian BN.
        targets_num (int): Number of target nodes.
        targets_type (str): Type of target relations ("random", "identifiable", "explicit").
        samples_num (int): Number of samples to generate.
        bk_rate (float): The overall rate of background knowledge. If > 0, it overrides other rates.
        adj_rate (float): The rate of adjacency background knowledge.
        ori_rate (float): The rate of orientation background knowledge.
        gap_rate (float): The rate of gap background knowledge.
        bk_error (float): The error rate of background knowledge.
        adj_error (float): The error rate of adjacency background knowledge.
        ori_error (float): The error rate of orientation background knowledge.
        gap_error (float): The error rate of gap background knowledge.
        seed (int): Random seed for reproducibility.
        save (bool): Whether to save the generated data to a file.

    Returns:
        dict: A dictionary containing the generated model, data, and targets.
    """
    args_dict = locals()
    args_dict.pop("save")
    try:  # Try to load previously generated experimental data
        return load_experiment(**args_dict)
    except FileNotFoundError:  # Generate new experimental data
        if model == "synthetic":
            assert (
                nodes is not None and exp_degree is not None and max_degree is not None
            ), "Missing nodes, exp_degree and max_degree."
            data = generate_synthetic_experiment(
                nodes=nodes,
                exp_degree=exp_degree,
                max_degree=max_degree,
                binary=binary,
                targets_num=targets_num,
                targets_type=targets_type,
                samples_num=samples_num,
                bk_rate=bk_rate,
                adj_rate=adj_rate,
                ori_rate=ori_rate,
                gap_rate=gap_rate,
                bk_error=bk_error,
                adj_error=adj_error,
                ori_error=ori_error,
                gap_error=gap_error,
                seed=seed,
            )
        else:
            model = to_int_model(get_example_model(model))
            data = generate_experiment_from_model(
                model=model,
                targets_num=targets_num,
                targets_type=targets_type,
                samples_num=samples_num,
                bk_rate=bk_rate,
                adj_rate=adj_rate,
                ori_rate=ori_rate,
                gap_rate=gap_rate,
                bk_error=bk_error,
                adj_error=adj_error,
                ori_error=ori_error,
                gap_error=gap_error,
                seed=seed,
            )
        if save:
            save_experiment(data, **args_dict)
        return data
