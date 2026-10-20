from collections import defaultdict
from functools import cache
from itertools import combinations
from typing import Callable

from pathlib import Path

import networkx as nx
import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import (
    orient_by_background_knowledge,
)

from algorithms.orient import v_struc_pc, cpdag2mpdag


def tests_remain(order: int, g: np.ndarray, bk: np.ndarray | None = None) -> bool:
    """
    Check whether any CI test of the given order is still possible in the skeleton.

    Args:
        order (int): The order of CI tests.
        g (np.ndarray): The current skeleton.
        bk (np.ndarray | None): Background knowledge matrix.

    Returns:
        bool: Whether a CI test of the given order can still be performed.
    """
    if bk is None:
        bk = np.zeros_like(g)

    # Whether a node may be conditioned on (since it is a possible parent of the other node)
    poss_pa = (g != 0) & (bk.T != 1)
    # Number of nodes that may be conditioned on for each node
    num_poss_pa = poss_pa.sum(axis=1, keepdims=True)
    # Edges that are still to be tested
    to_test = (g != 0) & (bk == 0)
    # Check if (num_poss_pa - 1) for any to-test edge is >= order
    return bool(np.any(to_test & (num_poss_pa - 1 >= order)))


def skeleton_step(
    order: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    sep_set: dict[frozenset, list],
    bk: np.ndarray | None = None,
) -> np.ndarray:
    """
    Skeleton step at a given order.

    Args:
        order (int): The order of CI tests.
        g (np.ndarray): The current skeleton.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        sep_set (dict[frozenset, list]): The separating sets.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        np.ndarray: The updated (in-place) skeleton.
    """
    if bk is None:
        bk = np.zeros_like(g)

    for x in range(len(g)):
        poss_pa_x = np.where((g[x, :] != 0) & (bk[:, x] != 1))[0]
        if len(poss_pa_x) <= order:  # No conditioning set of this order without y
            continue
        # Skip any edge that we "know" from background knowledge
        neigh_x = np.where((g[x, :] != 0) & (bk[x, :] == 0))[0]
        for y in neigh_x:
            for S in combinations(poss_pa_x[poss_pa_x != y], order):
                if ci_test(x, y, S) >= alpha:
                    g[x, y] = g[y, x] = 0
                    # Keep the possible parents in sync with the skeleton
                    poss_pa_x = poss_pa_x[poss_pa_x != y]
                    sep_set[frozenset((x, y))].append(S)
                    break
    return g


def find_missing_separating_sets(
    g: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    sep_set: dict[frozenset, list],
    bk: np.ndarray,
) -> np.ndarray:
    """
    Find missing separating sets for unshielded triples in the skeleton.
    Gaps for which no separating set was found in the skeleton search are filled.

    Args:
        g (np.ndarray): The current skeleton.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        sep_set (dict[frozenset, list]): The separating sets.
        bk (np.ndarray): Background knowledge matrix.
    Returns:
        np.ndarray: The updated (in-place) skeleton.
    """
    # For all gaps in BK
    for x, y in zip(*np.where((bk == 1) & (bk.T == 1))):
        if frozenset((x, y)) in sep_set:
            g[x, y] = g[y, x] = 0
            continue
        # Search among neighbors of x
        found = False
        poss_pa_x = np.where((g[x, :] != 0) & (bk[:, x] != 1))[0]
        poss_pa_x = poss_pa_x[poss_pa_x != y]
        for order in range(len(poss_pa_x) + 1):
            for S in combinations(poss_pa_x, order):
                if ci_test(x, y, S) >= alpha:
                    g[x, y] = g[y, x] = 0
                    sep_set[frozenset((x, y))].append(S)
                    found = True
                    break
            if found:
                break
        if found:
            continue
        # Search among neighbors of y
        poss_pa_y = np.where((g[y, :] != 0) & (bk[:, y] != 1))[0]
        poss_pa_y = poss_pa_y[poss_pa_y != x]
        for order in range(len(poss_pa_y) + 1):
            for S in combinations(poss_pa_y, order):
                if ci_test(x, y, S) >= alpha:
                    g[x, y] = g[y, x] = 0
                    sep_set[frozenset((x, y))].append(S)
                    found = True
                    break
            if found:
                break
    return g


def pc(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict[str, list[list[int]]]:
    """
    PC algorithm with background knowledge.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        bk (np.ndarray | None): Background knowledge matrix.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, list[list[int]]]: The adjacency matrix of the MPDAG.
    """
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # Initialize skeleton with -1 for edge tail, 1 for edge head, and 0 for no edge
    skeleton = np.full((data.shape[1], data.shape[1]), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    # Skeleton search
    order = 0
    while tests_remain(order, skeleton, bk):
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set, bk)
        order += 1

    # Find missing separating sets for gaps in BK
    skeleton = find_missing_separating_sets(skeleton, ci_test, alpha, sep_set, bk)

    # Orient v-structures
    pdag = v_struc_pc(skeleton, sep_set)

    # Orient background knowledge and Meek rules
    mpdag = cpdag2mpdag(pdag, bk)

    # Convert CPDAG to networkx adjacency matrix
    amat = np.zeros_like(mpdag, dtype=int)
    amat[mpdag == -1] = 1
    return {"amat": amat.tolist()}


def pc_pp(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict[str, list[list[int]]]:
    """
    PC algorithm where background knowledge is only used for post-processing.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        bk (np.ndarray | None): Background knowledge matrix.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, list[list[int]]]: The adjacency matrix of the MPDAG.
    """
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # Initialize skeleton with -1 for edge tail, 1 for edge head, and 0 for no edge
    skeleton = np.full((data.shape[1], data.shape[1]), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    # Skeleton search
    order = 0
    while tests_remain(order, skeleton):
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        order += 1

    # Orient v-structures
    pdag = v_struc_pc(skeleton, sep_set)

    # Orient background knowledge and Meek rules
    mpdag = cpdag2mpdag(pdag, bk)

    # Convert CPDAG to networkx adjacency matrix
    amat = np.zeros_like(mpdag, dtype=int)
    amat[mpdag == -1] = 1
    return {"amat": amat.tolist()}


def cl_pc(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    bk: np.ndarray | None = None,
    stable: bool = False,
    uc_rule: int = 0,
    uc_priority: int = 2,
    **kwargs,
) -> dict[str, list[list[int]]]:
    """
    PC algorithm with background knowledge as implemented in causal-learn.
    Gaps are passed as forbidden edges and orientations as required edges;
    causal-learn cannot express adjacencies without an orientation.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        bk (np.ndarray | None): Background knowledge matrix.
        stable (bool): Whether to run the order-independent PC-stable skeleton search.
        uc_rule (int): How unshielded colliders are oriented
            (0: uc_sepset, 1: maxP, 2: definiteMaxP).
        uc_priority (int): How conflicts between unshielded colliders are resolved
            (-1: default of uc_rule, 0: overwrite, 1: orient bi-directed,
            2: prioritize existing colliders, 3: prioritize stronger colliders,
            4: prioritize stronger* colliders).
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, list[list[int]]]: The adjacency matrix of the MPDAG.
    """
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # Convert BK
    nodes = CausalGraph(data.shape[1]).G.get_nodes()
    background_knowledge = BackgroundKnowledge()
    for x, y in zip(*np.where((bk == 1) & (bk.T == 1))):
        background_knowledge.add_forbidden_by_node(nodes[x], nodes[y])
        background_knowledge.add_forbidden_by_node(nodes[y], nodes[x])
    for x, y in zip(*np.where((bk == -1) & (bk.T == 1))):
        background_knowledge.add_required_by_node(nodes[x], nodes[y])

    cg_1 = SkeletonDiscovery.skeleton_discovery(
        data,
        alpha,
        ci_test,
        stable=stable,
        background_knowledge=background_knowledge,
        verbose=False,
        show_progress=False,
        node_names=None,
    )
    orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(
                cg_1, uc_priority, background_knowledge=background_knowledge
            )
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)
    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(
                cg_1, uc_priority, background_knowledge=background_knowledge
            )
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)
    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(
                cg_1, alpha, uc_priority, background_knowledge=background_knowledge
            )
        else:
            cg_2 = UCSepset.definite_maxp(
                cg_1, alpha, background_knowledge=background_knowledge
            )
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")

    # Convert MPDAG to networkx adjacency matrix
    mpdag = cg.G.graph
    amat = np.zeros_like(mpdag, dtype=int)
    amat[mpdag == -1] = 1
    return {"amat": amat.tolist()}


@cache
def load_pcalg():
    """
    Load the R implementation of the pcalg wrapper.
    Imported lazily so that the other algorithms run without R installed.
    """
    from rpy2.robjects import r

    r.source(str(Path(__file__).with_name("pcalg_pc.R")))
    return r["pcalg_pc"]


def pcalg_pc(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    bk: np.ndarray | None = None,
    stable: bool = False,
    **kwargs,
) -> dict[str, list[list[int]]]:
    """
    PC algorithm with background knowledge as implemented in the R package pcalg.
    Adjacencies and gaps are passed as fixed edges and gaps,
    and orientations are added to the CPDAG afterwards.
    pcalg runs its own CI tests, which are counted in R.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test whose method and true DAG are passed on to pcalg.
        alpha (float): Significance level.
        bk (np.ndarray | None): Background knowledge matrix.
        stable (bool): Whether to run the order-independent PC-stable skeleton search.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, list[list[int]]]: The adjacency matrix of the MPDAG.
    """
    from rpy2.robjects import default_converter, numpy2ri

    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    skel_method = "stable" if stable else "original"
    dag = nx.to_numpy_array(ci_test.cit.true_dag, nodelist=range(data.shape[1]))
    dag = (dag != 0).astype(int)
    with (default_converter + numpy2ri.converter).context():
        res = load_pcalg()(
            data.astype(np.float64), ci_test.method, alpha, bk, skel_method, dag
        )
        # pcalg encodes i -> j as amat[j, i] = 1, so transpose to networkx format
        amat = np.asarray(res[0], dtype=int).T
        tests = np.asarray(res[1], dtype=int)
    # Report the tests counted in R instead of the (unused) Python counter
    ci_test.get_tests_per_order = lambda: tests
    return {"amat": amat.tolist()}
