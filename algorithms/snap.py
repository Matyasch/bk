from collections import defaultdict
from itertools import combinations
from typing import Callable

import numpy as np

from algorithms.orient import v_struc_pc, v_struc_rfci, cpdag2mpdag
from algorithms.pc import skeleton_step


def get_b_poss_anc(g: np.ndarray, targets: list[int]) -> set[int]:
    """
    Get b-possible ancestors of all targets via depth-first search.

    Args:
        g (np.ndarray): The causal graph.
        targets (list[int]): The target nodes.

    Returns:
        list[int]: Possible ancestors of any target.
    """

    def dfs(g: np.ndarray, current: int, visited: list[int]) -> set[int]:
        """
        Depth-first search to find all b-possible ancestors from the current.

        Args:
            g (np.ndarray): The partialy oriented graph.
            current (int): The current node.
            visited (set[int]): The set of visited nodes.
        """
        # Get possible parents...
        poss_pa = g[:, current] == -1
        # that were not visited before...
        poss_pa[visited] = False
        # and that are on unshieded paths (no edge to previous node)
        if len(visited) > 0:
            poss_pa[g[visited[-1]] != 0] = False
        # Recurse on possible parents
        b_poss_anc = {current}
        for v in np.where(poss_pa)[0]:
            b_poss_anc |= dfs(g, v, visited + [current])
        return b_poss_anc

    return set().union(*[dfs(g, t, []) for t in targets])


def get_poss_anc(g: np.ndarray, targets: list[int]) -> set[int]:
    """
    Get possible ancestors of any target.

    Args:
        g (np.ndarray): The causal graph.
        targets (list[int]): The target nodes.

    Returns:
        list[int]: Possible ancestors of any target.
    """
    g = -g.copy()
    g[g == -1] = 0
    np.fill_diagonal(g, 1)
    g = g.astype(bool)
    # Reachability matrix
    reach = np.linalg.matrix_power(g, g.shape[0] - 1)
    # Nodes that reach at least one target
    poss_anc = np.any(reach[:, targets], axis=1)
    return set(np.where(poss_anc)[0])


def find_missing_separating_sets_at_order(
    order: int,
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
        order (int): The order of CI tests.
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
        if x < y or frozenset((x, y)) in sep_set:
            continue
        # Search among neighbors of x
        found = False
        poss_pa_x = np.where((g[x, :] != 0) & (bk[:, x] != 1))[0]
        poss_pa_x = poss_pa_x[poss_pa_x != y]
        for S in combinations(poss_pa_x, order):
            if ci_test(x, y, S) >= alpha:
                g[x, y] = g[y, x] = 0
                sep_set[frozenset((x, y))].append(S)
                found = True
                break
        if found:
            continue
        # Search among neighbors of y
        poss_pa_y = np.where((g[y, :] != 0) & (bk[:, y] != 1))[0]
        poss_pa_y = poss_pa_y[poss_pa_y != x]
        for S in combinations(poss_pa_y, order):
            if ci_test(x, y, S) >= alpha:
                g[x, y] = g[y, x] = 0
                sep_set[frozenset((x, y))].append(S)
                found = True
                break
    return g


def snap(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    targets: list[int],
    max_order: int = -1,
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Sequential Non-Ancestor Pruning by Schubert et al. (2025) with background knowledge.

    The background knowledge is represented as a matrix bk where:
        - bk[i, j] = -1 and bk[j, i] = -1 indicates adjacency: i - j.
        - bk[i, j] = -1 and bk[j, i] =  1 indicates orientation: i -> j.
        - bk[i, j] =  1 and bk[j, i] =  1 indicates gap: i -/- j.
        - bk[i, j] =  0 and bk[j, i] =  0 indicates no knowledge about i and j.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        max_order (int): The maximum order of CI tests for SNAP(k). If negative, run SNAP(infinity) until completion (default: -1).
        bk (np.ndarray | None): Background knowledge matrix.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, np.ndarray]: The adjacency matrix amat of the MPDAG pruned to the possible ancestors of the targets, where
            amat[i, j] = 1 and amat[j, i] = 0 indicates i -> j, and
            amat[i, j] = 1 and amat[j, i] = 1 indicates i - j.
    """
    num_nodes = data.shape[1]
    if bk is None:
        bk = np.zeros((num_nodes, num_nodes), dtype=int)

    if max_order < 0:  # Run SNAP(infinity) until completion
        max_order = num_nodes - 1

    # Initialize skeleton with -1 for edge tail, 1 for edge head, and 0 for no edge
    skeleton = np.full((num_nodes, num_nodes), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    all_nodes = set(range(num_nodes))
    for order in range(max_order + 1):
        if np.amax(np.sum(skeleton != 0, axis=1)) <= order:
            break
        # Perform skeleton step
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set, bk)
        # Find missing separating sets at current order and fill gaps when not found
        skeleton = find_missing_separating_sets_at_order(
            order, skeleton, ci_test, alpha, sep_set, bk
        )
        # Orient v-structures
        if order < 1:
            g = v_struc_pc(skeleton, sep_set, bidirected=True)
        else:
            g, skeleton = v_struc_rfci(skeleton, ci_test, alpha, sep_set)

        # Apply background knowledge
        ori_bk_row, ori_bk_col = np.where((bk == -1) & (bk.T == 1))  # i -> j
        g[ori_bk_row, ori_bk_col] = -1
        g[ori_bk_col, ori_bk_row] = 1
        g[np.logical_and(bk == 1, bk.T == 1)] = 0
        # Get b-possibly ancestral set
        poss_anc = get_poss_anc(g, targets)
        # Prune other nodes
        non_anc = list(all_nodes - poss_anc)
        skeleton[non_anc, :] = skeleton[:, non_anc] = 0
        bk[non_anc, :] = bk[:, non_anc] = 0

    # If ran until completion
    if max_order == num_nodes - 1:
        # Orient v-structures without bidirected edges
        pdag = v_struc_pc(skeleton, sep_set)
        # Orient background knowledge and Meek rules
        res = cpdag2mpdag(pdag, bk)
        # Get b-possibly ancestral set
        poss_anc = get_b_poss_anc(res, targets)
        # Prune other nodes
        non_anc = list(all_nodes - poss_anc)
        res[non_anc, :] = res[:, non_anc] = 0
    else:
        res = pdag

    # Convert to networkx format
    amat = np.zeros_like(res, dtype=int)
    amat[res == -1] = 1
    return {"amat": amat}
