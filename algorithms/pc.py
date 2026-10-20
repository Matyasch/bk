from collections import defaultdict
from itertools import combinations
from typing import Callable

import numpy as np

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
) -> dict[str, np.ndarray]:
    """
    PC algorithm by Spirtes et al. (2000) with background knowledge.

    The background knowledge is represented as a matrix bk where:
        - bk[i, j] = -1 and bk[j, i] = -1 indicates adjacency: i - j.
        - bk[i, j] = -1 and bk[j, i] =  1 indicates orientation: i -> j.
        - bk[i, j] =  1 and bk[j, i] =  1 indicates gap: i -/- j.
        - bk[i, j] =  0 and bk[j, i] =  0 indicates no knowledge about i and j.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        bk (np.ndarray | None): Background knowledge matrix.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, np.ndarray]: The adjacency matrix amat of the MPDAG, where
            amat[i, j] = 1 and amat[j, i] = 0 indicates i -> j, and
            amat[i, j] = 1 and amat[j, i] = 1 indicates i - j.
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
    return {"amat": amat}
