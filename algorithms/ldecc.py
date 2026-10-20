"""
This code was adapted from the public code for
Local Causal Discovery for Estimating Causal Effects (CLeaR 2023)
by Gupta, Shantanu and Childers, David and Lipton, Zachary C.
Original code can be found at https://github.com/acmi-lab/local-causal-discovery
"""

import collections
from itertools import combinations, permutations
from typing import Callable, Sequence

import networkx as nx
import numpy as np

from algorithms.mb_by_mb import poss_mb


class MNS:
    """Instantiates a Minimal Neighbor Separator (MNS)."""

    def __init__(self, is_valid, mns=None):
        self._is_valid = is_valid
        self._mns = mns

    def __str__(self):
        if not self._is_valid:
            return "Invalid MNS"
        else:
            return str(self._mns)

    def is_valid(self):
        return self._is_valid

    def mns(self):
        if not self.is_valid:
            raise ValueError("MNS is not valid")

        return self._mns

    @staticmethod
    def equals(m1, m2):
        if (not m1.is_valid()) or (not m2.is_valid()):
            return False

        return m1.mns() == m2.mns()


def find_mb(g, sep_set, nodes, target, ci_test, alpha, bk=None):
    """
    Find the Markov blanket of a target node in a graph
    using the IAMB algorithm (without prioritization).
    """
    # Start from everything BK already places in the blanket
    mb = poss_mb(target, bk)
    # forward pass.
    cont = True
    while cont:
        cont = False
        mb_copy = set(mb)
        nodes_to_check = nodes - {target} - mb_copy
        for n in nodes_to_check:
            if ci_test(n, target, mb - {n}) < alpha:
                mb.add(n)
                cont = True
            elif g.has_edge(n, target):
                g.remove_edge(n, target)
                sep_set[n][target] = mb - {n}
                sep_set[target][n] = mb - {n}

    # backward pass.
    mb_copy = set(mb)
    for n in mb_copy:
        if ci_test(n, target, mb - {n}) >= alpha:
            if g.has_edge(n, target):
                g.remove_edge(n, target)
                sep_set[n][target] = mb - {n}
                sep_set[target][n] = mb - {n}

            mb -= {n}

    return g, sep_set, mb


def find_neighbors(g, sep_set, mb, target, ci_test, alpha, bk):
    to_check = set(mb) - set(np.where((bk[target] == -1) | (bk[:, target] == -1))[0])
    size = 0
    for size in range(len(mb)):
        for n in list(to_check):
            for S in combinations(mb - {n}, size):
                S = set(S)
                if ci_test(n, target, S) >= alpha:
                    to_check.remove(n)
                    if g.has_edge(n, target):
                        g.remove_edge(n, target)
                    sep_set[n][target] |= S
                    sep_set[target][n] |= S
                    break
        size += 1
    return g, sep_set


def get_all_neighbor_separators(g, x, y, ci_test, alpha):
    neighbors_to_check = set(g.neighbors(x))
    for size in range(0, len(neighbors_to_check) + 1):
        for possible_sep in combinations(neighbors_to_check, size):
            possible_sep = set(possible_sep)
            if ci_test(x, y, possible_sep) >= alpha:
                yield possible_sep


def mark_children_unshielded_colliders(
    g, target, sep_set, to_be_oriented, mb, neighbors, ci_test, alpha, bk
):
    children = set(np.where((bk[target] == -1) & (bk[:, target] == 1))[0])
    spouses = mb - neighbors
    for cand_ch in to_be_oriented - children:
        for spouse in spouses:
            mark_as_child = False
            for S in get_all_neighbor_separators(g, target, spouse, ci_test, alpha):
                if cand_ch in S or ci_test(cand_ch, spouse, S) >= alpha:
                    mark_as_child = False
                    break
                else:
                    mark_as_child = True
            if mark_as_child:
                children.add(cand_ch)
                break
    return g, sep_set, children


def get_mns(g, target, node, mns_cache, ci_test, alpha):
    def cache_and_get(node, mns):
        mns_cache[node] = mns
        return mns

    if mns_cache[node] is not None:
        return mns_cache[node]

    neighbors_X = set(g.neighbors(target))

    if node == target:
        raise ValueError("Can't get MNS for treatment node.")

    if node in neighbors_X:
        return cache_and_get(node, MNS(is_valid=True, mns={node}))

    for size in range(0, len(neighbors_X) + 1):
        for possible_mns in combinations(neighbors_X, size):
            possible_mns = set(possible_mns)
            if ci_test(target, node, possible_mns) >= alpha:
                return cache_and_get(node, MNS(is_valid=True, mns=possible_mns))

    return cache_and_get(node, MNS(is_valid=False))


def eager_collider_check(
    g,
    target,
    i,
    j,
    sep_set,
    t_neighbors,
    mns_cache,
    ci_test,
    alpha,
    ldecc_do_checks: bool = False,
):
    parents = set()
    if target in sep_set[i][j]:
        return g, parents

    if i in t_neighbors and j in t_neighbors:
        if ci_test(i, j, {target} | sep_set[i][j]) < alpha:
            # This means that there is an unshielded collider i -> X <- j and
            # so mark i, j as parents.
            parents = {i, j}
            # This is to account for Meek rule 3.
            parents |= sep_set[i][j] & t_neighbors
        return g, parents

    # Run an Eager collider check.
    if ci_test(i, j, {target} | sep_set[i][j]) < alpha:
        mns_i = get_mns(g, target, i, mns_cache, ci_test, alpha)
        mns_j = get_mns(g, target, j, mns_cache, ci_test, alpha)

        if mns_i.is_valid() and mns_j.is_valid():

            if len({i, j} & t_neighbors) == 1:
                if i in t_neighbors:
                    nbr = i
                    mns_non_nbr = mns_j.mns()
                else:
                    nbr = j
                    mns_non_nbr = mns_i.mns()

                if len(mns_non_nbr) < 2 or (nbr not in mns_non_nbr):
                    return g, parents

                parents = mns_i.mns() | mns_j.mns()
            else:
                mns_check_pass = True
                if ldecc_do_checks:
                    mns_check_pass = mns_i.mns() == mns_j.mns()

                if mns_check_pass:
                    parents = ((mns_i.mns() | mns_j.mns()) | {i, j}) & t_neighbors

    return g, parents


def get_non_collider_nodes(neighbors, non_colliders):
    res = set()
    for n in neighbors:
        res |= non_colliders[n]
    return res


def ldecc_alg(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    target: int,
    ldecc_do_checks: bool = False,
    bk: np.ndarray | None = None,
):
    # Prepare data structures
    mns_cache = collections.defaultdict(lambda: None)
    non_colliders = collections.defaultdict(set)
    nodes = set(range(data.shape[1]))
    num_nodes = data.shape[1]
    sep_set = [[set() for _ in range(num_nodes)] for _ in range(num_nodes)]
    g = nx.complete_graph(nodes)
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # Find and orient Markov blanket of target
    g, sep_set, mb = find_mb(g, sep_set, nodes, target, ci_test, alpha, bk)
    g, sep_set = find_neighbors(g, sep_set, mb, target, ci_test, alpha, bk)
    neighbors = set(g.neighbors(target))
    parents = set(np.where((bk[target] == 1) & (bk[:, target] == -1))[0])
    g, sep_set, children = mark_children_unshielded_colliders(
        g, target, sep_set, neighbors - parents, mb, neighbors, ci_test, alpha, bk
    )
    to_be_oriented = neighbors - parents - children

    # Run PC-style tests with Eager Collider Checks
    l = 0
    debug_skel = g.copy()
    while True:
        cont = False
        nodes_sorted = sorted(nodes, key=lambda n: 0 if n in neighbors else 1)
        for i, j in permutations(nodes_sorted, 2):
            if len(to_be_oriented) == 0:
                break
            if i == target or j == target:
                continue
            # Skip neighbors in background knowledge
            if bk[i, j] == -1 or bk[j, i] == -1:
                continue
            nb_i = set(g.neighbors(i))
            if j not in nb_i:
                continue
            else:
                nb_i.remove(j)
            poss_pa = nb_i - set(np.where((bk[:, i] == 1))[0].tolist())
            if len(poss_pa) >= l:
                poss_pa = sorted(poss_pa, key=lambda n: 0 if n == target else 1)
                for S in combinations(poss_pa, l):
                    S = set(S)
                    if len(to_be_oriented) == 0:
                        break
                    if ci_test(i, j, S) >= alpha:
                        if g.has_edge(i, j):
                            g.remove_edge(i, j)
                            debug_skel.remove_edge(i, j)
                        sep_set[i][j] |= set(S)
                        sep_set[j][i] |= set(S)
                        mark_non_collider = False
                        if target in S and i in neighbors and j in neighbors:
                            mark_non_collider = True
                            non_colliders[i].add(j)
                            non_colliders[j].add(i)

                            for nbr in to_be_oriented - {i, j} - S:
                                if ci_test(i, j, S | {nbr}) < alpha:
                                    children.add(nbr)

                        debug_skel, detected_parents = eager_collider_check(
                            debug_skel,
                            target,
                            i,
                            j,
                            sep_set,
                            neighbors,
                            mns_cache,
                            ci_test,
                            alpha,
                            ldecc_do_checks,
                        )
                        if mark_non_collider or len(detected_parents) > 0:
                            parents |= detected_parents
                            children |= get_non_collider_nodes(parents, non_colliders)
                            to_be_oriented -= detected_parents | children

                        break
                cont = True
        if len(nodes) < 2:
            break
        l += 1
        if cont is False:
            break

    # arbitrarily resolve contradictions.
    parents -= children
    return {
        "parents": set(parents),
        "children": set(children),
        "unoriented": set(to_be_oriented),
        "non_colliders": (
            {
                key: {v for v in non_colliders[key]}
                for key in parents | children | to_be_oriented
            }
        ),
    }


def get_all_combinations(unoriented: set, non_colliders: dict | None = None):
    """
    Get all locally valid parent adjustment sets.

    Args:
        unoriented (set): The set of unoriented neighbors.
        non_colliders (dict | None): Information about non-colliders.

    Returns:
        list: A list of all locally valid parent adjustment sets.
    """

    def is_valid_parent_set(par):
        par = set(par)
        for p in par:
            if len(par & non_colliders[p]) != 0:
                return False
        return True

    res = []
    for l in range(len(unoriented) + 1):
        for c in combinations(unoriented, l):
            if non_colliders is not None and not is_valid_parent_set(c):
                continue
            res.append(list(c))
    return res


def is_explicit_ancestor(
    x: int,
    y: int,
    result: dict,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is an explicit ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        result (dict): Local information of x.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.

    Returns:
        bool: True if x is an explicit ancestor of y, False otherwise.
    """
    if y in result["children"]:
        return True
    elif y in result["parents"] | result["unoriented"]:
        return False
    return ci_test(x, y, result["parents"] | result["unoriented"]) < alpha


def is_possible_ancestor(
    x: int,
    y: int,
    result: dict,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is a possible ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        result (dict): Local information of x.
        ci_test (Callable): Conditional independence test.
        alpha (float): Significance level.

    Returns:
        bool: True if x is a possible ancestor of y, False otherwise.
    """
    if y in result["children"] | result["unoriented"]:
        return True
    elif y in result["parents"]:
        return False
    return ci_test(x, y, result["parents"]) < alpha


def ldecc_plus(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: Sequence[int],
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    LDECC algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        bk: np.ndarray | None = None,
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned causal relationships.
    """
    t1, t2 = targets
    res_t1 = ldecc_alg(data, ci_test, alpha, t1, bk=bk)
    res_t2 = ldecc_alg(data, ci_test, alpha, t2, bk=bk)

    adj_sets = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, res_t1, ci_test, alpha):
        combs = get_all_combinations(res_t1["unoriented"], res_t1["non_colliders"])
        adj_sets[(t1, t2)] = [set(s) | res_t1["parents"] for s in combs]
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, res_t2, ci_test, alpha):
        combs = get_all_combinations(res_t2["unoriented"], res_t2["non_colliders"])
        adj_sets[(t2, t1)] = [set(s) | res_t2["parents"] for s in combs]
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, res_t1, ci_test, alpha):
            combs = get_all_combinations(res_t1["unoriented"], res_t1["non_colliders"])
            adj_sets[(t1, t2)] = [set(s) | res_t1["parents"] for s in combs]
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, res_t2, ci_test, alpha):
            combs = get_all_combinations(res_t2["unoriented"], res_t2["non_colliders"])
            adj_sets[(t2, t1)] = [set(s) | res_t2["parents"] for s in combs]
            ancestry[(t2, t1)] = "possible"

    return {"adj_sets": str(adj_sets), "ancestry": str(ancestry)}


def ldecc_pp(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: Sequence[int],
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    LDECC algorithm where background knowledge is only used for post-processing.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        bk: np.ndarray | None = None,
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned causal relationships.
    """
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    t1, t2 = targets
    res_t1 = ldecc_alg(data, ci_test, alpha, t1)
    res_t2 = ldecc_alg(data, ci_test, alpha, t2)

    gaps_t1 = set(np.where((bk[t1, :] == 1) & (bk[:, t1] == 1))[0])
    res_t1["unoriented"] -= gaps_t1
    res_t1["children"] -= gaps_t1
    res_t1["parents"] -= gaps_t1

    children_t1 = set(np.where((bk[t1, :] == -1) & (bk[:, t1] == 1))[0])
    res_t1["unoriented"] -= children_t1
    res_t1["children"] |= children_t1
    res_t1["parents"] -= children_t1

    parents_t1 = set(np.where((bk[t1, :] == 1) & (bk[:, t1] == -1))[0])
    res_t1["unoriented"] -= parents_t1
    res_t1["children"] -= parents_t1
    res_t1["parents"] |= parents_t1

    adj_sets = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, res_t1, ci_test, alpha):
        combs = get_all_combinations(res_t1["unoriented"], res_t1["non_colliders"])
        adj_sets[(t1, t2)] = [set(s) | res_t1["parents"] for s in combs]
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, res_t2, ci_test, alpha):
        combs = get_all_combinations(res_t2["unoriented"], res_t2["non_colliders"])
        adj_sets[(t2, t1)] = [set(s) | res_t2["parents"] for s in combs]
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, res_t1, ci_test, alpha):
            combs = get_all_combinations(res_t1["unoriented"], res_t1["non_colliders"])
            adj_sets[(t1, t2)] = [set(s) | res_t1["parents"] for s in combs]
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, res_t2, ci_test, alpha):
            combs = get_all_combinations(res_t2["unoriented"], res_t2["non_colliders"])
            adj_sets[(t2, t1)] = [set(s) | res_t2["parents"] for s in combs]
            ancestry[(t2, t1)] = "possible"

    return {"adj_sets": str(adj_sets), "ancestry": str(ancestry)}
