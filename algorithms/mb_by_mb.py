from collections import defaultdict
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from algorithms.load import (
    get_locally_valid_parent_sets,
    is_explicit_ancestor,
    is_possible_ancestor,
)
from algorithms.orient import v_struc_pc


def poss_mb(target: int, bk: np.ndarray | None, ignore: set | None = None) -> set[int]:
    """
    Collect every node in the Markov blanket according to BK.

    Args:
        target (int): The target node.
        bk (np.ndarray | None): Background knowledge matrix.
        ignore (set | None): Nodes to leave out.

    Returns:
        set[int]: The initial Markov blanket of the target given the BK.
    """
    if bk is None:
        return set()
    if ignore is None:
        ignore = set()
    # V - T, V -> T and T -> V
    mb = set(np.where((bk[target] == -1) | (bk[:, target] == -1))[0])
    # V with T -> V' <- V for some V'
    children = np.where((bk[target] == -1) & (bk[:, target] == 1))[0]
    mb |= set(np.where((bk[children] == 1) & (bk.T[children] == -1))[1])
    return {int(v) for v in mb} - {target} - set(ignore)


def grow_shrink(
    data: np.ndarray,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    ignore: set | None = None,
    bk: np.ndarray | None = None,
) -> set:
    """
    Find the Markov blanket of a target node in a graph using the Grow-Shrink algorithm.
    Adapted from https://github.com/acmi-lab/local-causal-discovery.

    Args:
        data (np.ndarray): The data matrix.
        target (int): Target node for which to find the Markov blanket.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        set: The Markov blanket of the target node.
    """
    if ignore is None:
        ignore = set()
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    # Start from everything BK already places in the blanket
    mb = poss_mb(target, bk, ignore)
    # Forward pass
    nodes = set(range(data.shape[1])) - {target} - ignore
    cont = True
    while cont:
        cont = False
        mb_copy = set(mb)
        nodes_to_check = nodes - mb_copy
        for n in nodes_to_check:
            if ci_test(n, target, mb - {n}) < alpha:
                mb.add(n)
                cont = True

    # Backward pass
    mb_copy = set(mb)
    for n in mb_copy:
        if ci_test(n, target, mb - {n}) >= alpha:
            mb.remove(n)

    return mb


def total_conditioning(
    data: np.ndarray,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    ignore: set | None = None,
    bk: np.ndarray | None = None,
) -> set:
    """
    Find the Markov blanket of a target node in a graph using the Total Conditioning algorithm
    from Using markov blankets for causal structure learning by
    Jean-Philippe Pellet and André Elisseeff.

    Args:
        data (np.ndarray): The data matrix.
        target (int): Target node for which to find the Markov blanket.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        set: The Markov blanket of the target node.
    """
    if ignore is None:
        ignore = set()
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    # Start from everything BK already places in the blanket
    mb = poss_mb(target, bk, ignore)

    nodes = set(range(data.shape[1])) - ignore
    for n in nodes - {target} - mb:
        if ci_test(n, target, nodes - {target, n}) < alpha:
            mb.add(n)
    return mb


def learn_local_structure(
    data: np.ndarray,
    target: int,
    observed: set,
    sep_set: dict[frozenset, list],
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    bk: np.ndarray | None = None,
) -> np.ndarray:
    """
    Learn the local structure over a set of observed nodes using the PC algorithm.

    Args:
        data (np.ndarray): The data matrix.
        target (int): The target node.
        observed (set): The observed nodes.
        sep_set (dict[frozenset, list]): The separating sets.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        np.ndarray: The adjacency matrix of the learned local structure.
    """
    # Initialization
    nodes = set(range(data.shape[1]))
    g = np.full((len(nodes), len(nodes)), -1, dtype=int)
    np.fill_diagonal(g, 0)
    ignore = list(nodes.difference(observed))
    g[ignore, :] = g[:, ignore] = 0

    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    # Only consider background knowledge around target
    bk_local = np.zeros_like(g)
    observed = list(observed)
    bk_local[target, observed] = bk[target, observed]
    bk_local[observed, target] = bk[observed, target]

    # Skeleton search
    order = 0
    while np.amax(np.sum(g != 0, axis=1)) > order:
        for x in range(len(g)):
            neigh_x = np.where(g[x, :] != 0)[0]
            if len(neigh_x) < order - 1:
                continue
            for y in neigh_x:
                # Skip adjacencies from global BK
                if bk[x, y] == -1 or bk[y, x] == -1:
                    continue
                poss_pa_x = np.where((g[x, :] != 0) & (bk_local[:, x] != 1))[0]
                poss_pa_x = poss_pa_x[poss_pa_x != y]
                for S in combinations(poss_pa_x, order):
                    if ci_test(x, y, S) >= alpha:
                        g[x, y] = g[y, x] = 0
                        sep_set[frozenset((x, y))].append(S)
                        break
        order += 1
    # Orient v-structures
    g = v_struc_pc(g, sep_set)
    return g


def copy_local_structure(g: np.ndarray, observed: set) -> np.ndarray:
    """
    Create a copy of the local structure over the observed nodes.

    Args:
        g (np.ndarray): The input graph.
        observed (set): The observed nodes.

    Returns:
        np.ndarray: A copy of the local structure.
    """
    local = np.zeros_like(g, dtype=int)
    observed = list(observed)
    local[np.ix_(observed, observed)] = g[np.ix_(observed, observed)]
    return local


def update_graph(g: np.ndarray, target: int, local: np.ndarray) -> np.ndarray:
    """
    Update the graph with the local structure around the target node.

    Args:
        g (np.ndarray): The global graph.
        target (int): The target node.
        local (np.ndarray): Local structure for the target node.
    Returns:
        np.ndarray: The updated global graph.
    """
    # Put the edges connected to target in local to g
    g[target, :] = local[target, :]
    g[:, target] = local[:, target]
    # Put v-structures containing target as a parent in local to g.
    coll = np.where(local.T[target] == 1)[0]  # target *-> coll
    spouse = np.where((local[coll] == 1) & (local[target] == 0))
    coll, spouse = coll[spouse[0]], spouse[1]
    mask = spouse != target
    coll, spouse = coll[mask], spouse[mask]
    g[target, coll] = local[target, coll]
    g[coll, target] = local[coll, target]
    g[spouse, coll] = local[spouse, coll]
    g[coll, spouse] = local[coll, spouse]

    return g


def meek(g: np.ndarray, sep_set: dict[frozenset, list]) -> np.ndarray:
    """
    Orient the undirected edges in g by a revision of Meek's approach,
    as described in the MB-by-MB algorithm.

    Args:
        g (np.ndarray): The graph to orient.
        sep_set (dict[frozenset, list]): The separating sets.
    Returns:
        np.ndarray: The oriented graph.
    """

    def rule1(g: np.ndarray, sep_set: dict[frozenset, list]) -> bool:
        """
        For (a -> b - c) in g,
        if g.sepset[a, c] exists and b in g.sepset[a, c], orient b -> c.

        Args:
            g (np.ndarray): The graph to orient.
        Returns:
            bool: Whether the graph was changed.
        """
        a, b = np.where((g == -1) & (g.T == 1))  # a -> b
        c = np.where(((g[b] == -1) & (g.T[b] == -1)) & (g[a] == 0))  # b - c  # a -/- c
        # Extract actual indices
        a, b, c = a[c[0]], b[c[0]], c[1]
        # Orient as b -> c
        changed = False
        for a_, b_, c_ in zip(a, b, c):
            if frozenset((a_, c_)) in sep_set and all(
                b_ in S for S in sep_set[frozenset((a_, c_))]
            ):
                g[c_, b_] = 1
                changed = True

        return changed

    def rule2(g: np.ndarray) -> bool:
        """
        For (a -> b -> c - a) in g, orient a -> c.
        Args:
            g (np.ndarray): The graph to orient.
        Returns:
            bool: Whether the graph was changed.
        """
        a, b = np.where((g == -1) & (g.T == 1))  # a -> b
        c = np.where(
            ((g[b] == -1) & (g.T[b] == 1))  # b -> c
            & ((g[a] == -1) & (g.T[a] == -1))  # a - c
        )
        # Extract actual indices
        a, b, c = a[c[0]], b[c[0]], c[1]
        # Orient as a -> c
        g[c, a] = 1
        # return whether graph changed
        return len(a) > 0

    def rule3(g: np.ndarray, sep_set: dict[frozenset, list]) -> bool:
        """
        For a - b, a - c -> b and a - d -> b in g,
        if g.sepset[c, d] exists and a in g.sepset[c, d], orient a -> b.
        """
        a, c = np.where((g == -1) & (g.T == -1))  # a - c
        d = np.where(((g[a] == -1) & (g.T[a] == -1)) & (g[c] == 0))  # a - d -/- c
        # Extract actual indices
        a, c, d = a[d[0]], c[d[0]], d[1]
        mask = c < d
        a, c, d = a[mask], c[mask], d[mask]
        b = np.where(
            ((g[a] == -1) & (g.T[a] == -1))  # a - b
            & ((g[c] == -1) & (g.T[c] == 1))  # c -> b
            & ((g[d] == -1) & (g.T[d] == 1))  # d -> b
        )
        # Extract actual indices
        a, c, d, b = a[b[0]], c[b[0]], d[b[0]], b[1]
        # Orient as b -> a
        changed = False
        for a_, b_, c_, d_ in zip(a, b, c, d):
            if frozenset((c_, d_)) in sep_set and all(
                a_ in S for S in sep_set[frozenset((c_, d_))]
            ):
                g[b_, a_] = 1
                changed = True
        return changed

    def rule4(pdag: np.ndarray) -> bool:
        """
        Orient a - b with a - c -> d -> b and a - d -> b and c -/- b as a -> b in place.
        Args:
            pdag (np.ndarray): The partially directed graph.
        Returns:
            bool: Whether the graph changed.
        """
        a, c = np.where((pdag == -1) & (pdag.T == -1))  # a - c
        k = np.where(
            ((pdag[a] == -1) & (pdag.T[a] == -1))  # a - k
            & ((pdag[c] == -1) & (pdag.T[c] == 1))  # c -> k
        )
        # Extract actual indices
        a, c, d = a[k[0]], c[k[0]], k[1]
        # No need for masking because j and k are not interchangeable
        b = np.where(
            ((pdag[a] == -1) & (pdag.T[a] == -1))  # a - b
            & ((pdag[c] == 0))  # c -/- b
            & ((pdag[d] == -1) & (pdag.T[d] == 1))  # d -> b
        )
        # Extract actual indices
        a, c, d, b = a[b[0]], c[b[0]], d[b[0]], b[1]
        # Orient as a -> b
        pdag[b, a] = 1
        # return whether graph changed
        return len(a) > 0

    g = g.copy()
    while rule1(g, sep_set) or rule2(g) or rule3(g, sep_set) or rule4(g):
        continue
    return g


def reach_with_undirected(g: np.ndarray, target: int) -> np.ndarray:
    """
    Find all nodes that can reach the target node with an undirected path.
    Args:
        g (np.ndarray): The partially directed graph.
        target (int): The target node.
    Returns:
        np.ndarray: An array of nodes that can reach the target node with an undirected path
    """
    undir = (g == -1) & (g.T == -1)
    np.fill_diagonal(undir, 1)
    reach = np.linalg.matrix_power(undir, undir.shape[0] - 1)
    mask = reach[target] != 0
    mask[target] = False
    nodes = np.where(mask)[0]
    return nodes


def mb_by_mb_alg(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    mb_algorithm: str = "grow_shrink",
    mb: dict[int, set] | None = None,
    L: dict[int, np.ndarray] | None = None,
    sep_set: dict[frozenset, list] | None = None,
    ignore: set | None = None,
    bk: np.ndarray | None = None,
) -> np.ndarray:
    """
    MB-by-MB algorithm for learning the local network around a target node.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        target (int): The target node for which to learn the local network.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        mb (dict[int, set]): Pre-computed Markov blankets of nodes.
        L (dict[int, np.ndarray]): Pre-computed local structures of nodes.
        sep_set (dict[frozenset, list]): Pre-computed separating sets.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        np.ndarray: The learned local network around the target node.
    """
    # Input
    if mb_algorithm == "grow_shrink":
        mb_algorithm = grow_shrink
    elif mb_algorithm == "total_conditioning":
        mb_algorithm = total_conditioning
    else:
        raise ValueError(f"Unknown mb_algorithm: {mb_algorithm}")

    if mb is None:
        mb = dict()
    if L is None:
        L = dict()
    if sep_set is None:
        sep_set = defaultdict(list)
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    if ignore is None:
        ignore = set()
    else:
        # Remove BK about ignored nodes
        bk[list(ignore), :] = bk[:, list(ignore)] = 0
        # Remove gaps
        bk[(bk == 1) & (bk.T == 1)] = 0

    # 1. Initialization
    # Nodes whose MBs have been found
    done_list = set()
    # Nodes whose MBs will be found
    wait_list = [target]
    # The constructed local network around target
    # with -1 for edge tail, 1 for edge head, and 0 for no edge
    g = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # 2. Repeat
    while len(wait_list) > 0:  # 7. Until wait_list is empty
        # Take a node x from the head of wait_list
        x = wait_list.pop(0)
        # Find mb[x]
        if x not in mb:
            mb[x] = mb_algorithm(data, x, ci_test, alpha, ignore, bk)
        mb_x = mb[x].union({x})
        # Add [mb[x] \ done_list \ wait_list] to the tail of wait_list
        wait_list.extend([v for v in mb[x] if v not in list(done_list) + wait_list])
        # Add x to done_list
        done_list.add(x)

        # 3. Learn the local structure L[x] over mb+(x)
        # If L[x] is already learned, skip
        if x in L:
            pass
        # If mb+(x) is a subset of mb+(n) for some n in done_list
        elif (
            n := next((n for n in done_list if n != x and mb_x < mb[n] | {n}), None)
        ) is not None:
            # Set L[x] equal to the substructure of L[n] over mb+(x)
            L[x] = copy_local_structure(L[n], mb_x)
        # Else If mb(x) is a subset of done_list
        elif mb[x] < done_list:
            # Set L[x] equal to the substructure of g over mb+(x)
            L[x] = copy_local_structure(g, mb_x)
        else:
            # learn L[x] from observed data of mb[x]+
            L[x] = learn_local_structure(data, x, mb_x, sep_set, ci_test, alpha, bk)

        # 4. Put the edges connected to x and the v-structures containing x in L[x] to g
        g = update_graph(g, x, L[x])
        # 5. Orient undirected edges in G
        ori_bk_row, ori_bk_col = np.where((bk == -1) & (bk.T == 1))
        g[ori_bk_row, ori_bk_col] = -1
        g[ori_bk_col, ori_bk_row] = 1
        g = meek(g, sep_set)
        # 6. Remove all nodes from wait_list whose paths to target in g are blocked by directed edges.
        connected = reach_with_undirected(g, target)
        wait_list = [w for w in wait_list if w in connected]

    # Output: the local network g around the target
    return g


def mb_by_mb_alg_pp(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    mb_algorithm: str = "grow_shrink",
    mb: dict[int, set] | None = None,
    L: dict[int, np.ndarray] | None = None,
    sep_set: dict[frozenset, list] | None = None,
    ignore: set | None = None,
    bk: np.ndarray | None = None,
) -> np.ndarray:
    """
    MB-by-MB algorithm for learning the local network around a target node,
    where background knowledge is only used for post-processing.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        target (int): The target node for which to learn the local network.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        mb (dict[int, set]): Pre-computed Markov blankets of nodes.
        L (dict[int, np.ndarray]): Pre-computed local structures of nodes.
        sep_set (dict[frozenset, list]): Pre-computed separating sets.
        ignore (set | None): Nodes to leave out.
        bk (np.ndarray | None): Background knowledge matrix.
    Returns:
        np.ndarray: The learned local network around the target node.
    """
    # Input
    if mb_algorithm == "grow_shrink":
        mb_algorithm = grow_shrink
    elif mb_algorithm == "total_conditioning":
        mb_algorithm = total_conditioning
    else:
        raise ValueError(f"Unknown mb_algorithm: {mb_algorithm}")

    if mb is None:
        mb = dict()
    if L is None:
        L = dict()
    if sep_set is None:
        sep_set = defaultdict(list)
    if bk is None:
        bk = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    if ignore is None:
        ignore = set()
    else:
        # Remove BK about ignored nodes
        bk[list(ignore), :] = bk[:, list(ignore)] = 0
        # Remove gaps
        bk[(bk == 1) & (bk.T == 1)] = 0

    # 1. Initialization
    # Nodes whose MBs have been found
    done_list = set()
    # Nodes whose MBs will be found
    wait_list = [target]
    # The constructed local network around target
    # with -1 for edge tail, 1 for edge head, and 0 for no edge
    g = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # 2. Repeat
    while len(wait_list) > 0:  # 7. Until wait_list is empty
        # Take a node x from the head of wait_list
        x = wait_list.pop(0)
        # Find mb[x]
        if x not in mb:
            mb[x] = mb_algorithm(data, x, ci_test, alpha, ignore)
        mb_x = mb[x].union({x})
        # Add [mb[x] \ done_list \ wait_list] to the tail of wait_list
        wait_list.extend([v for v in mb[x] if v not in list(done_list) + wait_list])
        # Add x to done_list
        done_list.add(x)

        # 3. Learn the local structure L[x] over mb+(x)
        # If L[x] is already learned, skip
        if x in L:
            pass
        # If mb+(x) is a subset of mb+(n) for some n in done_list
        elif (
            n := next((n for n in done_list if n != x and mb_x < mb[n] | {n}), None)
        ) is not None:
            # Set L[x] equal to the substructure of L[n] over mb+(x)
            L[x] = copy_local_structure(L[n], mb_x)
        # Else If mb(x) is a subset of done_list
        elif mb[x] < done_list:
            # Set L[x] equal to the substructure of g over mb+(x)
            L[x] = copy_local_structure(g, mb_x)
        else:
            # learn L[x] from observed data of mb[x]+
            L[x] = learn_local_structure(data, x, mb_x, sep_set, ci_test, alpha)

        # 4. Put the edges connected to x and the v-structures containing x in L[x] to g
        g = update_graph(g, x, L[x])
        # 5. Orient undirected edges in G
        ori_bk_row, ori_bk_col = np.where((bk == -1) & (bk.T == 1))
        g[ori_bk_row, ori_bk_col] = -1
        g[ori_bk_col, ori_bk_row] = 1
        g = meek(g, sep_set)
        # 6. Remove all nodes from wait_list whose paths to target in g are blocked by directed edges.
        connected = reach_with_undirected(g, target)
        wait_list = [w for w in wait_list if w in connected]

    # Output: the local network g around the target
    return g


def mb_by_mb_plus(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    MB-by-MB algorithm with extra steps to determine causal relationships.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        bk (np.ndarray | None): Background knowledge matrix.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Locally valid parent adjustment sets and learned causal relationships.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    t1, t2 = targets
    G[t1] = mb_by_mb_alg(data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set, bk=bk)
    G[t2] = mb_by_mb_alg(data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set, bk=bk)

    adj_sets = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
            ancestry[(t2, t1)] = "possible"

    return {"adj_sets": str(adj_sets), "ancestry": str(ancestry)}


def mb_by_mb_pp(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    bk: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    MB-by-MB algorithm with extra steps to determine causal relationships,
    where background knowledge is only used for post-processing.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        bk (np.ndarray | None): Background knowledge matrix.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Locally valid parent adjustment sets and learned causal relationships.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    t1, t2 = targets
    G[t1] = mb_by_mb_alg_pp(
        data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set, bk=bk
    )
    G[t2] = mb_by_mb_alg_pp(
        data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set, bk=bk
    )

    adj_sets = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
            ancestry[(t2, t1)] = "possible"

    return {"adj_sets": str(adj_sets), "ancestry": str(ancestry)}
