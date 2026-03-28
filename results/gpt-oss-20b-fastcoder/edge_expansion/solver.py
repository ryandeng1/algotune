from typing import Any

def solve(problem: dict[str, Any]) -> dict[str, float]:
    """
    Compute the directed edge expansion of the subset S in the given graph.

    The edge expansion is defined as:
        |E(S, V\S)| / |S|
    where:
        - E(S, V\S) is the set of directed edges with tail in S and head in the complement of S.
        - |S| is the number of nodes in S.

    If S is empty or contains all nodes, the expansion is defined to be 0.0.

    Parameters
    ----------
    problem : dict[str, Any]
        Should contain:
        - 'adjacency_list': a list of lists of out-neighbor indices.
        - 'nodes_S': a list of indices representing the subset S.

    Returns
    -------
    dict[str, float]
        Dictionary with a single key 'edge_expansion' mapping to the computed value.
    """
    adj = problem.get("adjacency_list", [])
    nodes_S = problem.get("nodes_S", [])

    n = len(adj)
    if n == 0 or not nodes_S:
        return {"edge_expansion": 0.0}

    S_set = set(nodes_S)
    if len(S_set) == n:
        return {"edge_expansion": 0.0}

    # Count edges from S to its complement
    out_edges = 0
    for u in S_set:
        for v in adj[u]:
            if v not in S_set:
                out_edges += 1

    expansion = out_edges / len(S_set)
    return {"edge_expansion": float(expansion)}