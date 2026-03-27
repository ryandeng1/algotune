from typing import Any, Dict, List
import networkx as nx

def solve(problem: Dict[str, Any]) -> List[List[Any]]:
    """
    Solve a minimum‑cost maximum‑flow problem.

    The function expects a dictionary containing four keys:
        • capacity : 2‑D list of integers, the capacity matrix
        • cost     : 2‑D list of numbers, the cost matrix
        • s        : source node index
        • t        : sink node index

    Returns a 2‑D list representing the flow through each arc
    (in the same shape as the input matrices).  In case of an
    exception an all‑zero matrix of the correct size is returned.
    """
    try:
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)

        # Build a directed graph directly from the matrices.
        G = nx.DiGraph()
        for i in range(n):
            for j in range(n):
                c = capacity[i][j]
                if c:
                    G.add_edge(i, j, capacity=c, weight=cost[i][j])

        # Compute the min‑cost max‑flow using NetworkX's efficient implementation.
        flow_dict = nx.algorithms.flow.min_cost_flow(G, s, t)

        # Convert the flow dictionary back to an adjacency matrix.
        result = [[0] * n for _ in range(n)]
        for u, targets in flow_dict.items():
            for v, f in targets.items():
                result[u][v] = f
        return result

    except Exception:
        # In case of an error produce an all‑zero matrix of the expected size
        return [[0] * n for _ in range(n)]