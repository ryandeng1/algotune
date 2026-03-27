from typing import Any
import networkx as nx


def dict_to_graph(data):
    capacity = data["capacity"]
    cost = data["cost"]
    s_idx = data["s"]
    t_idx = data["t"]
    n = len(capacity)

    # Create empty directed graph
    G = nx.DiGraph()

    # Add nodes with integer labels
    for i in range(n):
        G.add_node(i)

    # Add edges with capacity and cost
    for i in range(n):
        for j in range(n):
            if capacity[i][j] > 0:  # optional: only add existing edges
                G.add_edge(i, j, capacity=capacity[i][j], cost=cost[i][j])

    return G, s_idx, t_idx


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[Any]]:
        """
        Solves the minimum weight assignment problem using scipy.sparse.csgraph.

        :param problem: A dictionary representing the max flow min cost.
        :return: A 2-d list containing the flow for each edge (adjacency matrix format).
        """
        try:
            n = len(problem["capacity"])
            G, s, t = dict_to_graph(problem)
            mincostFlow = nx.max_flow_min_cost(G, s, t)
            solution = [[0 for _ in range(n)] for _ in range(n)]

            for i in range(n):
                if i not in mincostFlow:
                    continue
                for j in range(n):
                    if j not in mincostFlow[i]:
                        continue
                    solution[i][j] = mincostFlow[i][j]

        except Exception as e:
            return [[0 for _ in range(n)] for _ in range(n)]  # Indicate failure

        return solution
