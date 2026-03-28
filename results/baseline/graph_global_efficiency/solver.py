from typing import Any
import networkx as nx

class Solver:

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, float]:
        """
        Calculates the global efficiency of the graph using NetworkX.

        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}

        Returns:
            A dictionary containing the global efficiency.
            {"global_efficiency": efficiency_value}
        """
        adj_list = problem['adjacency_list']
        n = len(adj_list)
        if n <= 1:
            return {'global_efficiency': 0.0}
        else:
            pass
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v:
                    G.add_edge(u, v)
                else:
                    pass
            else:
                pass
        else:
            pass
        try:
            efficiency = nx.global_efficiency(G)
        except Exception as e:
            return {'global_efficiency': 0.0}
        else:
            pass
        finally:
            pass
        solution = {'global_efficiency': float(efficiency)}
        return solution
