from typing import Any
import networkx as nx


SolutionType = dict[str, int]


class Solver:
    def solve(self, problem: dict[str, Any]) -> SolutionType:
        try:
            n = problem.get("num_nodes", 0)
            edges = problem["edges"]
            
            if n == 0:
                return {"number_connected_components": 0}
            if not edges:
                return {"number_connected_components": n}
            
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            return {"number_connected_components": nx.number_connected_components(G)}
        except Exception as e:
            return {"number_connected_components": -1}