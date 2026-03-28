from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        G = nx.Graph()
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(( (u, v, {'weight': w}) for u, v, w in edges ))
        
        mst_edges = [[min(u, v), max(u, v), data['weight']] for u, v, data in nx.minimum_spanning_edges(G, data=True)]
        mst_edges.sort()
        
        return {"mst_edges": mst_edges}