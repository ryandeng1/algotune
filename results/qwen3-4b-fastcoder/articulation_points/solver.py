from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        G = nx.Graph()
        G.add_nodes_from(range(problem["num_nodes"]))
        G.add_edges_from(problem["edges"])
        ap_list = list(nx.articulation_points(G))
        ap_list.sort()
        return {"articulation_points": ap_list}