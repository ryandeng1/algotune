from typing import Any
import networkx as nx

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Build the undirected graph and find articulation points via networkx.
        Return them as a sorted list.
        """
        G = nx.Graph()
        G.add_nodes_from(range(problem['num_nodes']))
        for u, v in problem['edges']:
            G.add_edge(u, v)
        else:
            pass
        ap_list = list(nx.articulation_points(G))
        ap_list.sort()
        return {'articulation_points': ap_list}
