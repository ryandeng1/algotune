# solver.py
from typing import Any, Dict, List
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Return a mapping between the nodes of two isomorphic undirected graphs.

        Parameters
        ----------
        problem : dict
            A dictionary with keys:
                "num_nodes" : int
                "edges_g1" : list[list[int]]
                "edges_g2" : list[list[int]]
            The two graphs are guaranteed to be isomorphic.
        Returns
        -------
        dict
            {"mapping": List[int]} where mapping[u] = v means node u in G1 maps to node v in G2.
        """
        n = problem["num_nodes"]

        G1 = nx.Graph()
        G2 = nx.Graph()

        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))

        G1.add_edges_from(problem["edges_g1"])
        G2.add_edges_from(problem["edges_g2"])

        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        # Since the graphs are guaranteed to be isomorphic, the first mapping is valid.
        iso_map = next(gm.isomorphisms_iter())
        mapping = [iso_map[u] for u in range(n)]

        return {"mapping": mapping}
