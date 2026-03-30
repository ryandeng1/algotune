# solver.py

import networkx as nx
from typing import Any, Dict, List

class Solver:
    """
    Very small graph isomorphism solver using NetworkX's VF2 implementation.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Compute an isomorphism from G1 to G2 when it exists. If no isomorphism
        exists, return a list of -1 of length n.

        Parameters
        ----------
        problem : dict
            Must contain:
                - 'num_nodes' : int
                - 'edges_g1' : iterable of (u, v)
                - 'edges_g2' : iterable of (u, v)

        Returns
        -------
        dict
            A single key 'mapping' whose value is a list of length n.
            mapping[u] == v means node u in G1 maps to node v in G2.
        """
        n = problem['num_nodes']
        # Build the two graphs
        G1 = nx.Graph()
        G1.add_nodes_from(range(n))
        G1.add_edges_from(problem['edges_g1'])

        G2 = nx.Graph()
        G2.add_nodes_from(range(n))
        G2.add_edges_from(problem['edges_g2'])

        # Try to get the first isomorphism
        try:
            iso_map = next(nx.algorithms.isomorphism.GraphMatcher(G1, G2).isomorphisms_iter())
        except StopIteration:
            return {'mapping': [-1] * n}

        # Convert mapping to a list indexed by G1 node
        mapping = [iso_map[u] for u in range(n)]
        return {'mapping': mapping}