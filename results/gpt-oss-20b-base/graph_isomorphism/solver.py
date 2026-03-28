from typing import Any, Dict, List
import networkx as nx

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """Find the isomorphism mapping from G1 to G2 using NetworkX VF2."""
        n = problem['num_nodes']
        # Build graphs quickly using built‑in list constructors
        G1 = nx.Graph()
        G1.add_nodes_from(range(n))
        G1.add_edges_from(problem['edges_g1'])

        G2 = nx.Graph()
        G2.add_nodes_from(range(n))
        G2.add_edges_from(problem['edges_g2'])

        # Run VF2 isomorphism test
        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not gm.is_isomorphic():
            return {'mapping': [-1] * n}

        # Obtain the first (and only) mapping
        iso_map = next(gm.isomorphisms_iter())
        mapping = [iso_map[u] for u in range(n)]
        return {'mapping': mapping}