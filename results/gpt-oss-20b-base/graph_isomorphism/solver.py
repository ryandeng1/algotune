# solver.py
import networkx as nx
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Find a node mapping from G1 to G2 that preserves adjacency.
        G1 and G2 are guaranteed to be isomorphic.
        """
        n = problem["num_nodes"]

        # Build graphs
        G1 = nx.Graph()
        G1.add_nodes_from(range(n))
        G1.add_edges_from(problem["edges_g1"])

        G2 = nx.Graph()
        G2.add_nodes_from(range(n))
        G2.add_edges_from(problem["edges_g2"])

        # Use VF2 algorithm from NetworkX
        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        # Since the graphs are isomorphic, we can take the first mapping
        iso_map = next(gm.isomorphisms_iter())

        # Convert dict mapping to list
        mapping = [iso_map[u] for u in range(n)]

        return {"mapping": mapping}
