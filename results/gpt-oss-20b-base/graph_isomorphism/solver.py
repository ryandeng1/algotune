from typing import Any, Dict, List
import networkx as nx
from itertools import islice


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Find an isomorphism from G1 to G2 using NetworkX's VF2 algorithm.
        Returns a list `mapping` where mapping[u] = v indicates node u in G1
        maps to node v in G2.
        """
        n = problem["num_nodes"]

        # Fast graph construction: create from edge lists directly
        G1 = nx.from_edgelist(problem["edges_g1"], create_using=nx.Graph())
        G2 = nx.from_edgelist(problem["edges_g2"], create_using=nx.Graph())

        # Ensure all nodes 0..n-1 are present even if isolated
        if len(G1) < n:
            G1.add_nodes_from(range(n))
        if len(G2) < n:
            G2.add_nodes_from(range(n))

        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)

        if not gm.is_isomorphic():
            return {"mapping": [-1] * n}

        # Get the first found isomorphism
        try:
            iso_map = next(islice(gm.isomorphisms_iter(), 1))
        except StopIteration:
            return {"mapping": [-1] * n}

        mapping = [iso_map[u] for u in range(n)]
        return {"mapping": mapping}