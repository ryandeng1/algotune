from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """Find an isomorphism between two equal‑size graphs using
        NetworkX's GraphMatcher.  The function is written without
        superfluous control flow to keep it minimal and fast for
        the expected input sizes.
        """
        import networkx as nx

        n = problem["num_nodes"]
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))

        G1.add_edges_from(problem["edges_g1"])
        G2.add_edges_from(problem["edges_g2"])

        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not gm.is_isomorphic():
            return {"mapping": [-1] * n}

        iso_map = next(gm.isomorphisms_iter())
        return {"mapping": [iso_map[u] for u in range(n)]}