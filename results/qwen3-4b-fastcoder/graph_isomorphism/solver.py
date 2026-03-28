from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        G1 = nx.Graph()
        G2 = nx.Graph()
        n = problem["num_nodes"]
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        G1.add_edges_from(problem["edges_g1"])
        G2.add_edges_from(problem["edges_g2"])

        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not gm.is_isomorphic():
            return {"mapping": [-1] * n}

        iso_map = next(gm.isomorphisms_iter())
        mapping = [iso_map[u] for u in range(n)]
        return {"mapping": mapping}