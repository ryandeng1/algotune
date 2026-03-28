from typing import Any
import networkx as nx

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Find an isomorphism mapping from graph G1 to graph G2.
        If G1 and G2 are not isomorphic, return list of -1.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'num_nodes' (int): number of nodes in each graph.
            - 'edges_g1' (Iterable[tuple[int, int]]): edges of the first graph.
            - 'edges_g2' (Iterable[tuple[int, int]]): edges of the second graph.

        Returns
        -------
        dict
            Mapping from nodes of G1 to nodes of G2:
            { 'mapping': List[int] }
        """
        n = problem['num_nodes']

        # Build the graphs
        G1 = nx.empty_graph(n)
        G2 = nx.empty_graph(n)

        G1.add_edges_from(problem['edges_g1'])
        G2.add_edges_from(problem['edges_g2'])

        # Quick cardinality check
        if G1.number_of_edges() != G2.number_of_edges():
            return {'mapping': [-1] * n}

        # Use a lightweight isomorphism test that stops after the first mapping
        matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        try:
            iso_map = next(matcher.isomorphisms_iter())
        except StopIteration:
            return {'mapping': [-1] * n}

        # Convert dict to list (order by node index)
        mapping = [iso_map[u] for u in range(n)]
        return {'mapping': mapping}