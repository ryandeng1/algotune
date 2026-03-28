from typing import Any, Dict, List
import networkx as nx

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Find a graph isomorphism between two undirected graphs with the same
        number of nodes using NetworkX's VF2 algorithm.

        Parameters
        ----------
        problem : dict
            Must contain:
                'num_nodes' : int
                'edges_g1'  : list[tuple[int, int]]
                'edges_g2'  : list[tuple[int, int]]

        Returns
        -------
        dict
            { 'mapping': List[int] } where mapping[u] is the vertex in G2
            that vertex u in G1 is mapped to. If no isomorphism exists,
            the list contains -1 for every vertex.
        """
        n = problem['num_nodes']
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))

        G1.add_edges_from(problem['edges_g1'])
        G2.add_edges_from(problem['edges_g2'])

        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)

        try:
            iso_map = next(gm.isomorphisms_iter())
            return {'mapping': [iso_map[u] for u in range(n)]}
        except StopIteration:
            return {'mapping': [-1] * n}