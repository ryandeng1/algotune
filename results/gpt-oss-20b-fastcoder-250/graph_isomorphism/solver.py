from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Find a mapping that demonstrates the isomorphism between two graphs
        described by the problem dictionary.  The mapping is returned as a
        list where mapping[u] == v means node `u` in the first graph maps to
        node `v` in the second graph.

        The implementation builds two networkx graphs, uses the
        :class:`networkx.algorithms.isomorphism.GraphMatcher` to find an
        isomorphism, and returns the first mapping found.  It never
        constructs more than the minimal amount of data required.

        :param problem: dict with keys:
            * 'num_nodes'   – number of nodes in each graph,
            * 'edges_g1'    – iterable of edge tuples for graph G1,
            * 'edges_g2'    – iterable of edge tuples for graph G2.
        :return: dict with key 'mapping' pointing to a list of integers.
        """
        n = problem["num_nodes"]
        G1 = nx.Graph()
        G2 = nx.Graph()

        # Create nodes; they are simply integer indices 0 .. n-1.
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))

        # Bulk add edges – this is much faster than calling add_edge
        # repeatedly in a tight loop.
        G1.add_edges_from(problem["edges_g1"])
        G2.add_edges_from(problem["edges_g2"])

        # Build the matcher and check isomorphism.  If the input is not
        # isomorphic we return a vector of -1 to signal failure.
        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not gm.is_isomorphic():
            return {"mapping": [-1] * n}

        # Grab the first mapping the iterator yields – this is
        # deterministic for a given input and the problem guarantees at
        # least one mapping exists.
        iso_map = next(gm.isomorphisms_iter())

        # Convert the dictionary mapping into a list indexed by nodes.
        mapping = [iso_map[u] for u in range(n)]
        return {"mapping": mapping}