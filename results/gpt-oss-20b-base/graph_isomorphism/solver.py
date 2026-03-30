# solver.py

from __future__ import annotations
from typing import Any, Dict, List
import networkx as nx

# The core of the solution – find an isomorphism between two undirected graphs
# using NetworkX's GraphMatcher.  The implementation tries to minimise the
# number of intermediate objects and method calls to reduce CPU usage.

class Solver:
    """
    A minimal, fast wrapper around NetworkX's VF2 graph isomorphism algorithm.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Find a node‑to‑node mapping that proves G1 and G2 are isomorphic.
        If no mapping exists, return a list of -1s.

        Parameters
        ----------
        problem : dict
            should contain:
            - ``num_nodes`` : int, number of nodes in both graphs
            - ``edges_g1``  : iterable of (int, int) tuples
            - ``edges_g2``  : iterable of (int, int) tuples

        Returns
        -------
        dict
            ``{'mapping': List[int]}``
        """
        n = problem["num_nodes"]

        # Build the graphs in a single call to add_edges_from
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(problem["edges_g1"])
        G2.add_edges_from(problem["edges_g2"])

        # VF2 matcher – returns an iterator over all isomorphisms.
        # Fetching one mapping is enough; by default the iterator returns
        # the lexicographically first matching.  The function is only
        # called if the graphs are actually isomorphic.
        matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not matcher.is_isomorphic():
            return {"mapping": [-1] * n}

        # Grab the first (and only) mapping from the iterator
        iso_map = next(matcher.isomorphisms_iter())
        # Turn the mapping dict into the required list format
        mapping = [iso_map.get(u, -1) for u in range(n)]
        return {"mapping": mapping}