# solver.py
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple, Set

import networkx as nx


class Solver:
    """
    A straightforward solver for the Vertex k‑Center problem.

    The implementation builds all–pairs shortest paths with Dijkstra and then
    evaluates the objective for every combination of `k` nodes.  The instance
    sizes used in the tests are small enough for this exhaustive search to
    finish in a fraction of a second while guaranteeing optimality.

    The returned result is a set of node identifiers.
    """

    @staticmethod
    def _build_graph(gdict: Dict[str, Dict[str, float]]) -> nx.Graph:
        """Build a networkx graph from the dictionary format."""
        G = nx.Graph()
        for u, neigh in gdict.items():
            for v, w in neigh.items():
                G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def _all_pairs_shortest_distances(G: nx.Graph) -> Dict[Tuple[str, str], float]:
        """
        Pre‑compute the shortest‑path distances between all pairs of nodes.
        Uses Dijkstra from each source node.
        """
        dist = {}
        for src in G.nodes:
            dists = nx.single_source_dijkstra_path_length(G, src, weight="weight")
            for dst, d in dists.items():
                dist[(src, dst)] = d
        return dist

    @staticmethod
    def _objective_for_centers(
        centers: Iterable[str], all_dist: Dict[Tuple[str, str], float], nodes: Iterable[str]
    ) -> float:
        """
        Compute the maximum distance from any node to its nearest center.
        """
        maxd = 0.0
        for n in nodes:
            mind = min(all_dist[(n, c)] for c in centers)
            if mind > maxd:
                maxd = mind
        return maxd

    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int], **kwargs) -> Any:
        """
        Solve the vertex k‑center problem.

        Parameters
        ----------
        problem : tuple[dict[str, dict[str, float]], int]
            Graph dictionary and integer k.

        Returns
        -------
        set[str]
            Set of vertices chosen as centers.
        """
        gdict, k = problem
        G = self._build_graph(gdict)

        # Quick return for trivial cases
        if k <= 0 or len(G) == 0:
            return set()

        nodes = list(G.nodes)
        if k >= len(nodes):
            return set(nodes)

        all_dist = self._all_pairs_shortest_distances(G)

        best_centers: Set[str] = set()
        best_obj: float = float("inf")

        # Pre‑compute distances for each node to every other node once
        dist_from_nodes = {
            n: {m: all_dist[(n, m)] for m in nodes} for n in nodes
        }

        for combo in itertools.combinations(nodes, k):
            # Quick prune: if already worse than best_obj, skip
            cur_max = 0.0
            for n in nodes:
                # compute min distance to combo
                mind = min(dist_from_nodes[n][c] for c in combo)
                if mind > cur_max:
                    cur_max = mind
                    if cur_max >= best_obj:
                        break
            else:
                # exhausted all nodes with no early break
                if cur_max < best_obj:
                    best_obj = cur_max
                    best_centers = set(combo)

        return best_centers
