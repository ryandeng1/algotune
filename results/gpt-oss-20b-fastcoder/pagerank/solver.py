from __future__ import annotations
from typing import Dict, List
import numpy as np

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-6

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """
        Compute PageRank with a small custom implementation that avoids NetworkX
        (which is slow for large inputs).  The algorithm is the standard power
        iteration with damping factor `alpha`, using efficient NumPy operations.

        Args:
            problem: {"adjacency_list": adj_list}
                     adj_list is a list of lists, each inner list contains
                     neighbors of node i.

        Returns:
            {"pagerank_scores": [score_0, ..., score_{n-1}]}
            Handles empty or single‑node graphs explicitly.
        """
        adj = problem["adjacency_list"]
        n = len(adj)

        # Special cases: empty or one node
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Pre‑compute out‑degree and the transition matrix in compressed form
        out_deg = np.array([len(neigh) for neigh in adj], dtype=int)
        # Replace zero out‑degree with self‑loop to avoid dangling sinks
        dangling = out_deg == 0
        out_deg[dangling] = 1

        # Build adjacency as list of (source, target) tuples for fast access
        src = np.empty(0, dtype=int)
        dst = np.empty(0, dtype=int)
        for u, neigh in enumerate(adj):
            if neigh:
                src = np.concatenate((src, np.full(len(neigh), u, dtype=int)))
                dst = np.concatenate((dst, np.array(neigh, dtype=int)))
        # Transition probabilities for outgoing edges: 1/out_deg[source]
        trans_prob = 1.0 / out_deg[src]

        # Initialize PageRank vector uniformly
        pr = np.full(n, 1.0 / n, dtype=float)

        # Power iteration
        teleport = (1.0 - self.alpha) / n
        for _ in range(self.max_iter):
            pr_new = np.full(n, teleport, dtype=float)

            # Contribution from links
            contributions = pr[src] * trans_prob
            np.add.at(pr_new, dst, self.alpha * contributions)

            # Handle dangling nodes (no outgoing links) by distributing uniformly
            dangling_pr = pr[dangling].sum()
            pr_new += self.alpha * dangling_pr / n

            # Check convergence
            if np.linalg.norm(pr_new - pr, 1) < self.tol:
                pr = pr_new
                break
            pr = pr_new

        return {"pagerank_scores": pr.tolist()}