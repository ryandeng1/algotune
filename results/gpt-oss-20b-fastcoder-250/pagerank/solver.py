from typing import Any

class Solver:
    def __init__(self) -> None:
        # default PageRank parameters
        self.alpha = 0.85
        self.max_iter = 200
        self.tol = 1e-6

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        """
        Calculates PageRank scores using a manual power‑iteration implementation.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        # Handle trivial graphs
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build the out‑degree list and the reversed adjacency list
        out_deg = [len(neigh) for neigh in adj_list]
        rev_adj = [[] for _ in range(n)]
        for u, neigh in enumerate(adj_list):
            for v in neigh:
                rev_adj[v].append(u)

        # Pre‑compute the teleport probability
        teleport = 1.0 / n

        # Initialize PageRank vector uniformly
        pr = [teleport] * n
        pr_new = [0.0] * n

        for _ in range(self.max_iter):
            # Accumulate contributions from incoming links
            for v in range(n):
                sink_sum = 0.0
                for u in rev_adj[v]:
                    if out_deg[u] > 0:
                        sink_sum += pr[u] / out_deg[u]
                pr_new[v] = self.alpha * sink_sum + (1 - self.alpha) * teleport

            # Check convergence
            diff = sum(abs(pr_new[i] - pr[i]) for i in range(n))
            pr, pr_new = pr_new, pr  # swap references for next iteration
            if diff < self.tol:
                break

        return {"pagerank_scores": [float(x) for x in pr]}