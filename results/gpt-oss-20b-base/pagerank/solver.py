from typing import List, Dict

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-06

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """
        Calculate PageRank scores for the directed graph specified by an adjacency list,
        without using NetworkX.  The algorithm is a standard power‑iteration
        implementation with a damping factor.

        Parameters
        ----------
        problem : dict
            {"adjacency_list": adj_list}
        Returns
        -------
        dict
            {"pagerank_scores": [score0, score1, ...]}
        """
        adj_list: List[List[int]] = problem["adjacency_list"]
        n: int = len(adj_list)

        # Trivial cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Pre‑compute outgoing‑link counts for efficiency
        out_counts = [len(neigh) for neigh in adj_list]
        # Create reverse adjacency list: inbound neighbors for each node
        inbound = [[] for _ in range(n)]
        for u, neigh in enumerate(adj_list):
            for v in neigh:
                if 0 <= v < n:          # ignore dangling references
                    inbound[v].append(u)

        # Initialise rank uniformly
        vec = [1.0 / n] * n
        new_vec = [0.0] * n
        teleport = (1.0 - self.alpha) / n

        for _ in range(self.max_iter):
            # Reset new_vec
            for i in range(n):
                new_vec[i] = teleport

            # Contributions from inbound edges
            for v in range(n):
                if out_counts[v] == 0:
                    # dangling node: distribute its rank uniformly
                    for i in range(n):
                        new_vec[i] += self.alpha * vec[v] / n
                    continue
                share = self.alpha * vec[v] / out_counts[v]
                for u in inbound[v]:
                    new_vec[u] += share

            # Check convergence
            diff = sum(abs(new_vec[i] - vec[i]) for i in range(n))
            if diff < self.tol:
                vec, new_vec = new_vec, vec
                break
            vec, new_vec = new_vec, vec

        return {"pagerank_scores": vec}