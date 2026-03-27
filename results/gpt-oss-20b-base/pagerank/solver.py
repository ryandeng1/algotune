from typing import Any, Dict, List

class Solver:
    alpha: float = 0.85
    max_iter: int = 100
    tol: float = 1e-6

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """
        Calculates the PageRank scores for the graph using a self‑contained
        power iteration implementation, avoiding the NetworkX dependency.

        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}

        Returns:
            A dictionary containing the PageRank scores as a list, ordered by node index.
            {"pagerank_scores": [score_node_0, score_node_1, ..., score_node_n-1]}
            Returns {"pagerank_scores": []} for n=0.
            Returns {"pagerank_scores": [1.0]} for n=1.
        """
        adjacency_list = problem["adjacency_list"]
        n = len(adjacency_list)

        # Trivial cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build out‑degree array and transition matrix as a list of sparse rows
        out_degree = [0] * n
        for u, neighbors in enumerate(adjacency_list):
            out_degree[u] = len(neighbors)

        # Pre‑compute the transition probabilities for each node with outgoing edges
        transition = []
        for u, neighbors in enumerate(adjacency_list):
            if out_degree[u]:
                prob = self.alpha / out_degree[u]
                row = [(v, prob) for v in neighbors]
            else:
                # Dangling node: will be handled by the teleportation term
                row = []
            transition.append(row)

        # Initialize PageRank vector uniformly
        pr = [1.0 / n] * n
        damp = 1.0 - self.alpha
        teleport = damp / n

        for iteration in range(self.max_iter):
            next_pr = [teleport] * n  # start with teleportation contribution
            # Accumulate contributions from predecessors
            for u in range(n):
                u_weight = pr[u]
                if out_degree[u]:
                    for v, prob in transition[u]:
                        next_pr[v] += u_weight * prob
                else:
                    # Distribute dangling weight uniformly
                    for v in range(n):
                        next_pr[v] += u_weight * self.alpha / n

            # Check convergence
            diff = sum(abs(next_pr[i] - pr[i]) for i in range(n))
            if diff < self.tol:
                pr = next_pr
                break
            pr = next_pr

        return {"pagerank_scores": pr}