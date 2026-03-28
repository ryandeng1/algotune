import collections

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, float]:
        """
        Calculates the global efficiency of an undirected unweighted graph
        given as an adjacency list. The implementation avoids NetworkX and
        computes the mean of 1/distance for all unordered pairs of
        distinct vertices.

        Parameters
        ----------
        problem : dict
            {"adjacency_list": List[List[int]]}

        Returns
        -------
        dict
            {"global_efficiency": float}
        """
        adj = problem["adjacency_list"]
        n = len(adj)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Pre‑allocate distances array to reuse during BFS
        dist = [0] * n
        total = 0.0

        for src in range(n):
            # BFS from src
            for i in range(n):
                dist[i] = -1
            q = collections.deque([src])
            dist[src] = 0
            while q:
                u = q.popleft()
                d = dist[u] + 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d
                        q.append(v)
            # accumulate contributions for pairs (src, v) with src < v
            for tgt in range(src + 1, n):
                d = dist[tgt]
                if d > 0:
                    total += 1.0 / d
                # if d == 0 or -1 (disconnected), contribution is zero

        # Number of unordered node pairs
        pair_count = n * (n - 1) / 2.0
        efficiency = total / pair_count if pair_count else 0.0
        return {"global_efficiency": float(efficiency)}