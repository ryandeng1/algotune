from collections import deque
from typing import Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
        """
        Computes the global efficiency of an undirected graph without using NetworkX.
        Global efficiency = (1 / (n * (n - 1))) * Σ_{i≠j} 1 / d(i, j), 
        where d(i, j) is the shortest path length between i and j.
        Unreachable pairs contribute 0 to the sum.
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Ensure adjacency lists are 0‑based and symmetric
        for u, neigh in enumerate(adj_list):
            for v in neigh:
                if v < 0 or v >= n:
                    return {"global_efficiency": 0.0}
                # Add symmetric edge implicitly; we don't need to modify because BFS
                # will traverse as is

        reciprocal_sum = 0.0
        # BFS from each node to compute distances
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                du = dist[u]
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
            # Accumulate reciprocals for j > src to avoid double counting
            for tgt in range(src + 1, n):
                d = dist[tgt]
                if d > 0:  # reachable and not same node
                    reciprocal_sum += 2.0 / d  # account for both (src, tgt) and (tgt, src)

        total_pairs = n * (n - 1)
        global_efficiency = reciprocal_sum / total_pairs if total_pairs else 0.0
        return {"global_efficiency": float(global_efficiency)}