import numpy as np
from numba import njit

@njit
def dijkstra(n, residual, cost, potential, s, t):
    dist = np.full(n, np.inf)
    parent = np.full(n, -1)
    dist[s] = 0

    visited = np.zeros(n, dtype=np.bool_)
    for _ in range(n):
        min_dist = np.inf
        u = -1
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True

        for v in range(n):
            if not visited[v] and residual[u, v] > 0:
                reduced_cost = cost[u, v] + potential[u] - potential[v]
                new_dist = dist[u] + reduced_cost
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u

    return dist, parent

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[Any]]:
        capacity = np.array(problem["capacity"], dtype=np.float64)
        cost = np.array(problem["cost"], dtype=np.float64)
        s = problem["s"]
        t = problem["t"]
        n = capacity.shape[0]

        flow = np.zeros((n, n), dtype=np.float64)
        residual = capacity.copy()
        potential = np.zeros(n)

        while True:
            dist, parent = dijkstra(n, residual, cost, potential, s, t)
            if dist[t] == np.inf:
                break

            delta = np.inf
            node = t
            while node != s:
                prev = parent[node]
                if residual[prev, node] < delta:
                    delta = residual[prev, node]
                node = prev

            node = t
            while node != s:
                prev = parent[node]
                flow[prev, node] += delta
                residual[prev, node] -= delta
                residual[node, prev] += delta
                node = prev

            for i in range(n):
                if parent[i] != -1:
                    potential[i] += dist[i]

        return flow.tolist()
