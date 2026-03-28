import heapq
from typing import Any, Dict, List

def solve(problem: Dict[str, Any]) -> List[List[Any]]:
    """
    Minimum‑cost maximum‑flow (restricted to a unit amount of flow) for a dense
    adjacency matrix description.  The function is fully implemented in pure
    Python but uses heaps and pre‑allocated lists for speed.  It should evaluate
    faster than the original NetworkX solution for the typical test sizes.
    """
    n = len(problem["capacity"])
    cap = problem["capacity"]
    cost = problem["cost"]
    s = problem["s"]
    t = problem["t"]

    # adjacency list: for each node store list of (to, capacity, cost, rev_index)
    graph = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            c = cap[u][v]
            if c > 0:
                graph[u].append([v, c, cost[u][v], len(graph[v])])
                graph[v].append([u, 0, -cost[u][v], len(graph[u]) - 1])

    flow = 0
    INF = 10 ** 18
    potential = [0] * n  # for reduced costs
    while True:
        dist = [INF] * n
        dist[s] = 0
        prevnode = [-1] * n
        prevedge = [-1] * n
        inqueue = [False] * n
        pq = [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for i, (v, cap_e, cost_e, rev) in enumerate(graph[u]):
                if cap_e == 0:
                    continue
                nd = d + cost_e + potential[u] - potential[v]
                if nd < dist[v]:
                    dist[v] = nd
                    prevnode[v] = u
                    prevedge[v] = i
                    heapq.heappush(pq, (nd, v))
        if dist[t] == INF:
            break  # no augmenting path

        # augment one unit of flow (since all capacities are ints, push as much as possible)
        aug = INF
        v = t
        while v != s:
            u = prevnode[v]
            e = graph[u][prevedge[v]]
            aug = min(aug, e[1])
            v = u
        v = t
        while v != s:
            u = prevnode[v]
            e = graph[u][prevedge[v]]
            e[1] -= aug
            graph[v][e[3]][1] += aug
            v = u
        flow += aug

        # update potentials for reduced cost correctness
        for i in range(n):
            if dist[i] < INF:
                potential[i] += dist[i]

    # build output matrix
    result = [[0] * n for _ in range(n)]
    for u in range(n):
        for v, cap_e, cost_e, rev in graph[u]:
            # reverse edges contain the flow that we sent
            if cap_e > 0 and graph[v][rev][1] > 0:
                result[u][v] = graph[v][rev][1]
    return result