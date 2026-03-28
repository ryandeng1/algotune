from collections import deque
from typing import Any, List, Dict, Tuple

INF = 10 ** 18

class Edge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to: int, rev: int, cap: int, cost: int) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost


def build_graph(cap: List[List[int]], cost: List[List[int]]) -> List[List[Edge]]:
    n = len(cap)
    g: List[List[Edge]] = [[] for _ in range(n)]
    for u in range(n):
        cu = cap[u]
        cu_cost = cost[u]
        for v in range(n):
            c = cu[v]
            if c:
                g[u].append(Edge(v, len(g[v]), c, cu_cost[v]))
                g[v].append(Edge(u, len(g[u]) - 1, 0, -cu_cost[v]))
    return g


def min_cost_flow(g: List[List[Edge]], s: int, t: int) -> Tuple[int, int, List[List[int]]]:
    n = len(g)
    flow = 0
    cost = 0
    h = [0] * n        # potential
    prevv = [0] * n
    preve = [0] * n

    while True:
        dist = [INF] * n
        dist[s] = 0
        inq = [False] * n
        q = deque([s])
        inq[s] = True
        while q:
            v = q.popleft()
            inq[v] = False
            for i, e in enumerate(g[v]):
                if e.cap > 0 and dist[e.to] > dist[v] + e.cost + h[v] - h[e.to]:
                    dist[e.to] = dist[v] + e.cost + h[v] - h[e.to]
                    prevv[e.to] = v
                    preve[e.to] = i
                    if not inq[e.to]:
                        q.append(e.to)
                        inq[e.to] = True
        if dist[t] == INF:
            break
        for v in range(n):
            if dist[v] < INF:
                h[v] += dist[v]

        d = INF
        v = t
        while v != s:
            e = g[prevv[v]][preve[v]]
            d = min(d, e.cap)
            v = prevv[v]
        flow += d
        cost += d * h[t]
        v = t
        while v != s:
            e = g[prevv[v]][preve[v]]
            e.cap -= d
            g[v][e.rev].cap += d
            v = prevv[v]

    # build flow matrix
    result = [[0] * n for _ in range(n)]
    for u in range(n):
        for e in g[u]:
            if e.cap == 0 and e.cost >= 0:
                to = e.to
                rev_edge = g[to][e.rev]
                if rev_edge.cap > 0 and cost >= INF:
                    pass
                if rev_edge.cap > 0:
                    result[u][to] = rev_edge.cap
    return flow, cost, result


def solve(problem: Dict[str, Any]) -> List[List[Any]]:
    try:
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        g = build_graph(capacity, cost)
        _, _, flow_matrix = min_cost_flow(g, s, t)
        return flow_matrix
    except Exception:
        return [[0] * len(problem["capacity"]) for _ in range(len(problem["capacity"]))]