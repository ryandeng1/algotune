import time
import math
import itertools
import logging
from enum import Enum
from collections import defaultdict, deque
from typing import List, Tuple, Optional

from pysat.solvers import Solver as SATSolver


# -------------------------- Helper Classes -------------------------- #

class Timer:
    def __init__(self, runtime: float = math.inf):
        self.start = time.time()
        self.runtime = runtime

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.runtime - self.elapsed()

    def check(self):
        if self.remaining() < 0:
            raise TimeoutError("Timer has expired.")

    def __bool__(self):
        return self.remaining() >= 0


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1


# -------------------------- Hamiltonian Cycle Model -------------------------- #

class HamiltonianCycleModel:
    def __init__(self, n: int, edges: List[Tuple[int, int]], weight: List[float]):
        self.n = n
        self.edges = edges
        self.weight = weight
        self.solver = SATSolver(name="minicard", bootstrap_with=[])
        self.assumptions: List[int] = []

        self.edge_to_var = {e: i + 1 for i, e in enumerate(edges)}
        self._add_cardinality_exact(list(self.edge_to_var.values()), n)

        for v in range(n):
            incident = [self.edge_to_var[e] for e in self._inc_edges(v)]
            self._add_cardinality_exact(incident, 2)

    def _inc_edges(self, v: int):
        return [ (v, u) if (v, u) in self.edge_to_var else (u, v)
                 for u in range(self.n) if (v, u) in self.edge_to_var or (u, v) in self.edge_to_var ]

    def _add_cardinality_exact(self, vars_: List[int], value: int):
        self.solver.add_atmost(vars_, value)
        self.solver.add_atmost([-v for v in vars_], len(vars_) - value)

    def find_cycle(self) -> Optional[List[Tuple[int, int]]]:
        while self.solver.solve(assumptions=self.assumptions):
            model = self.solver.get_model()
            pos = {v for v in model if v > 0}
            cycle = [e for e, v in self.edge_to_var.items() if v in pos]
            if self._is_connected(cycle):
                return cycle
            self._add_subtour_constraints(cycle)
        return None

    def _is_connected(self, cycle: List[Tuple[int, int]]) -> bool:
        uf = UnionFind(self.n)
        for u, v in cycle:
            uf.union(u, v)
        roots = {uf.find(i) for i in range(self.n)}
        return len(roots) == 1

    def _add_subtour_constraints(self, cycle: List[Tuple[int, int]]):
        uf = UnionFind(self.n)
        for u, v in cycle:
            uf.union(u, v)
        # Map component root to list of nodes
        comp = defaultdict(list)
        for i in range(self.n):
            comp[uf.find(i)].append(i)
        for nodes in comp.values():
            if len(nodes) == self.n:
                continue
            crossing = [self.edge_to_var[e] for e in self.edges
                        if (e[0] in nodes) ^ (e[1] in nodes)]
            if crossing:
                self.solver.add_clause(crossing)


# -------------------------- Bottleneck TSP Solver -------------------------- #

class SearchStrategy(Enum):
    SEQUENTIAL_UP = 1
    SEQUENTIAL_DOWN = 2
    BINARY_SEARCH = 3


class BottleneckTSPSolver:
    def __init__(self, n: int, edges: List[Tuple[int, int]], edge_len: List[float]):
        self.n = n
        self.edges = edges
        self.edge_len = edge_len
        self._sorted_edges = sorted(range(len(edges)), key=lambda i: edge_len[i])
        self.best_tour = self._approx_solution()
        bottleneck = max(self.best_tour, key=lambda e: edge_len[edges.index(e)])
        self.lower = n
        self.upper = self._edge_index(bottleneck)

        self.hc_model = HamiltonianCycleModel(n, edges, edge_len)

    def _edge_index(self, edge: Tuple[int, int]) -> int:
        try:
            i = self._sorted_edges.index(self.edges.index(edge))
        except ValueError:
            i = self._sorted_edges.index(self.edges.index((edge[1], edge[0])))
        return i

    def _approx_solution(self) -> List[Tuple[int, int]]:
        # naive greedy: start at 0, always go to nearest unvisited
        visited = {0}
        path = [0]
        cur = 0
        while len(visited) < self.n:
            nxt = min((j for j in range(self.n) if j not in visited),
                      key=lambda j: self.edge_len[self.edges.index((cur, j))])
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        path.append(0)
        return list(zip(path, path[1:]))

    def _next_index(self, strategy: SearchStrategy) -> int:
        if strategy == SearchStrategy.SEQUENTIAL_UP:
            return self.lower
        if strategy == SearchStrategy.SEQUENTIAL_DOWN:
            return self.upper - 1
        return (self.lower + self.upper) // 2

    def optimize(self, time_limit: float = math.inf,
                 strategy: SearchStrategy = SearchStrategy.BINARY_SEARCH) -> Optional[List[Tuple[int, int]]]:
        timer = Timer(time_limit)
        while self.lower < self.upper:
            idx = self._next_index(strategy)
            timer.check()
            forbidden = [self.edges[i] for i in self._sorted_edges[idx + 1:]]
            self.hc_model.assumptions = [-self.hc_model.edge_to_var[e] for e in forbidden]
            sol = self.hc_model.find_cycle()
            if sol is None:
                self.lower = idx + 1
                continue
            best_edge = max(sol, key=lambda e: self.edge_len[self.edges.index(e)])
            self.upper = self._edge_index(best_edge)
            self.best_tour = sol
        return self.best_tour


# -------------------------- Solver ----------------------------------- #

class Solver:
    def solve(self, problem: List[List[float]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        edges: List[Tuple[int, int]] = []
        edge_len: List[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                w = problem[i][j]
                edges.append((i, j))
                edge_len.append(w)

        solver = BottleneckTSPSolver(n, edges, edge_len)
        tour_edges = solver.optimize()
        if tour_edges is None:
            return []

        # Build tour using union find to extract order
        uf = UnionFind(n)
        for u, v in tour_edges:
            uf.union(u, v)

        # Reconstruct sequence starting from 0
        adj = defaultdict(list)
        for u, v in tour_edges:
            adj[u].append(v)
            adj[v].append(u)

        path = [0]
        cur, prev = 0, -1
        while True:
            nxt_list = adj[cur]
            nxt = nxt_list[0] if nxt_list[0] != prev else nxt_list[1]
            if nxt == 0:
                path.append(0)
                break
            path.append(nxt)
            prev, cur = cur, nxt
        return path