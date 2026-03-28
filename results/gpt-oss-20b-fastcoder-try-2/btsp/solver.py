import itertools
import math
import time
from enum import Enum

import networkx as nx
import numpy as np
from pysat.solvers import Minisat22


class Timer:
    def __init__(self, runtime: float = math.inf):
        self.runtime = runtime
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.runtime - self.elapsed()

    def check(self):
        if self.remaining() < 0:
            raise TimeoutError("Timer has expired.")


class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.this = Minisat22()
        self.n = self.graph.number_of_nodes()
        self.nodes = list(self.graph.nodes())
        # map edges to positive ints
        self.edge_var = {e: i + 1 for i, e in enumerate(self.graph.edges())}
        self.var_edge = {v: e for e, v in self.edge_var.items()}

        # constrain to pick exactly n edges
        all_vars = list(self.edge_var.values())
        self.this.add_clause([-v for v in all_vars])  # at most n
        self.this.add_clause(all_vars)  # at least n

        # degree constraints: exactly two edges per node
        for u in self.nodes:
            inc = [self.edge_var[(u, v)] if (u, v) in self.edge_var else self.edge_var[(v, u)]
                   for v in self.graph.neighbors(u)]
            self.this.add_clause([-v for v in inc])  # at most 2
            self.this.add_clause(inc)  # at least 2

        self.assumptions = []

    def get_edgevar(self, edge):
        return self.edge_var.get(edge) or self.edge_var[(edge[1], edge[0])]

    def _extract_solution_edges(self):
        model = self.this.get_model()
        if model is None:
            raise ValueError("no model")
        chosen = {self.var_edge[v] for v in model if v > 0}
        return [e for e, v in self.edge_var.items() if v in chosen]

    def _add_subtour_constraint(self, component):
        comp = set(component)
        cross = [self.get_edgevar((u, v))
                 for u in comp
                 for v in set(self.nodes) - comp
                 if self.graph.has_edge(u, v)]
        self.this.add_clause(cross)

    def _check_and_extend(self):
        edges = self._extract_solution_edges()
        subgraph = self.graph.edge_subgraph(edges)
        comps = list(nx.connected_components(subgraph))
        if len(comps) == 1:
            return edges
        for c in comps:
            self._add_subtour_constraint(c)
        return None

    def find_cycle(self):
        while self.this.solve(assumptions=self.assumptions):
            cand = self._check_and_extend()
            if cand is not None:
                return cand
        return None


class SearchStrategy(Enum):
    SEQUENTIAL_UP = 1
    SEQUENTIAL_DOWN = 2
    BINARY_SEARCH = 3


class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.edges_sorted = sorted(graph.edges, key=lambda e: graph.edges[e]["weight"])
        self.all_idx = {e: i for i, e in enumerate(self.edges_sorted)}
        self.hmodel = HamiltonianCycleModel(graph)
        self.best = self.approx_solution()
        self.lower = self.n
        self.upper = self.all_idx[max(self.best, key=graph.edges.get)["weight"]]

    def _edge_len(self, e):
        return self.graph.edges[e]["weight"]

    def approx_solution(self):
        cycle = nx.approximation.traveling_salesman.christofides(self.graph)
        cycle = list(cycle)
        return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    def try_bottleneck(self, idx):
        forbid = [self.hmodel.get_edgevar(e) for e in self.edges_sorted[idx + 1 :]]
        self.hmodel.assumptions = [-v for v in forbid]
        res = self.hmodel.find_cycle()
        return res

    def optimize(self, limit: float = math.inf, strategy=SearchStrategy.BINARY_SEARCH):
        timer = Timer(limit)
        while self.lower < self.upper:
            timer.check()
            if strategy == SearchStrategy.SEQUENTIAL_UP:
                idx = self.lower
            elif strategy == SearchStrategy.SEQUENTIAL_DOWN:
                idx = self.upper - 1
            else:  # binary
                idx = (self.lower + self.upper) // 2

            sol = self.try_bottleneck(idx)
            if sol is None:
                self.lower = idx + 1
            else:
                max_w = max(self._edge_len(e) for e in sol)
                self.upper = self.all_idx[max_w]
                self.best = sol
        return self.best


class Solver:
    def solve(self, problem: list[list[float]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        G = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=problem[i][j])
        solver = BottleneckTSPSolver(G)
        tour_edges = solver.optimize()
        if tour_edges is None:
            return []
        path = nx.dfs_preorder_nodes(nx.Graph(tour_edges), source=0)
        return list(path) + [0]