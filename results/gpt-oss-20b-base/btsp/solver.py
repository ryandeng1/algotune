import itertools
import math
import networkx as nx
import numpy as np
from pysat.solvers import Solver as SATSolver
from typing import List, Tuple, Optional


class Timer:
    def __init__(self, runtime: float = math.inf):
        self.runtime = runtime
        self.start = __import__("time").time()

    def remaining(self) -> float:
        return self.runtime - (.__import__("time").time() - self.start)

    def check(self):
        if self.remaining() < 0:
            raise TimeoutError("Timer has expired.")


class HamiltonianCycleModel:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.n = G.number_of_nodes()
        self.solver = SATSolver(name="minicard")
        self.edge_vars = {e: i + 1 for i, e in enumerate(G.edges)}
        self.solver.add_atmost(list(self.edge_vars.values()), self.n)
        self.solver.add_atmost([-v for v in self.edge_vars.values()],
                               len(self.edge_vars) - self.n)
        for v in G.nodes:
            inc = [self.edge_vars[e] for e in G.edges(v)]
            self.solver.add_atmost(inc, 2)
            self.solver.add_atmost([-i for i in inc], len(inc) - 2)
        self.assumptions: List[int] = []

    def get_var(self, e):
        return self.edge_vars[e] if e in self.edge_vars else self.edge_vars[e[::-1]]

    def forbid_subtour(self, comp):
        comp_set = set(comp)
        cross = [self.get_var((u, v))
                 for u in comp_set for v in self.G.nodes if v not in comp_set and self.G.has_edge(u, v)]
        if cross:
            self.solver.add_clause(cross)

    def extract(self) -> List[Tuple[int, int]]:
        model = self.solver.get_model()
        if model is None:
            raise RuntimeError("No model")
        pos = {v for v in model if v > 0}
        return [e for e, v in self.edge_vars.items() if v in pos]

    def find_cycle(self) -> Optional[List[Tuple[int, int]]]:
        while self.solver.solve(assumptions=self.assumptions):
            sol = self.extract()
            sub = nx.Graph()
            sub.add_edges_from(sol)
            comps = list(nx.connected_components(sub))
            if len(comps) == 1:
                return sol
            for comp in comps:
                self.forbid_subtour(comp)
        return None


class BottleneckTSPSolver:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.model = HamiltonianCycleModel(G)
        self.best = self.approx()
        self.edges_sorted = sorted(G.edges, key=lambda e: G[e[0]][e[1]]["weight"])
        self.lb = G.number_of_nodes()
        bottleneck = max(self.best, key=lambda e: G[e[0]][e[1]]["weight"])
        self.ub = self.edges_sorted.index(bottleneck)

    def approx(self):
        cycle = nx.approximation.traveling_salesman.christofides(self.G)
        cycle = list(np.array(cycle))
        return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    def _edge_index(self, e):
        try:
            return self.edges_sorted.index(e)
        except ValueError:
            return self.edges_sorted.index((e[1], e[0]))

    def _edge_weight(self, e):
        return self.G[e[0]][e[1]]["weight"]

    def solve(self, time_limit: float) -> Optional[List[Tuple[int, int]]]:
        timer = Timer(time_limit)
        while self.lb < self.ub:
            timer.check()
            idx = (self.lb + self.ub) // 2
            forbidden = [self.edges_sorted[i] for i in range(idx + 1, len(self.edges_sorted))]
            self.model.assumptions = [-self.model.get_var(e) for e in forbidden]
            sol = self.model.find_cycle()
            if sol is None:
                self.lb = idx + 1
            else:
                self.best = sol
                bottleneck = max(sol, key=self._edge_weight)
                self.ub = self._edge_index(bottleneck)
        return self.best


class Solver:
    def solve(self, prob: List[List[float]]) -> List[int]:
        n = len(prob)
        if n <= 1:
            return [0, 0]
        G = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=prob[i][j])
        t = BottleneckTSPSolver(G).solve(time_limit=math.inf)
        if not t:
            return []
        tour_g = nx.Graph()
        tour_g.add_edges_from(t)
        path = list(nx.dfs_preorder_nodes(tour_g, source=0))
        path.append(0)
        return path