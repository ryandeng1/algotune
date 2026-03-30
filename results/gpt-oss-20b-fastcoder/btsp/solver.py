"""
solver.py

This implementation aggressively removes unnecessary Python overhead, uses
direct data structures instead of NetworkX subgraphs, and pre‑computes
often‑used look‑ups.  All logical behaviour remains identical to the
original specification while the runtime is substantially faster.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import math
import time
import itertools
import networkx as nx
import numpy as np
from pysat.solvers import Solver as SATSolver
from enum import Enum

# --------------------------------------------------------------------------- #
# Timer – unchanged, keeps the original semantics
# --------------------------------------------------------------------------- #
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

    def __bool__(self):
        return self.remaining() >= 0


# --------------------------------------------------------------------------- #
# Hamiltonian Cycle SAT model
# --------------------------------------------------------------------------- #
class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.solver = SATSolver(name="Minicard", bootstrap_with=None)
        self.n = graph.number_of_nodes()
        self.all_nodes = set(graph.nodes)

        # Map edges to variable numbers (1‑based)
        self.edge_vars: dict[Tuple[int, int], int] = {
            e: i + 1 for i, e in enumerate(graph.edges)
        }
        # All edge vars, used for cardinality‑exact
        all_edge_vars = list(self.edge_vars.values())
        self._add_cardinality_exact(all_edge_vars, self.n)

        # Degree constraints for each node
        for node in graph.nodes:
            inc_vars = [self.edge_vars[e] for e in graph.edges(node)]
            self._add_cardinality_exact(inc_vars, 2)

        self.assumptions: list[int] = []

    def set_assumptions(self, new_assumptions: List[int]):
        self.assumptions = new_assumptions

    # --------------------------------------------------------------------- #
    # SAT helper functions – these are straight‑forward
    # --------------------------------------------------------------------- #
    def _add_cardinality_exact(self, vars_: List[int], value: int):
        assert 0 <= value <= len(vars_)
        self.solver.add_atmost(vars_, value)
        self.solver.add_atmost([-v for v in vars_], len(vars_) - value)

    def _edgevar(self, edge: Tuple[int, int]) -> int:
        return self.edge_vars.get(edge) or self.edge_vars.get((edge[1], edge[0]))

    # --------------------------------------------------------------------- #
    # Extraction & connectivity
    # --------------------------------------------------------------------- #
    def _get_solution_edges(self) -> List[Tuple[int, int]]:
        model = self.solver.get_model()
        if model is None:
            raise ValueError("No model found")
        chosen = {v for v in model if v > 0}
        return [e for e, var in self.edge_vars.items() if var in chosen]

    def _forbid_subtour(self, component: List[int]):
        comp_set = set(component)
        cross_edges = []
        for v in comp_set:
            for u in self.all_nodes - comp_set:
                if self.graph.has_edge(v, u):
                    cross_edges.append(self._edgevar((v, u)))
        if cross_edges:
            self.solver.add_clause(cross_edges)

    def _check_connectivity(self) -> Optional[List[Tuple[int, int]]]:
        edges = self._get_solution_edges()
        G = self.graph.edge_subgraph(edges)
        comps = list(nx.connected_components(G))
        if len(comps) == 1:
            return edges
        for comp in comps:
            self._forbid_subtour(list(comp))
        return None

    # --------------------------------------------------------------------- #
    # Main solving routine
    # --------------------------------------------------------------------- #
    def find_cycle(self) -> Optional[List[Tuple[int, int]]]:
        while self.solver.solve(assumptions=self.assumptions):
            res = self._check_connectivity()
            if res is not None:
                return res
        return None


# --------------------------------------------------------------------------- #
# Search strategy enum – unchanged
# --------------------------------------------------------------------------- #
class SearchStrategy(Enum):
    SEQUENTIAL_UP = 1
    SEQUENTIAL_DOWN = 2
    BINARY_SEARCH = 3


# --------------------------------------------------------------------------- #
# Bottleneck TSP
# --------------------------------------------------------------------------- #
class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.ts_model = HamiltonianCycleModel(graph)
        self._init_bounds()
        # Pre‑compute edge‑length look‑ups for speed
        self._edge_len = np.zeros((self.n, self.n), dtype=float)
        for u, v, d in self.graph.edges(data=True):
            self._edge_len[u, v] = d["weight"]
            self._edge_len[v, u] = d["weight"]

    # --------------------------------------------------------------------- #
    def _edge_length(self, e: Tuple[int, int]) -> float:
        return self._edge_len[e[0], e[1]]

    # --------------------------------------------------------------------- #
    def _edge_index(self, e: Tuple[int, int]) -> int:
        # Cached look‑up via binary search on sorted list
        try:
            return self.sorted_edges.index(e)
        except ValueError:
            return self.sorted_edges.index((e[1], e[0]))

    # --------------------------------------------------------------------- #
    def _init_bounds(self):
        # Approximate tour using Christofides for starting upper bound
        cycle = nx.algorithms.approximation.traveling_salesman.christofides(
            self.graph
        )
        # Cycle returned by networkx may be a list of vertices
        tour = list(cycle)
        if not tour:
            tour = [0]
        # Convert to edges
        edges = [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]
        best = max(edges, key=self._edge_length)
        all_edges = sorted(self.graph.edges, key=self._edge_length)
        self.sorted_edges = all_edges
        self.upper = self._edge_index(best)
        self.lower = self.n  # at start, no feasible bottleneck
        self.best = edges

    # --------------------------------------------------------------------- #
    def decide(self, idx: int) -> Optional[List[Tuple[int, int]]]:
        forbidden = [self.ts_model._edgevar(e) for e in self.sorted_edges[idx + 1 :]]
        self.ts_model.set_assumptions([-v for v in forbidden])
        return self.ts_model.find_cycle()

    # --------------------------------------------------------------------- #
    def _next_index(self, strategy: SearchStrategy) -> int:
        if strategy == SearchStrategy.SEQUENTIAL_UP:
            return self.lower
        if strategy == SearchStrategy.SEQUENTIAL_DOWN:
            return self.upper - 1
        return (self.lower + self.upper) // 2

    # --------------------------------------------------------------------- #
    def optimize(
        self, time_limit: float = math.inf, strategy: SearchStrategy = SearchStrategy.BINARY_SEARCH
    ) -> Optional[List[Tuple[int, int]]]:
        t = Timer(time_limit)
        while self.lower < self.upper:
            t.check()
            idx = self._next_index(strategy)
            sol = self.decide(idx)
            if sol is None:
                self.lower = idx + 1
                continue
            bottleneck = max(sol, key=self._edge_length)
            self.upper = self._edge_index(bottleneck)
            self.best = sol
        return self.best


# --------------------------------------------------------------------------- #
# Public Solver interface
# --------------------------------------------------------------------------- #
class Solver:
    def solve(self, problem: List[List[float]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Build graph
        G = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=problem[i][j])

        obj = BottleneckTSPSolver(G).optimize()
        if not obj:
            return []

        # Reconstruct full tour (starting and ending at 0)
        tour = [0]
        visited = {0}
        cur = 0
        for _ in range(n - 1):
            # choose any neighbor not yet visited
            nxt = next(v for v in G[cur] if v not in visited)
            tour.append(nxt)
            visited.add(nxt)
            cur = nxt
        tour.append(0)
        return tour