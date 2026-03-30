import time
import itertools
import math
import networkx as nx
from pysat.solvers import Solver as SATSolver

# ----------------------------------------------------------------------
#  Helper tools
# ----------------------------------------------------------------------
class Timer:
    def __init__(self, runtime: float = math.inf):
        self._runtime = runtime
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def remaining(self) -> float:
        return self._runtime - self.elapsed()

    def check(self):
        if self.remaining() < 0:
            raise TimeoutError("Time limit exceeded")

# ----------------------------------------------------------------------
#  Hamiltonian cycle solver (SAT based)
# ----------------------------------------------------------------------
class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph):
        self._g = graph
        self._solver = SATSolver(name="Minicard")
        self._n = graph.number_of_nodes()
        self._edges = list(graph.edges)
        self._edge_vars = {e: i + 1 for i, e in enumerate(self._edges)}
        self._all_vars = list(self._edge_vars.values())
        # Exactly n edges
        self._solver.add_atmost(self._all_vars, self._n)
        self._solver.add_atmost([-v for v in self._all_vars], self._n)
        # Degree-2 constraints
        for node in graph.nodes:
            inc = [self._edge_vars[e] for e in graph.edges(node)]
            self._solver.add_atmost(inc, 2)
            self._solver.add_atmost([-v for v in inc], 2)

        self._assumptions = []

    # ------------------------------------------------------------------
    def set_assumptions(self, assumptions: list[int]):
        self._assumptions = assumptions

    def _edge_var(self, e: tuple[int, int]) -> int:
        return self._edge_vars.get(e) or self._edge_vars[(e[1], e[0])]

    def _extract(self) -> list[tuple[int, int]] | None:
        m = self._solver.get_model()
        if m is None:
            return None
        chosen = {v for v in m if v > 0}
        return [e for e, v in self._edge_vars.items() if v in chosen]

    def _reject_subtour(self, comp: set[int]):
        # forbid any edge crossing from comp to its complement
        cross = [self._edge_var((u, v)) for u in comp for v in self._g.nodes if v not in comp if self._g.has_edge(u, v)]
        if cross:
            self._solver.add_clause(cross)

    def find_cycle(self) -> list[tuple[int, int]] | None:
        while self._solver.solve(assumptions=self._assumptions):
            sol = self._extract()
            if sol is None:
                return None
            subg = self._g.edge_subgraph(sol)
            comps = list(nx.connected_components(subg))
            if len(comps) == 1:
                return sol
            for comp in comps:
                self._reject_subtour(comp)
        return None

# ----------------------------------------------------------------------
#  Bottleneck TSP with binary search
# ----------------------------------------------------------------------
class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph):
        self._g = graph
        self._hc = HamiltonianCycleModel(graph)
        self._init_bounds()

    def _edge_len(self, e: tuple[int, int]) -> float:
        return self._g.edges[e]["weight"]

    def _init_bounds(self):
        best = self._approx()
        bottleneck = max(best, key=self._edge_len)
        sorted_edges = sorted(self._g.edges(keys=False), key=self._edge_len)
        self._edge_index = {e: i for i, e in enumerate(sorted_edges)}
        self._lower_idx = self._g.number_of_nodes()
        self._upper_idx = self._edge_index[bottleneck]
        self._best_tour = best

    def _approx(self) -> list[tuple[int, int]]:
        cycle = nx.approximation.traveling_salesman.christofides(self._g)
        cycle = list(cycle)
        return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    def _play(self, idx: int) -> list[tuple[int, int]] | None:
        forbidden = [-self._hc._edge_var(e) for e in self._g.edges if self._edge_index[e] > idx]
        self._hc.set_assumptions(forbidden)
        return self._hc.find_cycle()

    def optimize(self, time_limit: float = math.inf) -> list[tuple[int, int]] | None:
        timer = Timer(time_limit)
        while self._lower_idx < self._upper_idx:
            mid = (self._lower_idx + self._upper_idx) // 2
            timer.check()
            sol = self._play(mid)
            if sol is None:          # infeasible
                self._lower_idx = mid + 1
            else:
                self._upper_idx = self._edge_index[max(sol, key=self._edge_len)]
                self._best_tour = sol
        return self._best_tour

# ----------------------------------------------------------------------
#  Main solver interface
# ----------------------------------------------------------------------
class Solver:
    """
    Solver for the Bottleneck TSP.
    """
    def solve(self, problem: list[list[float]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        G = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=problem[i][j])

        solver = BottleneckTSPSolver(G)
        tour_edges = solver.optimize()
        if not tour_edges:
            return []

        # Build tour order starting at 0
        tour_graph = nx.Graph()
        tour_graph.add_edges_from(tour_edges)
        order = list(nx.dfs_preorder_nodes(tour_graph, source=0))
        order.append(0)
        return order