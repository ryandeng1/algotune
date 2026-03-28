from typing import List, Tuple, Optional
import itertools
import networkx as nx
from pysat.solvers import Solver as SATSolver


class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.solver = SATSolver("minicard")
        self.edge_vars = {edge: i + 1 for i, edge in enumerate(graph.edges)}
        self._add_cardinality_exact(list(self.edge_vars.values()), self.n)
        for node in graph.nodes:
            inc = [self.edge_vars[e] for e in graph.edges(node)]
            self._add_cardinality_exact(inc, 2)

    def _add_cardinality_exact(self, vars_: List[int], val: int) -> None:
        self.solver.add_atmost(vars_, val)
        self.solver.add_atmost([-v for v in vars_], len(vars_) - val)

    def _find_cycle(self) -> Optional[List[Tuple[int, int]]]:
        if not self.solver.solve():
            return None
        model = set(v for v in self.solver.get_model() if v > 0)
        cycle = [e for e, v in self.edge_vars.items() if v in model]
        subg = self.graph.edge_subgraph(cycle)
        if list(nx.connected_components(subg)) == [set(subg.nodes)]:
            return cycle
        # remove subtours
        for comp in nx.connected_components(subg):
            forb = [self.edge_vars[(u, v)] for u in comp for v in self.graph.nodes() if v not in comp and self.graph.has_edge(u, v)]
            self.solver.add_clause(forb)
        return self._find_cycle()

    def find(self) -> Optional[List[Tuple[int, int]]]:
        return self._find_cycle()


class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.h = HamiltonianCycleModel(graph)
        self.edges_sorted = sorted(graph.edges(data=True), key=lambda e: e[2]["weight"])
        self.best = self._approx()
        self.lower = self.graph.number_of_nodes()
        self.upper = self._edge_index(self._max_edge(self.best))

    def _approx(self) -> List[Tuple[int, int]]:
        tour = nx.approximation.traveling_salesman.christofides(self.graph)
        tour = list(tour)
        return [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]

    def _max_edge(self, tour: List[Tuple[int, int]]) -> Tuple[int, int]:
        return max(tour, key=lambda e: self.graph.edges[e]["weight"])

    def _edge_index(self, edge: Tuple[int, int]) -> int:
        for i, (u, v, _) in enumerate(self.edges_sorted):
            if (u, v) == edge or (v, u) == edge:
                return i
        return len(self.edges_sorted)

    def _prohibit(self, idx: int) -> Optional[List[Tuple[int, int]]]:
        forb = [-self.h.edge_vars[e] for e, _, _ in self.edges_sorted[idx + 1:]]
        self.h.solver.add_clause(forb)
        return self.h.find()

    def optimize(self) -> List[Tuple[int, int]]:
        while self.lower < self.upper:
            mid = (self.lower + self.upper) // 2
            sol = self._prohibit(mid)
            if sol is None:
                self.lower = mid + 1
            else:
                self.upper = self._edge_index(self._max_edge(sol))
                self.best = sol
        return self.best


class Solver:
    def solve(self, problem: List[List[float]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        g = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            g.add_edge(i, j, weight=problem[i][j])
        tour_edges = BottleneckTSPSolver(g).optimize()
        tour_graph = nx.Graph()
        tour_graph.add_edges_from(tour_edges)
        path = list(nx.dfs_preorder_nodes(tour_graph, 0))
        path.append(0)
        return path