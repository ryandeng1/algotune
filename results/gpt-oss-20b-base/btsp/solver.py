import itertools
import math
import time
import networkx as nx
from pysat.solvers import Solver as SATSolver

class Timer:
    def __init__(self, runtime: float = math.inf):
        self.start = time.time()
        self.runtime = runtime

    def remaining(self):
        return self.runtime - (time.time() - self.start)

    def check(self):
        if self.remaining() < 0:
            raise TimeoutError()


class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.solver = SATSolver('Minicard')
        self.n = graph.number_of_nodes()
        self.nodes = list(graph.nodes)
        self.edge_vars = {e: i + 1 for i, e in enumerate(graph.edges)}
        self._add_cardinality_exact(list(self.edge_vars.values()), self.n)
        for node in graph.nodes:
            inc = [self.edge_vars[e] for e in graph.edges(node)]
            self._add_cardinality_exact(inc, 2)

    def _add_cardinality_exact(self, vars, val):
        self.solver.add_atmost(vars, val)
        self.solver.add_atmost([-v for v in vars], len(vars) - val)

    def find(self, max_edge_index: int, all_edges_sorted: list):
        forbidden = all_edges_sorted[max_edge_index + 1 :]
        self.solver.add_clause([-self.edge_vars[e] for e in forbidden])
        if self.solver.solve():
            model = self.solver.get_model()
            sel = set(v for v in model if v > 0)
            edges = [e for e, v in self.edge_vars.items() if v in sel]
            # check connectivity
            sub = nx.Graph()
            sub.add_edges_from(edges)
            if len(list(nx.connected_components(sub))) == 1:
                return edges
            # forbid subtours
            for comp in nx.connected_components(sub):
                other = set(self.nodes) - comp
                for u in comp:
                    for v in other:
                        if self.graph.has_edge(u, v):
                            self.solver.add_clause([self.edge_vars[(u, v)]])
        return None


class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.model = HamiltonianCycleModel(graph)
        self.edges_sorted = sorted(graph.edges, key=lambda e: graph.edges[e]["weight"])
        self.n = graph.number_of_nodes()
        self.lower = self.n
        best = self._approx()
        bottleneck = max(best, key=lambda e: graph.edges[e]["weight"])
        self.upper = self.edges_sorted.index(bottleneck)

    def _approx(self):
        cycle = nx.approximation.traveling_salesman.christofides(self.graph)
        cycle = [int(x) for x in cycle]
        return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    def solve(self, limit: float = math.inf):
        timer = Timer(limit)
        while self.lower < self.upper:
            mid = (self.lower + self.upper) // 2
            timer.check()
            res = self.model.find(mid, self.edges_sorted)
            if res is None:
                self.lower = mid + 1
            else:
                self.upper = mid
                self.best = res
        return self.best


class Solver:
    def solve(self, problem: list[list[float]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        G = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=problem[i][j])
        sol_edges = BottleneckTSPSolver(G).solve()
        if not sol_edges:
            return []
        tour = nx.Graph()
        tour.add_edges_from(sol_edges)
        path = list(nx.dfs_preorder_nodes(tour, source=0))
        path.append(0)
        return path