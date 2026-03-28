import time
import itertools
import networkx as nx
from pysat.solvers import Solver as SATSolver

# ----- Timer ---------------------------------------------------------

class Timer:
    def __init__(self, runtime=float("inf")):
        self.runtime = runtime
        self.start = time.time()

    def remaining(self):
        return self.runtime - (time.time() - self.start)

    def check(self):
        if self.remaining() < 0:
            raise TimeoutError("Time limit exceeded")

# ----- Hamiltonian Cycle model ---------------------------------------

class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.solver = SATSolver(name="minicard")
        self.n = graph.number_of_nodes()
        self.nodes = set(graph.nodes)
        self._init_vars()
        self._degree_constraints()

    def _init_vars(self):
        # edge -> positive variable id
        self.e2v = {e: i + 1 for i, e in enumerate(self.graph.edges)}
        # enforce exactly n edges selected
        vars = list(self.e2v.values())
        self.solver.add_atmost(vars, self.n)
        self.solver.add_atmost([-v for v in vars], len(vars) - self.n)

    def _degree_constraints(self):
        for v in self.graph.nodes:
            inc = [self.e2v[e] for e in self.graph.edges(v)]
            self.solver.add_atmost(inc, 2)
            self.solver.add_atmost([-i for i in inc], len(inc) - 2)

    def _extract_edges(self) -> list[tuple[int, int]]:
        m = self.solver.get_model()
        if m is None:
            raise RuntimeError("No model")
        pos = {v for v in m if v > 0}
        return [e for e, var in self.e2v.items() if var in pos]

    def _enforce_subtour(self, comp):
        comp_set = set(comp)
        cross = []
        for u in comp_set:
            for v in self.nodes - comp_set:
                if self.graph.has_edge(u, v):
                    cross.append(self.e2v[tuple(sorted((u, v)))])
        if cross:
            self.solver.add_clause(cross)

    def find_cycle(self):
        while self.solver.solve():
            edges = self._extract_edges()
            g = self.graph.edge_subgraph(edges)
            comps = list(nx.connected_components(g))
            if len(comps) == 1:
                return edges
            for c in comps:
                self._enforce_subtour(c)
        return None

# ----- Bottleneck TSP solver ----------------------------------------

class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.model = HamiltonianCycleModel(graph)
        self._prepare()

    def _prepare(self):
        # sorted by weight
        self.edges_sorted = sorted(self.graph.edges, key=lambda e: self.graph[e[0]][e[1]]["weight"])
        self.e2idx = {e: i for i, e in enumerate(self.edges_sorted)}
        self.lower = self.n            # no solution yet
        self.upper = len(self.edges_sorted) - 1
        self.best = self._approximate()

    def _approximate(self):
        # 2‑approx from networkx
        tour = nx.approximation.traveling_salesman.christofides(self.graph).edges
        return list(tour)

    def _edge_weight(self, e):
        return self.graph[e[0]][e[1]]["weight"]

    def _check(self, idx):
        forb = [self.model.e2v[e] for e in self.edges_sorted[idx + 1:]]
        if forb:
            self.model.solver.add_clause([-f for f in forb])
        return self.model.find_cycle()

    def optimize(self, time_limit=float("inf")):
        timer = Timer(time_limit)
        while self.lower < self.upper:
            timer.check()
            mid = (self.lower + self.upper) // 2
            sol = self._check(mid)
            if sol is None:
                self.lower = mid + 1
            else:
                self.upper = mid
                self.best = sol
        return self.best

# ----- Solver wrapper -----------------------------------------------

class Solver:
    def solve(self, problem):
        n = len(problem)
        if n <= 1:
            return [0] * (2 if n == 1 else 1)
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=problem[i][j])
        solver = BottleneckTSPSolver(G)
        tour_edges = solver.optimize()
        if not tour_edges:
            return []
        Gt = nx.Graph()
        Gt.add_edges_from(tour_edges)
        path = list(nx.dfs_preorder_nodes(Gt, 0))
        path.append(0)
        return path