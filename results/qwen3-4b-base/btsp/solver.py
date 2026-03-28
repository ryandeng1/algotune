from typing import Any
import itertools
import math
import time
from enum import Enum
import networkx as nx
import numpy as np
from pysat.solvers import Solver as SATSolver


class Timer:
    """
    A simple timer for measuring time.
    """

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


class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.solver = SATSolver("Minicard")
        self.all_nodes = set(graph.nodes)
        self.n = graph.number_of_nodes()
        self.set_assumptions([])
        self._make_edge_vars()
        self._enforce_degree_constraints()

    def set_assumptions(self, new_assumptions: list[int]):
        self.assumptions = new_assumptions

    def get_edgevar(self, edge: tuple[int, int]) -> int:
        u, v = edge
        if edge in self.edge_vars:
            return self.edge_vars[edge]
        return self.edge_vars[v, u]

    def _add_cardinality_exact(self, vars: list[int], value: int):
        assert 0 <= value <= len(vars)
        self.solver.add_atmost(vars, value)
        self.solver.add_atmost([-i for i in vars], len(vars) - value)

    def _make_edge_vars(self):
        self.edge_vars = {edge: i for i, edge in enumerate(self.graph.edges, start=1)}
        all_edgevars = list(self.edge_vars.values())
        self._add_cardinality_exact(all_edgevars, self.n)

    def _enforce_degree_constraints(self):
        for node in self.graph.nodes:
            incident_edges_vars = [self.get_edgevar(e) for e in self.graph.edges(node)]
            self._add_cardinality_exact(incident_edges_vars, 2)

    def _extract_solution_edges(self) -> list[tuple[int, int]]:
        model = self.solver.get_model()
        if model is None:
            raise ValueError("No model found!")
        assignment = {v for v in model if v > 0}
        return [e for e, var in self.edge_vars.items() if var in assignment]

    def _forbid_subtour(self, component: list[int]):
        component_set = set(component)
        crossing_edges = itertools.product(component_set, self.all_nodes - component_set)
        clause = [self.get_edgevar(e) for e in crossing_edges]
        self.solver.add_clause(clause)

    def _check_tour_connectivity_or_add_lazy_constraints(
        self,
    ) -> list[tuple[int, int]] | None:
        sol_edges = self._extract_solution_edges()
        sol_graph = self.graph.edge_subgraph(sol_edges)
        components = list(nx.connected_components(sol_graph))
        if len(components) == 1:
            return sol_edges
        for subtour in components:
            self._forbid_subtour(list(subtour))
        logging.info(f"[HC] Intermediate solution contained {len(components)} subtours.")
        return None

    def find_hamiltonian_cycle(self) -> list[tuple[int, int]] | None:
        iterations = 0
        while self.solver.solve(assumptions=self.assumptions):
            iterations += 1
            solution = self._check_tour_connectivity_or_add_lazy_constraints()
            if solution:
                logging.info(f"[HC] Found a Hamiltonian cycle after {iterations} iterations.")
                return solution
        logging.info("[HC] Infeasibility proven: Graph has no Hamiltonian cycle!")
        return None


class SearchStrategy(Enum):
    SEQUENTIAL_UP = 1  # Try smallest possible k first.
    SEQUENTIAL_DOWN = 2  # Try any improvement.
    BINARY_SEARCH = 3  # Try a binary search for the optimal k.

    def __str__(self):
        return self.name.title()


class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.hamiltonian = HamiltonianCycleModel(graph)
        self._initialize_bounds()

    def _edge_length(self, edge: tuple[int, int]) -> float:
        return self.graph.edges[edge]["weight"]

    def _edge_index(self, edge: tuple[int, int]) -> int:
        try:
            return self.all_edges_sorted.index(edge)
        except ValueError:
            return self.all_edges_sorted.index((edge[1], edge[0]))

    def _initialize_bounds(self):
        self.best_tour = self.approx_solution()
        bottleneck_edge = max(self.best_tour, key=self._edge_length)
        self.all_edges_sorted = sorted(self.graph.edges, key=self._edge_length)
        # Lower bound index: at least n edges are needed
        self.lower_bound_index = self.graph.number_of_nodes()
        # Upper bound index: index of the bottleneck edge in the current tour
        self.upper_bound_index = self._edge_index(bottleneck_edge)

    def decide_bottleneck(self, index: int) -> list[tuple[int, int]] | None:
        forbidden_edges = self.all_edges_sorted[index + 1 :]
        assumptions = [-self.hamiltonian.get_edgevar(e) for e in forbidden_edges]
        self.hamiltonian.set_assumptions(assumptions)
        return self.hamiltonian.find_hamiltonian_cycle()

    def approx_solution(self) -> list[tuple[int, int]]:
        cycle = nx.approximation.traveling_salesman.christofides(self.graph)
        cycle = list(np.array(cycle))
        edges = list(zip(cycle, cycle[1:]))
        assert len(edges) == self.graph.number_of_nodes()
        return edges

    def _next_bottleneck_index(self, search_strategy: SearchStrategy) -> int:
        if search_strategy == SearchStrategy.SEQUENTIAL_UP:
            return self.lower_bound_index
        elif search_strategy == SearchStrategy.SEQUENTIAL_DOWN:
            return self.upper_bound_index - 1
        elif search_strategy == SearchStrategy.BINARY_SEARCH:
            return (self.lower_bound_index + self.upper_bound_index) // 2
        else:
            raise ValueError(f"Invalid search strategy: {search_strategy}")

    def lower_bound(self) -> float:
        return self._edge_length(self.all_edges_sorted[self.lower_bound_index])

    def _best_bottleneck(self) -> float:
        return max(map(self._edge_length, self.best_tour))

    def optimize_bottleneck(
        self,
        time_limit: float = math.inf,
        search_strategy: SearchStrategy = SearchStrategy.BINARY_SEARCH,
    ) -> list[tuple[int, int]] | None:
        timer = Timer(time_limit)
        try:
            while self.lower_bound_index < self.upper_bound_index:
                index = self._next_bottleneck_index(search_strategy)
                timer.check()
                bottleneck_sol = self.decide_bottleneck(index)
                if bottleneck_sol is None:
                    logging.info(
                        f"[BTSP] Bottleneck {self._edge_length(self.all_edges_sorted[index])} at index {index} is infeasible!"
                    )
                    self.lower_bound_index = index + 1
                    continue
                bottleneck_edge = max(bottleneck_sol, key=self._edge_length)
                self.upper_bound_index = self._edge_index(bottleneck_edge)
                self.best_tour = bottleneck_sol
                logging.info(
                    f"[BTSP] Found new bottleneck of value {self._best_bottleneck()} at index {index}!"
                )
        except TimeoutError:
            logging.info("[BTSP] Reached timeout!")

        if self.lower_bound_index < self.upper_bound_index:
            logging.info(
                f"[BTSP] Returning best found solution with objective {self._best_bottleneck()}!"
            )
        else:
            logging.info(f"[BTSP] Bottleneck {self._best_bottleneck()} is optimal!")
        return self.best_tour


class Solver:
    def solve(self, problem: list[list[float]]) -> list[int]:
        """
        Solve the BTSP problem.
        Returns a tour as a list of city indices starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Create a complete graph from the distance matrix.
        graph = nx.Graph()
        for i, j in itertools.combinations(range(n), 2):
            graph.add_edge(i, j, weight=problem[i][j])

        solver = BottleneckTSPSolver(graph)
        sol = solver.optimize_bottleneck()
        if sol is None:
            return []

        # Build the tour graph from the solution edges.
        tour_graph = nx.Graph()
        for i, j in sol:
            tour_graph.add_edge(i, j, weight=problem[i][j])

        # Recover the tour using depth-first search starting at node 0.
        path = list(nx.dfs_preorder_nodes(tour_graph, source=0))
        path.append(0)
        return path