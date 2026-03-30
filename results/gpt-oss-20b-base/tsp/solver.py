# solver.py
from typing import List
from ortools.sat.python import cp_model

# --------------------------------------------------------------------------
# Minimal setup for the CP-SAT solver – all configuration that affects
# the runtime of *solve* is done here, so that the user-facing method stays
# clean and fast.
# --------------------------------------------------------------------------
_model = None  #: shared model used for all instances
_solver = None  #: shared solver instance

def _init_solver() -> None:
    """
    Create a shared CP-SAT solver instance with optimised default parameters.
    This function is called exactly once by the first call to Solver.solve.
    """
    global _solver
    _solver = cp_model.CpSolver()
    # Turn off all optional logging – it only increases solver time.
    _solver.parameters.log_search_progress = False
    # Enable parallel search to speed up the search on multi‑core machines.
    _solver.parameters.num_search_workers = 0
    # Tell the solver to exit when an optimum is found – this is already the
    # default but it makes the intent explicit.
    _solver.parameters.enumerate_all_solutions = False
    # A moderate timeout (in seconds) ensures we never block for days on a
    # pathological instance.
    _solver.parameters.max_time_in_seconds = 600

def _build_model(problem: List[List[int]]) -> cp_model.CpModel:
    """
    Build a CP‑SAT model for the Travelling Salesman problem with the given
    distance matrix. The model is optimised for speed:

    * one BoolVar per edge,
    * Circuit constraint (already optimal for TSP),
    * objective linear combination expanded only once.
    """
    n = len(problem)
    model = cp_model.CpModel()

    # create the Boolean variables only for the directed edges
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = model.NewBoolVar(f"x[{i},{j}]")

    # Circuit constraint constructs a Hamiltonian circuit automatically.
    circuit_edges = [(i, j, x[(i, j)]) for i in range(n) for j in range(n) if i != j]
    model.AddCircuit(circuit_edges)

    # objective: minimise total travel distance
    obj = sum(problem[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)
    model.Minimize(obj)

    return model, x

# --------------------------------------------------------------------------
# Solver implementation
# --------------------------------------------------------------------------
class Solver:
    """
    A highly optimised solver for the Travelling Salesman Problem using
    Google's OR‑Tools CP‑SAT solver.
    """

    def __init__(self) -> None:
        # Just a guard to make sure the solver is initialised once.
        global _solver
        if _solver is None:
            _init_solver()

    def solve(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Build a new model for each instance – models are lightweight.
        model, x = _build_model(problem)

        # Solve the model
        status = _solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Recover the path using the solution values.
            path = [0]
            current = 0
            for _ in range(n - 1):
                # pick the unique outgoing edge that is set to 1
                for nxt in range(n):
                    if current != nxt and _solver.Value(x[(current, nxt)]) == 1:
                        path.append(nxt)
                        current = nxt
                        break
            path.append(0)
            return path

        # No feasible solution found – return an empty tour.
        return []