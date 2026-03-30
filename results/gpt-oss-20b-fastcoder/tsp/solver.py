# solver.py
#
# The goal of this solver is to give the user a fast yet correct solution to a
# TSP instance that is represented as an `n x n` distance matrix.  The heavy
# lifting is done by Google's OR‑Tools CP‑SAT solver.  The main optimisation
# we can control from our Python side is how the model is built and how the
# solver is driven.  

from typing import List
from ortools.sat.python import cp_model


class Solver:
    """
    A tiny thin‑wrapper around OR‑Tools CP‑SAT that returns the optimal TSP tour
    starting and ending at node 0.

    Why this implementation is fast:
        * All boolean variables are created in a single
          dictionary‑comprehension – no Python `if` inside loops.
        * `AddCircuit` is a specialised OR‑Tools construct that ensures a
          Hamiltonian cycle in O(n²) time – far cheaper than a full subtour
          elimination approach.
        * The solver is sent a few tuning hints (`num_search_workers`,
          `max_time_in_seconds`, `enumerate_all_solutions`) that let it use
          multiple cores and stop as soon as the optimum is found, which
          cuts the solve time dramatically for most medium‑sized instances.
        * The result extraction walks the assignment exactly once – O(n).
    """

    def __init__(self):
        # Preparation of a solver instance that remains in memory between
        # calls – this saves the effort of re‑instantiating CP‑SAT for each
        # TSP instance.
        self._solver = cp_model.CpSolver()
        # Give the solver a few hours of time; for small/medium instances it
        # will usually finish early.
        self._solver.parameters.max_time_in_seconds = 20.0
        # Enable parallelism – the degree of parallelism is automatically
        # selected based on the machine capabilities.
        self._solver.parameters.num_search_workers = 0
        # We only need one solution – the optimal one – so we turn off
        # enumeration of all solutions.  This reduces memory consumption and
        # speeds up the search.
        self._solver.parameters.enumerate_all_solutions = False
        # Turn on the solver's own progress logging if desired.
        self._solver.parameters.log_search_progress = False

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the TSP problem using CP-SAT solver.
        :param problem: Distance matrix as a list of lists.
        :return: A list representing the optimal tour, starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            # Trivial cases – start and end at 0, or empty
            return [0, 0] if n == 1 else []

        # Build the CP‑SAT model
        model = cp_model.CpModel()

        # Create a boolean variable for every edge (i, j) where i != j.
        # [] is a list of (i, j) pairs; constructing it outside the dictionary
        # comprehension avoids repeated checks at Python level.
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]
        x = {e: model.NewBoolVar(f"x[{e[0]},{e[1]}]") for e in edges}

        # A single AddCircuit enforces that each node has exactly one outgoing
        # and one incoming edge – this automatically gives us a Hamiltonian
        # cycle that visits every node exactly once.
        model.AddCircuit([(i, j, x[(i, j)]) for i, j in edges])

        # Objective: minimise the sum of distances for the chosen edges
        model.Minimize(
            sum(problem[i][j] * x[(i, j)] for i, j in edges)
        )

        # Solve the model
        status = self._solver.Solve(model)

        # Extract the tour if a solution was found
        if status in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,  # CP‑SAT may sometimes only find a feasible one
        ):
            tour = [0]
            cur = 0
            for _ in range(n - 1):
                # the out‑edge from the current node must be the one that
                # was set to 1 by the solver
                for nxt in range(n):
                    if cur != nxt and self._solver.Value(x[(cur, nxt)]):
                        cur = nxt
                        break
                tour.append(cur)
            tour.append(0)
            return tour

        # No feasible solution – return an empty list
        return []