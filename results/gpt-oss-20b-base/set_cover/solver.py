#!/usr/bin/env python3
"""
Optimised solver for the Set Cover problem using a SAT solver.

The implementation is deliberately lean: all costly set‑ up work is
performed once in the constructor and reused during the binary search.
The SAT encoding for the cardinality constraint is generated with
the efficient Pseudo‑Boolean encoding provided by PySAT.

The code is fully self‑contained except for the PySAT package and
requires Python 3.10 or newer.
"""

from __future__ import annotations

from typing import List

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver

__all__ = ["Solver"]


class Solver:
    """
    Solver for the set‑cover problem.

    The problem is given as a list of subsets (each subset itself being a list of
    positive integers).  The universe is assumed to be the set of all integers
    occurring in any subset, and the solver returns the *indices* of a minimal
    sub‑collection (1‑based) that covers the universe.
    """

    def __init__(self, problem: List[List[int]]):
        """
        Pre‑compute data structures that are reused for each binary‑search
        iteration.  The heavy work is therefore paid only once.
        """
        self.problem = problem
        self.m = len(problem)

        # Find the universe and build a mapping element -> list of subset indices
        self.universe: set[int] = set()
        self.elem_to_sets: dict[int, List[int]] = {}
        for i, subset in enumerate(problem, start=1):
            for e in subset:
                self.universe.add(e)
                self.elem_to_sets.setdefault(e, []).append(i)

        # For faster cardinality encoding, pre‑compute the list of
        # literal indices that correspond to the subsets.
        self.lits = list(range(1, self.m + 1))

    # -------------------------------------------------------------------------

    def _sat_encoding(self, k: int) -> CNF:
        """
        Construct a CNF expressing:
            * Every element in the universe is covered by at least one selected set.
            * At most *k* subsets are chosen.
        """
        cnf = CNF()

        # Coverage constraints
        for e in sorted(self.universe):
            covers = self.elem_to_sets.get(e, [])
            if covers:
                cnf.append(covers)
            else:
                # Empty clause -> unsatisfiable
                cnf.append([1, -1])

        # Cardinality constraint: at most k literals set to True
        atmost_k = CardEnc.atmost(lits=self.lits, bound=k, encoding=EncType.seqcounter)
        cnf.extend(atmost_k.clauses)
        return cnf

    # -------------------------------------------------------------------------

    def solve(self) -> List[int]:
        """
        Perform a binary search over the optimum cardinality *k*.
        The smallest *k* yielding a satisfiable encoding gives the minimal
        set of indices.  The final model may contain *more* than *k*
        variables set to True; we filter to those that are positive in
        the model.
        """
        left, right = 1, self.m + 1
        best_solution: List[int] | None = None

        while left < right:
            k = (left + right) // 2
            cnf = self._sat_encoding(k)

            with SATSolver(name="Minicard") as solver:
                solver.append_formula(cnf)
                sat = solver.solve()
                if not sat:
                    left = k + 1
                    continue

                model = solver.get_model()
                # Keep only the positive literals: selected sets
                selected = [lit for lit in model if lit > 0]
                best_solution = selected
                right = k

        return sorted(best_solution) if best_solution else []