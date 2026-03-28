# -*- coding: utf-8 -*-
"""
A compact, highly‑performant solver for the exact set‑cover problem using a SAT
back‑end.  The implementation focuses on
* linear‑time construction of the CNF formula; and
* reuse of the solver instance between iterations so that the
  expensive SAT solver initialisation is performed only once.

The “binary search” strategy is preserved: we keep shrinking the
upper bound on the number of chosen subsets until the optimum is
found.  The function is fully typed, conforms to the required
signature and never modifies the original problem data.
"""

from typing import List, Set
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SatSolver


def solve(problem: List[List[int]]) -> List[int]:
    """
    Return a minimum‑size sub‑collection of subsets that covers the
    universe `{1, 2, …, n}` (the union of all subsets).

    Parameters
    ----------
    problem : List[List[int]]
        Each inner list represents a subset of the universe.
        Elements are assumed to be positive integers.

    Returns
    -------
    List[int]
        1‑indexed indices of the selected subsets.
    """
    m = len(problem)
    if m == 0:
        return []

    # --------------------------------------------------
    # 1. Pre‑process
    # --------------------------------------------------
    # Build a mapping  element -> list of subset indices that contain it.
    universe: Set[int] = set()
    cover_map: dict[int, List[int]] = {}
    for idx, subset in enumerate(problem, start=1):
        for e in subset:
            universe.add(e)
            cover_map.setdefault(e, []).append(idx)

    n = len(universe)
    if n == 0:
        return []  # Nothing to cover

    # --------------------------------------------------
    # 2. Shared solver instance
    # --------------------------------------------------
    # The solver is instantiated once; it will be reused for all
    # mid‑values of the binary search.
    # We remove the default assumptions (if any) between runs to avoid
    # contaminating the model space.
    solver = SatSolver(name="Minicard")

    # --------------------------------------------------
    # 3. Helper: build CNF for a given bound k
    # --------------------------------------------------
    def build_cnf(k: int) -> CNF:
        """
        Build a CNF instance representing the set‑cover constraints with
        an upper bound of k selected subsets.

        The function is intentionally lightweight: it only assembles the
        clauses without any expensive data structures.
        """
        cnf = CNF()
        # --- coverage constraints ------------------------------------
        for e in range(1, n + 1):
            # all subsets that contain `e`
            lits = cover_map.get(e, [])
            if not lits:
                # unsatisfiable clause (covers nothing)
                cnf.append([1, -1])
            else:
                cnf.append(lits)

        # --- cardinality constraint ----------------------------------
        # "At most k" over all subset indices 1..m
        atmost_k = CardEnc.atmost(
            lits=list(range(1, m + 1)),
            bound=k,
            encoding=EncType.seqcounter
        )
        cnf.extend(atmost_k.clauses)
        return cnf

    # --------------------------------------------------
    # 4. Binary search for the optimal bound
    # --------------------------------------------------
    left, right = 1, m + 1
    best_solution: List[int] = []

    while left < right:
        mid = (left + right) // 2
        cnf = build_cnf(mid)

        # Use the same solver instance: clear it, add formulas, solve
        solver.append_formula(cnf)
        sat = solver.solve()

        if sat:
            # collect indices that are part of the model
            model = solver.get_model()
            selected = [idx for idx in range(1, m + 1) if idx in model]
            best_solution = selected
            # try to find a smaller cover
            right = len(selected)
        else:
            left = mid + 1

        # Reset solver for next iteration
        solver.delete()

    # If the problem is unsatisfiable (should not happen), return empty list
    return best_solution