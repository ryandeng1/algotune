from __future__ import annotations

from typing import List, Set

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


class Solver:
    """Optimised solver for the minimum set‑cover problem using a SAT back‑end."""

    # ---------------------------------------------------------------------
    # Helper: Pre–compute the universe and a mapping from elements to the
    #      subsets that contain them.  This work is done once per problem
    #      instance and speeds up the SAT encoding dramatically.
    # ---------------------------------------------------------------------
    @staticmethod
    def _preprocess(subsets: List[List[int]]) -> tuple[Set[int], List[List[int]]]:
        universe: Set[int] = set()
        element_to_sets: List[Set[int]] = []

        # Build the mapping element → set indices
        for idx, subset in enumerate(subsets):
            universe.update(subset)

        # Reverse mapping: for each element in the universe, collect the indices
        elem_to_subset: dict[int, List[int]] = {e: [] for e in universe}
        for idx, subset in enumerate(subsets):
            for e in subset:
                elem_to_subset[e].append(idx + 1)        # 1‑based variables

        # Convert dict values to lists for faster iteration
        element_to_sets = [elem_to_subset[e] for e in sorted(universe)]
        return universe, element_to_sets

    # ---------------------------------------------------------------------
    # Encode the set‑cover problem as a CNF formula with an upper bound `k`
    # on the number of selected subsets.
    # ---------------------------------------------------------------------
    @staticmethod
    def _encode(subsets: List[List[int]],
                element_to_sets: List[List[int]],
                k: int) -> CNF:
        cnf = CNF()

        # Coverage constraints – one clause per element
        for subset_list in element_to_sets:
            cnf.append(subset_list)

        # Cardinality constraint – at most `k` subsets can be selected
        num_subsets = len(subsets)
        lits = list(range(1, num_subsets + 1))
        atmost_k = CardEnc.atmost(lits=lits, bound=k, encoding=EncType.seqcounter)
        cnf.extend(atmost_k.clauses)

        return cnf

    # ---------------------------------------------------------------------
    # Main API – find a minimal set of subset indices (1‑based) covering the universe.
    # ---------------------------------------------------------------------
    def solve(self, problem: List[List[int]]) -> List[int]:
        m = len(problem)
        if not m:
            return []

        # Pre‑process once
        _, element_to_sets = self._preprocess(problem)

        left, right = 1, m          # k ∈ [1, m]
        best_k = m
        best_solution: List[int] | None = None

        # Binary search on the minimum size `k`.
        while left <= right:
            mid = (left + right) // 2
            cnf = self._encode(problem, element_to_sets, mid)

            # Use a context‑manager for the solver to ensure timely cleanup.
            with Solver(name="Minicard") as solver:
                solver.append_formula(cnf)
                sat = solver.solve()

                if sat:
                    # Extract the subset indices from the model.
                    model = solver.get_model()
                    selected = [i + 1 for i, v in enumerate(model) if v > 0]
                    best_k = mid
                    best_solution = selected
                    right = mid - 1    # try a smaller k
                else:
                    left = mid + 1     # need more subsets

        return best_solution if best_solution is not None else []