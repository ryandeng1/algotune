import sys
from typing import List, Set
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

# ---------------------------------------------------------------------------

def solve(problem: List[List[int]]) -> List[int]:
    """
    Find a minimum-cardinality subcollection of `problem` that covers the
    universe.  Returns the 1‑based indices of the chosen subsets.
    """
    # Pre‑compute universe and the subsets that contain each element.
    universe: Set[int] = set()
    for subset in problem:
        universe.update(subset)
    universe = sorted(universe)                 # for deterministic order
    n_elements = len(universe)
    element_to_subsets = {e: [] for e in universe}
    for idx, subset in enumerate(problem, start=1):   # 1‑based indices
        for e in subset:
            element_to_subsets[e].append(idx)

    # Build the CNF once and reuse it, simply altering the cardinality part.
    base_cnf = CNF()
    for e in universe:
        base_cnf.append(element_to_subsets[e])     # at least one covering set

    subsets_count = len(problem)
    all_lits = list(range(1, subsets_count + 1))

    # Binary search for the minimum cardinality.
    left, right = 1, subsets_count
    best_solution = []

    while left <= right:
        mid = (left + right) // 2
        # Re‑use the base CNF and add the at‑most‑mid constraint.
        cnf = CNF()
        cnf.extend(base_cnf.clauses)
        atmost_mid = CardEnc.atmost(lits=all_lits, bound=mid, encoding=EncType.seqcounter)
        cnf.extend(atmost_mid.clauses)

        with Solver(name="minicard", bootstrap_with=cnf) as s:
            if s.solve():
                model = s.get_model()
                selected = [i for i in range(1, subsets_count + 1) if model[i - 1] > 0]
                best_solution = selected
                right = mid - 1
            else:
                left = mid + 1

    return best_solution