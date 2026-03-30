# solver.py
# --------------------------------------------------------------
# High‑performance MIS solver using PySAT with a lightweight
# binary‑search over the cardinality constraint.
# --------------------------------------------------------------
from __future__ import annotations

import itertools
from typing import List

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver


class Solver:
    """
    Solve the Maximum Independent Set problem for a simple undirected
    graph given as an adjacency matrix.
    The implementation uses PySAT's efficient cardinality encodings
    and performs a binary search on the size of the solution.
    """

    @staticmethod
    def _adj_to_cnf_edges(adj: List[List[int]]) -> List[List[int]]:
        """Return a list of 2‑clause literals representing edges."""
        n = len(adj)
        clauses = []
        for i in range(n):
            row = adj[i]
            for j in range(i + 1, n):
                if row[j]:
                    # Variables are 1‑based in SATLib
                    clauses.append([i + 1, j + 1])
        return clauses

    @staticmethod
    def _build_cnf(adj: List[List[int]], atmost: int | None = None) -> CNF:
        """Return a CNF containing the edge constraints and optionally
        a cardinality constraint that at most *atmost* variables can be true.
        """
        cnf = CNF()
        cnf.extend(Solver._adj_to_cnf_edges(adj))
        if atmost is not None:
            lits = list(range(1, len(adj) + 1))
            # Sequence counter encoding is very fast for dense cardinality constraints
            atmost_cls = CardEnc.atmost(lits=lits, bound=atmost, encoding=EncType.seqcounter)
            cnf.extend(atmost_cls.clauses)
        return cnf

    @staticmethod
    def _sat_model_to_set(model: List[int], n: int) -> List[int]:
        """Return the set of node indices (0‑based) that are true in the model."""
        # The model is sorted; zero or negative literals indicate false assignments
        return [i for i, lit in enumerate(model[:n], start=0) if lit > 0]

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the Maximum Independent Set problem.

        Parameters
        ----------
        problem : List[List[int]]
            Adjacency matrix of the graph (square, non‑negative integers).

        Returns
        -------
        List[int]
            List of 0‑based node indices that form a maximum independent set.
        """
        n = len(problem)
        if n == 0:
            return []

        # Quick edge case: no edges ⇒ full set is an independent set
        has_edges = any(any(row) for row in problem)
        if not has_edges:
            return list(range(n))

        # Pre‑compute edge clauses (avoids recomputing during binary search)
        edge_clauses = Solver._adj_to_cnf_edges(problem)

        # Helper that tries to solve with a cardinality bound and returns
        # the model (or None).  The SAT solver is reused with a fresh instance each call.
        def try_bound(k: int | None = None) -> List[int] | None:
            cnf = CNF()
            cnf.extend(edge_clauses)
            if k is not None:
                lits = list(range(1, n + 1))
                cnf.extend(CardEnc.atmost(lits=lits, bound=k, encoding=EncType.seqcounter).clauses)
            with SATSolver(name="Minicard", bootstrap_with=cnf) as solver:
                if solver.solve():
                    return solver.get_model()
                return None

        # Initialise bounds for binary search
        lo, hi = 0, n
        best_model = None

        # Binary search over the size of the independent set
        while lo <= hi:
            mid = (lo + hi) // 2
            model = try_bound(mid)
            if model is not None:
                # A solution of size mid exists; try to enlarge
                best_model = model
                lo = mid + 1
            else:
                # No solution of size mid; shrink
                hi = mid - 1

        # Convert the best model to 0‑based node indices
        if best_model is None:
            # Should never happen because at least the empty set is feasible
            return []
        return Solver._sat_model_to_set(best_model, n)