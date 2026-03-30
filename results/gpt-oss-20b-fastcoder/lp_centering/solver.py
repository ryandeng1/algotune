# solver.py
"""
Highly‑optimised solver for the LP centering problem.

* The problem is formulated once during construction.  
* A CVXPY problem instance with symbolic parameters is reused for every
  call to `solve`, so matrix allocations and solver initialisation are
  avoided after the first run.

The implementation keeps the external API identical to the original
version while gaining a large performance boost.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

class Solver:
    """
    A fast solver for the LP centering problem.
    
    Parameters are held as ``cvxpy.Param`` objects and only re‑assigned
    during ``solve``.  The underlying CVXPY problem is constructed
    with a single symbolic variable and a single constraint; the
    Re‑use of the problem instance eliminates repeated construction
    overhead and reduces the overall runtime per call.
    """

    def __init__(self):
        # Dummy arrays – they will be overridden during the first solve.
        self._c_param: cp.Parameter | None = None
        self._A_param: cp.Parameter | None = None
        self._b_param: cp.Parameter | None = None
        self._prob: cp.Problem | None = None
        self._x: cp.Variable | None = None

    def _build_problem(self, n: int, m: int) -> None:
        """
        Construct the symbolic CVXPY problem for an n‑variable, m‑constraint
        equality LP centering task.
        """
        # Parameters for objective and constraints
        self._c_param = cp.Parameter(shape=(n,), name="c", nonneg=False)
        self._b_param = cp.Parameter(shape=(m,), name="b")

        # The matrix A is the same shape for every solve, but CVXPY
        # requires a parameter for each element – so we create a single
        # Parameter and reshape during solve.
        self._A_param = cp.Parameter(shape=(m, n), name="A")

        self._x = cp.Variable(n, name="x")

        # Symbolic objective and constraint
        objective = cp.Minimize(self._c_param @ self._x - cp.sum(cp.log(self._x)))
        constraint = [self._A_param @ self._x == self._b_param]

        # Keep the problem instance for reuse
        self._prob = cp.Problem(objective, constraint)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP centering problem as specified by *problem*.

        Parameters
        ----------
        problem : dict
            Must contain keys 'c', 'A', 'b' with array‑compatible values.

        Returns
        -------
        dict
            Mapping ``"solution"`` to a list of floating point values.
        """
        # Extract numpy arrays
        c = np.asarray(problem["c"], dtype=np.float64, order="C")
        A = np.asarray(problem["A"], dtype=np.float64, order="C")
        b = np.asarray(problem["b"], dtype=np.float64, order="C")

        n = c.shape[0]
        m = b.shape[0]  # assume A is compatible

        # Build the symbolic problem on first use (or if dimensions change)
        if self._prob is None or self._c_param.shape[0] != n or self._A_param.shape[0] != m:
            self._build_problem(n, m)

        assert self._prob is not None

        # Reset parameter values for the current solve
        self._c_param.value = c
        self._b_param.value = b
        # CVXPY expects 2‑D arrays for matrix parameters
        self._A_param.value = A.reshape((m, n))

        # Solve using CLARABEL.  Warm–starting is triggered automatically
        # by CVXPY once the same problem instance is reused.
        self._prob.solve(solver=cp.CLARABEL, warm_start=True)

        if self._prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Solver failed with status {self._prob.status}")

        # Convert solution to plain Python list
        return {"solution": self._x.value.ravel().tolist()}