#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb

# ------------------------------------------------------------------
# Helper functions (vectorised via numba)
# ------------------------------------------------------------------
@nb.njit
def f_vec(x, a0, a1, a2, a3, a4, a5):
    """
    Computes the polynomial values for each element of x.
    """
    return a0 * x**5 + a1 * x**4 + a2 * x**3 + a3 * x**2 + a4 * x + a5

@nb.njit
def fprime_vec(x, a0, a1, a2, a3, a4, a5):
    """
    Computes the derivative of the polynomial for each element of x.
    """
    return 5*a0*x**4 + 4*a1*x**3 + 3*a2*x**2 + 2*a3*x + a4

# ------------------------------------------------------------------
# Solver Class
# ------------------------------------------------------------------
class Solver:
    """
    Fast vectorised Newton–Raphson solver for a quintic polynomial.
    """
    def __init__(self):
        # Coefficients of the polynomial (fixed constants)
        self.a2 = 1e-09
        self.a3 = 0.004
        self.a4 = 10.0
        self.a5 = 0.27456
        self.max_iter = 30
        self.tol = 1e-12

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        """
        Finds the root of the polynomial f(x)=0 for every entry using
        a vectorised and highly optimised Newton iteration.

        Parameters
        ----------
        problem : dict
            Must contain lists 'x0', 'a0', 'a1' of equal length.

        Returns
        -------
        dict
            Dictionary with key 'roots': list of roots (NaN on failure).
        """
        try:
            x0_arr = np.asarray(problem['x0'], dtype=np.float64)
            a0_arr = np.asarray(problem['a0'], dtype=np.float64)
            a1_arr = np.asarray(problem['a1'], dtype=np.float64)
        except Exception:
            return {'roots': []}

        if not (x0_arr.shape == a0_arr.shape == a1_arr.shape):
            return {'roots': []}

        n = x0_arr.size
        # Initialise output with NaNs (in case of divergence)
        roots = np.full(n, np.nan, dtype=np.float64)

        # Prepare constants for numba routine
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        max_iter = self.max_iter
        tol = self.tol

        # ------------------------------------------------------------------
        # Vectorised Newton loop (numba accelerated)
        # ------------------------------------------------------------------
        @nb.njit
        def newton_vec(x, a0, a1, a2, a3, a4, a5, max_iter, tol):
            root = np.empty_like(x)
            convergence = np.full(x.shape, False, dtype=nb.boolean)
            for i in range(max_iter):
                f_val = f_vec(x, a0, a1, a2, a3, a4, a5)
                f_der = fprime_vec(x, a0, a1, a2, a3, a4, a5)
                # Avoid division by zero
                mask = (f_der != 0.0) & (~convergence)
                if not mask.any():
                    break
                dx = np.empty_like(x)
                dx[mask] = f_val[mask] / f_der[mask]
                x[mask] -= dx[mask]
                con = np.abs(dx) <= tol
                convergence[mask] = con[mask]
            # Collect converged roots
            root[convergence] = x[convergence]
            root[~convergence] = np.nan
            return root

        roots = newton_vec(np.copy(x0_arr),
                           a0_arr,
                           a1_arr,
                           a2,
                           a3,
                           a4,
                           a5,
                           max_iter,
                           tol)

        # Ensure list view for the answer
        return {'roots': roots.tolist()}