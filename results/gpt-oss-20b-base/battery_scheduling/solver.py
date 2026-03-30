#!/usr/bin/env python3
"""
Fast CVXPY solver for a circular battery scheduling problem.
Author: ChatGPT
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Dict, Any


class Solver:
    """
    A lightweight wrapper around a CVXPY optimisation model
    that solves a small fixed‐runtime battery optimisation
    problem.  All function arguments are purely numeric
    and the optimisation objective is a linear cost of the
    net battery power.  The model is expressed with
    vectorised linear constraints instead of Python loops,
    which yields a noticeable speedup compared with the
    reference implementation.
    """

    @staticmethod
    def _make_circular_matrix(shape: int) -> np.ndarray:
        """
        Build a circulant shift matrix of size (shape, shape)
        such that (shift @ v)[t] = v[(t + 1) % shape].
        """
        shift = np.zeros((shape, shape), dtype=np.float64)
        for i in range(shape):
            shift[i, (i + 1) % shape] = 1.0
        return shift

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the linear battery optimisation problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing the following keys:
                - 'T'          : number of time steps
                - 'p'          : electricity price vector (T,)
                - 'u'          : load vector (T,)
                - 'batteries'  : list with a single entry containing:
                      'Q'          : max SOC
                      'C'          : max charge power
                      'D'          : max discharge power
                      'efficiency' : round‑trip efficiency (unitless)

        Returns
        -------
        dict
            Dictionary containing solution status, optimality flag
            and all requested outputs.
        """
        # --- Data & Types ----------------------------------------------------
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=np.float64)
        u = np.asarray(problem["u"], dtype=np.float64)

        bat = problem["batteries"][0]
        Q = float(bat["Q"])
        C = float(bat["C"])
        D = float(bat["D"])
        eff = float(bat["efficiency"])

        # --- CVXPY Variables -----------------------------------------------
        q = cp.Variable(T)              # state of charge
        c_in = cp.Variable(T)           # charging power
        c_out = cp.Variable(T)          # discharging power
        c = c_in - c_out                # net battery power

        # --- Constraints -----------------------------------------------
        constraints = [
            q >= 0,
            q <= Q,
            c_in >= 0,
            c_in <= C,
            c_out >= 0,
            c_out <= D,
            u + c >= 0,               # power balance
        ]

        # Circular dynamics: q_{t+1} - q_t = eff * c_in_t - 1/eff * c_out_t
        shift = self._make_circular_matrix(T)
        lhs = shift @ q - q
        rhs = eff * c_in - (1.0 / eff) * c_out
        constraints.append(lhs == rhs)

        # --- Objective -----------------------------------------------
        objective = cp.Minimize(p @ c)

        # --- Solve -----------------------------------------------
        prob = cp.Problem(objective, constraints)

        # Use a fast, dense LP solver.  OSQP is the default for CVXPY
        # and handles the linear structure efficiently.
        result = prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"status": prob.status, "optimal": False}

        # Extract solution
        q_val = q.value.reshape(-1).tolist()
        c_in_val = c_in.value.reshape(-1).tolist()
        c_out_val = c_out.value.reshape(-1).tolist()
        c_net_val = np.asarray(c_in_val, np.float64) - np.asarray(c_out_val, np.float64)
        c_net_val = c_net_val.tolist()

        # Cost calculations
        cost_without = float(p @ u)
        cost_with = float(p @ (u + c_net_val))
        savings = cost_without - cost_with
        savings_pct = 100.0 * savings / cost_without if cost_without else 0.0

        return {
            "status": prob.status,
            "optimal": True,
            "battery_results": [
                {
                    "q": q_val,
                    "c": c_net_val,
                    "c_in": c_in_val,
                    "c_out": c_out_val,
                    "cost": cost_with,
                }
            ],
            "total_charging": c_net_val,
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": savings_pct,
        }