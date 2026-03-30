# solver.py
#   -----------------------
#   Optimised CP‑SAT solver for the DAP problem
#   -----------------------

from __future__ import annotations

from typing import Any, List

from ortools.sat.python import cp_model


class Solver:
    """Exact DAP solver based on OR‑Tools CP‑SAT with main performance
    improvements:

    * Linear provisioning of all variables in a flat list (no dict look‑ups).
    * Problems for each period and bid are combined directly in the objective.
    * Skipping unnecessary ``else: pass`` statements to reduce byte‑code.
    * Using integer variables for bids (if needed) is avoided – CP‑SAT
      handles the binary domain efficiently.
    * On failure we simply return a list of ``-1``es immediately.
    """

    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Parameters
        ----------
        problem
            Dictionary containing:
            - ``'T'``  : number of periods
            - ``'N'``  : number of bids
            - ``'prices'``   : 1‑D list of raw bid prices
            - ``'capacities'``: 1‑D list of maximum allocation per bid
            - ``'probs'``: 2‑D list of probability weights per period and bid

        Returns
        -------
        List[int]
            offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period t.
        """

        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        # ------------------------------------------------------------------
        # Linear representation: x[t, i] → idx = t*N + i
        # ------------------------------------------------------------------
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{t}_{i}") for t in range(T) for i in range(N)]

        # One bid per period
        for t in range(T):
            model.Add(sum(x[t * N + i] for i in range(N)) <= 1)

        # Capacity constraints per bid
        for i in range(N):
            model.Add(sum(x[t * N + i] for t in range(T)) <= capacities[i])

        # Objective: maximise sum(prices[i] * probs[t][i] * x_ti)
        objective_lits: List[int] = []
        objective_coeffs: List[int] = []

        for t in range(T):
            for i in range(N):
                coeff = int(prices[i] * probs[t][i] * 1_000_000)  # scale to avoid float
                if coeff:  # skip zero coefficients to lighten the model
                    objective_lits.append(x[t * N + i])
                    objective_coeffs.append(coeff)

        model.Maximize(
            sum(
                coeff * lit for lit, coeff in zip(objective_lits, objective_coeffs)
            )
        )

        solver = cp_model.CpSolver()
        # Disable time limit – we want the exact optimum
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        # Build solution list
        offer: List[int] = []
        for t in range(T):
            chosen = -1
            base = t * N
            for i in range(N):
                if solver.Value(x[base + i]) == 1:
                    chosen = i
                    break
            offer.append(chosen)

        return offer