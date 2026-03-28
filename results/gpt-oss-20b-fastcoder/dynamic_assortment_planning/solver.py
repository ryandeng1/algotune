from ortools.sat.python import cp_model
from typing import Any, List

def solve(problem: dict[str, Any]) -> List[int]:
    """
    Solve the DAP exactly with a binary integer program (CP‑SAT).

    Returns
    -------
    List[int]
        offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period t.
    """
    T, N = problem["T"], problem["N"]
    prices = problem["prices"]
    probs = problem["probs"]
    capacities = problem["capacities"]

    model = cp_model.CpModel()

    # Create all variables at once and store in a 2‑D list for fast access
    x = [[model.NewBoolVar(f"x_{t}_{i}") for i in range(N)] for t in range(T)]

    # At most one offer per time period
    for t in range(T):
        model.Add(sum(x[t]) <= 1)

    # Do not offer more than the capacity of a product
    for i in range(N):
        model.Add(sum(x[t][i] for t in range(T)) <= capacities[i])

    # Objective : maximize expected revenue
    model.Maximize(
        sum(prices[i] * probs[t][i] * x[t][i] for t in range(T) for i in range(N))
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [-1] * T

    # Extract solution
    offer = []
    for t in range(T):
        chosen = -1
        for i in range(N):
            if solver.Value(x[t][i]):
                chosen = i
                break
        offer.append(chosen)
    return offer