import numpy as np
import cvxpy as cp

# Optimised `solve` function
def solve(problem: dict) -> dict:
    """
    Solve the optimal advertising problem in a lightweight way.
    """
    # Convert inputs to NumPy arrays (memory efficient)
    P = np.asarray(problem["P"], dtype=float)
    R = np.asarray(problem["R"], dtype=float)
    B = np.asarray(problem["B"], dtype=float)
    c = np.asarray(problem["c"], dtype=float)
    T = np.asarray(problem["T"], dtype=float)

    m, n = P.shape

    # Decision variables
    D = cp.Variable((m, n), nonneg=True)

    # Auxiliary variables for the revenue caps
    y = cp.Variable(m, nonneg=True)

    # Constraints
    cons = [
        cp.sum(D, axis=0) <= T,           # Budget per impression type
        cp.sum(D, axis=1) >= c,           # Minimum displays per ad
        y <= R * (P @ D.T).T,             # y_i <= R_i * (P_i * D_i)
        y <= B,                           # y_i <= B_i
    ]

    # Objective: maximise total revenue
    prob = cp.Problem(cp.Maximize(cp.sum(y)), cons)

    try:
        prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
        status = prob.status
        if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"status": status, "optimal": False}

        D_val = D.value
        clicks = np.sum(P * D_val, axis=1)          # vectorised multiplication
        revenue = np.minimum(R * clicks, B)

        return {
            "status": status,
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue.tolist(),
            "total_revenue": float(np.sum(revenue)),
            "objective_value": float(prob.value),
        }

    except cp.SolverError as exc:
        return {"status": "solver_error", "optimal": False, "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "optimal": False, "error": str(exc)}