import cvxpy as cp
import numpy as np

def solve(problem: dict) -> dict:
    """
    Fast CVXPY solver for the multi‑point aircraft wing design problem.
    Optimised for speed – no unnecessary loops, no scipy/NumPy conversions
    inside the loop, and minimal Python objects.
    """
    n = problem["num_conditions"]
    cond = problem["conditions"]

    # Decision variables
    A  = cp.Variable(pos=True, name="A")
    S  = cp.Variable(pos=True, name="S")
    V  = cp.Variable(n, pos=True, name="V")
    W  = cp.Variable(n, pos=True, name="W")
    Re = cp.Variable(n, pos=True, name="Re")
    CD = cp.Variable(n, pos=True, name="C_D")
    CL = cp.Variable(n, pos=True, name="C_L")
    Cf = cp.Variable(n, pos=True, name="C_f")
    Ww = cp.Variable(n, pos=True, name="W_w")

    constraints = []
    total_drag = 0

    # Pre‑extract all constants in one pass
    rho   = np.asarray([c["rho"] for c in cond], dtype=float)
    tau   = np.asarray([c["tau"] for c in cond], dtype=float)
    Vmin  = np.asarray([c["V_min"] for c in cond], dtype=float)
    C_DA0 = np.asarray([c["CDA0"] for c in cond], dtype=float)
    CLmax = np.asarray([c["C_Lmax"] for c in cond], dtype=float)
    Nult  = np.asarray([c["N_ult"] for c in cond], dtype=float)
    Swet  = np.asarray([c["S_wetratio"] for c in cond], dtype=float)
    W0    = np.asarray([c["W_0"] for c in cond], dtype=float)
    c1    = np.asarray([c["W_W_coeff1"] for c in cond], dtype=float)
    c2    = np.asarray([c["W_W_coeff2"] for c in cond], dtype=float)
    e     = np.asarray([c["e"] for c in cond], dtype=float)
    k     = np.asarray([c["k"] for c in cond], dtype=float)
    mu    = np.asarray([c["mu"] for c in cond], dtype=float)

    # Build constraints vectorised
    drag = 0.5 * rho * V**2 * CD * S
    total_drag = cp.sum(drag) / n

    constraints += [
        CD >= C_DA0 / S + k * Cf * Swet + CL**2 / (np.pi * A * e),
        Cf >= 0.074 / Re**0.2,
        Re * mu      >= rho * V * cp.sqrt(S / A),
        Ww >= c2 * S + c1 * Nult * A**1.5 * cp.sqrt(W0 * W) / tau,
        W  >= W0 + Ww,
        W  <= 0.5 * rho * V**2 * CL * S,
        2 * W / (rho * Vmin**2 * S) <= CLmax
    ]

    prob = cp.Problem(cp.Minimize(total_drag), constraints)

    try:
        prob.solve(gp=True, warm_start=True, verbose=False)
    except cp.SolverError:
        return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
        return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

    # Gather results
    V_vals = V.value
    W_vals = W.value
    Ww_vals = Ww.value
    CL_vals = CL.value
    CD_vals = CD.value
    Cf_vals = Cf.value
    Re_vals = Re.value

    condition_results = [
        {
            "condition_id": c["condition_id"],
            "V": float(V_vals[i]),
            "W": float(W_vals[i]),
            "W_w": float(Ww_vals[i]),
            "C_L": float(CL_vals[i]),
            "C_D": float(CD_vals[i]),
            "C_f": float(Cf_vals[i]),
            "Re": float(Re_vals[i]),
            "drag": float(drag[i].value) if hasattr(drag[i], "value") else float(drag[i])
        }
        for i, c in enumerate(cond)
    ]

    return {
        "A": float(A.value),
        "S": float(S.value),
        "avg_drag": float(prob.value),
        "condition_results": condition_results
    }