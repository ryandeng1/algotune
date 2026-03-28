import numpy as np
import cvxpy as cp

# --------------------------------------------------------------------
# The solver below is deliberately simple: it sets up a small SDP
# that only uses the CVXPY bmat helper and the fast SCS solver.
# --------------------------------------------------------------------
def solve(problem: dict) -> dict:
    """
    Solves a linear-quadratic regulator (LQR) style SDP for feedback gain K
    and Lyapunov matrix P.
    """
    # ----- 1. Prepare data ------------------------------------------------
    A = np.asarray(problem["A"])
    B = np.asarray(problem["B"])
    n, m = A.shape[0], B.shape[1]

    # ----- 2. Declare CVXPY variables ------------------------------------
    Q = cp.Variable((n, n), PSD=True)   # PD ensures constraint is positive definite
    L = cp.Variable((m, n))

    # ----- 3. Constraints -------------------------------------------------
    # Block-matrix inequality: [ Q,  Q Aᵀ + Lᵀ Bᵀ ;
    #                            A Q + B L,  Q ]  >> I_{2n}
    blk = cp.bmat([[Q, Q @ A.T + L.T @ B.T],
                   [A @ Q + B @ L,     Q]])
    constraints = [blk >> np.eye(2 * n),   # block LMI
                   Q >> np.eye(n)]        # ensure Q is bounded away from singular

    # ----- 4. Problem definition -----------------------------------------
    prob = cp.Problem(cp.Minimize(0), constraints)

    # ----- 5. Solve ------------------------------------------------------
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=5000)
    except Exception:
        return {"is_stabilizable": False, "K": None, "P": None}

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return {"is_stabilizable": False, "K": None, "P": None}

    # ----- 6. Post‑processing ---------------------------------------------
    try:
        Q_val = Q.value
        L_val = L.value
        P_mat = np.linalg.inv(Q_val)
        K_mat = L_val @ P_mat
    except Exception:
        return {"is_stabilizable": False, "K": None, "P": None}

    return {"is_stabilizable": True,
            "K": K_mat.tolist(),
            "P": P_mat.tolist()}