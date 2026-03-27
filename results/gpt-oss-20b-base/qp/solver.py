from typing import Any
import numpy as np
import osqp
import scipy.sparse as sp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert data to numpy arrays of float dtype
        P = np.asarray(problem["P"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)
        G = np.asarray(problem["G"], dtype=float)
        h = np.asarray(problem["h"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        n = P.shape[0]
        mG = G.shape[0]
        mA = A.shape[0]

        # Ensure P is symmetric
        P = (P + P.T) / 2.0

        # OSQP expects a (n+n) x n QP: 0.5 x'Px + q'x
        #   Gx <= h  -> l = -inf, u = h
        #   Ax = b   -> l = u = b
        if mG == 0 and mA == 0:
            # Unconstrained problem – solve directly
            L = np.linalg.cholesky(P)
            x_sol = np.linalg.solve(L.T, np.linalg.solve(L, -q))
            return {"solution": x_sol.tolist(), "objective": float(0.5 * x_sol @ P @ x_sol + q @ x_sol)}

        # Build constraint matrix
        if mG > 0 and mA > 0:
            # For efficiency, build sparse blocks
            Data = []
            Rows = []
            Cols = []

            # G block
            if sp.issparse(G):
                G_sp = G.tocoo()
            else:
                G_sp = sp.coo_matrix(G)
            Data.extend(G_sp.data)
            Rows.extend(G_sp.row)
            Cols.extend(G_sp.col)

            # A block
            if sp.issparse(A):
                A_sp = A.tocoo()
            else:
                A_sp = sp.coo_matrix(A)
            Data.extend(A_sp.data)
            Rows.extend(A_sp.row + mG)
            Cols.extend(A_sp.col)

            AA = sp.coo_matrix((Data, (Rows, Cols)), shape=(mG + mA, n))
        elif mG > 0:
            AA = sp.csc_matrix(G) if sp.issparse(G) else sp.csc_matrix(G)
        else:
            # Only equality constraints
            AA = sp.csc_matrix(A) if sp.issparse(A) else sp.csc_matrix(A)

        # Construct bounds
        l = np.empty(mG + mA)
        u = np.empty(mG + mA)
        if mG > 0:
            l[:mG] = -np.inf
            u[:mG] = h
        if mA > 0:
            l[mG:] = b
            u[mG:] = b

        # Convert P to sparse format
        if sp.issparse(P):
            P_sp = sp.csc_matrix(P)
        else:
            P_sp = sp.csc_matrix(P)

        # Setup OSQP problem
        prob = osqp.OSQP()
        prob.setup(
            P=P_sp, q=q,
            A=AA, l=l, u=u,
            eps_abs=1e-8, eps_rel=1e-8,
            verbose=False, warm_start=True
        )

        res = prob.solve()

        if res.info.status_val not in (200, 201, 202, 203):  # OSQP status codes for optimal
            raise ValueError(f"OSQP solver failed (status = {res.info.status})")

        x_sol = res.x
        obj_val = 0.5 * x_sol @ P @ x_sol + q @ x_sol

        return {"solution": x_sol.tolist(), "objective": float(obj_val)}