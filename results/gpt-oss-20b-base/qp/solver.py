import numpy as np
from scipy import sparse
import osqp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to numpy arrays
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        G = np.asarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = P.shape[0]
        # Ensure P is symmetric positive definite
        P = (P + P.T) / 2.0

        # Build OSQP problem
        # OSQP expects a sparse QP in the form:
        #   Min 0.5 x^T P x + q^T x
        #   s.t. A x == l = u = b
        #        G x <= h
        prob = osqp.OSQP()
        P_csc = sparse.csc_matrix(P)
        A_csc = sparse.csc_matrix(A) if A.size else None
        G_csc = sparse.csc_matrix(G) if G.size else None

        if A_csc is not None and G_csc is not None:
            # Concatenate equality and inequality constraints
            A_mat = sparse.vstack([A_csc, G_csc])
            l = np.concatenate([b, -np.inf * np.ones(G_csc.shape[0])])
            u = np.concatenate([b, h])
        elif G_csc is not None:
            A_mat = G_csc
            l = -np.inf * np.ones(G_csc.shape[0])
            u = h
        else:  # only equality constraints
            A_mat = A_csc
            l = b
            u = b

        prob.setup(
            P=P_csc,
            q=q,
            A=A_mat,
            l=l,
            u=u,
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=10000,
        )
        res = prob.solve()
        if res.info.status != "solved":
            raise ValueError(f"Solver failed (status = {res.info.status})")

        return {"solution": res.x.tolist(), "objective": float(res.info.obj_val)}