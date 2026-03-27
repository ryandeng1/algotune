import numpy as np
import osqp
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Load data as numpy arrays
        P = np.asarray(problem["P"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)
        G = np.asarray(problem["G"], dtype=float)
        h = np.asarray(problem["h"], dtype=float)
        A = np.asarray(problem["A"], dtype=float) if "A" in problem else np.empty((0, P.shape[0]))
        b = np.asarray(problem["b"], dtype=float) if "b" in problem else np.empty(0)

        # Preprocess P to be symmetric
        P = 0.5 * (P + P.T)

        n = P.shape[0]

        # Build constraint matrix [G; A] and bounds
        if G.size:
            # Inequalities Gx <= h
            Gs = sparse.csc_matrix(G)
            gs_l = np.full(G.shape[0], -np.inf)
            gs_u = h
        else:
            Gs = None
            gs_l = np.array([])
            gs_u = np.array([])

        if A.size:
            # Equalities Ax = b
            As = sparse.csc_matrix(A)
            as_l = b
            as_u = b
        else:
            As = None
            as_l = np.array([])
            as_u = np.array([])

        # Combine
        if Gs is not None and As is not None:
            A_mat = sparse.vstack([Gs, As]).tocsr()
        elif Gs is not None:
            A_mat = Gs.tocsr()
        elif As is not None:
            A_mat = As.tocsr()
        else:
            A_mat = sparse.csc_matrix((0, n))

        l = np.concatenate([gs_l, as_l])
        u = np.concatenate([gs_u, as_u])

        # Convert P to sparse form
        P_sparse = sparse.csc_matrix(P)

        # Setup OSQP
        prob = osqp.OSQP()
        prob.setup(P=P_sparse, q=q, A=A_mat, l=l, u=u,
                   eps_abs=1e-8, eps_rel=1e-8, verbose=False)

        res = prob.solve()
        if res.info.status_val not in [osqp.constant('OSQP_SOLVED'), osqp.constant('OSQP_SOLVED_INACCURATE')]:
            raise ValueError(f"Solver failed (status = {res.info.status})")

        x = res.x
        objective = 0.5 * np.dot(x, P @ x) + np.dot(q, x)

        return {"solution": x.tolist(), "objective": float(objective)}