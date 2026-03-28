import numpy as np
from osqp import OSQP
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Parse problem data
        P = np.asarray(problem['P'], dtype=np.float64)
        q = np.asarray(problem['q'], dtype=np.float64).reshape(-1)
        G = np.asarray(problem['G'], dtype=np.float64)
        h = np.asarray(problem['h'], dtype=np.float64).reshape(-1)
        A = np.asarray(problem['A'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64).reshape(-1)

        n = P.shape[0]
        # Ensure P is symmetric
        P = (P + P.T) / 2.0

        # OSQP works with sparse A, G
        def to_csc(mat):
            return sparse.csc_matrix(mat)

        P_csc = to_csc(P)
        G_csc = to_csc(G)
        A_csc = to_csc(A)

        # Construct QP matrices
        # OSQP solver: minimize 0.5*x.T*P*x + q.T*x
        # Constraints: Gx <= h  and  Ax == b
        # Combine inequality and equality matrices
        if G_csc.size:
            A_comb = sparse.vstack([G_csc, A_csc], format='csc')
            l_comb = np.hstack([np.full(G_csc.shape[0], -np.inf), b])
            u_comb = np.hstack([h, b])
        else:
            A_comb = A_csc
            l_comb = b
            u_comb = b

        # Setup OSQP problem
        prob = OSQP()
        prob.setup(P=P_csc, q=q, A=A_comb, l=l_comb, u=u_comb,
                   verbose=False, eps_abs=1e-9, eps_rel=1e-9)

        # Solve
        res = prob.solve()
        if not res.success:
            raise ValueError(f'Solver failed (code = {res.info["status"]})')

        return {
            'solution': res.x.tolist(),
            'objective': float(res.info['obj_val'])
        }