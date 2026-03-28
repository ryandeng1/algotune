import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert data to contiguous float arrays
        P = np.ascontiguousarray(problem['P'], dtype=float)
        q = np.ascontiguousarray(problem['q'], dtype=float)
        G = np.ascontiguousarray(problem['G'], dtype=float)
        h = np.ascontiguousarray(problem['h'], dtype=float)
        A = np.ascontiguousarray(problem['A'], dtype=float)
        b = np.ascontiguousarray(problem['b'], dtype=float)

        n = P.shape[0]
        # Force symmetry to aid the solver
        P = (P + P.T) * 0.5

        x = cp.Variable(n)
        obj = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
        constr = [G @ x <= h, A @ x == b]
        prob = cp.Problem(cp.Minimize(obj), constr)

        # Use OSQP with tight tolerances
        opt_val = prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            warm_start=True,
            verbose=False,
        )

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f'Solver failed (status = {prob.status})')

        return {'solution': x.value.tolist(), 'objective': float(opt_val)}