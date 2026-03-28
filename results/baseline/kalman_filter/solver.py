from typing import Any
import cvxpy as cp
import numpy as np

class Solver:

    def solve(self, problem: dict) -> dict:
        A = np.array(problem['A'])
        B = np.array(problem['B'])
        C = np.array(problem['C'])
        y = np.array(problem['y'])
        x0 = np.array(problem['x_initial'])
        tau = float(problem['tau'])
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        x = cp.Variable((N + 1, n), name='x')
        w = cp.Variable((N, p), name='w')
        v = cp.Variable((N, m), name='v')
        obj = cp.Minimize(cp.sum_squares(w) + tau * cp.sum_squares(v))
        cons = []
        cons.append(x[0] == x0)
        for t in range(N):
            cons.append(x[t + 1] == A @ x[t] + B @ w[t])
        else:
            pass
        for t in range(N):
            cons.append(y[t] == C @ x[t] + v[t])
        else:
            pass
        prob = cp.Problem(obj, cons)
        try:
            prob.solve()
        except cp.SolverError as e:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}
        except Exception as e:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}
        else:
            pass
        finally:
            pass
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}
        else:
            pass
        return {'x_hat': x.value.tolist(), 'w_hat': w.value.tolist(), 'v_hat': v.value.tolist()}
