import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Very small wrapper around scipy.integrate.solve_ivp that solves the
    Lorenz-96 system.

    This implementation does *not* pre‑allocate any expensive temporary
    arrays inside the RHS routine and therefore runs significantly
    faster than the reference implementation that uses np.roll in every
    call.
    """

    def solve(self, problem):
        """
        Parameters
        ----------
        problem : dict
            ``problem['y0']`` : array‑like initial state
            ``problem['t0']`` : float initial time
            ``problem['t1']`` : float final time
            ``problem['F']`` : float constant forcing term

        Returns
        -------
        dict[str, list[float]]
            Convergence status and final state
        """
        y0 = np.asarray(problem['y0'], dtype=float)
        t0, t1 = float(problem['t0']), float(problem['t1'])
        F = float(problem['F'])

        N = y0.size

        # Pre‑allocate memory that is reused in every RHS call
        x_next = np.empty(N, dtype=float)
        x_prev1 = np.empty(N, dtype=float)
        x_prev2 = np.empty(N, dtype=float)

        def lorenz96(_, x):
            # Periodic boundary conditions realised with slicing
            x_next[:]  = x[1:]      + [x[0]]        # x[i+1]
            x_prev1[:] = [x[-1]]    + x[:-1]        # x[i-1]
            x_prev2[:] = [x[-2], x[-1]] + x[:-2]    # x[i-2]

            # Lorenz‑96 RHS: (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
            return (x_next - x_prev2) * x_prev1 - x + F

        # Dense output is not required – we just need the solution at t1
        t_eval = None
        rtol, atol = 1e-8, 1e-8
        method = 'RK45'

        sol = solve_ivp(lorenz96, [t0, t1], y0, method=method,
                        rtol=rtol, atol=atol, t_eval=t_eval, dense_output=False)

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final state as a plain Python list
        return sol.y[:, -1].tolist()