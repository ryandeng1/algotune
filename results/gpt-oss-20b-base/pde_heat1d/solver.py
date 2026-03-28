import numpy as np
from typing import Any, Dict, List

class Solver:

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: Dict[str, Any], debug: bool = True) -> Any:
        # Extract problem data
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]

        # Discretization parameters
        steps = 1000 if debug else 100  # adjustable for speed vs precision
        dt = (t1 - t0) / steps
        n = y0.size

        # Helper to compute RHS of heat equation
        def rhs(u: np.ndarray) -> np.ndarray:
            u_xx = np.empty_like(u)
            # interior points
            u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            # boundaries with Dirichlet BCs (u=0 outside)
            u_xx[0] = (u[1] - 2 * u[0]) / dx**2
            u_xx[-1] = (-2 * u[-1] + u[-2]) / dx**2
            return alpha * u_xx

        # RK4 loop
        u = y0.copy()
        for _ in range(steps):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Mimic the structure of solve_ivp result
        class Result:
            success = True
            message = ""
            y = np.column_stack([u])
        return Result()