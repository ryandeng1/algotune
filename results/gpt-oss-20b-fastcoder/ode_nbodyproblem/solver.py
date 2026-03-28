import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0: np.ndarray = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        masses: np.ndarray = np.asarray(problem["masses"], dtype=np.float64)
        softening: float = problem["softening"]
        N = int(problem["num_bodies"])

        def nbody(t: float, y: np.ndarray) -> np.ndarray:
            # Reshape once
            positions = y[: N * 3].reshape(N, 3)
            velocities = y[N * 3 :].reshape(N, 3)

            # Pairwise displacement, shape (N,N,3)
            diff = positions[:, None, :] - positions[None, :, :]
            # squared distance
            dist_sq = np.sum(diff ** 2, axis=2) + softening ** 2
            # 1 / dist^3
            inv_dist3 = np.divide(
                masses, dist_sq ** 1.5, out=np.zeros_like(masses), where=~np.eye(N, dtype=bool)
            )

            # Acceleration: sum over j
            # shape (N,3)
            accel = np.sum(diff * inv_dist3[:, None, None], axis=1)
            # Concatenate derivatives
            dp_dt = velocities.reshape(-1)
            dv_dt = accel.reshape(-1)
            return np.concatenate([dp_dt, dv_dt])

        rtol = 1e-8
        atol = 1e-8
        method = "RK45"
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            nbody,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol