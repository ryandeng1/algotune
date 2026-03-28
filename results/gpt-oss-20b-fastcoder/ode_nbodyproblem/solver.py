import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List, Union

class Solver:
    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"solution": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, Union[np.ndarray, float]], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.asarray(problem["masses"], dtype=float)
        softening = problem["softening"]
        n = int(problem["num_bodies"])

        # Pre‑compute a universal index mask to exclude self interactions
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)

        def nbody(t: float, y: np.ndarray) -> np.ndarray:
            # Split state vector into positions and velocities
            pos = y[: n * 3].reshape(n, 3)
            vel = y[n * 3 :].reshape(n, 3)

            # Compute pairwise separation vectors (n, n, 3)
            r = pos[:, None, :] - pos[None, :, :]  # j - i

            # Squared distances with softening
            r2 = np.sum(r * r, axis=2) + softening ** 2   # (n, n)
            inv_r3 = np.where(mask, 1 / (r2 * np.sqrt(r2)), 0)  # (n, n)

            # Multiply by masses of target bodies (m_j)
            mass_matrix = masses[None, :]                    # (1, n)
            weight = inv_r3 * mass_matrix                    # (n, n)

            # Acceleration on each body i: sum over j ≠ i (m_j * r_ij / |r_ij|^3)
            acc = np.sum(weight[:, :, None] * r, axis=1)     # (n, 3)

            return np.concatenate([vel.ravel(), acc.ravel()])

        rtol, atol = 1e-8, 1e-8
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        return solve_ivp(
            nbody,
            (t0, t1),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )