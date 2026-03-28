import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.asarray(problem["masses"], dtype=np.float64)
        softening = problem["softening"]
        n = problem["num_bodies"]

        def nbodyproblem(t, y):
            # extract positions and velocities
            pos = y[: n * 3].reshape(n, 3)
            vel = y[n * 3 :].reshape(n, 3)

            # pairwise relative vectors (j - i)
            rij = pos[None, :, :] - pos[:, None, :]  # shape (n, n, 3)
            # squared distances with softening
            dist2 = np.sum(rij ** 2, axis=2) + softening ** 2  # (n, n)
            # avoid self-interaction by masking
            np.fill_diagonal(dist2, np.inf)

            # 1 / r^3 for each pair
            inv_r3 = 1.0 / (dist2 * np.sqrt(dist2))  # (n, n)
            # mass[j] * inv_r3[i, j]
            coeff = inv_r3 * masses  # broadcast masses to rows (n, n)

            # vector forces on each body: sum over j
            acc = np.einsum("nij,ij->ni", rij, coeff)  # (n, 3)

            # derivative of state vector
            return np.concatenate([vel.reshape(-1), acc.reshape(-1)])

        # integration settings
        rtol, atol = 1e-8, 1e-8
        method = "RK45"
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        return solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )