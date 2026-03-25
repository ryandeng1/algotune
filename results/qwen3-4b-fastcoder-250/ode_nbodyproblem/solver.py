import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        t0 = problem['t0']
        t1 = problem['t1']
        y0 = np.array(problem['y0'])
        masses = np.array(problem['masses'])
        softening = problem['softening']
        num_bodies = problem['num_bodies']
        
        def nbodyproblem(t, y):
            positions = y[:num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3:].reshape(num_bodies, 3)
            dp_dt = velocities.reshape(-1)
            
            diff = positions[:, None] - positions[None, :]
            dist_sq = np.sum(diff**2, axis=2) + softening**2
            denom = dist_sq ** (3/2)
            contribs = masses[:, None] * diff / denom
            accelerations = np.sum(contribs, axis=1)
            dv_dt = accelerations.reshape(-1)
            
            return np.concatenate([dp_dt, dv_dt])
        
        rtol = 1e-8
        atol = 1e-8
        sol = solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method='RK45',
            rtol=rtol,
            atol=atol,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()
