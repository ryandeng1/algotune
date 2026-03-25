import numpy as np
import jax
import jax.numpy as jnp
from diffrax import solve

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        F = problem["F"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"], dtype=float)
        
        y0 = jnp.array(y0)
        
        def lorenz96(t, x):
            N = x.shape[0]
            ip1 = jnp.roll(jnp.arange(N), -1)
            im1 = jnp.roll(jnp.arange(N), 1)
            im2 = jnp.roll(jnp.arange(N), 2)
            return (x[ip1] - x[im2]) * x[im1] - x + F
        
        sol = solve(
            lorenz96,
            y0,
            t0,
            t1,
            rtol=1e-8,
            atol=1e-8,
            solver="RK45"
        )
        
        return sol.y[-1].tolist()
