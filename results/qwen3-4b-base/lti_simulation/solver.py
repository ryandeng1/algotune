import numpy as np
import jax
from jax import numpy as jnp
from jax.scipy import odeint

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        
        n = len(den) - 1
        A = np.zeros((n, n))
        for i in range(n-1):
            A[i, i+1] = 1.0
        A[n-1, :] = -np.array(den[:-1]) / den[0]
        
        B = np.zeros((n, 1))
        B[n-1] = 1.0 / den[0]
        
        C = np.array(num)
        
        t_jax = jnp.array(t)
        u_jax = jnp.array(u)
        x0 = jnp.zeros(n)
        
        def ode_func(t, x):
            return A @ x + B @ u_jax
        
        sol = odeint(ode_func, x0, t_jax)
        yout = C @ sol
        
        return {"yout": yout.tolist()}
