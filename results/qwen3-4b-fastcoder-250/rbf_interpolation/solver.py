import jax
import jax.numpy as jnp
from jax import jit

class Solver:
    def solve(self, problem: dict) -> dict:
        x_train = jnp.array(problem["x_train"], dtype=jnp.float64)
        y_train = jnp.array(problem["y_train"], dtype=jnp.float64).reshape(-1)
        x_test = jnp.array(problem["x_test"], dtype=jnp.float64)
        
        rbf_config = problem["rbf_config"]
        kernel = rbf_config["kernel"]
        epsilon = rbf_config["epsilon"]
        smoothing = rbf_config["smoothing"]
        
        @jit
        def gaussian(r):
            return jnp.exp(-r**2 / epsilon**2)
        
        @jit
        def multiquadric(r):
            return jnp.sqrt(r**2 + epsilon**2)
        
        @jit
        def inverse_multiquadric(r):
            return 1.0 / jnp.sqrt(r**2 + epsilon**2)
        
        @jit
        def thin_plate_spline(r):
            r = jnp.where(r > 0, r, 1e-10)
            return r**2 * jnp.log(r)
        
        @jit
        def linear(r):
            return r
        
        @jit
        def cubic(r):
            return r**3
        
        @jit
        def quintic(r):
            return r**5
        
        kernel_func = {
            "gaussian": gaussian,
            "multiquadric": multiquadric,
            "inverse_multiquadric": inverse_multiquadric,
            "thin_plate_spline": thin_plate_spline,
            "linear": linear,
            "cubic": cubic,
            "quintic": quintic
        }[kernel]
        
        dists = jnp.linalg.norm(x_train[:, None, :] - x_train[None, :, :], axis=-1)
        A = kernel_func(dists)
        A = A + smoothing * jnp.eye(x_train.shape[0])
        
        c = jnp.linalg.solve(A, y_train)
        
        test_dists = jnp.linalg.norm(x_test[:, None, :] - x_train[None, :, :], axis=-1)
        rbf_vals = kernel_func(test_dists)
        y_pred = jnp.dot(c, rbf_vals.T)
        
        return {
            "y_pred": y_pred.tolist(),
            "rbf_config": rbf_config
        }
