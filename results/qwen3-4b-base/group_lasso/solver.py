import jax
import jax.numpy as jnp
import numpy as np

class Solver:
    def solve(self, problem):
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = problem["lba"]
        
        # Precompute group information
        unique_groups, inverse_inds, pjs = np.unique(gl, return_inverse=True, return_counts=True)
        m = len(unique_groups)
        group_sizes = np.sqrt(pjs)
        group_size_vector = group_sizes[inverse_inds]
        
        # Convert to JAX arrays
        X = jnp.array(X)
        y = jnp.array(y)
        lba = jnp.array(lba)
        
        # Define gradient function
        def grad(beta):
            z = jnp.dot(X, beta)
            sigmoid_z = jnp.exp(z) / (1 + jnp.exp(z))
            grad_loss = jnp.dot(X.T, sigmoid_z - y)
            penalty_grad = 2 * lba * group_size_vector * beta[1:]
            return jnp.concatenate([jnp.array([0.0]), penalty_grad])
        
        # Initial guess
        beta = jnp.zeros(X.shape[1])
        
        # Run gradient descent
        n_iter = 100
        step_size = 0.1
        for _ in range(n_iter):
            grad_val = grad(beta)
            beta = beta - step_size * grad_val
        
        # Convert to numpy for output
        beta_np = beta.numpy()
        optimal_value = -np.sum(y * (X @ beta_np)) + np.sum(np.log(1 + np.exp(X @ beta_np)))
        penalty = lba * np.sum(group_size_vector * np.linalg.norm(beta_np[1:], axis=0)**2)
        return {
            "beta0": beta_np[0],
            "beta": beta_np[1:].tolist(),
            "optimal_value": optimal_value + penalty
        }
