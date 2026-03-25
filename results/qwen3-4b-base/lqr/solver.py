import jax
import jax.numpy as jnp
from jax import jit

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = jnp.array(problem["A"])
        B = jnp.array(problem["B"])
        Q = jnp.array(problem["Q"])
        R = jnp.array(problem["R"])
        P = jnp.array(problem["P"])
        T = problem["T"]
        x0 = jnp.array(problem["x0"])
        
        @jit
        def riccati_recursion(A, B, Q, R, P, T):
            n, m = B.shape
            S = jnp.zeros((T + 1, n, n))
            K = jnp.zeros((T, m, n))
            S = S.at[T].set(P)
            for t in range(T - 1, -1, -1):
                St1 = S[t + 1]
                M1 = R + jnp.dot(B.T, St1) @ B
                M2 = jnp.dot(B.T, St1) @ A
                Kt = jnp.linalg.solve(M1, M2)
                Acl = A - jnp.dot(B, Kt)
                S = S.at[t].set(Q + jnp.dot(Kt.T, R) @ Kt + jnp.dot(Acl.T, St1) @ Acl)
                S = S.at[t].set((S + S.T) / 2)
            return S, K
        
        S, K = riccati_recursion(A, B, Q, R, P, T)
        
        @jit
        def forward_sim(K, A, B, x0, T):
            x = x0
            U = jnp.zeros((T, B.shape[1]))
            for t in range(T):
                u = -K[t] @ x
                U = U.at[t].set(u)
                x = A @ x + B @ u
            return U
        
        U = forward_sim(K, A, B, x0, T)
        return {"U": U.tolist()}
