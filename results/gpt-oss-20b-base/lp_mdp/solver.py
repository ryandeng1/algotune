import numpy as np
from typing import Any, Dict, List

class Solver:
    """
    Fast discounted MDP solver using value iteration.
    """
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        # Extract data
        ns, na, gamma = (
            problem["num_states"],
            problem["num_actions"],
            problem["discount"],
        )
        # Ensure numpy arrays with proper dtype
        trans = np.asarray(problem["transitions"], dtype=np.float64)  # shape: (ns, na, ns)
        rew = np.asarray(problem["rewards"], dtype=np.float64)       # shape: (ns, na, ns)

        # Pre‑compute expected rewards for each state-action pair
        # E[r | s,a] = sum_sp P(s'|s,a)*r(s,a,s')
        exp_r = np.sum(rew * trans, axis=2)  # shape: (ns, na)

        # Value iteration parameters
        V = np.zeros(ns, dtype=np.float64)
        # Use a single buffer to avoid repeated allocations
        V_next = np.empty_like(V)

        # Value iteration loop
        for _ in range(1000):
            # Compute RHS for all states and actions in one go
            # shape: (ns, na)
            rhs = exp_r + gamma * trans @ V
            # Bellman optimality: V[s] = max_a RHS[s,a]
            np.max(rhs, axis=1, out=V_next)
            # Check convergence
            delta = np.max(np.abs(V_next - V))
            V[:] = V_next
            if delta < 1e-8:
                break

        # Extract greedy policy
        # For each state find action that attains the maximum (breaking ties by first)
        policy = np.argmax(rhs, axis=1).tolist()

        return {"value_function": V.tolist(), "policy": policy}