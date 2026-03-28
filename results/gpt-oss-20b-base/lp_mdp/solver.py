import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the discounted MDP via iterative value‑iteration.
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        # Convert to numpy arrays (S, A, S)
        transitions = np.array(problem["transitions"], dtype=np.float64)
        rewards = np.array(problem["rewards"], dtype=np.float64)

        V = np.zeros(num_states, dtype=np.float64)
        eps = 1e-8
        max_iter = 2000  # safety bound
        for _ in range(max_iter):
            # For each action compute Q = r + gamma * P @ V
            # transitions shape (S, A, S)
            # rewards shape (S, A, S)
            # Use einsum for efficient batch dot product
            M = gamma * np.einsum("sas,s->sa", transitions, V)  # (S, A)
            Q = rewards.sum(axis=2) + M  # sum over sp in reward already includes next-state reward?
            # Since rewards[s,a,sp] is immediate reward, sum over sp gives total immediate reward for transition distribution
            best_action = np.argmax(Q, axis=1)
            V_new = Q[np.arange(num_states), best_action]
            diff = np.max(np.abs(V_new - V))
            if diff < eps:
                V = V_new
                break
            V = V_new

        # Policy extraction
        M = gamma * np.einsum("sas,s->sa", transitions, V)
        Q = rewards.sum(axis=2) + M
        policy = np.argmax(Q, axis=1).tolist()
        return {"value_function": V.tolist(), "policy": policy}