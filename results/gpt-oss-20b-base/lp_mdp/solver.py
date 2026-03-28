import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the discounted MDP with value iteration.
        """
        num_states = problem['num_states']
        num_actions = problem['num_actions']
        gamma = problem['discount']
        transitions = np.asarray(problem['transitions'], dtype=np.float64)  # shape: S x A x S
        rewards = np.asarray(problem['rewards'], dtype=np.float64)          # shape: S x A x S

        # Pre‑compute expected immediate rewards for every (s, a)
        exp_reward = np.sum(transitions * rewards, axis=2)  # shape: S x A

        V = np.zeros(num_states, dtype=np.float64)
        for _ in range(5000):  # safety cap
            # Compute expected value of following each action
            V_next = np.max(
                exp_reward + gamma * np.tensordot(transitions, V, axes=([2], [0])),
                axis=1,
            )
            if np.max(np.abs(V_next - V)) < 1e-9:
                V = V_next
                break
            V = V_next

        # Derive deterministic policy
        policy = []
        EV = exp_reward + gamma * np.tensordot(transitions, V, axes=([2], [0]))
        for s in range(num_states):
            best_a = int(np.argmax(EV[s]))
            policy.append(best_a)

        return {"value_function": V.tolist(), "policy": policy}