import numpy as np
from typing import Any, Dict, List

class Solver:
    """
    Fast discounted MDP solver using value iteration.

    Parameters
    ----------
    problem : dict
        Dictionary with the keys:
            - num_states (int)
            - num_actions (int)
            - discount (float)
            - transitions (np.ndarray, shape [S, A, S])
            - rewards (np.ndarray, shape [S, A, S])

    Returns
    -------
    dict
        Keys:
            - value_function : list[float]
            - policy : list[int]
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        # convert to numpy arrays for faster ops
        transitions = np.array(problem["transitions"], dtype=np.float64)
        rewards = np.array(problem["rewards"], dtype=np.float64)

        # Pre‑compute the expected returns for each state–action pair
        # shape: (S, A)
        exp_returns = np.einsum(
            "s a s', s a s' -> s a",
            transitions,
            rewards + gamma * np.eye(num_states)[None, None, :],  # placeholder for V
        )

        # Value iteration
        V = np.zeros(num_states, dtype=np.float64)
        threshold = 1e-8
        max_iter = 10000
        for _ in range(max_iter):
            # Update expected returns with current V
            exp_returns = np.einsum(
                "s a s', s a s' -> s a",
                transitions,
                rewards + gamma * V[None, None, :],
            )
            new_V = exp_returns.max(axis=1)
            if np.max(np.abs(new_V - V)) < threshold:
                V = new_V
                break
            V = new_V

        # Extract deterministic policy
        policy = np.argmax(exp_returns, axis=1).tolist()
        return {"value_function": V.tolist(), "policy": policy}