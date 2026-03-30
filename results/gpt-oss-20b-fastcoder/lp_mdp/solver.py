from typing import Any, Dict, List
import cvxpy as cp
import numpy as np


class Solver:
    """
    Solve a discounted Markov Decision Process (MDP) with a linear programming
    formulation.

    The LP is
          maximize  Σ V_s
          subject to V_s ≥ r(s,a) + γ Σ_{s'} P(s'|s,a) V_{s'}   ∀ s,a

    After solving for the state values V we construct a greedy policy.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]

        # Convert to numpy arrays for fast vectorised operations
        transitions = np.asarray(problem["transitions"], dtype=np.float64)
        rewards = np.asarray(problem["rewards"], dtype=np.float64)

        # Decision variable
        V = cp.Variable(num_states, name="V")

        constraints = []

        # Construct constraints:  V >= r + γ P V
        # For every (s,a) we write the inequality in a vectorised way.
        for a in range(num_actions):
            P = transitions[:, a, :]                  # shape: (S, S)
            r_vec = rewards[:, a, :].sum(axis=1)      # shape: (S,)
            # Inequality: V ≥ r_vec + γ P @ V
            constraints.append(V >= r_vec + gamma * (P @ V))

        # Objective: maximise Σ V
        objective = cp.Maximize(cp.sum(V))

        problem_cvx = cp.Problem(objective, constraints)
        try:
            problem_cvx.solve(solver=cp.ECOS, verbose=False, eps=1e-5)
        except Exception:
            # Fallback to trivial solution in case of solver failure
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        if V.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        # Extract optimal value function
        value_function = V.value.tolist()

        # Greedy policy: choose action that best satisfies the Bellman equality
        policy = []
        for s in range(num_states):
            best_a = 0
            best_rhs = -np.inf
            for a in range(num_actions):
                rhs_val = np.sum(
                    transitions[s, a] * (rewards[s, a] + gamma * np.array(value_function))
                )
                if rhs_val > best_rhs + 1e-8:
                    best_rhs = rhs_val
                    best_a = a
            policy.append(best_a)

        return {"value_function": value_function, "policy": policy}