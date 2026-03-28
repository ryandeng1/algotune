from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]

        transitions = np.array(problem["transitions"])
        rewards = np.array(problem["rewards"])

        # Precompute Q for constraints: Q[s,a] = sum_sp transitions[s,a,sp] * rewards[s,a,sp]
        Q = np.sum(transitions * rewards, axis=2)

        V_vars = cp.Variable(shape=(num_states), name="V")
        constraints = []
        for s in range(num_states):
            for a in range(num_actions):
                constraints.append(V_vars[s] >= Q[s, a] + gamma * cp.sum(transitions[s, a] * V_vars))

        objective = cp.Minimize(cp.sum(V_vars))
        prob_cvx = cp.Problem(objective, constraints)
        try:
            prob_cvx.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        except Exception as e:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        if V_vars.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        val_func = V_vars.value.tolist()
        rewards_gamma = rewards + gamma * val_func.reshape(num_states, 1, num_states)
        Q_val = np.sum(transitions * rewards_gamma, axis=2)

        policy = []
        for s in range(num_states):
            best_a = 0
            best_rhs = -1e10
            for a in range(num_actions):
                if Q_val[s, a] > best_rhs + 1e-8:
                    best_rhs = Q_val[s, a]
                    best_a = a
            policy.append(best_a)

        return {"value_function": val_func, "policy": policy}