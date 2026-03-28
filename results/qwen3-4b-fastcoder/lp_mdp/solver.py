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

        V_vars = cp.Variable(shape=(num_states), name="V")
        constraints = []
        for s in range(num_states):
            for a in range(num_actions):
                rhs_expr = cp.sum(transitions[s, a] * (rewards[s, a] + gamma * V_vars))
                constraints.append(V_vars[s] >= rhs_expr)
        
        objective = cp.Minimize(cp.sum(V_vars))
        prob_cvx = cp.Problem(objective, constraints)
        try:
            prob_cvx.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        except Exception:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        if V_vars.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        val_func = V_vars.value.tolist()
        val_func_np = np.array(val_func)
        
        term1 = np.einsum('sas, sas -> sa', transitions, rewards)
        term2 = gamma * np.einsum('sas, s -> sa', transitions, val_func_np)
        Q = term1 + term2
        
        policy = np.argmax(Q, axis=1).tolist()
        
        return {"value_function": val_func, "policy": policy}