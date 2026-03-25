import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        transitions = np.array(problem["transitions"])
        rewards = np.array(problem["rewards"])
        
        R = np.einsum('sas, sas -> sa', transitions, rewards)
        b = R.flatten()
        
        transitions_flat = transitions.reshape(num_states * num_actions, num_states)
        D = np.zeros((num_states * num_actions, num_states))
        D[np.arange(num_states * num_actions), np.arange(num_states * num_actions) // num_actions] = 1.0
        A = D - gamma * transitions_flat
        
        V_vars = cp.Variable(num_states)
        constraints = [cp.matmul(A, V_vars) >= b]
        objective = cp.Minimize(cp.sum(V_vars))
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        except Exception:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        if V_vars.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        val_func = V_vars.value.tolist()
        val_func_np = np.array(val_func)
        T = np.einsum('sas, s -> sa', transitions, val_func_np)
        Q = R + gamma * T
        policy = np.argmax(Q, axis=1).tolist()
        
        return {"value_function": val_func, "policy": policy}
