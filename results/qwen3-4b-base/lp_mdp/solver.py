import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        transitions = np.array(problem["transitions"])
        rewards = np.array(problem["rewards"])
        
        transitions_reshaped = transitions.reshape(num_states * num_actions, num_states)
        rewards_reshaped = rewards.reshape(num_states * num_actions, num_states)
        
        total_constraints = num_states * num_actions
        s_indices = np.arange(total_constraints) // num_actions
        B = np.zeros((total_constraints, num_states))
        B[np.arange(total_constraints), s_indices] = 1.0
        B = B - gamma * transitions_reshaped
        
        c = np.einsum('ij,ij->i', transitions_reshaped, rewards_reshaped)
        
        V = cp.Variable(num_states)
        objective = cp.Minimize(cp.sum(V))
        constraints = [B @ V >= c]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        except Exception:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        if V.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        val_func = V.value.tolist()
        
        policy = []
        for s in range(num_states):
            best_a = 0
            best_rhs = -1e10
            for a in range(num_actions):
                rhs_value = np.sum(transitions[s, a] * (rewards[s, a] + gamma * np.array(val_func)))
                if rhs_value > best_rhs + 1e-8:
                    best_rhs = rhs_value
                    best_a = a
            policy.append(best_a)
        
        return {"value_function": val_func, "policy": policy}
