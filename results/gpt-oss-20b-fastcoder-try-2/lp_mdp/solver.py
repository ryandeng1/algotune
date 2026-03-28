import numpy as np

class Solver:
    def solve(self, problem: dict):
        """
        Solve a discounted MDP using fast value iteration.
        """
        ns, na, gamma = (
            problem["num_states"],
            problem["num_actions"],
            problem["discount"],
        )
        T = np.array(problem["transitions"])  # shape (ns, na, ns)
        R = np.array(problem["rewards"])     # shape (ns, na, ns)

        # Pre‑compute expected rewards for each (state, action)
        # E[R | s, a] = sum_{s'} P(s'|s,a) * R(s,a,s')
        exp_R = np.sum(T * R, axis=2)  # shape (ns, na)

        V = np.zeros(ns)  # initial value function
        eps = 1e-8
        max_iter = 2000
        for _ in range(max_iter):
            # Compute Bellman backup for all actions in one matrix op
            Q = exp_R + gamma * (T @ V)  # shape (ns, na)
            new_V = np.max(Q, axis=1)
            if np.max(np.abs(new_V - V)) < eps:
                V = new_V
                break
            V = new_V

        # Extract deterministic policy
        policy = np.argmax(Q, axis=1).tolist()
        return {"value_function": V.tolist(), "policy": policy}