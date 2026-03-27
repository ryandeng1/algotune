import numpy as np
from typing import Any
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        """
        Solve a discounted MDP via a linear program written out explicitly
        and solved with SciPy's simplex method. This avoids the overhead of
        CVXPY by building the constraints as a dense matrix and using a
        fast, compiled solver.
        """
        # --- 1. Read problem data ------------------------------------------
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]

        transitions = np.asarray(problem["transitions"], dtype=np.float64)
        rewards = np.asarray(problem["rewards"], dtype=np.float64)

        # --- 2. Build linear program ---------------------------------------
        # Variables: V[0..S-1]
        # Constraints: for each state s and action a
        #   V_s - gamma * sum_{s'} P(s'|s,a) * V_{s'} >= sum_{s'} P(s'|s,a) * r(s,a,s')
        # We create inequalities of the form A_ineq @ V >= b_ineq
        A_ineq = np.zeros((num_states * num_actions, num_states), dtype=np.float64)
        b_ineq = np.zeros(num_states * num_actions, dtype=np.float64)

        for s in range(num_states):
            for a in range(num_actions):
                idx = s * num_actions + a
                # left-hand side coefficients
                A_ineq[idx, s] = 1.0
                A_ineq[idx] -= gamma * transitions[s, a, :]
                # right-hand side
                b_ineq[idx] = np.dot(transitions[s, a, :], rewards[s, a, :])

        # Objective: minimize sum(V)
        c = np.ones(num_states, dtype=np.float64)

        # --- 3. Solve ------------------------------------------------------
        res = linprog(c, A_ub=-A_ineq, b_ub=-b_ineq, bounds=(None, None), method='highs')
        if res.success:
            V = res.x
        else:
            # fallback to zero solution if the LP fails
            V = np.zeros(num_states, dtype=np.float64)

        # --- 4. Recover policy ---------------------------------------------
        policy = []
        for s in range(num_states):
            best_a = 0
            best_val = -np.inf
            for a in range(num_actions):
                expected = np.dot(transitions[s, a, :], rewards[s, a, :] + gamma * V)
                if expected > best_val + 1e-8:
                    best_val = expected
                    best_a = a
            policy.append(best_a)

        # --- 5. Return results ---------------------------------------------
        return {"value_function": V.tolist(), "policy": policy}