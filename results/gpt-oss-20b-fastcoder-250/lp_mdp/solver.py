import numpy as np
from typing import Any, Dict, List
from scipy.optimize import linprog


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]

        trans = np.array(problem["transitions"], dtype=np.float64)  # shape (S, A, S)
        rewards = np.array(problem["rewards"], dtype=np.float64)   # shape (S, A, S)

        # Build LP matrices
        # For each state-action pair we have a constraint:
        #   V_s - gamma * sum_{s'} P(s'|s,a) V_{s'} >= sum_{s'} P(s'|s,a) * r(s,a,s')
        # Arrange as: A @ V >= b
        constraints = []
        rhs = []

        for s in range(num_states):
            for a in range(num_actions):
                row = np.zeros(num_states, dtype=np.float64)
                row[s] = 1.0
                row -= gamma * trans[s, a]  # subtract gamma * P
                constraints.append(row)
                rhs.append(np.sum(trans[s, a] * rewards[s, a]))

        A = np.vstack(constraints)              # shape (S*A, S)
        b = np.array(rhs, dtype=np.float64)     # shape (S*A,)

        # Convert to standard form for scipy.optimize.linprog: minimize c.T x, subject to A_ub @ x <= b_ub
        c = -np.ones(num_states, dtype=np.float64)  # maximize sum V == minimize -sum V
        A_ub = -A
        b_ub = -b

        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', options={'presolve': True})

        if not res.success or res.x is None:
            # Fallback in case of failure
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        V = res.x

        # Recover policy: choose action maximizing Q(s,a)
        policy = []
        for s in range(num_states):
            best_a = 0
            best_q = -np.inf
            for a in range(num_actions):
                q = np.sum(trans[s, a] * (rewards[s, a] + gamma * V))
                if q > best_q + 1e-10:
                    best_q = q
                    best_a = a
            policy.append(best_a)

        return {"value_function": V.tolist(), "policy": policy}
