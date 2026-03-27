import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict, List


def solve(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Solve the discounted MDP by linear programming using a dense matrix
    representation and SciPy's efficient linprog solver instead of CVXPY.
    """
    num_states = problem["num_states"]
    num_actions = problem["num_actions"]
    gamma = problem["discount"]

    transitions = np.asarray(problem["transitions"])  # (S, A, S)
    rewards = np.asarray(problem["rewards"])          # (S, A, S)

    # Build matrix A (S*A x S) and RHS b (S*A)
    A = np.empty((num_states * num_actions, num_states), dtype=np.float64)
    b = np.empty(num_states * num_actions, dtype=np.float64)

    row = 0
    for a in range(num_actions):
        for s in range(num_states):
            A[row, :] = np.eye(1, num_states, s, dtype=np.float64)[0] - gamma * transitions[s, a, :]
            b[row] = np.sum(rewards[s, a, :])
            row += 1

    # objective: minimize sum V => c = [1]*S
    c = np.ones(num_states, dtype=np.float64)

    # Solve LP: minimize c^T V  subject to A V >= b
    # linprog works with inequality in form A_ub @ x <= b_ub,
    # so multiply by -1 to convert.
    res = linprog(
        c=c,
        A_ub=-A,
        b_ub=-b,
        bounds=[(None, None)] * num_states,
        method="highs",
        options={"tol": 1e-8},
    )

    if not res.success or res.x is None:
        return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

    val_func = res.x.tolist()

    # Recover policy by evaluating Q(s,a) with the obtained value function
    policy = []
    V_np = np.array(val_func)
    for s in range(num_states):
        best_a = 0
        best_val = -np.inf
        for a in range(num_actions):
            q_val = np.sum(transitions[s, a, :] * (rewards[s, a, :] + gamma * V_np))
            if q_val > best_val + 1e-8:
                best_val = q_val
                best_a = a
        policy.append(best_a)

    return {"value_function": val_func, "policy": policy}