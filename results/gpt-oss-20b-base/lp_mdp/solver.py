from typing import Any, Dict, List
import numpy as np
import cvxpy as cp

# ------------------------------------------------------------
# Solver implementation
# ------------------------------------------------------------
class Solver:
    """
    Solve a discounted Markov Decision Process (MDP) via a linear program.
    The LP is formulated as:

        Minimise   sum_s V_s
        Subject to V_s >= r(s,a) + γ * Σ_{s'} P(s'|s,a) * V_{s'}   ∀s,a

    The policy is derived by selecting, for each state, the action that
    (approximately) saturates the corresponding constraint.
    """

    # ------------------------------------------------------------------
    #   constructor / initialisation
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        pass  # no expensive setup required

    # ------------------------------------------------------------------
    #   core routine
    # ------------------------------------------------------------------
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        # ------------------------------------------------------------------
        #   Unpack problem
        # ------------------------------------------------------------------
        num_states: int = problem["num_states"]
        num_actions: int = problem["num_actions"]
        gamma: float = problem["discount"]

        transitions: np.ndarray = np.asarray(problem["transitions"], dtype=np.float64)
        rewards: np.ndarray = np.asarray(problem["rewards"], dtype=np.float64)

        # ------------------------------------------------------------------
        #   Build the linear program
        #   --------------------------------------------------------------
        #   Each action for each state generates a scalar inequality.
        #   We express the left‑hand side explicitly (V_s) and the
        #   right‑hand side as a linear combination of the V_vars plus a constant.
        # ------------------------------------------------------------------
        V_vars = cp.Variable(shape=(num_states,), name="V")

        # Pre‑compute constants for each (s,a):
        #   const[s,a] = Σ_{sp} P(s'|s,a) * r(s,a,s')
        con_actions = np.einsum("ska,ska->sa", transitions, rewards)
        # Pre‑compute γ * P(s'|s,a) for each (s,a) as an array of shape
        # (num_states, num_actions, num_states)
        # Transpose later to use with V_vars.
        gamma_P = gamma * transitions  # shape (s,a,sp)

        # Build constraints:
        constraints = []
        for s in range(num_states):
            Vs = V_vars[s]
            for a in range(num_actions):
                # RHS: const + (γ P) · V_vars
                rhs = con_actions[s, a] + gamma_P[s, a].dot(V_vars)
                constraints.append(Vs >= rhs)

        # Objective: minimise Σ V_s
        objective = cp.Minimize(cp.sum(V_vars))

        # Solve the LP
        problem_lp = cp.Problem(objective, constraints)

        try:
            # ECOS is often faster for small LPs; set_silent=true suppresses output.
            problem_lp.solve(solver=cp.ECOS, verbose=False, max_iters=3000)
        except Exception:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        # If solver fails to find a V we return a safe default.
        if V_vars.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}

        val_func = V_vars.value.ravel().tolist()

        # ------------------------------------------------------------------
        #   Construct optimal policy
        #   --------------------------------------------------------------
        #   For each state we compute the expected return for every action
        #   using the obtained value function.
        # ------------------------------------------------------------------
        policy: List[int] = []

        # Pre‑compute expected reward part for each action:
        #  exp_r[s,a] = Σ_{sp} P(s'|s,a) * r(s,a,s')
        exp_r = con_actions  # already computed above

        V_arr = np.asarray(val_func, dtype=np.float64)

        for s in range(num_states):
            # Compute RHS for all actions in one vectorised operation.
            # : gamma_P[s] shape (num_actions, num_states), dot with V_arr.
            rhs_actions = exp_r[s] + gamma * np.dot(gamma_P[s], V_arr)

            best_a = int(np.argmax(rhs_actions))
            policy.append(best_a)

        return {"value_function": val_func, "policy": policy}