import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve a discounted finite‑state MDP by value iteration.
        :param problem: dict with keys
            'num_states', 'num_actions', 'discount',
            'transitions' (S×A×S), 'rewards' (S×A×S)
        :return: dict with 'value_function' and 'policy'
        """
        N = problem['num_states']
        A = problem['num_actions']
        gamma = problem['discount']
        trans = np.asarray(problem['transitions'], dtype=np.float64)
        rew   = np.asarray(problem['rewards'],   dtype=np.float64)

        # Pre‑compute the expectation of reward + discounted next value
        # for each state, action: shape (N, A, S)
        exp_rew = rew + 0.0  # copy to avoid modifying input

        V = np.zeros(N, dtype=np.float64)
        max_iter = 2000
        eps = 1e-10

        for _ in range(max_iter):
            V_next = np.empty_like(V)
            # compute the right‑hand side for all actions simultaneously
            rhs = np.einsum('nas,as->na', trans, V)          # shape (N, A)
            rhs += np.sum(exp_rew, axis=2) / trans.sum(axis=2)  # reward term

            V_next = np.max(rhs, axis=1)
            if np.max(np.abs(V_next - V)) < eps:
                V = V_next
                break
            V = V_next

        # derive greedy policy
        rhs = np.einsum('nas,as->na', trans, V)          # shape (N, A)
        rhs += np.sum(exp_rew, axis=2) / trans.sum(axis=2)
        policy = np.argmax(rhs, axis=1).tolist()
        return {'value_function': V.tolist(), 'policy': policy}