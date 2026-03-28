import numpy as np

class Solver:
    def solve(self, problem: dict[str, float | np.ndarray]) -> dict[str, list[float]]:
        """
        Solve a discounted Markov Decision Process using value iteration.
        The algorithm iteratively improves the value function until the
        maximum change is below a tolerance.  The resulting policy chooses
        an action that maximises the Bellman backup for each state.

        Args:
            problem: Dictionary containing the keys:
                - 'num_states': number of states (int)
                - 'num_actions': number of actions (int)
                - 'discount': discount factor gamma (float)
                - 'transitions': 3‑D array of shape (S, A, S) giving
                                 transition probabilities P(s'|s,a)
                - 'rewards': 3‑D array of shape (S, A, S) giving
                              immediate rewards r(s,a,s')

        Returns:
            A dictionary with keys:
                - 'value_function': list of state values
                - 'policy': list of chosen actions (0‑based indexing)
        """
        # Extract problem data
        num_states = problem['num_states']
        num_actions = problem['num_actions']
        gamma = problem['discount']
        transitions = np.asarray(problem['transitions'], dtype=np.float64)
        rewards = np.asarray(problem['rewards'], dtype=np.float64)

        # Value iteration
        V = np.zeros(num_states, dtype=np.float64)
        policy = np.zeros(num_states, dtype=int)

        # Convergence parameters
        eps = 1e-8   # tolerance
        max_iter = 10000

        for _ in range(max_iter):
            V_old = V.copy()

            # Compute Bellman backups for all states
            for s in range(num_states):
                # (A, S) tensors for this state
                trans_sa = transitions[s]          # shape (A, S)
                rewards_sa = rewards[s]           # shape (A, S)

                # Expected return for each action:
                # sum_{s'} P(s'|s,a) * [ r(s,a,s') + gamma * V_old[s'] ]
                tmp = trans_sa * (rewards_sa + gamma * V_old)
                act_vals = tmp.sum(axis=1)         # shape (A,)

                # Best action and value
                best_a = int(act_vals.argmax())
                V[s] = act_vals[best_a]
                policy[s] = best_a

            # Check convergence
            if np.max(np.abs(V - V_old)) < eps:
                break

        # Return results
        return {
            'value_function': V.tolist(),
            'policy': policy.tolist()
        }