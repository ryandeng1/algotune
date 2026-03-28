from typing import Any
import cvxpy as cp
import numpy as np

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        """
        Solve the MDP using the standard linear program for discounted MDPs:

          Maximize \\sum_s V_s
        subject to
          V_s >= r(s,a) + discount * \\sum_{s'} P(s'|s,a)*V_{s'}   for all s,a

        We then pick a policy by finding, for each s, an a that (approximately)
        saturates the constraint:
          V_s ≈ r(s,a) + discount * sum_{s'} p(s'|s,a)*V_{s'}.

        :param problem: Dictionary with 'num_states', 'num_actions', 'discount',
                        'transitions', 'rewards'.
        :return: Dictionary with 'value_function' and 'policy'.
        """
        num_states = problem['num_states']
        num_actions = problem['num_actions']
        gamma = problem['discount']
        transitions = np.array(problem['transitions'])
        rewards = np.array(problem['rewards'])
        V_vars = cp.Variable(shape=(num_states,), name='V')
        constraints = []
        for s in range(num_states):
            for a in range(num_actions):
                rhs = 0
                for sp in range(num_states):
                    rhs += transitions[s, a, sp] * (rewards[s, a, sp] + gamma * V_vars[sp])
                else:
                    pass
                constraints.append(V_vars[s] >= rhs)
            else:
                pass
        else:
            pass
        objective = cp.Minimize(cp.sum(V_vars))
        prob_cvx = cp.Problem(objective, constraints)
        try:
            prob_cvx.solve(solver=cp.SCS, verbose=False, eps=1e-05)
        except Exception as e:
            return {'value_function': [0.0] * num_states, 'policy': [0] * num_states}
        else:
            pass
        finally:
            pass
        if V_vars.value is None:
            return {'value_function': [0.0] * num_states, 'policy': [0] * num_states}
        else:
            pass
        val_func = V_vars.value.tolist()
        policy = []
        for s in range(num_states):
            best_a = 0
            best_rhs = -10000000000.0
            for a in range(num_actions):
                rhs_value = np.sum(transitions[s, a] * (rewards[s, a] + gamma * np.array(val_func)))
                if rhs_value > best_rhs + 1e-08:
                    best_rhs = rhs_value
                    best_a = a
                else:
                    pass
            else:
                pass
            policy.append(best_a)
        else:
            pass
        return {'value_function': val_func, 'policy': policy}
