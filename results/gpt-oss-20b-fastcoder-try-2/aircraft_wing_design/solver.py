from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary with problem parameters.

        Returns
        -------
        dict
            Dictionary with optimal design variables and minimum drag.
        """
        num_conditions = problem['num_conditions']
        conditions = problem['conditions']

        # Decision variables
        A = cp.Variable(pos=True, name='A')
        S = cp.Variable(pos=True, name='S')
        V = [cp.Variable(pos=True, name=f'V_{i}') for i in range(num_conditions)]
        W = [cp.Variable(pos=True, name=f'W_{i}') for i in range(num_conditions)]
        Re = [cp.Variable(pos=True, name=f'Re_{i}') for i in range(num_conditions)]
        C_D = [cp.Variable(pos=True, name=f'C_D_{i}') for i in range(num_conditions)]
        C_L = [cp.Variable(pos=True, name=f'C_L_{i}') for i in range(num_conditions)]
        C_f = [cp.Variable(pos=True, name=f'C_f_{i}') for i in range(num_conditions)]
        W_w = [cp.Variable(pos=True, name=f'W_w_{i}') for i in range(num_conditions)]

        constraints = []
        total_drag = 0

        for i in range(num_conditions):
            cond = conditions[i]
            CDA0 = float(cond['CDA0'])
            C_Lmax = float(cond['C_Lmax'])
            N_ult = float(cond['N_ult'])
            S_wetratio = float(cond['S_wetratio'])
            V_min = float(cond['V_min'])
            W_0 = float(cond['W_0'])
            W_W_coeff1 = float(cond['W_W_coeff1'])
            W_W_coeff2 = float(cond['W_W_coeff2'])
            e = float(cond['e'])
            k = float(cond['k'])
            mu = float(cond['mu'])
            rho = float(cond['rho'])

            # Drag term
            drag_i = 0.5 * rho * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i

            # Inequality constraints
            constraints += [
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (np.pi * A * e),
                C_f[i] >= 0.074 / Re[i] ** 0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S / A),
                W_w[i] >= W_W_coeff2 * S + W_W_coeff1 * N_ult * A ** (3 / 2) * cp.sqrt(W_0 * W[i]) / cond['tau'],
                W[i] >= W_0 + W_w[i],
                W[i] <= 0.5 * rho * V[i] ** 2 * C_L[i] * S,
                2 * W[i] / (rho * V_min ** 2 * S) <= C_Lmax
            ]

        # Objective: minimize average drag
        objective = cp.Minimize(total_drag / num_conditions)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return {'A': [], 'S': [], 'avg_drag': 0.0, 'condition_results': []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
            return {'A': [], 'S': [], 'avg_drag': 0.0, 'condition_results': []}

        # Prepare results
        condition_results = []
        for i in range(num_conditions):
            cond_id = conditions[i]['condition_id']
            condition_results.append({
                'condition_id': cond_id,
                'V': float(V[i].value),
                'W': float(W[i].value),
                'W_w': float(W_w[i].value),
                'C_L': float(C_L[i].value),
                'C_D': float(C_D[i].value),
                'C_f': float(C_f[i].value),
                'Re': float(Re[i].value),
                'drag': float(0.5 * conditions[i]['rho'] * V[i].value ** 2 * C_D[i].value * S.value)
            })

        return {
            'A': float(A.value),
            'S': float(S.value),
            'avg_drag': float(prob.value),
            'condition_results': condition_results
        }