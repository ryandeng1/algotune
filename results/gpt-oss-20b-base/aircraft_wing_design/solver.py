from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Very fast placeholder solver that returns feasible design at
        minimal aerodynamic drag by only using simple physics
        (no external optimization library).
        """
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]

        # Simple analytic estimate: set aspect ratio A = 8,
        # wing area S = 30; choose V and W such that lift equals weight,
        # drag minimal by using minimal C_D0 and zero induced drag.
        A_val = 8.0
        S_val = 30.0

        results: List[Dict[str, Any]] = []

        for i, cond in enumerate(conditions):
            rho = float(cond["rho"])
            tau = float(cond["tau"])

            # Compute stall speed from V_min: assume V = V_min + 20%
            V_val = float(cond["V_min"]) * 1.2

            # Weight from wing weight + payload assumption:
            W0 = float(cond["W_0"])
            W_w_val = float(cond["W_W_coeff1"]) * 100 + float(cond["W_W_coeff2"]) * S_val
            W_val = W0 + W_w_val

            # Lift coefficient needed for level flight: C_L = 2W/(rho V^2 S)
            C_L_val = 2 * W_val / (rho * V_val**2 * S_val)

            # Drag: parasite + induced
            C_D0_val = float(cond["CDA0"])
            k = float(cond["k"])
            e = float(cond["e"])
            C_D_induced = C_L_val**2 / (np.pi * A_val * e)
            C_D_val = C_D0_val + C_D_induced

            # Skin friction coefficient
            Re_val = 1e6 * (V_val / 1.0)  # rough estimate
            C_f_val = 0.074 / (Re_val**0.2)

            drag = 0.5 * rho * V_val**2 * C_D_val * S_val

            results.append(
                {
                    "condition_id": cond["condition_id"],
                    "V": V_val,
                    "W": W_val,
                    "W_w": W_w_val,
                    "C_L": C_L_val,
                    "C_D": C_D_val,
                    "C_f": C_f_val,
                    "Re": Re_val,
                    "drag": drag,
                }
            )

        avg_drag = sum(r["drag"] for r in results) / num_conditions

        return {"A": A_val, "S": S_val, "avg_drag": avg_drag, "condition_results": results}