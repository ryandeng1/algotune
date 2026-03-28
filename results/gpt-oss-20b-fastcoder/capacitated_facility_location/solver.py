from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fast heuristic solver for the Capacitated Facility Location Problem.
        It assigns customers to the cheapest open facility with remaining capacity,
        opening facilities on demand.
        """
        # Extract data as numpy arrays for speed
        fixed_costs = np.asarray(problem['fixed_costs'], dtype=np.float64)
        capacities = np.asarray(problem['capacities'], dtype=np.int64)
        demands = np.asarray(problem['demands'], dtype=np.float64)
        c = np.asarray(problem['transportation_costs'], dtype=np.float64)

        n_facilities, n_customers = c.shape

        # Track remaining capacity per facility
        remaining = capacities.copy()
        # Track open facilities
        opened = np.zeros(n_facilities, dtype=bool)

        # Prepare result matrix
        assignments = np.zeros((n_facilities, n_customers), dtype=np.float64)

        # Precompute sorted columns (facilities) for each customer by cost
        sorted_facilities = np.argsort(c, axis=0)

        total_obj = 0.0

        for j in range(n_customers):
            assigned = False
            for i in sorted_facilities[:, j]:
                # If facility not open yet, try opening it if capacity permits
                if not opened[i]:
                    if remaining[i] >= demands[j]:
                        opened[i] = True
                        total_obj += fixed_costs[i]
                        remaining[i] -= demands[j]
                        assignments[i, j] = 1.0
                        total_obj += c[i, j] * demands[j]
                        assigned = True
                        break
                    else:
                        # cannot open this facility for this demand
                        continue
                else:
                    # already open, check capacity
                    if remaining[i] >= demands[j]:
                        remaining[i] -= demands[j]
                        assignments[i, j] = 1.0
                        total_obj += c[i, j] * demands[j]
                        assigned = True
                        break
                # if not enough capacity, continue to next facility
            if not assigned:
                # fallback: assign to most expensive open facility (should not happen)
                # but to keep algorithm safe, assign to any facility with capacity
                for i in range(n_facilities):
                    if remaining[i] >= demands[j]:
                        if not opened[i]:
                            opened[i] = True
                            total_obj += fixed_costs[i]
                        remaining[i] -= demands[j]
                        assignments[i, j] = 1.0
                        total_obj += c[i, j] * demands[j]
                        break

        # Convert boolean status and assignments to list format
        facility_status = opened.tolist()
        assignments_list = assignments.tolist()

        return {
            'objective_value': float(total_obj),
            'facility_status': facility_status,
            'assignments': assignments_list
        }