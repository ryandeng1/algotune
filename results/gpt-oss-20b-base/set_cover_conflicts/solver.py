import logging
from typing import Any, List, Tuple

# We will use pulp which is available in the environment and is faster for many
# small to medium sized problems compared to CP-SAT for pure ILP tasks.
import pulp


class Solver:
    """
    Solver for set cover with conflicts.

    Parameters
    ----------
    None

    Methods
    -------
    solve(problem, **kwargs)
        Solves the problem and returns a list of selected set indices.
    """

    def solve(self, problem: Tuple[int, List[List[int]], List[List[int]]], **kwargs) -> List[int]:
        """
        Solve the set cover with conflicts problem.

        Parameters
        ----------
        problem : tuple
            A tuple (n, sets, conflicts) where:
                - n is the number of objects
                - sets is a list of sets (each set is a list of integers)
                - conflicts is a list of conflicts (each conflict is a list of set indices)

        Returns
        -------
        list[int]
            A list of set indices that form a valid cover with minimal size.

        Notes
        -----
        The implementation uses the PuLP library to formulate and solve the ILP.
        """
        # Unpack the problem.
        n, sets, conflicts = problem

        # Number of sets.
        m = len(sets)

        # Create binary decision variables: x_i = 1 if set i is selected.
        x = pulp.LpVariable.dicts("x", range(m), cat="Binary")

        # Create the problem instance.
        prob = pulp.LpProblem("SetCoverWithConflicts", pulp.LpMinimize)

        # Objective: minimize the number of selected sets.
        prob += pulp.lpSum(x[i] for i in range(m))

        # Coverage constraints: each object must be covered at least once.
        # We build a dictionary from object to the sets that contain it.
        obj_to_sets = {obj: [] for obj in range(n)}
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(i)

        for obj, idx_list in obj_to_sets.items():
            # If a object has no sets it would violate problem statement,
            # but we still keep the constraint for safety.
            prob += pulp.lpSum(x[i] for i in idx_list) >= 1, f"cover_{obj}"

        # Conflict constraints: at most one set per conflict group.
        for conflict_idx, conflict_sets in enumerate(conflicts):
            prob += pulp.lpSum(x[i] for i in conflict_sets) <= 1, f"conflict_{conflict_idx}"

        # Solve the problem using the default CBC solver.
        # Setting a small relative gap to speed up.
        solver = pulp.PULP_CBC_CMD(msg=False, gapRel=0.0, timeLimit=kwargs.get("time_limit", 0))
        prob.solve(solver)

        # Check the solver status.
        if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            raise ValueError(f"Solver did not find a feasible solution: {pulp.LpStatus[prob.status]}")

        # Extract the indices of the selected sets.
        solution = [i for i in range(m) if pulp.value(x[i]) > 0.5]
        return solution
