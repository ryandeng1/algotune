from typing import Any, NamedTuple, Tuple, List

class Instance(NamedTuple):
    n: int
    sets: List[List[int]]
    conflicts: List[List[int]]

class Solver:
    def __init__(self):
        # Nothing to precompute globally
        pass

    def solve(self, problem: Instance | Tuple) -> List[int]:
        if not isinstance(problem, Instance):
            problem = Instance(*problem)
        n, sets, conflicts = problem
        m = len(sets)

        # Convert sets to sorted lists for deterministic ordering
        sets = [sorted(s) for s in sets]

        # Build coverage bitmask for each set (bits 0..n-1)
        obj_mask = [0] * m
        for i, s in enumerate(sets):
            mask = 0
            for obj in s:
                mask |= 1 << obj
            obj_mask[i] = mask

        # Build conflict mask for each set (bits 0..m-1)
        conflict_mask = [0] * m
        for idx in range(m):
            conflict_mask[idx] |= 1 << idx  # self conflicts
        for c in conflicts:
            for i in c:
                for j in c:
                    conflict_mask[i] |= 1 << j

        # Precompute which sets cover each object
        obj_to_sets = [[] for _ in range(n)]
        for i, mask in enumerate(obj_mask):
            for obj in range(n):
                if mask >> obj & 1:
                    obj_to_sets[obj].append(i)

        # Order objects by fewest covering sets to speed up pruning
        order = sorted(range(n), key=lambda o: len(obj_to_sets[o]))

        best_solution: List[int] | None = None
        best_count = m + 1
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dfs(uncovered: int, chosen_mask: int, chosen_list: Tuple[int, ...]) -> None:
            nonlocal best_solution, best_count
            # Prune if already worse than best found
            if len(chosen_list) >= best_count:
                return
            if uncovered == 0:
                # All objects covered
                best_solution = list(chosen_list)
                best_count = len(chosen_list)
                return
            # Choose next uncovered object
            # Find lowest set bit in uncovered
            next_obj = (uncovered & -uncovered).bit_length() - 1
            # For each set covering it
            for si in obj_to_sets[next_obj]:
                # Skip if conflicts with already chosen
                if chosen_mask & conflict_mask[si]:
                    continue
                # New chosen mask
                new_chosen_mask = chosen_mask | (1 << si)
                # New uncovered mask
                new_uncovered = uncovered & ~obj_mask[si]
                dfs(new_uncovered, new_chosen_mask, chosen_list + (si,))
                # If optimal found (size 1) we can stop early
                if best_solution is not None and best_count == 1:
                    return

        ALL_UNCOVERED = (1 << n) - 1
        dfs(ALL_UNCOVERED, 0, tuple())
        if best_solution is None:
            raise ValueError("No feasible solution found.")
        return best_solution