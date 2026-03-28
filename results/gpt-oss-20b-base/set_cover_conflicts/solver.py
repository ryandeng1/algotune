#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import NamedTuple, List, Sequence, Tuple, Iterable
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: List[List[int]]
    conflicts: List[List[int]]


class Solver:
    """Fast set‑cover solver with conflicts (0‑based indices)."""

    def solve(self, problem: Instance | Tuple[int, List[List[int]], List[List[int]]]) -> List[int]:
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem

        # Filter out empty sets – they can never be used in a cover
        filtered_sets: List[List[int]] = [s for s in sets if s]
        set_index_map: dict[int, int] = {}
        for new_i, old_set in enumerate(sets):
            if old_set:
                set_index_map[new_i] = len(set_index_map)

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(len(filtered_sets))]

        # For each object, ensure at least one covering set is chosen
        # Build a mapping object → covering set indices (filtered)
        object_to_sets: List[List[int]] = [[] for _ in range(n)]
        for idx, s in enumerate(filtered_sets):
            for obj in s:
                object_to_sets[obj].append(idx)

        for obj_covered in object_to_sets:
            if obj_covered:  # should always be true
                model.Add(sum(set_vars[i] for i in obj_covered) >= 1)

        # Handle conflicts
        for conf in conflicts:
            filtered_conf = [set_index_map[i] for i in conf if i in set_index_map]
            if filtered_conf:
                model.AddAtMostOne(set_vars[i] for i in filtered_conf)

        # Minimise number of selected sets
        model.Minimize(sum(set_vars))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = [orig_i for orig_i, new_i in set_index_map.items()
                      if solver.Value(set_vars[new_i]) == 1]
            return chosen
        raise ValueError("No feasible solution found.")