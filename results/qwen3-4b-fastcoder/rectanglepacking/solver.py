from typing import Any
import itertools
from typing import NamedTuple
from ortools.sat.python import cp_model


class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool


class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: list[Rectangle]


class RectanglePlacement(NamedTuple):
    i: int
    x: int
    y: int
    rotated: bool


class Solver:
    def solve(self, problem: Instance) -> list[RectanglePlacement]:
        problem = self._typesafe_instance(problem)
        from ortools.sat.python import cp_model

        class RectangleKnapsackWithRotationsModel:
            def __init__(self, instance: Instance):
                self.instance = instance
                self.model = cp_model.CpModel()
                n = len(instance.rectangles)

                self.bottom_left_x_vars = [
                    self.model.new_int_var(0, instance.container_width, f"x1_{i}")
                    for i in range(n)
                ]
                self.bottom_left_y_vars = [
                    self.model.new_int_var(0, instance.container_height, f"y1_{i}")
                    for i in range(n)
                ]
                self.rotated_vars = [
                    self.model.new_bool_var(f"rotated_{i}") for i in range(n)
                ]
                self.placed_vars = [
                    self.model.new_bool_var(f"placed_{i}") for i in range(n)
                ]

                for i in range(n):
                    self.model.add(
                        self.bottom_left_x_vars[i] == 0
                    ).only_enforce_if(self.placed_vars[i].Not())
                    self.model.add(
                        self.bottom_left_y_vars[i] == 0
                    ).only_enforce_if(self.placed_vars[i].Not())

                for i, rect in enumerate(instance.rectangles):
                    if not rect.rotatable:
                        self.model.add(self.rotated_vars[i] == 0)

                for i in range(n):
                    for j in range(i + 1, n):
                        x2_i = self.bottom_left_x_vars[i] + (
                            self.instance.rectangles[i].width * (1 - self.rotated_vars[i]) +
                            self.instance.rectangles[i].height * self.rotated_vars[i]
                        )
                        y2_i = self.bottom_left_y_vars[i] + (
                            self.instance.rectangles[i].height * (1 - self.rotated_vars[i]) +
                            self.instance.rectangles[i].width * self.rotated_vars[i]
                        )
                        x2_j = self.bottom_left_x_vars[j] + (
                            self.instance.rectangles[j].width * (1 - self.rotated_vars[j]) +
                            self.instance.rectangles[j].height * self.rotated_vars[j]
                        )
                        y2_j = self.bottom_left_y_vars[j] + (
                            self.instance.rectangles[j].height * (1 - self.rotated_vars[j]) +
                            self.instance.rectangles[j].width * self.rotated_vars[j]
                        )

                        b_i_left_of_j = self.model.new_bool_var(f"{i}_left_of_{j}")
                        self.model.add(x2_i <= self.bottom_left_x_vars[j]).only_enforce_if(
                            [self.placed_vars[i], self.placed_vars[j], b_i_left_of_j]
                        )

                        b_i_right_of_j = self.model.new_bool_var(f"{i}_right_of_{j}")
                        self.model.add(x2_j <= self.bottom_left_x_vars[i]).only_enforce_if(
                            [self.placed_vars[i], self.placed_vars[j], b_i_right_of_j]
                        )

                        b_i_below_j = self.model.new_bool_var(f"{i}_below_{j}")
                        self.model.add(y2_i <= self.bottom_left_y_vars[j]).only_enforce_if(
                            [self.placed_vars[i], self.placed_vars[j], b_i_below_j]
                        )

                        b_i_above_j = self.model.new_bool_var(f"{i}_above_{j}")
                        self.model.add(y2_j <= self.bottom_left_y_vars[i]).only_enforce_if(
                            [self.placed_vars[i], self.placed_vars[j], b_i_above_j]
                        )

                        self.model.add(
                            b_i_left_of_j + b_i_right_of_j + b_i_below_j + b_i_above_j >= 1
                        ).only_enforce_if([self.placed_vars[i], self.placed_vars[j]])

            def _extract_solution(self, solver: cp_model.CpSolver) -> list[RectanglePlacement]:
                solution = []
                for i in range(len(self.instance.rectangles)):
                    if solver.Value(self.placed_vars[i]):
                        x = solver.Value(self.bottom_left_x_vars[i])
                        y = solver.Value(self.bottom_left_y_vars[i])
                        rotated = solver.Value(self.rotated_vars[i]) == 1
                        solution.append(RectanglePlacement(i, x, y, rotated))
                return solution

            def solve(self, time_limit: float = 900.0):
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = time_limit
                solver.parameters.log_search_progress = True
                status = solver.Solve(self.model)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    return self._extract_solution(solver)
                return []

        model = RectangleKnapsackWithRotationsModel(problem)
        return model.solve()