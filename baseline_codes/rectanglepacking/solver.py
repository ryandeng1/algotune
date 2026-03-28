from typing import Any
import itertools
from typing import NamedTuple


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
                self.bottom_left_x_vars = [self.model.new_int_var(0, instance.container_width, name=f'x1_{i}') for i, box in enumerate(instance.rectangles)]
                self.bottom_left_y_vars = [self.model.new_int_var(0, instance.container_height, name=f'y1_{i}') for i, box in enumerate(instance.rectangles)]
                self.upper_right_x_vars = [self.model.new_int_var(0, instance.container_width, name=f'x2_{i}') for i, box in enumerate(instance.rectangles)]
                self.upper_right_y_vars = [self.model.new_int_var(0, instance.container_height, name=f'y2_{i}') for i, box in enumerate(instance.rectangles)]
                self.rotated_vars = [self.model.new_bool_var(f'rotated_{i}') for i in range(len(instance.rectangles))]
                self.placed_vars = [self.model.new_bool_var(f'placed_{i}') for i in range(len(instance.rectangles))]
                for i, rect in enumerate(instance.rectangles):
                    if rect.rotatable:
                        self.model.add(self.upper_right_x_vars[i] == self.bottom_left_x_vars[i] + rect.width).only_enforce_if([self.placed_vars[i], self.rotated_vars[i].Not()])
                        self.model.add(self.upper_right_y_vars[i] == self.bottom_left_y_vars[i] + rect.height).only_enforce_if([self.placed_vars[i], self.rotated_vars[i].Not()])
                        self.model.add(self.upper_right_x_vars[i] == self.bottom_left_x_vars[i] + rect.height).only_enforce_if([self.placed_vars[i], self.rotated_vars[i]])
                        self.model.add(self.upper_right_y_vars[i] == self.bottom_left_y_vars[i] + rect.width).only_enforce_if([self.placed_vars[i], self.rotated_vars[i]])
                    else:
                        self.model.add(self.upper_right_x_vars[i] == self.bottom_left_x_vars[i] + rect.width).only_enforce_if(self.placed_vars[i])
                        self.model.add(self.upper_right_y_vars[i] == self.bottom_left_y_vars[i] + rect.height).only_enforce_if(self.placed_vars[i])
                        self.model.add(self.rotated_vars[i] == 0)
                    self.model.add(self.bottom_left_x_vars[i] == 0).only_enforce_if(self.placed_vars[i].Not())
                    self.model.add(self.bottom_left_y_vars[i] == 0).only_enforce_if(self.placed_vars[i].Not())
                    self.model.add(self.upper_right_x_vars[i] == 0).only_enforce_if(self.placed_vars[i].Not())
                    self.model.add(self.upper_right_y_vars[i] == 0).only_enforce_if(self.placed_vars[i].Not())
                for i, j in itertools.combinations(range(len(instance.rectangles)), 2):
                    b_i_left_of_j = self.model.new_bool_var(f'{i}_left_of_{j}')
                    self.model.add(self.upper_right_x_vars[i] <= self.bottom_left_x_vars[j]).only_enforce_if([self.placed_vars[i], self.placed_vars[j], b_i_left_of_j])
                    b_i_right_of_j = self.model.new_bool_var(f'{i}_right_of_{j}')
                    self.model.add(self.bottom_left_x_vars[i] >= self.upper_right_x_vars[j]).only_enforce_if([self.placed_vars[i], self.placed_vars[j], b_i_right_of_j])
                    b_i_below_j = self.model.new_bool_var(f'{i}_below_{j}')
                    self.model.add(self.upper_right_y_vars[i] <= self.bottom_left_y_vars[j]).only_enforce_if([self.placed_vars[i], self.placed_vars[j], b_i_below_j])
                    b_i_above_j = self.model.new_bool_var(f'{i}_above_{j}')
                    self.model.add(self.bottom_left_y_vars[i] >= self.upper_right_y_vars[j]).only_enforce_if([self.placed_vars[i], self.placed_vars[j], b_i_above_j])
                    self.model.add(b_i_left_of_j + b_i_right_of_j + b_i_below_j + b_i_above_j >= 1).only_enforce_if([self.placed_vars[i], self.placed_vars[j]])
                self.model.maximize(sum(self.placed_vars))

            def _extract_solution(self, solver: cp_model.CpSolver) -> list[RectanglePlacement]:
                """Extract the solution from the solver."""
                solution = []
                for i in range(len(self.instance.rectangles)):
                    if solver.Value(self.placed_vars[i]):
                        x = solver.Value(self.bottom_left_x_vars[i])
                        y = solver.Value(self.bottom_left_y_vars[i])
                        rotated = solver.Value(self.rotated_vars[i]) == 1
                        solution.append(RectanglePlacement(i, x, y, rotated))
                return solution

            def solve(self, time_limit: float=900.0):
                """Solve the model and return the solution."""
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = time_limit
                solver.parameters.log_search_progress = True
                status = solver.Solve(self.model)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    return self._extract_solution(solver)
                return []
        model = RectangleKnapsackWithRotationsModel(problem)
        return model.solve()

    def _typesafe_instance(self, instance) -> Instance:
        if isinstance(instance, Instance):
            return instance
        else:
            pass
        return Instance(instance[0], instance[1], [Rectangle(*r) for r in instance[2]])
