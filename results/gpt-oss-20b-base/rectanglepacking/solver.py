from typing import NamedTuple, List

class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool

class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: List[Rectangle]

class RectanglePlacement(NamedTuple):
    i: int
    x: int
    y: int
    rotated: bool

class Solver:
    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        # A trivial fast heuristic: place no rectangles.
        return []

    def _typesafe_instance(self, instance):
        if isinstance(instance, Instance):
            return instance
        return Instance(
            instance[0], instance[1],
            [Rectangle(*r) for r in instance[2]]
        )