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
        if not isinstance(problem, Instance):
            problem = Instance(problem[0], problem[1], [Rectangle(*r) for r in problem[2]])

        Cw, Ch, rects = problem.container_width, problem.container_height, problem.rectangles
        placements = []
        used = []

        # Simple row‑major packing, no rotation
        y = 0
        row_height = 0
        x = 0
        for idx, rect in enumerate(rects):
            w, h = rect.width, rect.height
            if x + w > Cw:
                # start new row
                y += row_height
                if y + h > Ch:
                    break
                x = 0
                row_height = h
            else:
                row_height = max(row_height, h)

            placements.append(RectanglePlacement(idx, x, y, False))
            used.append((x, y, x + w, y + h))
            x += w

        return placements